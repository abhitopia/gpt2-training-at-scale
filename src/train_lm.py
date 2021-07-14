import json
import os
import socket
import pprint
import git
import torch
from pathlib import Path

import click
import logging
from box import Box
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

from src.gpt2trainer import GPT2Trainer

from src.seq_dataset import get_json_dataset, batch_sequences_collate_fn

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def sanity_checks(args):
    """
    A bunch of args sanity checks to perform even starting...
    """
    assert args.alpha_clm > 0.0
    assert args.teacher_type == args.student_type

    # assert os.path.isfile(args.student_config)
    # if args.student_pretrained_weights is not None:
    #     assert os.path.isfile(args.student_pretrained_weights)

    if args.teacher is None:
        assert args.alpha_ce == 0.0 and args.alpha_mse == 0.0 and args.alpha_clm == 1.0

    assert args.alpha_ce >= 0.0
    assert args.alpha_clm >= 0.0
    assert args.alpha_mse >= 0.0
    assert args.alpha_ce + args.alpha_clm + args.alpha_mse > 0.0


def git_log(folder_path: str):
    """
    Log commit info.
    """
    repo = git.Repo(search_parent_directories=True)
    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
    }

    with open(os.path.join(folder_path, "git_log.json"), "w") as f:
        json.dump(repo_infos, f, indent=4)


def init_gpu_params(params):
    """
    Handle single and multi-GPU / multi-node.
    """
    if params.n_gpu <= 0:
        params.local_rank = 0
        params.master_port = -1
        params.is_master = True
        params.multi_gpu = False
        return

    assert torch.cuda.is_available()

    logger.info("Initializing GPUs")
    if params.n_gpu > 1:
        assert params.local_rank != -1

        params.world_size = int(os.environ["WORLD_SIZE"])
        params.n_gpu_per_node = int(os.environ["N_GPU_NODE"])
        params.global_rank = int(os.environ["RANK"])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
        params.multi_gpu = True

        assert params.n_nodes == int(os.environ["N_NODES"])
        assert params.node_id == int(os.environ["NODE_RANK"])

    # local job (single GPU)
    else:
        assert params.local_rank == -1

        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1
        params.multi_gpu = False

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # define whether this is the master process / if we are in multi-node distributed mode
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1

    # summary
    PREFIX = f"--- Global rank: {params.global_rank} - "
    logger.info(PREFIX + "Number of nodes: %i" % params.n_nodes)
    logger.info(PREFIX + "Node ID        : %i" % params.node_id)
    logger.info(PREFIX + "Local rank     : %i" % params.local_rank)
    logger.info(PREFIX + "World size     : %i" % params.world_size)
    logger.info(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    logger.info(PREFIX + "Master         : %s" % str(params.is_master))
    logger.info(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    logger.info(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    logger.info(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(params.local_rank)

    # initialize multi-GPU
    if params.multi_gpu:
        logger.info("Initializing PyTorch distributed")
        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
        )


@click.command()
@click.option('--input-dir', '-i', type=click.Path(file_okay=False, exists=True, dir_okay=True), required=True)
@click.option('--output-dir', '-o', type=click.Path(dir_okay=True, file_okay=False))
@click.option('--n-embd', '-ne', type=int, help="Size of embeddings", default=1280)
@click.option('--n-head', '-nh', type=int, help="Number of heads per layer", default=20)
@click.option('--n-layer', '-nl', type=int, help="Number of layers", default=36)
@click.option('--alpha_ce', type=float, help="Coeff Cross Entropy (for distillation)", default=0.0)
@click.option('--alpha_clm', type=float, help="Coeff Language Model Loss", default=1.0)
@click.option('--alpha_mse', type=float, help="Coeff MSE (for distillation)", default=0.0)
@click.option('--temperature', type=float, help="Temperature (for distillation)", default=1.0)
@click.option('--gradient-accumulation-steps', '-gas', type=int, default=50)
@click.option("--n_epoch", type=int, default=3, help="Number of pass on the whole dataset.")
@click.option("--batch_size", type=int, default=5, help="Batch size (for each process).")
@click.option("--group_by_size", is_flag=True, default=False, help="If true, group sequences that have similar length into the same batch. Default is False.")
@click.option("--warmup_prop", default=0.05, type=float, help="Linear warmup proportion.")
@click.option("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
@click.option("--learning_rate", default=5e-4, type=float, help="The initial learning rate for Adam.")
@click.option("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
@click.option("--max_grad_norm", default=5.0, type=float, help="Max gradient norm.")
@click.option("--initializer_range", default=0.02, type=float, help="Random initialization range.")
@click.option('--teacher', type=click.Path(file_okay=True, exists=True), default=None, help='Path to teacher '
                                                                                            'checkpoint (for '
                                                                                            'distillation)')
@click.option('--pretrained-weights', type=click.Path(file_okay=True, exists=True), default=None, help='Path to '
                                                                                                       'teacher '
                                                                                                       'checkpoint')
@click.option("--n-gpu", type=int, default=0, help="Number of GPUs in the node.")
@click.option('--fp16', is_flag=True, default=False, help="Use half precision")
@click.option('--elastic', is_flag=True, default=False, help="Enable torch elastic")
@click.option("--local-rank", type=int, default=-1, help="Distributed training - Local rank")
@click.option("--seed", type=int, default=56, help="Random seed")
@click.option("--log-interval", "-li",  type=int, default=500, help="Tensorboard logging interval.")
@click.option("--checkpoint-interval", "-ci",  type=int, default=4000, help="Checkpoint interval.")
def train(**args):
    args = Box(args)
    if args.output_dir is None:
        args.output_dir = str(Path(args.input_dir) / 'checkpoints')

    Path(args.output_dir).mkdir(exist_ok=True)
    args.tokenizer_name = args.student_type = args.teacher_type = "gpt2"
    args.vocab_size = 50257
    args.fp16_opt_level = "O1"
    args.cache_dir = f"{args.input_dir}/.cache"
    Path(args.cache_dir).mkdir(exist_ok=True)
    sanity_checks(args)
    init_gpu_params(args)

    logger.info(f"Experiment will be dumped and logged in {args.output_dir}")
    # SAVE PARAMS #
    logger.info(f"Param: {pprint.pformat(args)}")
    with open(os.path.join(args.output_dir, "parameters.json"), "w") as f:
        json.dump(args.to_dict(), f, indent=4)
    git_log(args.output_dir)

    config_class, model_class, tokenizer_class = GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

    # TOKENIZER #
    tokenizer = tokenizer_class.from_pretrained(args.teacher_type)
    special_tok_ids = {}
    for tok_name, tok_symbol in tokenizer.special_tokens_map.items():
        idx = tokenizer.all_special_tokens.index(tok_symbol)
        special_tok_ids[tok_name] = tokenizer.all_special_ids[idx]
    logger.info(f"Special tokens {special_tok_ids}")
    args.special_tok_ids = special_tok_ids
    args.max_model_input_size = tokenizer.max_model_input_sizes[args.teacher_type]

    ds = get_json_dataset(args.input_dir, args.cache_dir)
    logger.info("Data loaded!")

    # STUDENT #
    student_config = GPT2Config.from_pretrained("gpt2-large")
    student_config.n_head = args.n_head
    student_config.n_layer = args.n_layer
    student_config.n_embd = args.n_embd
    student_config.output_hidden_states = False
    # Fix because now the default is return_dict=True
    logger.info(f"Loaded model config: {pprint.pformat(student_config)}")
    if args.pretrained_weights is not None:
        logger.info(f"Loading pretrained weights from {args.pretrained_weights}")
        student = model_class.from_pretrained(args.pretrained_weights, config=student_config)
    else:
        student = model_class(student_config)

    # Fix because now the default is return_dict=True
    student.config.update(dict(return_dict=False))
    if args.n_gpu > 0:
        student.to(f"cuda:{args.local_rank}")
    logger.info("Model loaded.")

    # TEACHER #
    teacher = None
    if args.teacher is not None:
        teacher = model_class.from_pretrained(args.teacher)
        if args.n_gpu > 0:
            teacher.to(f"cuda:{args.local_rank}")
        logger.info(f"Teacher loaded from {args.teacher}.")

    torch.cuda.empty_cache()
    trainer = GPT2Trainer(params=args, dataset=ds, collate_fn=batch_sequences_collate_fn, student=student,
                          teacher=teacher)
    trainer.train()


if __name__ == '__main__':
    train()
