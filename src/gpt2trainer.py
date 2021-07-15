# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" The distiller to distil the student.
    Adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import math
import os
import time
from pathlib import Path

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from tqdm import tqdm

from .grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

import logging

logger = logging.getLogger(__name__)


class GPT2Trainer:
    def __init__(
            self, params: dict, dataset, collate_fn, student: nn.Module, teacher: nn.Module
    ):
        logger.info("Initializing Trainer")
        self.params = params
        self.dump_path = params.output_dir
        self.multi_gpu = params.multi_gpu
        self.fp16 = params.fp16

        self.student = student
        self.teacher = teacher

        self.student_config = student.config
        self.vocab_size = student.config.vocab_size

        if params.n_gpu <= 1:
            sampler = RandomSampler(dataset)
        elif params.elastic:
            sampler = ElasticDistributedSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)

        if params.group_by_size:
            groups = create_lengths_groups(lengths=dataset.lengths, k=params.max_model_input_size)
            sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.batch_size)
        else:
            sampler = BatchSampler(sampler=sampler, batch_size=params.batch_size, drop_last=False)

        self.dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=collate_fn)

        self.temperature = params.temperature
        assert self.temperature > 0.0

        self.alpha_ce = params.alpha_ce
        self.alpha_clm = params.alpha_clm
        self.alpha_mse = params.alpha_mse
        logger.info("Using CLM loss for LM step.")

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.total_clm_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_clm = 0
        if self.alpha_mse > 0.0:
            self.last_loss_mse = 0
        self.last_log = 0
        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if self.alpha_mse > 0.0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")

        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.dataloader)
        num_train_optimization_steps = (
                int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(0.9, 0.98)
        )

        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
            self.student, self.optimizer = amp.initialize(
                self.student, self.optimizer, opt_level=self.params.fp16_opt_level
            )
            if self.teacher is not None:
                self.teacher = self.teacher.half()

        if self.multi_gpu:
            if self.fp16:
                from apex.parallel import DistributedDataParallel

                logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(self.student)
            else:
                from torch.nn.parallel import DistributedDataParallel

                logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(
                    self.student,
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    find_unused_parameters=True,
                )

        self.is_master = params.is_master
        if self.is_master:
            logger.info("--- Initializing Tensorboard")
            self.tensorboard = SummaryWriter(log_dir=os.path.join(self.dump_path, "log", "train"))
            self.tensorboard.add_text(tag="config/training", text_string=str(self.params), global_step=0)
            self.tensorboard.add_text(tag="config/student", text_string=str(self.student_config), global_step=0)

    def prepare_batch_clm(self, batch):
        """
        Prepare the batch: from the token_ids and the lengths, compute the attention mask and the labels for CLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            clm_labels: `torch.tensor(bs, seq_length)` - The causal language modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, lengths = batch
        token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]
        clm_labels = token_ids.new(token_ids.size()).copy_(token_ids)
        clm_labels[~attn_mask] = -100  # previously `clm_labels[1-attn_mask] = -1`, cf pytorch 1.2.0 compatibility

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size

        return token_ids, attn_mask, clm_labels

    def round_batch(self, x: torch.tensor, lengths: torch.tensor):
        """
        For float16 only.
        Sub-sample sentences in a batch, and add padding, so that each dimension is a multiple of 8.

        Input:
        ------
            x: `torch.tensor(bs, seq_length)` - The token ids.
            lengths: `torch.tensor(bs, seq_length)` - The lengths of each of the sequence in the batch.

        Output:
        -------
            x:  `torch.tensor(new_bs, new_seq_length)` - The updated token ids.
            lengths: `torch.tensor(new_bs, new_seq_length)` - The updated lengths.
        """
        if not self.fp16 or len(lengths) < 8:
            return x, lengths

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[idx, :slen]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(1)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            if self.mlm:
                pad_id = self.params.special_tok_ids["pad_token"]
            else:
                pad_id = self.params.special_tok_ids["unk_token"]
            padding_tensor = torch.zeros(bs2, pad, dtype=torch.long, device=x.device).fill_(pad_id)
            x = torch.cat([x, padding_tensor], 1)
            assert x.size() == (bs2, ml2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths

    def _train(self, distill=False):
        """
        The real training loop.
        """
        if self.is_master:
            logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()
        if self.teacher:
            self.teacher.eval()

        self.load_checkpoint()
        for epoch in range(self.params.n_epoch):
            if epoch < self.epoch:
                continue

            if self.is_master:
                logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch - 1}")
            if self.multi_gpu:
                torch.distributed.barrier()

            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for bid, batch in enumerate(iter_bar):
                if bid < self.n_iter:
                    # iter_bar.update()
                    continue

                if self.params.n_gpu > 0:
                    batch = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)

                token_ids, attn_mask, lm_labels = self.prepare_batch_clm(batch=batch)
                if distill:
                    assert self.teacher is not None
                    self.step(input_ids=token_ids, attention_mask=attn_mask, lm_labels=lm_labels)
                else:
                    self.step_no_teacher(input_ids=token_ids, lm_labels=lm_labels)

                # iter_bar.update()
                post_fix = {"Last_loss": f"{self.last_loss:.2f}",
                            "Avg_cum_loss": f"{self.total_loss_epoch / self.n_iter:.2f}"}
                if self.alpha_clm > 0:
                    post_fix.update({
                        "CLM_loss": f"{self.last_loss_clm:.2f}",
                        "Avg_cum_CLM": f"{self.total_clm_loss_epoch / self.n_iter:.2f}"
                    })

                iter_bar.set_postfix(post_fix)
            iter_bar.close()

            if self.is_master:
                logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch - 1}")
            self.end_epoch()

        if self.is_master:
            logger.info("Save very last checkpoint as `pytorch_model.bin`.")
            self.save_checkpoint(checkpoint_name="pytorch_model.bin")
            logger.info("Training is finished")

    def train(self):
        self._train(distill=False)

    def distill(self):
        self._train(distill=True)

    def step_no_teacher(self, input_ids: torch.tensor, lm_labels: torch.tensor):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """

        s_logits, _ = self.student(
            input_ids=input_ids, attention_mask=None
        )  # (bs, seq_length, voc_size)

        loss = 0.0

        if self.alpha_clm > 0.0:
            shift_logits = s_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_clm = self.lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss += self.alpha_clm * loss_clm

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        if self.alpha_clm > 0.0:
            self.last_loss_clm = loss_clm.item()
            self.total_clm_loss_epoch += loss_clm.item()

        self.optimize(loss)
        self.n_sequences_epoch += input_ids.size(0)

    def step(self, input_ids: torch.tensor, attention_mask: torch.tensor, lm_labels: torch.tensor):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """
        s_logits, _ = self.student(
            input_ids=input_ids, attention_mask=None
        )  # (bs, seq_length, voc_size)

        with torch.no_grad():
            t_logits, _ = self.teacher(
                input_ids=input_ids, attention_mask=None
            )  # (bs, seq_length, voc_size)
        assert s_logits.size() == t_logits.size()

        # https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
        # https://github.com/peterliht/knowledge-distillation-pytorch/issues/2

        mask = attention_mask.unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)

        s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()

        loss_ce = (
                self.ce_loss_fct(
                    F.log_softmax(s_logits_slct / self.temperature, dim=-1),
                    F.softmax(t_logits_slct / self.temperature, dim=-1),
                )
                * (self.temperature) ** 2
        )
        loss = self.alpha_ce * loss_ce
        if self.alpha_clm > 0.0:
            shift_logits = s_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_clm = self.lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss += self.alpha_clm * loss_clm

        if self.alpha_mse > 0.0:
            loss_mse = self.mse_loss_fct(s_logits_slct, t_logits_slct) / s_logits_slct.size(
                0
            )  # Reproducing batchmean reduction
            loss += self.alpha_mse * loss_mse

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.alpha_clm > 0.0:
            self.last_loss_clm = loss_clm.item()
            self.total_clm_loss_epoch += loss_clm.item()

        if self.alpha_mse > 0.0:
            self.last_loss_mse = loss_mse.item()
        self.optimize(loss)
        self.n_sequences_epoch += input_ids.size(0)

    def optimize(self, loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.multi_gpu:
            loss = loss.mean()
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        if self.fp16:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self.iter()
        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            if self.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def iter(self):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        self.n_iter += 1
        self.n_total_iter += 1

        if (self.n_total_iter - 1) % self.params.log_interval == 0:
            self.log_tensorboard()
            self.last_log = time.time()
        if (self.n_total_iter - 1) % self.params.checkpoint_interval == 0:
            self.save_checkpoint()

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """
        if not self.is_master:
            return

        for param_name, param in self.student.named_parameters():
            self.tensorboard.add_scalar(
                tag="parameter_mean/" + param_name, scalar_value=param.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="parameter_std/" + param_name, scalar_value=param.data.std(), global_step=self.n_total_iter
            )
            if param.grad is None:
                continue
            self.tensorboard.add_scalar(
                tag="grad_mean/" + param_name, scalar_value=param.grad.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="grad_std/" + param_name, scalar_value=param.grad.data.std(), global_step=self.n_total_iter
            )

        self.tensorboard.add_scalar(
            tag="losses/cum_avg_loss_epoch",
            scalar_value=self.total_loss_epoch / self.n_iter,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(tag="losses/loss", scalar_value=self.last_loss, global_step=self.n_total_iter)
        self.tensorboard.add_scalar(
            tag="losses/loss_ce", scalar_value=self.last_loss_ce, global_step=self.n_total_iter
        )
        if self.alpha_clm > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_clm", scalar_value=self.last_loss_clm, global_step=self.n_total_iter
            )
        if self.alpha_mse > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_mse", scalar_value=self.last_loss_mse, global_step=self.n_total_iter
            )
        self.tensorboard.add_scalar(
            tag="learning_rate/lr", scalar_value=self.scheduler.get_lr()[0], global_step=self.n_total_iter
        )

        self.tensorboard.add_scalar(
            tag="global/memory_usage",
            scalar_value=psutil.virtual_memory()._asdict()["used"] / 1_000_000,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(
            tag="global/speed", scalar_value=time.time() - self.last_log, global_step=self.n_total_iter
        )

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        if self.is_master:
            self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")
            self.tensorboard.add_scalar(
                tag="epoch/loss", scalar_value=self.total_loss_epoch / self.n_iter, global_step=self.epoch
            )

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0
        self.total_clm_loss_epoch = 0

    def load_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):

        checkpoint_path = os.path.join(self.dump_path, checkpoint_name)

        if not Path(checkpoint_path).exists():
            return
        logger.info('---Loading checkpoint')
        if self.multi_gpu:
            device_id = self.params.local_rank
            state = torch.load(os.path.join(self.dump_path, checkpoint_name), map_location=f"cuda:{device_id}")
        elif self.params.n_gpu_per_node == 1:
            state = torch.load(os.path.join(self.dump_path, checkpoint_name), map_location=f"cuda")
        else:
            state = torch.load(os.path.join(self.dump_path, checkpoint_name), map_location=f"cpu")

        mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
        loaded_config = mdl_to_save.config.__class__.from_dict(state['config'])
        assert mdl_to_save.config.n_head == loaded_config.n_head and mdl_to_save.config.n_layer == loaded_config.n_layer and \
               mdl_to_save.config.n_embd == loaded_config.n_embd, "Checkpoint config doesn't match specified config!"

        mdl_to_save.config = loaded_config
        mdl_to_save.load_state_dict(state['model_state_dict'])

        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.epoch = state['epoch']
        self.n_iter = state['iteration']
        self.n_total_iter = state['total_iteration']
        logger.info(f'---Resuming from epoch: {self.epoch} and iteration: {self.n_iter}')

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """

        logger.info(f'---saving checkpoint at epoch:{self.epoch} and iteration: {self.n_iter}')
        if not self.is_master:
            return
        mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
        mdl_to_save.config.save_pretrained(self.dump_path)
        state = {
            'model_state_dict': mdl_to_save.state_dict(),
            'iteration': self.n_iter,
            'total_iteration': self.n_total_iter,
            'epoch': self.epoch,
            'config': mdl_to_save.config.to_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(state, os.path.join(self.dump_path, checkpoint_name))
