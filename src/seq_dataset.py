import torch
from pathlib import Path
from datasets import load_dataset
from transformers import GPT2Tokenizer
import multiprocessing as mp


def get_json_dataset(input_dir, cache_dir, cpu_count=None):
    input_dir = Path(input_dir)
    cache_dir = str(Path(cache_dir).absolute())
    assert input_dir.exists()
    files = list(str(f.absolute()) for f in input_dir.glob('*.json'))

    block_size_10MB = 10 << 20
    ds = load_dataset('json', data_files=files, field='data', block_size=block_size_10MB,
                      keep_in_memory=False, cache_dir=cache_dir, streaming=False)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    max_len = tokenizer.model_max_length
    bos = tokenizer.special_tokens_map["bos_token"]  # `<|endoftext|>`
    sep = tokenizer.special_tokens_map["eos_token"]  # `<|endoftext|>`

    def tokenize_function(examples):
        samples = {'input_ids': [], 'seq_lengths': []}

        def divide_chunks(l, n):
            return [l[i: i + n] for i in range(0, len(l), n)]

        cls_id, sep_id = tokenizer.bos_token_id, tokenizer.eos_token_id

        for article in examples['text']:
            if article[-2:] == ' .':
                article = article[:-2] + '.'

            article = f"{bos} {article.strip()} {sep}"
            seq_ = tokenizer.encode(article, add_special_tokens=False)
            len_ = len(seq_)

            # assert (seq_[0] == cls_id) and (seq_[-1] == sep_id), seq_
            if len_ <= 11:
                continue
            elif len_ <= max_len:
                samples['input_ids'].append(seq_)
                samples['seq_lengths'].append(len_)
            else:
                sub_seqs = []
                for sub_s in divide_chunks(seq_, max_len - 2):
                    if sub_s[0] != cls_id:
                        sub_s = [cls_id] + sub_s
                    if sub_s[-1] != sep_id:
                        sub_s = sub_s + [sep_id]
                    # assert len(sub_s) <= max_len
                    # assert (sub_s[0] == cls_id) and (sub_s[-1] == sep_id), sub_s
                    if len(sub_s) > 11:
                        sub_seqs.append(sub_s)

                samples['input_ids'].extend(sub_seqs)
                samples['seq_lengths'].extend([len(l) for l in sub_seqs])
        return samples

    tds = ds.map(tokenize_function,
                 batched=True,
                 batch_size=10,
                 num_proc=cpu_count or mp.cpu_count(),
                 remove_columns=['text'])

    lengths = [s['seq_lengths'] for s in tds['train']]
    ds = tds['train']
    ds.lengths = lengths
    return ds


def batch_sequences_collate_fn(batch):
    """
    Do the padding and transform into torch.tensor.
    """
    token_ids = [t['input_ids'] for t in batch]
    lengths = [t['seq_lengths'] for t in batch]
    # assert len(token_ids) == len(lengths)

    # Max for paddings
    max_seq_len_ = max(lengths)
    pad_idx = 50256  # GPT2 Tokenizer specific
    tk_ = [t + [pad_idx] * (max_seq_len_ - len(t)) for t in token_ids]
    # assert len(tk_) == len(token_ids)
    # assert all(len(t) == max_seq_len_ for t in tk_)

    tk_t = torch.tensor(tk_)  # (bs, max_seq_len_)
    lg_t = torch.tensor(lengths)  # (bs)
    return tk_t, lg_t


# if __name__ == '__main__':
#     ds = get_json_dataset('data/se/', 'data/se/.cache')
#
#     dl = DataLoader(ds, batch_size=32, collate_fn=batch_sequences_collate_fn)
#     for batch in dl:
#         doSomething = 1
