import json
from pathlib import Path

from tqdm import tqdm
from transformers import GPT2Tokenizer

from src.data.zendesk import ZendeskTicketGen

datapath = Path('/mnt/disks/training_disk/zendesk_datasets')
cache_dir = Path('.cache')
cache_dir.mkdir(exist_ok=True)
output_path = Path('/mnt/disks/training_disk/json_files')
zips = datapath.glob('*.zip')
bos_token = ' <|endoftext|> '

output_path.mkdir(exist_ok=True)

for f in tqdm(zips):
    output_file = output_path / (f.stem + '.json')

    if output_file.exists():
        continue
    print(f)
    gen = ZendeskTicketGen(paths=f, cache_dir=cache_dir, num_workers=2)

    try:
        data = {'data': []}
        for ticket in tqdm(gen, leave=True):
            line = bos_token.join([c.text.strip() for c in ticket.comments])
            data['data'].append({'text': line})

        json.dump(data, output_file.open('w', encoding='utf8'), indent=4, ensure_ascii=False)
    except Exception:
        print(f'Skipping {f}')
        continue
