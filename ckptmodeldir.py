from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

output_path = 'models/zd_cs_gpt2_v1/model_dir'
ckpt = torch.load("models/zd_cs_gpt2_v1/checkpoint.pth", map_location="cpu")

state_dict = ckpt['model_state_dict']
config = GPT2Config.from_dict(ckpt['config'])
model = GPT2LMHeadModel(config=config)
model.load_state_dict(state_dict=state_dict, strict=True)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
doSomething = 1
