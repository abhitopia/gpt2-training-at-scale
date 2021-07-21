from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

ckpt = torch.load("models/se_gpt2_v2/checkpoint.pth", map_location="cpu")

state_dict = ckpt['model_state_dict']
config = GPT2Config.from_dict(ckpt['config'])
model = GPT2LMHeadModel(config=config)
model.load_state_dict(state_dict=state_dict, strict=True)
model.save_pretrained('models/se_base')
tokenizer.save_pretrained('models/se_base')
doSomething = 1
