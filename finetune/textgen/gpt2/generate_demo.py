import sys
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = sys.argv[1]
device = torch.device("cuda")

# load tokenizer
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# load model
print("Loading model ...")
model = AutoModelForCausalLM.from_pretrained(sys.argv[1]).to(device)

# run model
print("Generating text ...")
prompt = sys.argv[2]
prompt_w_start = f"{prompt}<|startoftext|>"
encoding = tokenizer.encode(prompt_w_start, return_tensors='pt').to(device)
generated_ids = model.generate(encoding, max_new_tokens=100, eos_token_id=28895)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Input: {prompt}")
print(f"Output: {generated_text[len(prompt):]}")
