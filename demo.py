import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda")

tokenizer = GPT2Tokenizer.from_pretrained("stanford-crfm/pubmed_gpt_tokenizer")

model = GPT2LMHeadModel.from_pretrained("stanford-crfm/pubmedgpt").to(device)

input_ids = tokenizer.encode(
    "Photosynthesis is ", return_tensors="pt"
).to(device)

sample_output = model.generate(input_ids, do_sample=True, max_length=50, top_k=50)

print("Output:\n" + 100 * "-")
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
