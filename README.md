# BioMedLM

Code used for pre-training and fine-tuning the [BioMedLM](https://huggingface.co/stanford-crfm/pubmedgpt) model.

Note: This model was previously known as PubMedGPT, but the NIH has asked us to change the name since they hold the trademark on "PubMed", so the new name is BioMedLM!

### Links

[Blog](https://crfm.stanford.edu/2022/12/15/biomedlm.html)

[Model](https://huggingface.co/stanford-crfm/pubmedgpt/tree/main)

[MosaicML Composer](https://github.com/mosaicml/composer)

### Example Usage

```
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda")

tokenizer = GPT2Tokenizer.from_pretrained("stanford-crfm/BioMedLM")

model = GPT2LMHeadModel.from_pretrained("stanford-crfm/BioMedLM").to(device)

input_ids = tokenizer.encode(
    "Photosynthesis is ", return_tensors="pt"
).to(device)

sample_output = model.generate(input_ids, do_sample=True, max_length=50, top_k=50)

print("Output:\n" + 100 * "-")
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```
