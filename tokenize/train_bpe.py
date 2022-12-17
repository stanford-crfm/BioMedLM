import json
import os
import sys
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

input_files = sys.argv[1].split(",")
tokenizer_name = sys.argv[2]
os.system(f"mkdir {tokenizer_name}")

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())

# Customize pre-tokenization and decoding
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

# And then train
trainer = trainers.BpeTrainer(
    vocab_size=28896,
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)
tokenizer.train(input_files,trainer=trainer)

# And Save it
tokenizer.save(f"{tokenizer_name}/tokenizer.json", pretty=True)

# create vocab.json and merges.txt
with open(f"{tokenizer_name}/vocab.json", "w") as vocab_file:
    vocab_json = json.loads(open(f"{tokenizer_name}/tokenizer.json").read())["model"]["vocab"]
    vocab_file.write(json.dumps(vocab_json))

with open(f"{tokenizer_name}/merges.txt", "w") as merges_file:
    merges = "\n".join(json.loads(open(f"{tokenizer_name}/tokenizer.json").read())["model"]["merges"])
    merges_file.write(merges)
