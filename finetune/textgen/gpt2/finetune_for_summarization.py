import torch
from typing import Optional
from dataclasses import dataclass, field
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
    GPT2LMHeadModel,
    AutoModelForCausalLM,
)

from sum_data_collator import DataCollatorForSumLanguageModeling
from sum_dataset import LineByLineSumTextDataset

import torch.distributed as dist

import json

import sys

sys.path.insert(0, "../..")
from utils.hf_flash_gpt_2 import GPT2FlashLMHeadModel

@dataclass
class ModelArguments:
    """
    Arguments for the model
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Leave None if you want to train a model from"
                " scratch."
            )
        },
    )

    tokenizer_name: Optional[str] = field(
        default="gpt2", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )


@dataclass
class DataArguments:
    """
    Arguments for data
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_source_length: Optional[int] = field(
        default=510, metadata={"help": "the max source length of summarization data. "}
    )
    train_max_target_length: Optional[int] = field(
        default=510, metadata={"help": "the max target length for training data. "}
    )
    eval_max_target_length: Optional[int] = field(
        default=510, metadata={"help": "the max target length for dev data. "}
    )
    seq_prefix: Optional[str] = field(
        default="",
        metadata={"help": "A string to begin every sequence with."},
    )
    no_sep: bool = field(
        default=False, metadata={"help": "Don't use a separator token."}
    )
    block_size: int = field(
        default=-1,
        metadata={
            "help": (
                "Optional input sequence length after tokenization."
                "The training dataset will be truncated in block of this size for training."
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )


def get_dataset(
    args: DataArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
    training_args: TrainingArguments = None,
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    max_source_length = args.max_source_length
    max_target_length = args.train_max_target_length if not evaluate else args.eval_max_target_length
    dataset = LineByLineSumTextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=1024,
        bos_tok=tokenizer.bos_token,
        eos_tok=tokenizer.eos_token,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        seq_prefix=args.seq_prefix,
        no_sep=args.no_sep
    )

    return dataset


def finetune():
    # parse args
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # set seed
    set_seed(training_args.seed)
    # set up model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    print(config)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )
    # set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    # add extra pad token
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})
    tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    embedding_layer = model.resize_token_embeddings(len(tokenizer))
    # set up data collator
    data_collator = DataCollatorForSumLanguageModeling(tokenizer=tokenizer)
    # set up data sets
    train_dataset = get_dataset(data_args, tokenizer=tokenizer, training_args=training_args)
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
    # set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    # launch fine tuning
    trainer.train()
    # save final model
    trainer.save_model()
    trainer.save_state()

if __name__ == "__main__":
    finetune()
