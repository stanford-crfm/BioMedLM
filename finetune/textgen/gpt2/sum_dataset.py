import os
import pickle
import random
import time
import copy
import json
from typing import Dict, List, Optional
import ast
import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

from pathlib import Path
import linecache

# from transformers import BertTokenizer, BertForMaskedLM, BertModel, BertTokenizerFast
# from transformers import BertTokenizer,  BertTokenizerFast
logger = logging.get_logger(__name__)


class LineByLineSumTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str,
                 max_source_length:int, max_target_length:int, use_task_instruction:int=0, use_stream_mode:bool=True):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        self.src_file = file_path
        self.tgt_file = file_path[:-6] + 'target'
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        if use_task_instruction:
            self.instruction = "Summarize the following text: "
        else:
            self.instruction = None
        print (f'Task instruction: "{self.instruction}"')

        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        eos_idx = tokenizer(eos_tok, add_special_tokens=False)['input_ids'][0]

        self.bos_idx = separator
        self.eos_idx = eos_idx

        self.length = [len(x) for x in Path(self.tgt_file).open().readlines()]
        self.tokenizer = tokenizer

        self.use_stream_mode = use_stream_mode

        if self.use_stream_mode:
            return
        else:
            src_lines = []
            with open(self.src_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    line = self.instruction + line if self.instruction else line
                    if len(line) > 0 and not line.isspace():
                        src_lines.append(line)

                # print(len(list(f.read().splitlines())))
                # src_lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            print(len(src_lines))
            with open(self.tgt_file, encoding="utf-8") as f:
                tgt_lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

            print(self.tgt_file, len(tgt_lines), '\n', self.src_file, len(src_lines))

            assert len(tgt_lines) == len(src_lines)

            src_encoding = tokenizer(src_lines, add_special_tokens=True, truncation=True, max_length=max_source_length,
                                                                  is_split_into_words=False)['input_ids']

            tgt_encoding = tokenizer(tgt_lines, add_special_tokens=True, truncation=True, max_length=max_target_length,
                                     is_split_into_words=False)['input_ids']

            assert len(src_encoding) == len(tgt_encoding)
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            eos_idx = tokenizer(eos_tok, add_special_tokens=False)['input_ids'][0]

            edited_sents = []
            for src, tgt in zip(src_encoding, tgt_encoding):
                sent = src + [separator] + tgt + [eos_idx]
                # sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)

            # batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
            #                                                       is_split_into_words=False)

            self.examples = edited_sents

            self.labels = copy.deepcopy(self.examples)



            self.src_sent = []
            self.tgt_sent = []
            if True:
                separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
                for i, elem in enumerate(self.labels):
                    sep_idx = elem.index(separator) + 1
                    self.src_sent.append(self.examples[i][:sep_idx-1])
                    self.tgt_sent.append(self.examples[i][sep_idx-1:])
                    self.labels[i][:sep_idx] = [-100] * sep_idx


            print(self.labels[0])
            print(self.examples[0])
            print(edited_sents[0])
            print(self.src_sent[0])
            print(self.tgt_sent[0])
            # assert len(self.src_cat) == len(self.examples)




    def __len__(self):
        return len(self.length)


    def __getitem__(self, i):
        if not self.use_stream_mode:
            return (torch.tensor(self.examples[i], dtype=torch.long),
                    torch.tensor(self.labels[i], dtype=torch.long),
                    torch.tensor(self.src_sent[i], dtype=torch.long),
                    torch.tensor(self.tgt_sent[i], dtype=torch.long),
                    )
        else:
            index = i + 1  # linecache starts at 1
            source_line = linecache.getline(str(self.src_file), index).rstrip("\n")
            tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
            assert source_line, f"empty source line for index {index}"
            assert tgt_line, f"empty tgt line for index {index}"

            source_line = self.instruction + source_line if self.instruction else source_line

            src = self.tokenizer(source_line, add_special_tokens=True, truncation=True, max_length=self.max_source_length,
                                     is_split_into_words=False)['input_ids']

            tgt = self.tokenizer(tgt_line, add_special_tokens=True, truncation=True, max_length=self.max_target_length,
                                     is_split_into_words=False)['input_ids']

            sent = src + [self.bos_idx] + tgt + [self.eos_idx]

            sep_idx = sent.index(self.bos_idx) + 1

            label = copy.deepcopy(sent)
            label[:sep_idx] = [-100] * sep_idx
            src_sent = sent[:sep_idx - 1]
            tgt_sent = sent[sep_idx - 1:]

            return (torch.tensor(sent, dtype=torch.long),
                    torch.tensor(label, dtype=torch.long),
                    torch.tensor(src_sent, dtype=torch.long),
                    torch.tensor(tgt_sent, dtype=torch.long),
                    )


class LineByLineSumBatchGenTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str,
                 max_source_length:int, max_target_length:int, use_task_instruction:int=0):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        self.src_file = file_path
        self.tgt_file = file_path[:-6] + 'target'
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        if use_task_instruction:
            self.instruction = "Summarize the following text: "
        else:
            self.instruction = None
        print (f'Task instruction: "{self.instruction}"')

        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        eos_tok = "[SEP]"
        eos_idx = tokenizer(eos_tok, add_special_tokens=False)['input_ids'][0]

        self.bos_idx = separator
        self.eos_idx = eos_idx

        tokenizer.pad_token = "[PAD]"
        tokenizer.pad_token_id = 28896

        self.length = [len(x) for x in Path(self.tgt_file).open().readlines()]
        self.tokenizer = tokenizer
        return




    def __len__(self):
        return len(self.length)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        # return (torch.tensor(self.examples[i], dtype=torch.long),
        #         torch.tensor(self.labels[i], dtype=torch.long),
        #         torch.tensor(self.src_sent[i], dtype=torch.long),
        #         torch.tensor(self.tgt_sent[i], dtype=torch.long),
        #         )

        modegen = 1
        index = i + 1  # linecache starts at 1
        source_line = linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        source_line = self.instruction + source_line if self.instruction else source_line

        if modegen == 0:

            src = self.tokenizer(source_line, add_special_tokens=True, truncation=True, max_length=self.max_source_length,
                                     is_split_into_words=False)['input_ids']

            tgt = self.tokenizer(tgt_line, add_special_tokens=True, truncation=True, max_length=self.max_target_length,
                                     is_split_into_words=False)['input_ids']

            sent = src + [self.bos_idx] + tgt + [self.eos_idx]

            sep_idx = sent.index(self.bos_idx) + 1

            label = copy.deepcopy(sent)
            label[:sep_idx] = [-100] * sep_idx
            src_sent = sent[:sep_idx - 1]
            tgt_sent = sent[sep_idx - 1:]

            return (torch.tensor(sent, dtype=torch.long),
                    torch.tensor(label, dtype=torch.long),
                    )

        else:
            return (source_line, tgt_line)

