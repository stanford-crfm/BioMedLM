import torch

from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

@dataclass
class DataCollatorForSumLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizer
    mlm: bool = False
    format_mode: str = 'cat'
    mlm_probability: float = 0.15

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        # print(examples[0])
        # print(len(examples))
        input_ids, labels, src, tgt = zip(*examples)
        # print(len(input_ids), len(labels), len(weights))
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:

            # print(self.format_mode)

            if self.format_mode == 'peek' or self.format_mode == 'cat':
                mode_input = 1
            elif self.format_mode == 'nopeek':
                assert False, 'should use format_mode = peek or cat.'
                mode_input = 2
            elif self.format_mode == 'infix':
                assert False, 'should use format_mode = peek or cat.'
                mode_input = 4

            # mode_input = 1 # means that we take the input again.
            # mode_input = 2 # means that we do not peek at src again.
            # mode_input = 3 # means that we look at the categories, and see the input again.

            # print(self.format_mode, mode_input)

            if mode_input == 1:
                # input, batch
                batch = self._tensorize_batch(input_ids)
                labels = self._tensorize_batch(labels)
                src = self._tensorize_batch(src)

            labels[labels == self.tokenizer.pad_token_id] = -100 # tgt
            src_attn = (src != self.tokenizer.pad_token_id) # src
            tgt_attn = (batch != self.tokenizer.pad_token_id) # tgt

            return {"input_ids": batch, "labels": labels}


    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)


@dataclass
class DataCollatorForSumBatchGenLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    format_mode: str = 'cat'
    mlm_probability: float = 0.15
    max_source_length: int = 512
    max_target_length: int = 100


    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        # print(examples[0])
        # print(len(examples))

        mode_gen = 1

        if mode_gen == 0:
            input_ids, labels, src, tgt = zip(*examples)
            # print(len(input_ids), len(labels), len(weights))



            src = self._tensorize_batch(src) #src
            tgt = self._tensorize_batch(tgt)  # src

            src_attn = (src != self.tokenizer.pad_token_id) # src
            tgt_attn = (batch != self.tokenizer.pad_token_id) # tgt

            return {"input_ids": src, "labels": tgt, 'src_attn': src_attn, 'tgt_attn':tgt_attn,
                    'src':src}

        else:
            src, tgt = zip(*examples)
            bsz = len(src)
            self.tokenizer.padding_side = "left"
            src = self.tokenizer(src, return_tensors="pt", padding=True, truncation=True, max_length=self.max_source_length)
            tgt = self.tokenizer(tgt, return_tensors="pt", padding=True, truncation=True, max_length=self.max_target_length)
            bos_seq = torch.ones(bsz, 1).fill_(self.tokenizer.bos_token_id).long()
            src_input_ids = torch.cat([src['input_ids'], bos_seq], dim=-1)
            bos_mask = torch.ones(bsz, 1).long()
            src_mask = torch.cat([src["attention_mask"], bos_mask],dim=-1)

            return {"input_ids": src_input_ids, "labels": tgt['input_ids'], 'src_attn': src_mask,
                    'tgt_attn': tgt["attention_mask"]}




    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

