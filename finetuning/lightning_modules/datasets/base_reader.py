import json
import logging
import sys
import os
import torch

from typing import Dict, Iterable, List, Any, Optional, Union
from itertools import chain
from tqdm import tqdm

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from finetuning.lightning_modules.models.seq2seq_model_util import is_model_gpt_style, right_pad_sequences
from finetuning.lightning_modules.models.seq2seq_model_util import get_model, left_pad_sequences

from torch.utils.data import DataLoader

# set environment variable to avoid deadlocks, see: 
# https://docs.allennlp.org/main/api/data/data_loaders/multiprocess_data_loader/#multiprocessdataloader.common_issues
os.environ['TOKENIZERS_PARALLELISM']='0'

logger = logging.getLogger(__name__)

class NL2CodeDataset(Dataset):

    def __init__(
        self, 
        file_path: str,
        transformer_model_name: str, 
        max_instances: int = sys.maxsize,
        mode: str = "train", 
        mask_context_loss: bool = False,
        multi_instance_example: bool = False,
        enable_tqdm: bool = False,
        generation_length: int = 128,
        stats_keys: List[str] = ["total_instances", "input_too_long"],
        **kwargs):
        super().__init__(**kwargs)

        # mode is one of ["train", "test", "test_few_shot"]
        assert mode in ["train", "test", "test_few_shot"]

        self.transformer_model_name = transformer_model_name
        _, self.tokenizer = get_model(transformer_model_name, tokenizer_only=True)

        self.mask_context_loss = mask_context_loss
        assert not self.mask_context_loss or is_model_gpt_style(self.transformer_model_name), \
            "mask_context_loss is only supported for GPT-style models"

        self.max_instances = max_instances
        self.mode = mode
        self.multi_instance_example = multi_instance_example
        self.enable_tqdm = enable_tqdm
        self.generation_length = generation_length

        # use to report dataset statistics
        self.stats = dict()
        for key in stats_keys:
            self.stats[key] = 0

        self.instances = self.read(file_path)

    def get_example_dict_gpt(self, example: Dict[str, Any], context: str, code: str = "", 
                         train_mode: bool = True, length_cutoff: bool = True) -> Dict[str, Any]:
        example_dict = {"metadata": example}

        if train_mode:
            tokenizer_outputs = self.tokenizer("\n".join([context, code]), truncation=length_cutoff)
            context_len = len(self.tokenizer(context + "\n", truncation=length_cutoff)["input_ids"])
            if self.mask_context_loss:
                example_dict["labels"] = [-100] * context_len + tokenizer_outputs["input_ids"][context_len:]
            else:
                example_dict["labels"] = tokenizer_outputs["input_ids"].copy()
        else:
            tokenizer_outputs = self.tokenizer(context + "\n", truncation=length_cutoff)

        example_dict["input_ids"] = tokenizer_outputs["input_ids"]
        example_dict["attention_mask"] = tokenizer_outputs["attention_mask"]

        if train_mode:
            example_dict["input_ids"] += [self.tokenizer.eos_token_id]
            example_dict["labels"] += [self.tokenizer.eos_token_id]
            example_dict["attention_mask"] += [1]

        example_dict["metadata"]["pad_token_id"] = self.tokenizer.pad_token_id

        return example_dict
    
    def get_example_dict_enc_dec(self, example: Dict[str, Any], context: str, code: str = "", 
                         train_mode: bool = True, length_cutoff: bool = True) -> Dict[str, Any]:
        example_dict = {"metadata": example}

        context_tokenizer_outputs = self.tokenizer(context, truncation=length_cutoff)
        example_dict["input_ids"] = context_tokenizer_outputs["input_ids"]
        example_dict["attention_mask"] = context_tokenizer_outputs["attention_mask"]

        if train_mode:
            code_tokenizer_outputs = self.tokenizer(code, truncation=length_cutoff)
            example_dict["labels"] = code_tokenizer_outputs["input_ids"]

        example_dict["metadata"]["pad_token_id"] = self.tokenizer.pad_token_id

        return example_dict
    
    def get_example_dict(self, example: Dict[str, Any], context: str, code: str = "", 
                         train_mode: bool = True, length_cutoff: bool = True) -> Dict[str, Any]:
        if not is_model_gpt_style(self.transformer_model_name):
            example_dict = self.get_example_dict_enc_dec(example, context, code, train_mode, length_cutoff=length_cutoff)
        else:
            example_dict = self.get_example_dict_gpt(example, context, code, train_mode, length_cutoff=length_cutoff)
        
        # for verification only
        example_dict["metadata"]["correct_token_idx"] = 10932 if "bart" in self.transformer_model_name.lower() else 4273 # default for T5
        example_dict["metadata"]["incorrect_token_idx"] = 2362 if "bart" in self.transformer_model_name.lower() else 150 # default for T5
        
        if len(example_dict["input_ids"]) + self.generation_length > self.tokenizer.model_max_length:
            self.stats["input_too_long"] += 1
        
        return example_dict

    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError("the base class should not be used directly")

    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError("the base class should not be used directly")

    def read(self, file_path: str) -> Iterable[Dict[str, Any]]:
        print("Reading dataset files at %s", file_path)

        all_yield_instances = []

        # load the mathqa dataset with states
        mathqa_json_examples = []
        with open(file_path, 'r') as f:
            if self.mode == "test_few_shot":
                raise NotImplementedError("test_few_shot is not implemented yet")
            else:
                lines = f.readlines()[:self.max_instances]
            for line in lines:
                mathqa_json_examples.append(json.loads(line))

        iters = tqdm(mathqa_json_examples) if self.enable_tqdm else mathqa_json_examples
        for exp in iters:
            if self.mode == "train":
                example_dict = self.get_train_instance(exp)
            elif self.mode == "test":
                example_dict = self.get_test_instance(exp)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            # note that the returned example_dict might be a list of dicts
            all_yield_instances.extend(example_dict)

        logger.info(f"loaded {len(all_yield_instances)} instances")

        self.stats["total_instances"] = len(all_yield_instances)
        self.report_statistics()

        return all_yield_instances
    
    def report_statistics(self):
        total = self.stats["total_instances"]

        dataset_stats = "-" * 30 + "\nDataset statistics:\n"
        for key, value in self.stats.items():
            if key == "total_instances":
                continue
            dataset_stats += f"{key}: {value/total:.1%} \n"  
        dataset_stats += "-" * 30
        print(dataset_stats)

    def __getitem__(self, idx: int):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def truncate(self, max_instances):
        truncated_instances = self.instances[max_instances:]
        self.instances = self.instances[:max_instances]
        return truncated_instances

    def extend(self, instances):
        self.instances.extend(instances)

def customized_collate_fn_gpt(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    return customized_collate_fn(examples, is_left_pad=True)

def customized_collate_fn_enc_dec(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    return customized_collate_fn(examples, is_left_pad=False)

def customized_collate_fn(examples: List[Dict[str, Any]], is_left_pad: bool = True) -> Dict[str, Any]:
    result_dict = {}

    pad_token_id = examples[0]["metadata"]["pad_token_id"]

    pad_func = left_pad_sequences if is_left_pad else right_pad_sequences


    for k in examples[0].keys():
        if k == "metadata":
            result_dict[k] = [ex[k] for ex in examples]
        elif k == "input_ids":
            lists_to_pad = list(chain(*[[torch.tensor(t) for t in ex[k]] for ex in examples])) \
                if isinstance(examples[0][k][0], list) else [torch.tensor(ex[k]) for ex in examples]
            result_dict[k] = pad_func(lists_to_pad, batch_first=True, padding_value=pad_token_id)
        elif k == "attention_mask":
            lists_to_pad = list(chain(*[[torch.tensor(t) for t in ex[k]] for ex in examples])) \
                if isinstance(examples[0][k][0], list) else [torch.tensor(ex[k]) for ex in examples]
            result_dict[k] = pad_func(lists_to_pad, batch_first=True, padding_value=0)
        elif k == "labels":
            lists_to_pad = list(chain(*[[torch.tensor(t) for t in ex[k]] for ex in examples])) \
                if isinstance(examples[0][k][0], list) else [torch.tensor(ex[k]) for ex in examples]
            result_dict[k] = pad_func(lists_to_pad, batch_first=True, padding_value=-100)
        else:
            raise ValueError(f"Unknown key {k} in example instance")

    return result_dict

class NL2CodeDataModule(LightningDataModule):
    def __init__(self, 
                transformer_model_name: str,
                batch_size: int = 1, 
                val_batch_size: int = 1,
                train_set_init_args: Dict[str, Any] = {},
                val_set_init_args: Dict[str, Any] = {},
                set_common_init_args: Dict[str, Any] = {},
                train_max_instances: int = None,
                val_max_instances: int = None,
                ):
        super().__init__()
        self.transformer_model_name = transformer_model_name
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        # delegate the initialization of the train and val datasets to the dataset classes
        self.train_set_init_args = train_set_init_args
        self.train_set_init_args.update(set_common_init_args)
        self.val_set_init_args = val_set_init_args
        self.val_set_init_args.update(set_common_init_args) 

        if train_max_instances is not None:
            self.train_set_init_args["max_instances"] = train_max_instances
        if val_max_instances is not None:
            self.val_set_init_args["max_instances"] = val_max_instances

        self.train_data = None
        self.val_data = None

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError("the base class should not be used directly")

    def train_dataloader(self):
        if self.train_data is None:
            self.setup(stage="fit")
        
        collate_fn = customized_collate_fn_gpt if is_model_gpt_style(self.transformer_model_name) \
                                                else customized_collate_fn_enc_dec

        dtloader = DataLoader(self.train_data, batch_size=self.batch_size, 
                               shuffle=True, drop_last=True, collate_fn=collate_fn)
        return dtloader

    def val_dataloader(self):
        if self.val_data is None:
            self.setup(stage="validate")

        collate_fn = customized_collate_fn_gpt if is_model_gpt_style(self.transformer_model_name) \
                                                else customized_collate_fn_enc_dec

        dtloader = DataLoader(self.val_data, batch_size=self.val_batch_size, 
                               shuffle=False, drop_last=True, collate_fn=collate_fn)
        return dtloader

    def test_dataloader(self):
        raise NotImplementedError
    
    def get_gold_program_func(self, example_dict: Dict[str, Any]):
        raise NotImplementedError
    
    def get_gold_answer_func(self, example_dict: Dict[str, Any]):
        raise NotImplementedError