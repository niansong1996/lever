import re
import os
import pandas as pd

from overrides import overrides

from typing import Dict, Iterable, List, Any, Optional, Union, Tuple

from finetuning.lightning_modules.datasets.base_reader import NL2CodeDataset, NL2CodeDataModule
from execution.program_tracing import assertion_to_test

"""
The structure of an example of MBPP:
{
    'task_id': 1,
    'text': 'Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].',
    'code': 'R = 3\r\nC = 3\r\ndef min_cost(cost, m, n): \r\n\ttc = [[0 for x in range(C)] for x in range(R)] \r\n\ttc[0][0] = cost[0][0] \r\n\tfor i in range(1, m+1): \r\n\t\ttc[i][0] = tc[i-1][0] + cost[i][0] \r\n\tfor j in range(1, n+1): \r\n\t\ttc[0][j] = tc[0][j-1] + cost[0][j] \r\n\tfor i in range(1, m+1): \r\n\t\tfor j in range(1, n+1): \r\n\t\t\ttc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] \r\n\treturn tc[m][n]',
    'test_list': [
        'assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8',
        'assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12',
        'assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16'],
    'test_setup_code': '',
    'challenge_test_list': []
}

"""

def mbpp_example_to_demonstration(example: Dict[str, Any], train=True, 
                                  add_assertion_n: int = 0, test_input_only: bool = False) -> str:
    # get the assertions
    if not test_input_only:
        assertion_header = '# These are the assertions for your function:\n'
        for test_case in example['test_list'][:add_assertion_n]:
            assertion_header += test_case + '\n'
    else:
        assertion_header = '# These are the calls for your function:\n'
        for test_case in example['test_list'][:add_assertion_n]:
            assertion_header += assertion_to_test(test_case) + '\n'

    # separate the function header and the function body
    func_signature = example["func_signature"]
    func_body = example["func_body"]

    func_comment = f'""" {example["text"]} """'

    header = assertion_header + '\n' + func_comment if add_assertion_n > 0 else func_comment

    if train:
        return f'### Task Start ###\n{header}\n{example["code"]}\n### Task End ###'
    else:
        return f'### Task Start ###\n{header}'

def saved_promptify_mbpp(prompt_file: str, example: Dict[str, Any], add_assertion_n: int,
                         test_input_only: bool) -> str:
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    
    return prompt + "\n\n" + mbpp_example_to_demonstration(example, train=False, add_assertion_n=add_assertion_n, 
                                                           test_input_only=test_input_only)

def simple_val_str_func(val_name, val_dict) -> str:
    val_type = val_dict["type"].split(" ")[1].replace("'", "").replace(">", "")
    val_val = val_dict["str_value"] if "object" not in val_dict["str_value"] else "<object list>"
    return f"{val_name}({val_type})={val_val}"

def val_type_only_str_func(val_name, val_dict) -> str:
    val_type = val_dict["type"].split(" ")[1].replace("'", "").replace(">", "")
    return f"({val_type})"

def state_simple_str_func(program_exec_dict: Dict[str, Any]) -> str:
    val_strs = []
    if len(program_exec_dict["tracing_local_list"]) == 0:
        return "ERROR: tracing failed"
    else:
        for k, v in program_exec_dict["tracing_local_list"][-1].items():
            if k != "_return_val":
                val_strs.append(simple_val_str_func(k, v))
        
        return ", ".join(val_strs)


class FewShotMBPPDataset(NL2CodeDataset):

    def __init__(self, 
                 prompt_file: str,
                 add_assertion_n: int,
                 test_input_only: bool = False,
                 **kwargs):
        # init some dataset specific variables
        self.prompt_file = prompt_file
        self.add_assertion_n = add_assertion_n
        self.test_input_only = test_input_only

        super().__init__(**kwargs)

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise ValueError("Few shot datasets do not support training")

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        context = saved_promptify_mbpp(self.prompt_file, example, self.add_assertion_n, test_input_only=self.test_input_only)

        return [self.get_example_dict(example, context, train_mode=False)]

class MBPPEndVerificationDataset(NL2CodeDataset):
    def __init__(self, 
                 include_code: bool = True, 
                 state_str_func: str = "state_simple_str_func", 
                 use_final_exec_result_only: bool = False, 
                 use_exec_passing_only: bool = False,
                 use_exec_return_only: bool = False,
                 use_exec_type_only: bool = False,
                 include_exec_result: bool = True,
                 ignore_no_negative_example: bool = False,
                 ignore_no_positive_example: bool = False,
                 include_gold_program_in_train: bool = True,
                 max_programs_per_example: int = 100,
                 **kwargs):

        self.state_str_func = eval(state_str_func)
        self.include_code = include_code
        self.use_final_exec_result_only = use_final_exec_result_only
        self.use_exec_passing_only = use_exec_passing_only
        self.use_exec_return_only = use_exec_return_only
        self.use_exec_type_only = use_exec_type_only
        self.include_exec_result = include_exec_result
        self.ignore_no_negative_example = ignore_no_negative_example
        self.ignore_no_positive_example = ignore_no_positive_example
        self.include_gold_program_in_train = include_gold_program_in_train
        self.max_programs_per_example = max_programs_per_example

        super().__init__(multi_instance_example=True, enable_tqdm=True, \
                         stats_keys=["input_too_long", "positive_labels", "negative_labels", "upper_bound", "too_many_programs"], **kwargs)
    
    def get_execution_info(self, program_dict: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        exec_info = "# execution\n"

        # use only one word to represent the execution result
        if self.use_exec_passing_only:
            if isinstance(program_dict["exec_result"], str) and program_dict["exec_result"].startswith("ERROR"):
                exec_info += "error"
            elif not all([x['result'] == 'passed' for x in program_dict["exec_result"]]):
                exec_info += "error"
            else:
                exec_info += "passed"
            
            return exec_info
        
        # get the most informative execution result representation
        if isinstance(program_dict["exec_result"], str) and program_dict["exec_result"].startswith("ERROR"):
            exec_info += f"{program_dict['exec_result']}\n\n"
        else:
            assert len(metadata["test_list"]) == len(program_dict["exec_result"])
            for test, state in zip(metadata["test_list"], program_dict["exec_result"]):
                test = assertion_to_test(test)
                if "tracing_local_list" not in state or \
                    len(state["tracing_local_list"]) == 0 or \
                        "_return_val" not in state["tracing_local_list"][-1]:
                    return_value = "ERROR: tracing failed"
                else:
                    # deciding which format for the final return value
                    if self.use_exec_type_only:
                        return_value = val_type_only_str_func("", state["tracing_local_list"][-1]["_return_val"])
                    else:
                        return_value = simple_val_str_func("", state["tracing_local_list"][-1]["_return_val"])
                
                # deciding if only the return value or the whole function call is also added
                if self.use_exec_return_only:
                    test_info = f"# return: {return_value}\n"
                else:
                    test_info = f"# test: {test}\n# return: {return_value}\n"
                
                # deciding if intermediate results are also added
                if not self.use_final_exec_result_only and state["result"] == "passed":
                    test_info += f"# states: {self.state_str_func(state)}\n\n"
                else:
                    test_info += "\n"
                exec_info += test_info
        
        return exec_info
    
    def process_mbpp_outputs(self, example: Dict[str, Any], add_gold_sol: bool, train_mode: bool) -> Dict[str, Any]:
        metadata = example["metadata"]
        metadata["gold_program"] = example["gold_program"]
        metadata["generated_programs"] = example["generated_programs"]

        # add the gold program to the generated programs if needed
        programs = metadata["generated_programs"]
        if add_gold_sol:
            programs.append(example["gold_program"])

        processed_examples = []
        for program_dict in programs:
            program_dict["program_count"] = 1 # this is needed for the verification model to combine generation prob
            program_dict["exec_match"] = 1 if program_dict["exec_match"] == 1 else 0

            # construct the context
            nl_description = f"# description\n{metadata['text']}\n\n"
            program = f"# program\n{program_dict['code']}\n\n" if self.include_code else ""
            exec_info = self.get_execution_info(program_dict, metadata) if self.include_exec_result else ""

            # get the input_text and label to construct the example dict
            input_text = nl_description + program + exec_info
            output_label = "yes" if program_dict["exec_match"] else "no"
            example_dict = self.get_example_dict(metadata, input_text, output_label, train_mode=train_mode, length_cutoff=False)

            # save some dataset statistics
            if len(example_dict["input_ids"]) > self.tokenizer.model_max_length:
                self.stats["input_too_long"] += 1
            if program_dict["exec_match"]:
                self.stats["positive_labels"] += 1
            else:
                self.stats["negative_labels"] += 1

            # cut off from the right 
            example_dict["input_ids"] = example_dict["input_ids"][:self.tokenizer.model_max_length]
            example_dict["attention_mask"] = example_dict["attention_mask"][:self.tokenizer.model_max_length]
            if train_mode:
                example_dict["labels"] = example_dict["labels"][:self.tokenizer.model_max_length:]

            processed_examples.append(example_dict)
        
        return processed_examples

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        if all([program["exec_match"] for program in example["generated_programs"]]) and self.ignore_no_negative_example:
            return []

        if self.ignore_no_positive_example and \
            all([not program["exec_match"] for program in example["generated_programs"]]) and \
            (not self.include_gold_program_in_train or not example["gold_program"]["exec_match"]):
            return []

        processed_examples = self.process_mbpp_outputs(example, add_gold_sol=self.include_gold_program_in_train, train_mode=True)
        if len(processed_examples) == 0:
            return []

        # combine the instances
        return_dict = {}
        for k in processed_examples[0].keys():
            if k == "metadata":
                return_dict[k] = processed_examples[0][k]
            else:
                return_dict[k] = [ex[k] for ex in processed_examples]

        # this is because the base dataset module would assume returning a list with multi-instance on
        return [return_dict]

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        processed_examples = self.process_mbpp_outputs(example, add_gold_sol=False, train_mode=False)

        # combine the instances
        return_dict = {}
        for k in processed_examples[0].keys():
            if k == "metadata":
                return_dict[k] = processed_examples[0][k]
            else:
                return_dict[k] = [ex[k] for ex in processed_examples]

        # this is because the base dataset module would assume returning a list with multi-instance on
        return [return_dict]

class FewShotMBPPDataModule(NL2CodeDataModule):

    @overrides
    def setup(self, stage: Optional[str] = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["fit", "validate"]

        if stage == "fit":
            raise ValueError("Few shot datasets do not support training")

        if self.val_data is None:
            val_data = FewShotMBPPDataset(transformer_model_name=self.transformer_model_name,
                                    mode="test", **self.val_set_init_args)
            self.val_data = val_data 

class MBPPEndVerificationDataModule(NL2CodeDataModule):

    @overrides
    def setup(self, stage: Optional[str] = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["fit", "validate"]

        if stage == "fit":
            if self.train_data is None:
                train_data = MBPPEndVerificationDataset(transformer_model_name=self.transformer_model_name,
                                        mode="train", **self.train_set_init_args)
                self.train_data = train_data

        if self.val_data is None:
            val_data = MBPPEndVerificationDataset(transformer_model_name=self.transformer_model_name,
                                    mode="test", **self.val_set_init_args)
            self.val_data = val_data 