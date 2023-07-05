from typing import Dict, Iterable, List, Any, Optional, Union
from overrides import overrides

from finetuning.lightning_modules.datasets.base_reader import NL2CodeDataset, NL2CodeDataModule

def state_simple_str_func(state_dict: Dict[str, Any]):
    if isinstance(state_dict, str):
        return state_dict
    elif state_dict is not None and 'answer' in state_dict:
        return str(state_dict)
    else:
        return "ERROR"

def state_answer_only_func(state_dict: Dict[str, Any]):
    if isinstance(state_dict, str):
        return state_dict
    elif state_dict is not None and 'answer' in state_dict:
        return str(state_dict['answer'])
    else:
        return "ERROR"

def saved_promptify_mathqa(prompt_file_path: str, example: Dict, max_prompt_examples: int = 100) -> str:
    with open(prompt_file_path, 'r') as f:
        prompt = f.read()

    prompt += "\n\n## " + example["question"]

    return prompt

class MathQAEndVerificationDataset(NL2CodeDataset):
    def __init__(self, include_code: bool = True, 
                 state_str_func: str = "state_simple_str_func", 
                 include_exec_result: bool = True,
                 ignore_no_negative_example: bool = False,
                 ignore_no_positive_example: bool = False,
                 **kwargs):

        self.state_str_func = eval(state_str_func)
        self.include_code = include_code
        self.include_exec_result = include_exec_result

        self.ignore_no_negative_example = ignore_no_negative_example
        self.ignore_no_positive_example = ignore_no_positive_example

        super().__init__(multi_instance_example=True, enable_tqdm=True, \
                         stats_keys=["input_too_long", "positive_labels", "negative_labels", "upper_bound"], **kwargs)

    def process_math_outputs(self, example: Dict[str, Any],
                            keep_programs_in_metadata: bool = False, 
                            train_mode: bool = True) -> List[Dict[str, Any]]:
        # put the gold program and generated program results into metadata
        metadata = example["metadata"]
        if keep_programs_in_metadata:
            metadata["generated_programs"] = example["generated_programs"]

        # process all programs in the this example
        programs = example["generated_programs"]
        # programs = sorted(example["generated_programs"], key=lambda x: x["program_count"], reverse=True)[:5]
        # metadata["generated_programs"] = programs
        # if any([program["exec_match"] for program in programs]):
        #     self.stats["upper_bound"] += 1

        processed_examples = []

        for program_dict in programs: 
            code = program_dict["lower_code"]
            execution_match = program_dict["exec_match"]
            input_text = metadata["question"]
            if self.include_code:
                input_text += " | " + code
            if self.include_exec_result:
                input_text += " | " + self.state_str_func(program_dict["exec_result"])

            output_label = "yes" if execution_match else "no"
            example_dict = self.get_example_dict(metadata, input_text, output_label, train_mode=train_mode)
            
            # cut off the input text if it is too long
            if len(example_dict["input_ids"]) > self.tokenizer.model_max_length:
                self.stats["input_too_long"] += 1
            if execution_match:
                self.stats["positive_labels"] += 1
            else:
                self.stats["negative_labels"] += 1

            # cut off from the left since the execution result is the most important one
            example_dict["input_ids"] = example_dict["input_ids"][-self.tokenizer.model_max_length:]
            example_dict["attention_mask"] = example_dict["attention_mask"][-self.tokenizer.model_max_length:]
            if train_mode:
                example_dict["labels"] = example_dict["labels"][-self.tokenizer.model_max_length:]

            processed_examples.append(example_dict)
        
        return processed_examples

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        if all([program["exec_match"] for program in example["generated_programs"]]) and self.ignore_no_negative_example:
            return []

        processed_examples = self.process_math_outputs(example, keep_programs_in_metadata=True, train_mode=True)
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
        processed_examples = self.process_math_outputs(example, keep_programs_in_metadata=True, train_mode=False)

        # combine the instances
        return_dict = {}
        for k in processed_examples[0].keys():
            if k == "metadata":
                return_dict[k] = processed_examples[0][k]
            else:
                return_dict[k] = [ex[k] for ex in processed_examples]

        # this is because the base dataset module would assume returning a list with multi-instance on
        return [return_dict]

class MathQADataset(NL2CodeDataset):

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [self.get_example_dict(example, example["text"], example["code"], train_mode=True)]

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        # parse the answer and add the field
        example["original_answer"] = example["answer"]
        example["answer"] = example["answer"].split("\n####")[-1].strip()

        return [self.get_example_dict(example, example["text"], "", train_mode=False)]

class FewShotMathQADataset(NL2CodeDataset):

    def __init__(self, 
                 prompt_file: str,
                 prompt_examples: int = 100, 
                 **kwargs):
        # init some dataset specific variables
        self.prompt_examples = prompt_examples
        if self.prompt_examples < 100:
            print("WARNING: using less prompt examples are not supported for this dataset")
        self.prompt_file = prompt_file

        super().__init__(**kwargs)

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise ValueError("Few shot datasets do not support training")

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        context = saved_promptify_mathqa(self.prompt_file, example)

        # parse the answer and add the field
        example["original_answer"] = example["answer"]
        try:
            example["answer"] = float(example["answer"].split("\n####")[-1].strip().replace(",", ""))
        except Exception as e:
            example["answer"] = -100000.0

        return [self.get_example_dict(example, context, train_mode=False)]

class MathQADataModule(NL2CodeDataModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @overrides
    def setup(self, stage: Optional[str] = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["fit", "validate", "test"]

        if stage == "fit":
            if self.train_data is None:
                train_data = MathQADataset(transformer_model_name=self.transformer_model_name,
                                        mode="train", **self.train_set_init_args)
                self.train_data = train_data

        if self.val_data is None:
            val_data = MathQADataset(transformer_model_name=self.transformer_model_name,
                                    mode="test", **self.val_set_init_args)
            self.val_data = val_data 

class MathQAEndVerificationDataModule(NL2CodeDataModule):

    @overrides
    def setup(self, stage: Optional[str] = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["fit", "validate"]

        if stage == "fit":
            if self.train_data is None:
                train_data = MathQAEndVerificationDataset(transformer_model_name=self.transformer_model_name,
                                        mode="train", **self.train_set_init_args)
                self.train_data = train_data

        if self.val_data is None:
            val_data = MathQAEndVerificationDataset(transformer_model_name=self.transformer_model_name,
                                    mode="test", **self.val_set_init_args)
            self.val_data = val_data 

class FewShotMathQADataModule(NL2CodeDataModule):

    @overrides
    def setup(self, stage: Optional[str] = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["fit", "validate"]

        if stage == "fit":
            raise ValueError("Few shot datasets do not support training")

        if self.val_data is None:
            val_data = FewShotMathQADataset(transformer_model_name=self.transformer_model_name,
                                    mode="test", **self.val_set_init_args)
            self.val_data = val_data 