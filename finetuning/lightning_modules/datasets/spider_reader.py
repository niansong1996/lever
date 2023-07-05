import re
import os
import pandas as pd
import json

from overrides import overrides

from typing import Dict, Iterable, List, Any, Optional, Union

from finetuning.lightning_modules.datasets.base_reader import NL2CodeDataset, NL2CodeDataModule

DB_INFO_FILE = os.path.join(os.path.dirname(__file__), '../../../data/squall/db_info_wtq.json')

full_db_info = None

def pd_df_from_dict(dt: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(dt, orient='tight')

def example_to_demonstration_sql_bridge(example: Dict, train: bool = True, 
                                        include_schema: bool = True,
                                        include_sql: bool = True,
                                        lower_case_schema: bool = False) -> str:
    if full_db_info is None:
        with open(DB_INFO_FILE, "r") as f:
            full_db_info = json.load(f)

    # add an example column value to the schema
    if include_schema:
        db_id = example['db_id']
        db_info = full_db_info[db_id]
        text = f'-- Database {db_id}:\n'
        for table_name, columns in db_info['column_example_values'].items():
            column_representation = ', '.join([f"{name} ({str(val)[:50] + '...' if len(str(val)) > 50 else ''})" for name, val in columns])
            text += f'--  Table {table_name}: {column_representation}\n'
    else:
        text = ""
    
    if lower_case_schema:
        text = text.lower() + f'-- question: {example["question"]}\n'
    else:
        text += f'-- Question: {example["question"]}\n'
    
    if include_sql:
        if train:
            text += f'-- SQL:\n{example["query"]}'
        else:
            text += '-- SQL:\n'

    return text

def example_to_demonstration_sql(example: Dict, train: bool = True, 
                                 include_schema: bool = True,
                                 include_sql: bool = True,
                                 lower_case_schema: bool = False) -> str:
    if include_schema:
        raise NotImplementedError("This function is deprecated. Use example_to_demonstration_sql_bridge instead.")
        text = f'-- Database {example["db_id"]}:\n'
        for table_name, columns in example['db_table_headers'].items():
            column_representation = ', '.join(columns)
            text += f'--  Table {table_name}: {column_representation}\n'
    else:
        text = ""
    
    if lower_case_schema:
        text = text.lower() + f'-- question: {example["question"]}\n'
    else:
        text += f'-- Question: {example["question"]}\n'
    
    if include_sql:
        if train:
            text += f'-- SQL:\n{example["query"]}'
        else:
            text += '-- SQL:\n'

    return text

def saved_promptify_sql(prompt_file_path: str, example: Dict, promptify_func: callable, 
                        max_prompt_examples: int = 100, lower_case_schema: bool = False) -> str:
    with open(prompt_file_path, 'r') as f:
        prompt = f.read()

    # cut by the max prompt examples
    assert max_prompt_examples == 100, f"customizing the max prompt examples has been deprecated."
    prompt = prompt.strip()

    example_text = promptify_func(example, train=False, lower_case_schema=lower_case_schema)
    prompt += '\n\n-- Example:\n\n' + example_text.strip()

    return prompt

def state_add_meta_info_func(df: pd.DataFrame, program_dict: Dict[str, Any]):
    result = f"{df.shape[0]} rows" + f" (cutoff)\n" if program_dict["exec_result_cutoff"] else "\n"
    result += df.to_string(index=False)
    return result[:5000]

def state_simple_str_func(df: pd.DataFrame, program_dict: Dict[str, Any]):
    result = df.to_string(index=False)
    return result[:5000]

def state_to_csv_str_func(df: pd.DataFrame, program_dict: Dict[str, Any]):
    return df.to_csv(index=False)

def state_schema_only_func(df: pd.DataFrame, program_dict: Dict[str, Any]):
    return ", ".join(df.columns.tolist())

class SpiderEndVerificationDataset(NL2CodeDataset):
    def __init__(self, include_code: bool = True, 
                 include_schema: bool = False,
                 state_str_func: str = "state_simple_str_func", 
                 promptify_func: str = "example_to_demonstration_sql",
                 use_decomp_sql: bool = False, 
                 use_final_exec_result_only: bool = False, 
                 include_exec_result: bool = True,
                 ignore_no_negative_example: bool = False,
                 ignore_no_positive_example: bool = False,
                 include_gold_program_in_train: bool = True,
                 max_programs_per_example: int = 100,
                 **kwargs):

        self.state_str_func = eval(state_str_func)
        self.promptify_func = eval(promptify_func)
        self.include_code = include_code
        self.include_schema = include_schema
        self.use_decomp_sql = use_decomp_sql
        self.use_final_exec_result_only = use_final_exec_result_only
        self.include_exec_result = include_exec_result
        self.ignore_no_negative_example = ignore_no_negative_example
        self.ignore_no_positive_example = ignore_no_positive_example
        self.include_gold_program_in_train = include_gold_program_in_train
        self.max_programs_per_example = max_programs_per_example

        super().__init__(multi_instance_example=True, enable_tqdm=True, \
                         stats_keys=["input_too_long", "positive_labels", "negative_labels", "upper_bound", "too_many_programs"], **kwargs)
    
    def construct_input_text(self, program_dict: Dict[str, Any], context: str, code: str) -> str:
        execution_result = program_dict["exec_result"]
        execution_result_str = execution_result if isinstance(execution_result, str) \
                                                else self.state_str_func(pd_df_from_dict(execution_result), program_dict)
        
        # construct the input and output
        input_text = context + (code + "\n" if self.include_code else "") + \
                     "-- exec result:\n" + "/*\n" + (execution_result_str if self.include_exec_result else "") + "\n*/"
        input_text = input_text.replace("\n", "|") # some models (t5) can not process newline properly

        return input_text
    
    def construct_input_text_decomp(self, program_dict: Dict[str, Any], context: str, sqls: List[str]) -> str:
        execution_result_strs = [exec_result if isinstance(exec_result, str) else \
                        self.state_str_func(pd_df_from_dict(exec_result), program_dict) for exec_result in program_dict["exec_result"]]
        assert len(execution_result_strs) == len(sqls)

        # construct the input and output
        stmt_strs = []
        for i, (sql, exec_str) in enumerate(zip(sqls, execution_result_strs)):
            stmt_strs.append(sql + "\n" if self.include_code else "") 
            if not (self.use_final_exec_result_only and i != len(sqls) - 1) and self.include_exec_result:
                stmt_strs.append("/*\n" + exec_str + "\n*/\n")
        input_text = context + "".join(stmt_strs)
        input_text = input_text.replace("\n", "|") # some models (t5) can not process newline properly

        return input_text
    
    def process_sql_outputs(self, example: Dict[str, Any], add_gold_sol: bool = True, 
                            keep_programs_in_metadata: bool = False, 
                            train_mode: bool = True,
                            stats_avg: bool = False) -> List[Dict[str, Any]]:
        # put the gold program and generated program results into metadata
        metadata = example["metadata"]
        if keep_programs_in_metadata:
            metadata["gold_program"] = example["gold_program"]
            metadata["generated_programs"] = example["generated_programs"]

        # get the context
        include_sql = False if self.use_decomp_sql else self.include_code
        context = self.promptify_func(metadata, train=False, lower_case_schema=True, 
                                               include_schema=self.include_schema, include_sql=include_sql)

        # process all programs in the this example
        programs = example["generated_programs"]
        # programs = sorted(example["generated_programs"], key=lambda x: x["program_count"], reverse=True)[:5]
        # metadata["generated_programs"] = programs
        # if any([program["exec_match"] for program in programs]):
        #     self.stats["upper_bound"] += 1
        if add_gold_sol:
            # since during training, the predicted sql could be the gold sql, we need dedup
            lower_code_set = set([program["lower_code"] for program in programs])
            if example["gold_program"]["lower_code"] not in lower_code_set:
                programs.append(example["gold_program"]) # NOTE: this also adds to the original dictionary
        
        if self.ignore_no_negative_example and len(programs) == 1 and train_mode:
            return []

        processed_examples = []

        for program_dict in programs: 
            code = program_dict["lower_code"]
            execution_match = program_dict["exec_match"]
            if self.use_decomp_sql:
                sqls = code.split("\n")
                input_text = self.construct_input_text_decomp(program_dict, context, sqls)
            else:
                input_text = self.construct_input_text(program_dict, context, code)

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
        
        # use a limit of programs per example during training
        if train_mode and len(processed_examples) > self.max_programs_per_example:
            self.stats["too_many_programs"] += 1
            processed_examples = processed_examples[:self.max_programs_per_example]
        
        return processed_examples

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        if all([program["exec_match"] for program in example["generated_programs"]]) and self.ignore_no_negative_example:
            return []

        if self.ignore_no_positive_example and \
            all([not program["exec_match"] for program in example["generated_programs"]]) and \
            (not self.include_gold_program_in_train or not example["gold_program"]["exec_match"]):
            return []

        processed_examples = self.process_sql_outputs(example, add_gold_sol=self.include_gold_program_in_train, keep_programs_in_metadata=True, train_mode=True)
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
        processed_examples = self.process_sql_outputs(example, add_gold_sol=False, keep_programs_in_metadata=True, train_mode=False)

        # combine the instances
        return_dict = {}
        for k in processed_examples[0].keys():
            if k == "metadata":
                return_dict[k] = processed_examples[0][k]
            else:
                return_dict[k] = [ex[k] for ex in processed_examples]

        # this is because the base dataset module would assume returning a list with multi-instance on
        return [return_dict]

class FewShotSpiderDataset(NL2CodeDataset):

    def __init__(self, 
                 prompt_file: str,
                 prompt_examples: int = 100, 
                 promptify_func: str = "example_to_demonstration_sql",
                 **kwargs):
        # init some dataset specific variables
        self.prompt_examples = prompt_examples
        self.prompt_file = prompt_file
        self.promptify_func = eval(promptify_func)

        super().__init__(**kwargs)

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise ValueError("Few shot datasets do not support training")

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        context = saved_promptify_sql(self.prompt_file, example, self.promptify_func,
                                      max_prompt_examples=self.prompt_examples)

        return [self.get_example_dict(example, context, train_mode=False)]

class SpiderDataset(NL2CodeDataset):

    def __init__(self, promptify_func: str = "example_to_demonstration_sql", use_distinct: bool = False, 
                 use_decomp_sql: bool = False, use_skg_format: bool = False, **kwargs):
        # init some spider specific processing options
        self.promptify_func = eval(promptify_func)
        self.use_distinct = use_distinct
        self.use_skg_format = use_skg_format

        if use_decomp_sql:
            raise NotImplementedError("DecompSQL has been deprecated")

        super().__init__(**kwargs)

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:

        if self.use_skg_format:
            context = example["text_in"] + example["struct_in"]
            code: str = example["seq_out"]
            example["db_path"] = os.path.join("data/spider/database", example["db_id"], f'{example["db_id"]}.sqlite')
            example["answer"] = -10000 # this is a dummy value
        else:
            context = self.promptify_func(example, train=False, lower_case_schema=True) # we only need the context
            code: str = example["query"]

            # lower everything but the ones in the quote
            code = code.replace("\"", "'")
            code = re.sub(r"\b(?<!')(\w+)(?!')\b", lambda match: match.group(1).lower(), code) 

        return [self.get_example_dict(example, context, code, train_mode=True)]

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:

        if self.use_skg_format:
            context = example["text_in"] + example["struct_in"]
            code: str = example["seq_out"]

            example["db_path"] = os.path.join("data/spider/database", example["db_id"], f'{example["db_id"]}.sqlite')
            example["answer"] = -10000 # this is a dummy value since we don't need the answer to eval for spider
        else:
            context = self.promptify_func(example, train=False, lower_case_schema=True) # we only need the context
        
        return [self.get_example_dict(example, context, train_mode=False)]

class FewShotSQLDataModule(NL2CodeDataModule):

    @overrides
    def setup(self, stage: Optional[str] = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["fit", "validate"]

        if stage == "fit":
            raise ValueError("Few shot datasets do not support training")

        if self.val_data is None:
            val_data = FewShotSpiderDataset(transformer_model_name=self.transformer_model_name,
                                    mode="test", **self.val_set_init_args)
            self.val_data = val_data 

class SQLEndVerificationDataModule(NL2CodeDataModule):

    @overrides
    def setup(self, stage: Optional[str] = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["fit", "validate"]

        if stage == "fit":
            if self.train_data is None:
                train_data = SpiderEndVerificationDataset(transformer_model_name=self.transformer_model_name,
                                        mode="train", **self.train_set_init_args)
                self.train_data = train_data

        if self.val_data is None:
            val_data = SpiderEndVerificationDataset(transformer_model_name=self.transformer_model_name,
                                    mode="test", **self.val_set_init_args)
            self.val_data = val_data 

class Text2SqlDataModule(NL2CodeDataModule):

    @overrides
    def setup(self, stage: Optional[str] = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["fit", "validate"]

        if stage == "fit":
            train_data = SpiderDataset(transformer_model_name=self.transformer_model_name,
                                    mode="train", **self.train_set_init_args)
            self.train_data = train_data

        val_data = SpiderDataset(transformer_model_name=self.transformer_model_name,
                                 mode="test", **self.val_set_init_args)
        self.val_data = val_data 