import torch
import json
import os
import math
import torch.nn.functional as F
import pytorch_lightning as pl
import io, tokenize, re
import ast, astunparse

from types import ModuleType
from typing import Optional, Dict, Any, Tuple, List, Callable, Type
from transformers.optimization import Adafactor, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup

from torch.optim import AdamW

from torchmetrics import Metric, MeanMetric, MetricCollection
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

from .seq2seq_model_util import get_model, post_process_code, is_model_gpt_style
from finetuning.lightning_modules.patches.categorized_metric import CategorizedMetric
from execution.executors import BaseExecutor

# those are for eval() of the execution funcs, see line 59
import execution
import finetuning.lightning_modules.models
import finetuning.lightning_modules.datasets

class Seq2SeqModel(LightningModule):
    def __init__(self, 
                 transformer_model_name: str,
                 executor_cls: str,
                 transformer_model_init_args: Dict[str, Any] = {},
                 executor_init_args: Dict[str, Any] = {},
                 categorize_metric: str = "exec_acc",
                 categorize_func: str = None,
                 category_list: List[str] = None,
                 max_gen_len: int = 100,
                 sampling_temp: float = 0.2,
                 sampling_temp_at_k: float = 0.8,
                 beam_size: int = 1,
                 gradient_ckpt: bool = False,
                 pass_at_k: int = 1,
                 eval_pass_at_k_every_n_epochs: int = 1,
                 eval_pass_at_k_at_epoch_end: bool = True,
                 save_raw_generation_results: bool = False,
                 print_eval_every_n_batches: int = -1,
                 max_generation_batches: int = 100,
                 max_steps: int = -1,
                 warmup_steps: int = 0,
                 optimizer: Dict[str, Any] = None,
                 lr_scheduler: Dict[str, Any] = None,
                 load_ckpt_file: str = None,
                 ) -> None:
        super().__init__()

        self.max_gen_len = max_gen_len
        self.sampling_temp = sampling_temp
        self.sampling_temp_at_k = sampling_temp_at_k
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps

        self.pass_at_k = pass_at_k
        self.eval_pass_at_k_every_n_epochs = eval_pass_at_k_every_n_epochs
        self.max_generation_batches = max_generation_batches
        self.print_eval_every_n_batches = print_eval_every_n_batches
        self.beam_size = beam_size
        self.save_raw_generation_results = save_raw_generation_results

        # We only instantiate this when we need it.
        self.transformer_model_name = transformer_model_name
        self.model, self.tokenizer = get_model(transformer_model_name, gradient_ckpt=gradient_ckpt, 
                                               additional_init_args=transformer_model_init_args)

        # set the correct execution engine
        # NOTE: since lightning cli do not allow callable, we have to make class from str
        self.executor: BaseExecutor = eval(executor_cls)(**executor_init_args)

        # save the prediction results for every valiation epoch
        self.predictions: List[Dict[str, Any]] = []

        self.opt_params = optimizer["init_args"] if optimizer is not None else {}
        self.lrs_params = lr_scheduler if lr_scheduler is not None else {}
        assert lr_scheduler is None or lr_scheduler["name"] in ["linear", "cosine", "constant"], "lr_scheduler must be one of 'linear', 'cosine', 'constant'"

        # load the state dict from the checkpoint file
        if load_ckpt_file is not None:
            if os.path.isdir(load_ckpt_file):
                # this is a ZeRO checkpoint
                combined_file_path = os.path.join(load_ckpt_file, "combined.ckpt")
                if not os.path.isfile(combined_file_path) and self.global_rank == 0:
                    # combine the checkpoint files
                    print(f"combing checkpoint files in {load_ckpt_file} to {combined_file_path}")
                    convert_zero_checkpoint_to_fp32_state_dict(load_ckpt_file, combined_file_path)
                load_ckpt_file = combined_file_path
            
            # load the checkpoint from a single file
            checkpoint = torch.load(load_ckpt_file, map_location=torch.device("cpu"))
            self.load_state_dict(checkpoint["state_dict"], strict=False)
            print(f"loaded weights from {load_ckpt_file}")

        # init for the evaluation metric
        self.metrics_dict: Dict[str, Metric] = MetricCollection({})
        self.category_metrics = CategorizedMetric(categorize_metric, 
                    category_list=["tmp_cat"] if category_list is None else category_list,
                    cat_func = (lambda example: ['tmp_cat']) if categorize_func is None else eval(categorize_func))

        self.metrics_dict["exec_acc"] = MeanMetric()
        self.metrics_dict["exec_rate"] = MeanMetric()
        self.metrics_dict["program_len_diff"] = MeanMetric()

        if self.pass_at_k > 1:
            self.metrics_dict[f"acc@{self.pass_at_k}"]= MeanMetric()
            self.metrics_dict[f"pass@{self.pass_at_k}"]= MeanMetric()
            self.metrics_dict["unique_program_ratio"]= MeanMetric()
    
    def generation_score_to_log_prob(self, generated_token_ids: torch.Tensor, scores: Tuple[torch.Tensor]) -> Tuple[List[float], List[int]]:
        # generated_token_ids: (batch_size, seq_len)
        # scores: (seq_len, batch_size, vocab_size)
        logprobs = F.log_softmax(torch.stack(scores, dim=0).transpose(0, 1), dim=2) # (batch_size, seq_len, vocab_size)
        selected_token_logprobs = [torch.gather(logprobs[i], 1, generated_token_ids[i].unsqueeze(1)).squeeze(1).tolist() \
                                    for i in range(generated_token_ids.size(0))]

        return selected_token_logprobs
    
    def generate_and_post_process(self, 
                                  input_ids: torch.Tensor, 
                                  attention_mask: torch.Tensor, 
                                  temperature: float, 
                                  beam_search: bool = False,
                                  num_return_sequences: int = 1) -> Tuple[List[str], List[List[float]], List[List[int]]]:

        if beam_search:
            assert num_return_sequences == 1, "beam search only support num_return_sequences = 1"
            use_sample = False
            num_beam = self.beam_size
            temp = 1.0
        else:
            use_sample = True
            num_beam = 1
            temp = temperature

        generation_results = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=use_sample, 
                                                  max_new_tokens=self.max_gen_len, num_beams=num_beam,
                                                  temperature=temp, num_return_sequences=num_return_sequences,
                                                  return_dict_in_generate=True, output_scores=True)
        
        # unpack the results
        generated_token_ids = generation_results["sequences"]
        generated_scores = generation_results["scores"]

        if is_model_gpt_style(self.transformer_model_name):
            generated_token_ids = generated_token_ids[:, input_ids.shape[1]:]

        # NOTE: no need to skip special tokens because the model should learn to output eos at the end
        # NOTE: incoder can not correctly skip <pad> token thus skipping special token would cause error
        if "incoder" in self.transformer_model_name:
            generated_strs = self.tokenizer.batch_decode(generated_token_ids, clean_up_tokenization_spaces=False)
        elif "codex" in self.transformer_model_name:
            generated_strs = generated_token_ids # codex would directly output the string
        else:
            generated_strs = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
        
        if self.save_raw_generation_results:
            if "codex" in self.transformer_model_name:
                generation_log_probs = generated_scores
                generated_token_ids = generation_results["tokens"]
            else:
                generation_log_probs = self.generation_score_to_log_prob(generated_token_ids, generated_scores)
                generated_token_ids = [x.tolist() for x in generated_token_ids]
        
        # do some truncation
        generated_programs = [self.executor.process_output(s, self.tokenizer.eos_token) for s in generated_strs]

        if self.save_raw_generation_results:
            return generated_programs, generation_log_probs, generated_token_ids
        else:
            return generated_programs, None, None

    def forward(  # type: ignore
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        The inference time behavior of the model.

        Args:
            input_ids [torch.Tensor]: Tokens from the context. 
            metadata (Optional[List[Dict[str, Any]]], optional): All additional information, `List` for the batch. Defaults to None.

        Returns:
            Dict[str, Any]: results saved in a `Dict` object.
        """        
        generated_programs, generation_probs, generation_tokens = \
            self.generate_and_post_process(input_ids, attention_mask, self.sampling_temp, beam_search=(self.beam_size > 1))

        # construct the output dict with the basic information
        output_dicts = []
        for i in range(len(generated_programs)):
            output_dict = {}
            output_dict["generated_program"] = generated_programs[i]
            output_dict["metadata"] = metadata[i]
            if self.save_raw_generation_results:
                output_dict["generation_probs"] = generation_probs[i]
                output_dict["generation_tokens"] = generation_tokens[i]
            output_dicts.append(output_dict)

        # evaluate pass at k 
        if self.current_epoch % self.eval_pass_at_k_every_n_epochs == 0 and self.pass_at_k > 1:
            generated_strs_list = [[] for _ in range(len(metadata))]
            if self.save_raw_generation_results:
                generation_probs_list = [[] for _ in range(len(metadata))]
                generation_tokens_list = [[] for _ in range(len(metadata))]
            remaining_k = self.pass_at_k
            while remaining_k > 0:
                generate_batch_size = min(remaining_k, self.max_generation_batches)
                remaining_k -= generate_batch_size

                batch_generated_programs, batch_generation_probs, batch_generation_tokens = \
                    self.generate_and_post_process(input_ids, attention_mask, self.sampling_temp_at_k, 
                                                num_return_sequences=generate_batch_size)

                for i in range(len(metadata)):
                    generated_strs_list[i].extend(batch_generated_programs[i*generate_batch_size:(i+1)*generate_batch_size])
                    if self.save_raw_generation_results:
                        generation_probs_list[i].extend(batch_generation_probs[i*generate_batch_size:(i+1)*generate_batch_size])
                        generation_tokens_list[i].extend(batch_generation_tokens[i*generate_batch_size:(i+1)*generate_batch_size])


            for i in range(len(metadata)):
                output_dicts[i]["generated_k_programs"] =  generated_strs_list[i]
                if self.save_raw_generation_results:
                    output_dicts[i]["generation_probs"] = generation_probs_list[i]
                    output_dicts[i]["generation_tokens"] = generation_tokens_list[i]


        return output_dicts

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        model_result = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        self.log("loss", model_result.loss, on_step=True, on_epoch=True, sync_dist=True)
        return {"loss": model_result.loss}

    def on_fit_start(self) -> None:
        # save the code using wandb
        if self.logger: 
            # if logger is initialized, save the code
            self.logger.log_code()
        else:
            print("logger is not initialized, code will not be saved")  

        return super().on_fit_start()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        # input_tokens, target_mask, context_tokens, target_tokens, metadata = batch
        return self.forward(batch["input_ids"], batch["attention_mask"], batch["metadata"])
    
    def get_program_exec_dict(self, generated_program: str, exec_match: int, exec_result: Any) -> Dict[str, Any]:
        exec_acc = 1.0 if exec_match == 1 else 0.0
        exec_rate = 0.0 if exec_match == -1 else 1.0

        # save the results in the json output file
        save_metrics = {"exec_acc": float(exec_acc), 
                        "exec_rate": float(exec_rate)}

        # add more information to the program dict
        program_dict = {"program": generated_program, "exec_result": exec_result}
        program_dict.update(save_metrics)

        return program_dict

    def validation_step_end(self, outputs: List[Dict[str, Any]]) -> None:
        # update the evaluation metrics
        for output_dict in outputs:
            generated_program = output_dict["generated_program"]
            metadata = output_dict["metadata"]

            # obtain the execution results
            exec_match, exec_result = self.executor.exec_program(generated_program, metadata)
            program_len_diff = self.executor.program_len(generated_program) - self.executor.gold_program_len(metadata)
            program_dict = self.get_program_exec_dict(generated_program, exec_match, exec_result)
            output_dict["generated_program"] = program_dict

            # update the metrics
            self.metrics_dict["exec_acc"](program_dict["exec_acc"])
            self.metrics_dict["exec_rate"](program_dict["exec_rate"])
            self.metrics_dict["program_len_diff"](program_len_diff)
            self.category_metrics.update(program_dict["exec_acc"], metadata) # note that this can't be forward as compute will be called
        
        if self.print_eval_every_n_batches > 0:
            # compute the metrics
            eval_metrics_dict = {}
            for k in self.metrics_dict.keys():
                eval_metrics_dict[k] = float(self.metrics_dict[k].compute())
            print("eval metrics: ", eval_metrics_dict)

        # save the outputs to the model
        self.predictions.extend(outputs)

    def validation_epoch_end_extra(self, outputs: List[Dict[str, Any]]) -> None:
        # compute the eval_at_k metrics
        if self.current_epoch % self.eval_pass_at_k_every_n_epochs == 0 and self.pass_at_k > 1:
            print("evaluating pass at k...")

            all_generated_k_programs = [p["generated_k_programs"] for p in self.predictions]
            all_generated_k_programs_faltten = [item for sublist in all_generated_k_programs for item in sublist]
            all_metadata_flatten = [p["metadata"] for p in self.predictions]

            result_list = self.executor.batch_exec_programs(all_generated_k_programs_faltten, all_metadata_flatten, share_metadata_n=self.pass_at_k)
            print(f"Execution stats: {self.executor.get_exec_stats()}")
            assert len(result_list) == len(all_generated_k_programs_faltten)

            for i in range(0, len(result_list), self.pass_at_k):
                example_idx = i // self.pass_at_k
                programs = all_generated_k_programs_faltten[i:i+self.pass_at_k]
                exec_result_pairs = result_list[i:i+self.pass_at_k]

                program_dict_list = []
                for program, (exec_match, exec_result) in zip(programs, exec_result_pairs):
                    program_dict = self.get_program_exec_dict(program, exec_match, exec_result)
                    program_dict_list.append(program_dict)
                self.predictions[example_idx]["generated_k_programs"] = program_dict_list
                
                # calculate acc@k and pass@k and update the metrics
                acc_at_k = sum([p["exec_acc"] for p in program_dict_list]) / len(program_dict_list)
                pass_at_k = float(any([p["exec_acc"] == 1.0 for p in program_dict_list]))
                unique_program_ratio = len(set(programs)) / len(programs)

                self.predictions[example_idx]["metrics"] = {"acc_at_k": acc_at_k, "pass_at_k": pass_at_k,
                                                  "exec_acc": self.predictions[example_idx]["generated_program"]["exec_acc"],
                                                  "exec_rate": self.predictions[example_idx]["generated_program"]["exec_rate"]}

                self.metrics_dict[f"acc@{self.pass_at_k}"](acc_at_k)
                self.metrics_dict[f"pass@{self.pass_at_k}"](pass_at_k)
                self.metrics_dict["unique_program_ratio"](unique_program_ratio)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        # extra steps for using the predictions
        self.validation_epoch_end_extra(outputs)

        # compute the metrics
        eval_metrics_dict = {}
        for k in self.metrics_dict.keys():
            eval_metrics_dict[k] = float(self.metrics_dict[k].compute())
        
        # compute the categorized metricsa
        cat_metrics_dict = self.category_metrics.compute()
        eval_metrics_dict.update(cat_metrics_dict)
        
        # log and save the evalution metrics
        print(f"validation result: {eval_metrics_dict}")
        self.log_dict(eval_metrics_dict, sync_dist=True) 

        # reset all the metrics
        for k in self.metrics_dict.keys():
            self.metrics_dict[k].reset()
        self.category_metrics.reset()

        # save the predictions
        save_pred_file_path = os.path.join(self.trainer.log_dir,
                                f'predictions_step_{self.trainer.global_step}_rank_{self.trainer.global_rank}.jsonl')
        with open(save_pred_file_path, 'w+') as f:
            for prediction in self.predictions:
                f.write(json.dumps(prediction)+'\n')
        print(f"{len(self.predictions)} predictions saved to {save_pred_file_path}")

        # reset the predictions
        self.predictions = []
        
        # NOTE: debug setting only
        # self.sampling_temp += 0.1
        # self.sampling_temp_at_k += 0.2
        # print(f"sampling temp is now {self.sampling_temp}, sampling temp at k is now {self.sampling_temp_at_k}")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.opt_params)
        if self.lrs_params["name"] == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer, **self.lrs_params["init_args"])
        elif self.lrs_params["name"] == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, **self.lrs_params["init_args"])
        elif self.lrs_params["name"] == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(optimizer, **self.lrs_params["init_args"])
        else:
            raise ValueError(f"lr_scheduler {self.lrs_params} is not supported")

        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step"
                    }
                }