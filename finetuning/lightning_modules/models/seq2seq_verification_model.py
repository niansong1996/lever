import torch
import math

import numpy as np
import random

from collections import Counter

from typing import Dict, List, Any, Optional, Tuple

from .seq2seq_model import Seq2SeqModel
from .seq2seq_model_util import get_model, post_process_code, is_model_gpt_style, is_encoder_only_model

from torchmetrics import Metric, MeanMetric, MetricCollection
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.functional import log_softmax


class EndVerificationModel(Seq2SeqModel):
    def __init__(self, 
                 softmax_loss: bool = False,
                 binary_cls_threshold: float = 0.5,
                 avg_loss_per_example: bool = False,
                 contrastive: bool = False,
                 mml: bool = False,
                 eval_exec_result_agg: bool = False,
                 exec_result_agg: bool = False,
                 exec_result_agg_type: str = None,
                 exec_result_agg_threshold: float = None,
                 filtering_top_k: int = 100,
                 filtering_prob_threshold: float = 0.0,
                 gen_prob_coef: float = 1.0,
                 rerank_prob_coef: float = 1.0,
                 use_norm_gen_prob: bool = False,
                 max_batch_size: int = 100,
                 eval_max_batch_size: int = 100,
                 use_downsample_in_train: bool = True,
                 negative_cls_weight: float = 1.0,
                 positive_cls_weight: float = 1.0,
                 label_smoothing_coef: float = 0.0,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)

        assert not exec_result_agg or exec_result_agg_type is not None, "exec_result_agg_type must be specified if exec_result_agg is True"
        assert exec_result_agg_type is None or exec_result_agg_type in ["voting", "summing"]

        # training settings
        self.encoder_only_model = is_encoder_only_model(self.transformer_model_name)
        self.binary_cls_threshold = binary_cls_threshold
        self.softmax_loss = softmax_loss
        self.avg_loss_per_example = avg_loss_per_example
        self.contrastive = contrastive
        self.mml = mml
        self.max_batch_size = max_batch_size
        self.eval_max_batch_size = eval_max_batch_size
        self.use_downsample_in_train = use_downsample_in_train # this is used to control if downsampling or iterative pass is used when the batch size is too large
        self.negative_cls_weight = negative_cls_weight
        self.positive_cls_weight = positive_cls_weight
        self.label_smoothing_coef = label_smoothing_coef

        # inference settings
        self.eval_exec_result_agg = eval_exec_result_agg
        self.exec_result_agg = exec_result_agg
        self.exec_result_agg_type = exec_result_agg_type
        if exec_result_agg_threshold is None:
            self.exec_result_agg_threshold = 0.5 if self.exec_result_agg_type == "voting" else 0.0
        else:
            self.exec_result_agg_threshold = exec_result_agg_threshold
        self.filtering_top_k = filtering_top_k
        self.filtering_prob_threshold = filtering_prob_threshold
        self.gen_prob_coef = gen_prob_coef
        self.rerank_prob_coef = rerank_prob_coef
        self.use_norm_gen_prob = use_norm_gen_prob

        assert not self.softmax_loss or (self.contrastive != self.mml), "if softmax is used, either contrastive or mml must be used"

        # the metrics dict is created in the init function of the base class
        # self.metrics_dict["binary_acc"] = MeanMetric()
        self.metrics_dict["rerank_acc"] = MeanMetric()
        if self.softmax_loss:
            self.metrics_dict["marg_prob"] = MeanMetric()
        else:
            self.metrics_dict["binary_acc"] = MeanMetric()
        
        if self.eval_exec_result_agg:
            self.metrics_dict["summing_agg_rerank_acc"] = MeanMetric()
            self.metrics_dict["voting_agg_rerank_acc"] = MeanMetric()

        # remove some unused metrics
        self.metrics_dict.pop("exec_acc")
        self.metrics_dict.pop("exec_rate")
        self.metrics_dict.pop("program_len_diff")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        metadata = batch["metadata"]
        labels = batch["labels"][:, :1]

        if len(input_ids) > self.max_batch_size and self.use_downsample_in_train:
            assert len(metadata) == 1, "batch size must be 1 to use downsampling"

            # downsample the batch
            rand_idx = list(range(len(input_ids)))
            random.shuffle(rand_idx)
            rand_idx = rand_idx[:self.max_batch_size]
            input_ids = input_ids[rand_idx, :]
            attention_mask = attention_mask[rand_idx, :]
            labels = labels[rand_idx, :]
            metadata = [metadata[0].copy()]
            metadata[0]["generated_programs"] = [metadata[0]["generated_programs"][int(i)] for i in rand_idx]

        fake_loss = None # this is used to avoid the zero error case by using the loss as a dummy value w/ * 0.0
        if self.encoder_only_model:
            lm_logits_list = []
            # if len(input_ids) > self.max_batch_size:
            #     print(f"batch size is too large ({len(input_ids)}), splitting into smaller batches")
            for i in range(0, len(input_ids), self.max_batch_size):
                model_result = self.model(input_ids[i:i+self.max_batch_size], attention_mask[i:i+self.max_batch_size])
                fake_loss = model_result.loss
                lm_logits_list.append(model_result.logits)
            lm_logits = torch.cat(lm_logits_list, dim=0)
        else:
            correct_token_idx = metadata[0]["correct_token_idx"]
            incorrect_token_idx = metadata[0]["incorrect_token_idx"]

            lm_logits_list = []
            for i in range(0, len(input_ids), self.max_batch_size):
                model_result = self.model(input_ids[i:i+self.max_batch_size], attention_mask[i:i+self.max_batch_size], labels=labels[i:i+self.max_batch_size])
                fake_loss = model_result.loss
                # the only two logits we need is the one that represents the correct/incorrect tokens
                batch_logits = torch.stack([model_result.logits[:, 0, incorrect_token_idx], model_result.logits[:, 0, correct_token_idx]], dim=1)
                lm_logits_list.append(batch_logits)
            lm_logits = torch.cat(lm_logits_list, dim=0)

        # split by example to ensure fair loss 
        split_sizes = [len(x['generated_programs']) for x in metadata]
        split_lm_logits = lm_logits.split(split_sizes, dim=0)
        split_true_labels = [torch.tensor([program_info["exec_match"] for program_info in example_info["generated_programs"]], 
                                dtype=torch.long, device=self.device) for example_info in metadata]

        # depending on the loss selections
        if self.softmax_loss:
            split_lm_logits = [lm_logits[:, 1] for lm_logits in split_lm_logits] # for contrastive loss, we only need the positive logits
            if self.mml:
                example_log_probs = [log_softmax(exp_logits, dim=0) for exp_logits in split_lm_logits]
                positive_example_log_probs = [torch.masked_select(exp_logits, positive_mask.bool()) for exp_logits, positive_mask in zip(example_log_probs, split_true_labels)]
                marginal_positive_example_log_probs = torch.stack([torch.logsumexp(exp_logits, dim=0) for exp_logits in positive_example_log_probs])
                loss = -torch.mean(marginal_positive_example_log_probs)
            elif self.contrastive:
                loss_list = []
                for exp_logits, exp_labels in zip(split_lm_logits, split_true_labels):
                    pos_labels = exp_labels.bool()
                    neg_labels = (1.0 - exp_labels).bool()
                    pos_logits = torch.masked_select(exp_logits, pos_labels)
                    neg_logits = torch.masked_select(exp_logits, neg_labels)

                    if len(pos_logits) == 0 or len(neg_logits) == 0:
                        continue

                    # do contrastive pairing and computing the loss
                    example_pair_loss_list = []
                    for i in range(len(pos_logits)):
                        paired_logits = torch.cat((pos_logits[i:i+1], neg_logits), dim=0)
                        paired_log_probs = log_softmax(paired_logits, dim=0)
                        example_pair_loss_list.append(-paired_log_probs[0])
                    if self.avg_loss_per_example:
                        loss_list.append(torch.mean(torch.stack(example_pair_loss_list)))
                    else:
                        loss_list.extend(example_pair_loss_list)
                if len(loss_list) > 0:
                    loss = torch.mean(torch.stack(loss_list))
                else:
                    # print(f"warning - no loss found for batch {batch_idx}")
                    loss = 0.0 * fake_loss
            else:
                raise ValueError("if softmax is used, either contrastive or mml must be used")
        else:
            # performaning binary classification (verification) rather than softmax (reranking)
            reduction_method = "mean" if self.avg_loss_per_example else "sum"
            loss_weight = torch.tensor([self.negative_cls_weight, self.positive_cls_weight], device=self.device)
            loss_fct = CrossEntropyLoss(reduction=reduction_method, weight=loss_weight, label_smoothing=self.label_smoothing_coef)
            loss_total = sum([loss_fct(exp_logits, exp_labels) for exp_logits, exp_labels in zip(split_lm_logits, split_true_labels)])
            loss = loss_total / (len(split_sizes) if self.avg_loss_per_example else sum(split_sizes))

        self.log("loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return {"loss": loss}

    def forward(  # type: ignore
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if self.encoder_only_model:
            lm_logits_list = []
            for i in range(0, len(input_ids), self.eval_max_batch_size):
                model_result = self.model(input_ids=input_ids[i:i+self.eval_max_batch_size], 
                                          attention_mask=attention_mask[i:i+self.eval_max_batch_size])
                lm_logits_list.append(model_result.logits)
            lm_logits = torch.cat(lm_logits_list, dim=0)
        else:
            # run normal inference, it only needs to generate the yes or no symbol
            generation_scores = []
            for i in range(0, len(input_ids), self.eval_max_batch_size):
                model_result = self.model.generate(input_ids=input_ids[i:i+self.eval_max_batch_size], 
                                                   attention_mask=attention_mask[i:i+self.eval_max_batch_size], 
                                                   do_sample=False, max_new_tokens=2, return_dict_in_generate=True,
                                                   output_scores=True, num_beams=1)
                generation_scores.append(model_result.scores[0])
            generation_scores = torch.cat(generation_scores, dim=0)

            correct_token_idx = metadata[0]["correct_token_idx"]
            incorrect_token_idx = metadata[0]["incorrect_token_idx"]
            lm_logits = torch.stack([generation_scores[:, incorrect_token_idx], \
                                    generation_scores[:, correct_token_idx]], dim=1)

        # split to count per example
        split_sizes = [len(x['generated_programs']) for x in metadata]
        split_lm_logits = lm_logits.split(split_sizes, dim=0)
        split_true_labels = [torch.tensor([program_info["exec_match"] for program_info in example_info["generated_programs"]], 
                                dtype=torch.long, device=self.device) for example_info in metadata]
        
        if "gen_prob" in metadata[0]["generated_programs"][0]:
            prob_key_str = "gen_prob" if not self.use_norm_gen_prob else "norm_gen_prob"
            split_gen_log_probs = [torch.tensor([program_info[prob_key_str] for program_info in example_info["generated_programs"]], 
                                    dtype=torch.float, device=self.device) for example_info in metadata]
        else:
            generated_program_counts = [sum([y['program_count'] for y in x['generated_programs']]) for x in metadata]
            split_gen_log_probs = [torch.log(torch.tensor([x['program_count'] / total_program_count
                                        for x in example_info["generated_programs"]], device=self.device)) 
                                            for example_info, total_program_count in zip(metadata, generated_program_counts)]
        
        # depending on the loss selections
        split_true_labels = [torch.tensor([program_info["exec_match"] for program_info in example_info["generated_programs"]], 
                                dtype=torch.long, device=self.device) for example_info in metadata]
        if self.softmax_loss:
            # we only need the positive logits to do the softmax
            split_lm_logits = [lm_logits[:, 1] for lm_logits in split_lm_logits] 
            example_log_probs = [log_softmax(exp_logits, dim=0) for exp_logits in split_lm_logits]

            # evaluate the marginal prob
            positive_example_log_probs = [torch.masked_select(exp_logits, positive_mask.bool()) for exp_logits, positive_mask in zip(example_log_probs, split_true_labels)]
            marginal_positive_example_log_probs = torch.stack([torch.logsumexp(exp_logits, dim=0) for exp_logits in positive_example_log_probs])
            self.metrics_dict["marg_prob"](float(torch.mean(torch.exp(marginal_positive_example_log_probs))))
        else:
            # do the softmax over {pos, neg} logits and then only take the positive prob
            example_log_probs = [log_softmax(exp_logits, dim=1)[:, 1] for exp_logits in split_lm_logits]

            # evaluate binary classification accuracy
            example_preds = [log_prob > math.log(self.binary_cls_threshold) for log_prob in example_log_probs]
            binary_cls_accs = torch.stack([torch.sum(torch.eq(exp_pos_probs.int(), exp_true_labels)) / exp_pos_probs.shape[0] for exp_pos_probs, exp_true_labels in zip(example_preds, split_true_labels)])
            self.metrics_dict["binary_acc"](float(torch.mean(binary_cls_accs)))
        
        # how to selection the final answer
        output_dicts = [{"metrics": {}, "metadata": metadata[i]} for i in range(len(metadata))]
        for i, (rerank_log_probs, gen_log_probs, true_labels) in enumerate(zip(example_log_probs, split_gen_log_probs, split_true_labels)):
            # for threshold_str in ["0_001", "0_01", "0_05", "0_1", "0_5", "1_0", "2_0", "5_0", "10_0", "100_0"]:
            # for threshold_str in ["0_0001", "0_1", "0_2", "0_3", "0_4", "0_5", "0_6", "0_7", "0_8", "0_9", "0_95", "0_98"]:
                # self.exec_result_agg_threshold = float(threshold_str.replace("_", "."))
                # self.gen_prob_coef = float(threshold_str.replace("_", "."))
                # self.rerank_prob_coef = float(threshold_str.replace("_", "."))

            ####### indent starts here ########
            if self.exec_result_agg:
                pred_idx, final_log_probs = self.aggregate_exec_results(metadata[i]['generated_programs'], rerank_log_probs, gen_log_probs, 
                                                true_labels, self.gen_prob_coef, self.exec_result_agg_type, self.exec_result_agg_threshold)
            else:
                tuned_rerank_log_probs = self.rerank_prob_coef * rerank_log_probs
                final_log_probs = tuned_rerank_log_probs + self.gen_prob_coef * gen_log_probs
                pred_idx = int(torch.argmax(final_log_probs, dim=0))
            example_rerank_correct = float(true_labels[pred_idx])

            if self.eval_exec_result_agg:
                # evaluate summing
                pred_idx, final_log_probs = self.aggregate_exec_results(metadata[i]['generated_programs'], tuned_rerank_log_probs, gen_log_probs, 
                                                true_labels, self.gen_prob_coef, agg_type="summing", agg_threshold=0.0)
                summing_rerank_acc = float(true_labels[pred_idx])
                self.metrics_dict["summing_agg_rerank_acc"](summing_rerank_acc)

                # evaluate voting
                pred_idx, final_log_probs = self.aggregate_exec_results(metadata[i]['generated_programs'], tuned_rerank_log_probs, gen_log_probs, 
                                                true_labels, self.gen_prob_coef, agg_type="voting", agg_threshold=0.5)
                voting_rerank_acc = float(true_labels[pred_idx])
                self.metrics_dict["voting_agg_rerank_acc"](voting_rerank_acc)

            # metadata[i]["category"] = str(self.rerank_prob_coef).replace(".", "_")
            # metadata[i]["category"] = str(self.gen_prob_coef).replace(".", "_")
            # metadata[i]["category"] = str(self.exec_result_agg_threshold).replace(".", "_")
            # self.category_metrics.update(summing_rerank_acc, metadata[i]) # note that this can't be forward as compute will be called
            ####### indent ends here ########

            output_dicts[i]["metrics"]["rerank_acc"] = example_rerank_correct
            output_dicts[i]["reranking_scores"] = tuned_rerank_log_probs.tolist()
            output_dicts[i]["final_scores"] = final_log_probs.tolist()

            self.metrics_dict["rerank_acc"](example_rerank_correct)
            self.category_metrics.update(example_rerank_correct, metadata[i]) # note that this can't be forward as compute will be called

        return output_dicts
        

    def validation_step_end(self, outputs: List[Dict[str, Any]]) -> None:
        # save the outputs to the model
        self.predictions.extend(outputs)
    
    def aggregate_exec_results(self, generated_program_dicts: List[Dict[str, Any]], rerank_log_probs: torch.Tensor,
                               gen_log_probs: torch.Tensor, true_labels: List[bool], gen_prob_coef: float, agg_type: str,
                               agg_threshold: int) -> int:
        assert agg_type in ["summing", "voting"], f"Unknown aggregation type {agg_type}"

        # first build the final log probs, taking into account of the threshold
        gated_rerank_mask = (torch.exp(rerank_log_probs) < agg_threshold).float()
        if agg_type == "voting":
            # every program that is *higher* than the threshold would get a vote that is weighted by the generation prob
            final_log_probs = gated_rerank_mask * -1e10 + gen_log_probs * gen_prob_coef
        elif agg_type == "summing":
            final_log_probs = gated_rerank_mask * -1e10 + rerank_log_probs + gen_prob_coef * gen_log_probs
        else:
            raise ValueError(f"Unknown exec_result_agg_type: {agg_type}")

        # then build the groupings
        idx_groups = []
        for i, program_dict in enumerate(generated_program_dicts):
            if len(idx_groups) == 0:
                idx_groups.append([i])
                continue
                
            # try to fit into the existing groups
            same_group_idx = -1
            for group_i in range(len(idx_groups)):
                group_ref_program_dict = generated_program_dicts[idx_groups[group_i][0]]
                if self.executor.exec_result_eq(program_dict, group_ref_program_dict):
                    same_group_idx = group_i
                    idx_groups[group_i].append(i)
                    break

            # if not, create a new group
            if same_group_idx == -1:
                idx_groups.append([i])
        
        # then aggregate the log probs
        group_log_probs = []
        for group_indices in idx_groups:
            # an assertion to make sure that all the programs in the group are of the same true label
            group_true_labels = true_labels[group_indices].float()

            # if torch.sum(group_true_labels) > 0 and torch.prod(group_true_labels) == 0:
            #     print(f"Group true labels: {group_true_labels} has both 0 and 1, the exec results are:") 
                # for i in group_indices: 
                #     print(generated_program_dicts[i]['exec_result'])
            # assert not (torch.sum(group_true_labels) > 0 and torch.prod(group_true_labels) == 0), \
            #     f"Group true labels: {group_true_labels} has both 0 and 1, the exec results are {[generated_program_dicts[i]['exec_result'] for i in group_indices]}"
            group_log_probs.append(torch.logsumexp(final_log_probs[group_indices], dim=0))
        group_log_probs = torch.stack(group_log_probs, dim=0)

        # pick the best group and its representative program
        pred_group_idx = int(torch.argmax(group_log_probs, dim=0))
        return idx_groups[pred_group_idx][0], final_log_probs

