from typing import Any, Callable, List, Union, Dict, Tuple

import torch
from torch import Tensor

from torchmetrics import Metric, MetricCollection, MeanMetric, SumMetric
from torchmetrics.aggregation import BaseAggregator

class CategorizedMetric(BaseAggregator):

    def __init__(
        self,
        metric_name: str,
        nan_strategy: Union[str, float] = "warn",
        category_list: List[str] = ['tmp_cat'],
        cat_func: Callable[[Dict[str, Any]], List[str]] = lambda example: ['tmp_cat'],
        **kwargs: Any,
    ):
        super().__init__(
            "sum",
            torch.tensor(0.0),
            nan_strategy,
            **kwargs,
        )
        
        self.metric_name = metric_name 
        self.cat_func = cat_func # this function maps the examples to the category str
        self.cat_list = category_list
        self.cat_name_dict = {} # convert from name to name + percentage

        # init the categorized metric dictionary
        self.cat_metrics_dict: Dict[str, Metric] = MetricCollection({})
        self.cat_count_metrics_dict: Dict[str, Metric] = MetricCollection({})
        for cat_name in self.cat_list:
            self.cat_metrics_dict[cat_name] = MeanMetric()
            self.cat_count_metrics_dict[cat_name] = SumMetric()
        
        # record the total number of examples
        self.total_num_metric = SumMetric()

    def update(self, value: float, example: Dict[str, Any]) -> None:  # type: ignore
        # first count the number
        self.total_num_metric.update(1.0)

        # then update all the member metrics
        example_cats = self.cat_func(example)
        for example_cat in example_cats:
            assert example_cat in self.cat_list, f"{example_cat} not in {self.cat_list}"
            self.cat_metrics_dict[example_cat](value)
            self.cat_count_metrics_dict[example_cat](1.0)

    def compute(self) -> Dict[str, float]:
        """Compute the aggregated value."""
        cat_perf_dict = {}
        total_weight = self.total_num_metric.compute()
        if total_weight < 50:
            print(f"skipping counting category {self.metric_name} of {total_weight} val examples...")
            return cat_perf_dict

        for cat_name in self.cat_list:
            cat_perf = self.cat_metrics_dict[cat_name].compute()
            cat_count = self.cat_count_metrics_dict[cat_name].compute()
            cat_percentage_str = "{0:.0%}".format(float(cat_count / total_weight))

            # add category name and percentage to the dictionary
            cat_name_per = f"{self.metric_name}_{cat_name}({cat_percentage_str})"
            if cat_name not in self.cat_name_dict:
                self.cat_name_dict[cat_name] = cat_name_per
            else:
                assert self.cat_name_dict[cat_name] == cat_name_per, f"{self.cat_name_dict[cat_name]} != {cat_name_per}"

            # update the result dict 
            cat_perf_dict[cat_name_per] = float(cat_perf) if not torch.isnan(cat_perf) else 0.0

        return cat_perf_dict
    
    def reset(self) -> None:
        for cat_name in self.cat_list:
            self.cat_metrics_dict[cat_name].reset()
            self.cat_count_metrics_dict[cat_name].reset()
        self.total_num_metric.reset()
