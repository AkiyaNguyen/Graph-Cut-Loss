from .metrics import *
from typing import Callable
import torch

class metricWrapper:
    def __init__(self, MetricDict: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]):
        self.MetricDict = MetricDict
        self.EvalResult = {}
        for k in self.MetricDict.keys():
            self.EvalResult[k] = []
    def __getitem__(self, key: str):
        if key not in self.MetricDict.keys():
            raise ValueError("key not found in metricWrapper")
        return self.MetricDict[key], self.EvalResult[key]
    def update(self, preds, targets):
        for k, func in self.MetricDict.items():
            self.EvalResult[k].append(func(preds, targets))
    def eval_items(self):
        return self.EvalResult.items()
    def reset(self):
        """
        clear list inside self.EvalResult only
        """
        for key in self.MetricDict.keys():
            self.EvalResult[key] = []
