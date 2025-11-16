import torch
# from metrics import *
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

def preprocess_segmentation(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
):
    """
    Prepare TP, TN, FP, FN for binary segmentation metrics.

    preds   : (B, 1, H, W) or (B, H, W), probabilities after sigmoid in [0, 1]
    targets : (B, 1, H, W) or (B, H, W), binary mask {0, 1}
    threshold : probability threshold to binarize preds
    """
    preds = preds.to(torch.float32)
    targets = targets.to(torch.float32)

    preds_bin = (preds >= threshold).to(torch.float32)

    preds_flat = preds_bin.view(-1)
    targets_flat = targets.view(-1)

    # TP, TN, FP, FN
    TP = (preds_flat * targets_flat).sum()
    TN = ((1 - preds_flat) * (1 - targets_flat)).sum()
    FP = (preds_flat * (1 - targets_flat)).sum()
    FN = ((1 - preds_flat) * targets_flat).sum()

    return TP.float(), TN.float(), FP.float(), FN.float()


def iou_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5,
                eps: float = 1e-7) -> torch.Tensor:
    TP, TN, FP, FN = preprocess_segmentation(preds, targets, threshold)
    return TP / (TP + FP + FN + eps)


def dice_score(preds: torch.Tensor, targets: torch.Tensor,
            threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    TP, TN, FP, FN = preprocess_segmentation(preds, targets, threshold)
    return 2 * TP / (2 * TP + FP + FN + eps)


def sensitivity_score(preds: torch.Tensor, targets: torch.Tensor,
                      threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:

    TP, TN, FP, FN = preprocess_segmentation(preds, targets, threshold)
    return TP / (TP + FN + eps)


def specificity_score(preds: torch.Tensor, targets: torch.Tensor, 
                      threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    TP, TN, FP, FN = preprocess_segmentation(preds, targets, threshold)
    return TN / (TN + FP + eps)
