import torch 
import torch.nn
from typing import Callable
from tqdm import tqdm
from metrics.metricWrapper import metricWrapper
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def metric_eval(model, test_dataset, metric_wrapper: metricWrapper, eval_device='cpu'):
    model.eval()
    model = model.to(eval_device)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    for data, target in tqdm(test_loader):
        X, y = data.to(eval_device), target.to(eval_device)
        pred = model(X) ## (B, 1, H, W)
        metric_wrapper.update(pred, y)

    for k, v in metric_wrapper.eval_items():
        print(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    
