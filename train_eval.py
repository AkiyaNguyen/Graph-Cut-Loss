import torch
import torch.nn as nn
from typing import  List, Callable
import torch.optim as optim
from tqdm import tqdm
from metrics import metricWrapper
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



def train_one_epoch(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loader: DataLoader,
    optimizer,
    device: torch.device
):
    epoch_loss = 0.0
    model = model.to(device)
    model.train()

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / max(1, len(loader))


def train_model(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    log_every_epoch: int = 1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for ep in range(epochs):
        epoch_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        if log_every_epoch != 0 and (ep + 1) % log_every_epoch == 0:
            print(f"Epoch {ep+1}/{epochs}, Loss: {avg_loss:.4f}")

def metric_eval(model: nn.Module, test_loader: DataLoader, metric_wrapper: metricWrapper, eval_device='cpu'):
    model.eval()
    model.to(eval_device)
    
    metric_wrapper.reset()

    with torch.no_grad():  
        for data, target in test_loader:
            X = data.to(eval_device)
            y = target.to(eval_device)

            pred = model(X)  # (B,1,H,W)
            metric_wrapper.update(pred, y)

    results = {}
    for k, v in metric_wrapper.eval_items():
        v = np.array(v, dtype=float)
        results[k] = {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
        }

    return results
