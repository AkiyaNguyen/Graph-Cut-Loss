import torch
import torch.nn as nn
from typing import  List, Callable
import torch.optim as optim
from tqdm import tqdm
from metrics import metricWrapper
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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
        if (ep + 1) % log_every_epoch == 0:
            print(f"Epoch {ep+1}/{epochs}, Loss: {avg_loss:.4f}")

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
    
