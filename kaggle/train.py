import torch
from tqdm import tqdm
import torch.nn as nn

def mse_loss(pred, target):
    return nn.functional.mse_loss(pred, target)

def train_one_epoch(model, loader, optimiser, scaler, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(loader, desc='  train', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimiser.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            preds = model(imgs)
            loss  = mse_loss(preds, labels)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for imgs, labels in tqdm(loader, desc='  valid', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        total_loss += mse_loss(preds, labels).item() * imgs.size(0)
    return total_loss / len(loader.dataset)