"""PyTorch train/validation epoch loops; test loaders are intentionally unsupported."""
from __future__ import annotations
import torch

def _labels(y): return y.view(-1).long()
def train_epoch(model, loader, optimizer, criterion, device):
    model.train(); total=correct=n=0; loss_sum=0.0
    for x,y in loader:
        x,y=x.to(device),_labels(y).to(device); optimizer.zero_grad(set_to_none=True)
        out=model(x); loss=criterion(out,y); loss.backward(); optimizer.step()
        n+=y.numel(); loss_sum+=loss.item()*y.numel(); correct+=(out.argmax(1)==y).sum().item()
    return {"loss":loss_sum/n,"accuracy":correct/n}
@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval(); total=correct=n=0; loss_sum=0.; labels=[]; logits=[]
    for x,y in loader:
        x,y=x.to(device),_labels(y).to(device); out=model(x); loss=criterion(out,y)
        n+=y.numel(); loss_sum+=loss.item()*y.numel(); correct+=(out.argmax(1)==y).sum().item(); labels.extend(y.cpu().tolist()); logits.extend(out.cpu().tolist())
    return {"loss":loss_sum/n,"accuracy":correct/n,"labels":labels,"logits":logits}
