from collections import Counter
import torch

from src.utils import args, log_interval


# ----- Evaluation -----

@torch.no_grad()
def evaluate(model, criterion, valid_loader):
    model.eval()
    acc_losses = {}
    for i, (x, _) in enumerate(valid_loader):
        # forward pass
        x = x.to(args.device)
        output = model(x)
        _, diagnostics = criterion(x, output, model)
        # gather statistics
        acc_losses = Counter(acc_losses) + Counter(diagnostics)
        log_interval(i+1, len(valid_loader), acc_losses)
    avg_losses = {k: acc_losses[k] / len(valid_loader) for k in acc_losses}
    return avg_losses


# ----- Train -----

def train(model, criterion, optimizer, scheduler, train_loader):
    model.train()
    acc_losses = {}
    for i, (x, _) in enumerate(train_loader):
        optimizer.zero_grad()
        # forward pass
        x = x.to(args.device)
        output = model(x)
        loss, diagnostics = criterion(x, output, model)
        # back-prop
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        # gather statistics
        acc_losses = Counter(acc_losses) + Counter(diagnostics)
        log_interval(i+1, len(train_loader), acc_losses)
    avg_losses = {k: acc_losses[k] / len(train_loader) for k in acc_losses}
    return avg_losses


if __name__ == "__main__":
    pass
