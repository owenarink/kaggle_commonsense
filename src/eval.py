import torch

def accuracy(predictions, targets):
    if not torch.is_tensor(predictions):
        predictions = torch.tensor(predictions)
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)

    targets = targets.long().view(-1)
    pred_labels = torch.argmax(predictions, dim=1)
    return (pred_labels == targets).float().mean().item()

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(model.device)
            y = y.to(model.device).long().view(-1)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.numel()

    return correct / total if total > 0 else 0.0
