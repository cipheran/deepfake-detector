import torch

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).unsqueeze(1)

            outputs = model(x)
            probs = torch.sigmoid(outputs)

            preds = (probs > 0.5).float()

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = correct / total

    return {
        "accuracy": accuracy,
        "total_samples": total
    }