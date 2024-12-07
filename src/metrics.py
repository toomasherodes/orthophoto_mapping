import torch

def pix_acc(target, outputs, num_classes):
    labeled = (target > 0) * (target <= num_classes)
    _, preds = torch.max(outputs.data, 1)
    correct = ((preds == target) * labeled).sum().item()
    return labeled, correct