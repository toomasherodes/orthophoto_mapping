import torch

def pix_acc(target, outputs, num_classes, ignore_index=-1):
    valid_mask = (target != ignore_index)

    target = target[valid_mask]
    _, preds = torch.max(outputs.data, dim=1)
    preds = preds[valid_mask]

    labeled = target.numel()

    correct = (preds == target).sum().item()
    
    return labeled, correct