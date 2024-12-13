import torch
import numpy as np

from tqdm import tqdm
from utils import draw_translucent_seg_maps
from metrics import pix_acc
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1, ignore_index=-1):
    # Mask invalid pixels
    valid_mask = (target != -1)
    target[target==-1] = 0
    
    # Convert target to one-hot
    num_classes = pred.shape[1]  # Get the number of classes from predictions
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()  # Shape: [batch_size, num_classes, height, width]
    # Mask the predictions and target_one_hot
    valid_mask = valid_mask.unsqueeze(1)  # Add channel dimension: [batch_size, 1, height, width]
    pred = pred * valid_mask  # Apply mask to predictions
    target_one_hot = target_one_hot * valid_mask  # Apply mask to target

    # Convert predictions to probabilities
    pred = torch.softmax(pred, dim=1)
    
    # Compute Dice Loss
    intersection = (pred * target_one_hot).sum(dim=(0, 2, 3))  # Sum over spatial dimensions and batch
    union = pred.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def train(model, train_dataset, train_dataloader, device, optimizer, criterion, classes_to_train):
    model.train()
    train_running_loss = 0.0
    train_running_correct, train_running_label = 0, 0

    num_batches = int(len(train_dataset)/train_dataloader.batch_size)
    prog_bar = tqdm(train_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    counter = 0 # to keep track of batch counter
    num_classes = len(classes_to_train)

    for i, data in enumerate(prog_bar):
        counter += 1
        data, target = data[0].to(device), data[1].to(device)
        # print(f"target {target.shape}")
        optimizer.zero_grad()
        outputs = model(data)['out']

        ##### BATCH-WISE LOSS #####
        loss_ce = criterion(outputs, target)
        train_running_loss += loss_ce.item()
        ###########################

        # For pixel accuracy.
        labeled, correct = pix_acc(target, outputs, num_classes)
        train_running_label += labeled
        train_running_correct += correct
        train_running_pixacc = 1.0 * correct / (np.spacing(1) + labeled)
        #############################

        ##### BACKPROPAGATION AND PARAMETER UPDATION #####
        loss_ce.backward()
        optimizer.step()
        ##################################################

        prog_bar.set_description(desc=f"Loss: {loss_ce.detach().cpu().numpy():.4f} | PixAcc: {train_running_pixacc*100:.2f}")
        
    train_loss = train_running_loss / counter
    pixel_acc = ((1.0 * train_running_correct) / (np.spacing(1) + train_running_label)) * 100

    return train_loss, pixel_acc
    
def validate(model, valid_dataset, valid_dataloader, device, criterion, classes_to_train, label_colors_list, epoch, all_classes, save_dir):
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct, valid_running_label = 0, 0

    num_batches = int(len(valid_dataset)/valid_dataloader.batch_size)
    num_classes = len(classes_to_train)

    with torch.no_grad():
        prog_bar = tqdm(valid_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        counter = 0 # To keep track of batch counter.
        for i, data in enumerate(prog_bar):
            counter += 1
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)['out']
            
            # Save the validation segmentation maps every
            # last batch of each epoch
            if i == num_batches - 1:
                draw_translucent_seg_maps(
                    data, 
                    outputs, 
                    epoch, 
                    i, 
                    save_dir, 
                    label_colors_list,
                )

            ##### BATCH-WISE LOSS #####
            loss_ce = criterion(outputs, target)
            valid_running_loss += loss_ce.item()
            ###########################

            # For pixel accuracy.
            labeled, correct = pix_acc(target, outputs, num_classes)
            valid_running_label += labeled
            valid_running_correct += correct
            valid_running_pixacc = 1.0 * correct / (np.spacing(1) + labeled)
            #############################

            prog_bar.set_description(desc=f"Loss: {loss_ce.detach().cpu().numpy():.4f} | PixAcc: {valid_running_pixacc*100:.2f}")
    
    valid_loss = valid_running_loss / counter
    pixel_acc = ((1.0 * valid_running_correct) / (np.spacing(1) + valid_running_label)) * 100.
    
    return valid_loss, pixel_acc