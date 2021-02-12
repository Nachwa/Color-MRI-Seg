import torch
import torch.optim as optim
import time
import math
import copy
from modelT import UNet
from brats_dataset import Brats_db
#from brats_dataset import Brats_db_one_modality
import torch.nn as nn
from tqdm import tqdm
import os, sys
from collections import Counter
from statistics import mean

num_classes  = 2
threshold = 0.5

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
   
    def forward(self, inputs, targets, smooth=1):         
        #flatten label and prediction tensors
        #print(f'input: {inputs.shape} target: {targets.shape}')
        inputs = inputs.view(-1).to(device)
        targets = targets.view(-1).to(device)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

 
def compute_metric(inputs, targets, device, metric_phase, smooth=1):
    targetsUnique = list(range(num_classes))
    inputs = inputs.view(-1).to(device)
    targets = targets.view(-1).to(device)

    metrics = {'dice': [], 
                'iou': [], 
                'prec': [], 
                'rcll' : []}

    for i,label in enumerate(targetsUnique):
        inputs_i  = inputs==label
        targets_i = targets==label

        intersection = (inputs_i*targets_i).sum()
        intersection = intersection.item()
        union = inputs_i.sum() + targets_i.sum() - intersection
        union = union.item()

        dice  = 2.0 * intersection / (inputs_i.sum() + targets_i.sum() + smooth).item()
        iou   = intersection / (union + smooth)

        precision = intersection / (targets_i.sum() + smooth).item()
        recall    = intersection / (inputs_i.sum() + smooth).item() 

        metrics['dice'].append(dice)
        metrics['iou'].append(iou)
        metrics['rcll'].append(recall)
        metrics['prec'].append(precision)
    
    for m in metrics:
        if m in metric_phase:
            metrics[m] = [x + y for x, y in zip(metrics[m], metric_phase[m])]
    ''' to compute mean
    for m in metrics:
        mean_metric = metrics[m].mean()
        metrics[m].append(mean_metric)
    '''
    return metrics


def trainModel(model, criterion, optimizer, scheduler, dataloaders, num_epoch=1000, device=None, criterion2=None):
    since = time.time()
    best_res = float('inf')
    early_stop = []
    patience_early_stop = 5

    for epoch in range(num_epoch):
        
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print('-' * 10)
        
        metrics_dict = {'train': {}, 'val': {}}
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            
            dataloader = dataloaders[phase]
            for inputs, labels in tqdm(dataloader, leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) #output is between 0, 1
                    predictions = torch.zeros_like(outputs, dtype=torch.long)
                    predictions[outputs> threshold] = 1 
                    predictions[outputs<=threshold] = 0

                    outputs = outputs.contiguous()
                    labels = labels.contiguous()

                    loss = 0.5*criterion(outputs, labels) + 0.5*criterion2(outputs.squeeze(0), labels)
                    #print(f'input {outputs.shape} labels {labels.shape}')
                    #loss = 0.5*criterion(outputs.squeeze(1), labels) + 0.5*criterion2(outputs.squeeze(1), labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += 100 * torch.sum(predictions == labels.long()).float() / labels.numel()
                metrics_dict[phase] = compute_metric(predictions, labels, device, metrics_dict[phase]) 

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_corrects / len(dataloader)

            str_out = ''
            for k in metrics_dict[phase]: 
                metric = [v / len(dataloader) for v in metrics_dict[phase][k]]
                str_out += f'{k} [' 
                str_out += ''.join([f'({c:1d}): {v*100:0.2f}, ' for c, v in enumerate(metric)])
                str_out =  str_out[:-2] + ']; '
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}%')
            print(f'{str_out} \n')


            # Saving training info
            with open('checkpoints/metrics.txt', 'a') as f:
                if epoch == 0: f.write(' '.join(sys.argv) + '\n' ) #to save the running command
                f.write(f'{phase} epoch: {epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}% \n {str_out} \n')

            early_stop_break = False
            if phase == 'val':
                torch.save(model.state_dict(), f'checkpoints/current_ep.pth')
                scheduler.step(epoch_loss)

                epoch_diceMean = mean(metrics_dict['val'].get('dice'))
                if len(early_stop)<patience_early_stop:
                    early_stop.append(epoch_diceMean)
                else:
                    early_stop.pop(0) #remove the first element
                    early_stop.append(epoch_diceMean)
                    if early_stop[0] > early_stop[-1] :
                        print('early stopping ...', early_stop[0], early_stop[-1])
                        early_stop_break = True                  
                if epoch_loss < best_res:
                    best_res = epoch_loss
                    torch.save(model.state_dict(), f'checkpoints/best_mdl.pth')
            
        #if early_stop_break: break
        
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet_mdl = UNet(n_channels=3, n_classes=num_classes, device=device)

    unet_mdl.to(device)

    trainDL = torch.utils.data.DataLoader(Brats_db(subset_name='train'), num_workers=32, batch_size=1)
    valDL = torch.utils.data.DataLoader(Brats_db(subset_name='validation'), num_workers=32, batch_size=1)
    #trainDL = torch.utils.data.DataLoader(Brats_slice(subset_name='train'), num_workers=64, batch_size=2)
    #valDL = torch.utils.data.DataLoader(Brats_slice(subset_name='validation'), num_workers=64, batch_size=2)
    #trainDL = torch.utils.data.DataLoader(Brats_db_one_modality(modality='t1gd',subset_name='train'), num_workers=32, batch_size=1)
    #valDL = torch.utils.data.DataLoader(Brats_db_one_modality(modality='t1gd',subset_name='validation'), num_workers=32, batch_size=1)
    dataset ={"train": trainDL,"val": valDL}

    criterion = DiceLoss()
    criterion2 = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(unet_mdl.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    modelFt = trainModel(unet_mdl, criterion, optimizer, scheduler, dataset, num_epoch=100, device=device, criterion2 = criterion2)


    #testDL = torch.utils.data.DataLoader(LoadDataset('test'), num_workers=8)