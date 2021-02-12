from modelT import UNet
from brats_dataset import Brats_db
#from brats_dataset import Brats_db_one_modality
import torch
import time
from tqdm import tqdm
#from matplotlib import pyplot
import numpy as np


num_classes  = 2
threshold = 0.5 

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

def test(model, data, device=None):
    print('Testing is starting')
    since = time.time()
    metrics_dict = {'test': {}}
    with torch.no_grad():
        for inputs, labels in tqdm(data, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)#output is between 0, 1
            predictions = torch.zeros_like(output, dtype=torch.long)
            predictions[output> threshold] = 1 
            predictions[output<=threshold] = 0

            metrics_dict['test'] = compute_metric(predictions, labels, device, metrics_dict['test']) 

    str_out = ''
    for k in metrics_dict['test']: 
        metric = [v / len(data) for v in metrics_dict['test'][k]]
        str_out += f'{k} [' 
        str_out += ''.join([f'({c:1d}): {v*100:0.2f}, ' for c, v in enumerate(metric)])
        str_out =  str_out[:-2] + ']; '
    print(f'{str_out} \n')

    with open('checkpoints/test-metrics.txt', 'a') as f:
        #if epoch == 0: f.write(' '.join(sys.argv) + '\n' ) #to save the running command
        f.write(f' test:% \n {str_out} \n')

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #return outputs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet_mdl = UNet(n_channels=3, n_classes=num_classes, device=device)
    unet_mdl.to(device)

    testDL = torch.utils.data.DataLoader(Brats_db(subset_name='test'), num_workers=32, batch_size=1)
    #testDL = torch.utils.data.DataLoader(Brats_slice(subset_name='test'), num_workers=64, batch_size=2)
    #testDL = torch.utils.data.DataLoader(Brats_db_one_modality(modality='t1gd',subset_name='test'), num_workers=32, batch_size=1)

    unet_mdl.load_state_dict(torch.load(f'checkpoints/best_mdl.pth',map_location=device))
    unet_mdl.eval()

    test(unet_mdl,testDL,device=device)
