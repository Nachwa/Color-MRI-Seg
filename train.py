import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time, sys
from tqdm import tqdm

from model import ColorProjUNet
from brats_db import Brats_db
from utils import compute_metric, compute_rotation_matrices, BinaryDiceWithLogitsLoss
from opts import args


def train(model, criterion1, optimizer, scheduler, dataloaders, num_epoch=1000, device=None, criterion2=None):
    since = time.time()
    best_res = float('inf')

    unpadding = (22, 22, 22, 22)
    for epoch in range(num_epoch):

        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print('-' * 10)
        
        metrics_dict = {'train': {}, 'val': {}}
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_loss1, running_loss2 = 0.0, 0.0, 0.0

            dataloader = dataloaders[phase]
            dataloader_bar = tqdm(dataloader, leave=False, postfix={'BCE':0, 'Dice':0})
            for iter, (inputs, labels) in enumerate(dataloader_bar):
                inputs = inputs.to(device)
                labels = labels.to(device).squeeze()
                labels = labels.contiguous()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) 
                    outputs = outputs.squeeze() #(155, 240, 240)
                    outputs = outputs[:, unpadding[0]:-unpadding[1], unpadding[2]:-unpadding[3]]
                    outputs = outputs.contiguous()

                    #add the batch dim for loss
                    outputs = outputs.unsqueeze(0)
                    labels = labels.unsqueeze(0)

                    loss1 = criterion1(outputs, labels) 
                    loss2 = criterion2(outputs, labels)
                    loss = loss1 + loss2

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    optimizer.zero_grad()

                    out_data = torch.sigmoid(outputs).detach().data
                    predictions = torch.zeros_like(outputs, dtype=torch.long, device=device)
                    predictions[out_data> detection_threshold] = 1 
                    predictions[out_data<=detection_threshold] = 0
                    
                    running_loss += loss.item()
                    running_loss1 += loss1.item()
                    running_loss2 += loss2.item()
                    dataloader_bar.set_postfix({'ComboLoss':running_loss/(iter+1), 'Dice':running_loss1/(iter+1), 'BCE': running_loss2/(iter+1)})

                    metrics_dict[phase] = compute_metric(predictions, labels, device, metrics_dict[phase]) 
          
            num_iterations = len(dataloader)
            epoch_loss = running_loss / num_iterations
            epoch_loss1 = running_loss1 / num_iterations
            epoch_loss2 = running_loss2 / num_iterations

            str_out = ''
            for k in metrics_dict[phase]: 
                metric = [v / len(dataloader) for v in metrics_dict[phase][k]]
                str_out += f'{k} [' 
                str_out += ''.join([f'{c:1d}: {v*100:0.2f}, ' for c, v in enumerate(metric)])
                str_out =  str_out[:-2] + '] '
            print(f'{phase} Loss: {epoch_loss:.4f}, Dice loss: {epoch_loss1:.4f}, BCE loss: {epoch_loss2:.4f} ')
            print(f'\t {str_out}')

            # Saving training info
            with open('checkpoints/metrics.txt', 'a') as f:
                #to save the running command
                if epoch == 0: f.write(' '.join(sys.argv) + '\n' ) 
                f.write(f'{phase} epoch: {epoch} Loss: {epoch_loss:.4f}, Dice loss: {epoch_loss1:.4f}, BCE loss: {epoch_loss2:.4f} \n \t {str_out} \n')

            if phase == 'val':
                torch.save(model.state_dict(), f'checkpoints/current_ep.pth')
                scheduler.step(epoch_loss)

                if epoch_loss < best_res:
                    best_res = epoch_loss
                    torch.save(model.state_dict(), f'checkpoints/best_mdl.pth')        
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model



if __name__ == '__main__':
    
    num_classes  = 2
    detection_threshold = 0.5
    num_projections = 12
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = args.db_dir

    # create data loader
    train_db = Brats_db(subset_name='train', root_dir=data_dir, num_Projections=num_projections, save_batches=args.save_batches)
    trainDL = DataLoader(train_db, num_workers=0, batch_size=1, shuffle=True)
    valDL   = DataLoader(Brats_db(subset_name='validation', root_dir=data_dir, num_Projections=num_projections, save_batches=args.save_batches), num_workers=0, batch_size=1, shuffle=False)
    dataset = {"train": trainDL,"val": valDL}

    # create model
    model = ColorProjUNet(n_classes=num_classes, device=device)
    model.to(device)

    if args.checkpoint:
        model.load_state_dict(torch.load(f'{args.checkpoint}',map_location=device))

    # prepare loss 
    criterion1 = BinaryDiceWithLogitsLoss()
    criterion2 = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # start training
    train(model, criterion1, optimizer, scheduler, dataset, num_epoch=100, device=device, criterion2 = criterion2)