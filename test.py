import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

from model import ColorProjUNet
from brats_db import Brats_db
from utils import compute_metric
from opts import args


def test(model, data, device=None):
    print('Testing is starting')
    since = time.time()
    metrics_dict = {'test': {}}

    unpadding = (22, 22, 22, 22)
    with torch.no_grad():
        for inputs, labels in tqdm(data, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device).squeeze()
            labels = labels.contiguous()

            outputs = model(inputs) 
            outputs = outputs.squeeze()
            outputs = outputs[:, unpadding[0]:-unpadding[1], unpadding[2]:-unpadding[3]]
            outputs = outputs.contiguous()

            outputs = outputs.unsqueeze(0)
            labels  = labels.unsqueeze(0)

            outputs = torch.sigmoid(outputs)
            predictions = torch.zeros_like(outputs, dtype=torch.long)
            predictions[outputs> detection_threshold] = 1 
            predictions[outputs<=detection_threshold] = 0

            metrics_dict['test'] = compute_metric(predictions, labels, device, metrics_dict['test']) 

    str_out = ''
    for k in metrics_dict['test']: 
        metric = [v / len(data) for v in metrics_dict['test'][k]]
        str_out += f'{k} [' 
        str_out += ''.join([f'({c:1d}): {v*100:0.2f}, ' for c, v in enumerate(metric)])
        str_out =  str_out[:-2] + ']; '
    print(f'{str_out} \n')

    with open('checkpoints/test-metrics.txt', 'a') as f:
        f.write(f' test:% \n {str_out} \n')

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    num_classes  = 2
    detection_threshold = 0.5
    num_projections = 12
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = args.db_dir 
    

    # create model
    model = ColorProjUNet(n_classes=num_classes, device=device)
    model.to(device)

    testDB = Brats_db(subset_name='test', root_dir=data_dir, num_Projections=num_projections, save_batches=args.save_batches)
    testDL = DataLoader(testDB, num_workers=0, batch_size=1)

    assert (args.checkpoint), "Please specify the checkpoint file to evaluate"
    
    model.load_state_dict(torch.load(f'{args.checkpoint}',map_location=device))
    model.eval()

    test(model, testDL, device=device)
