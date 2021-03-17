import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_rot_mat, rotate_torch
from unet import UNet

class Filter_Func(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=(1,3)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(0, 1))

    def forward(self,x):
        return self.conv(x)

class Tune_Func(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=(2),stride=1, padding=(1, 1, 1))
        self.norm = nn.BatchNorm3d(num_features=1)

    def forward(self,x):
        poolL = self.pool(x)
        poolL = poolL[:,1:,1:,1:]
        shift = self.norm(poolL.reshape(1,poolL.shape[0],poolL.shape[1],poolL.shape[2],poolL.shape[3]))

        return shift


def linear_reconstruction(minibatch, device, out_size=(1, 155, 284, 284), num_Projections=12):
    ''' minibatch: 12 projections, width, height
        out: the 3D reconstrauction of the 12 projections
        out (155, 284, 284)
    '''
    max_rotation = num_Projections
    degrees = 180.0 // max_rotation

    d = out_size[1] #depth of output tensor
    out = torch.zeros(out_size, dtype=torch.float).to(device)
    for i in range(max_rotation):
        proj = minibatch[i].repeat([d, 1, 1]).to(device)
        proj = proj.reshape(1, 1, *proj.shape)
        rot_mat = get_rot_mat(-i*degrees)
        out += rotate_torch(proj, rot_mat, device).squeeze()

    return out

class ColorProjUNet(nn.Module):
    def __init__(self, n_classes, device, bilinear=True):
        super().__init__()
        self.device  = device
        self.unet = UNet(3, n_classes, bilinear)
        self.filtration_func = Filter_Func(n_classes, 1, (1,3))
        self.finetune_func = Tune_Func()

    def forward(self, x):
        composed_img = x.squeeze()
        logits = self.unet(composed_img)
        filtered = self.filtration_func(logits)
        reconstructed = linear_reconstruction(filtered, self.device, num_Projections=12)
        out = self.finetune_func(reconstructed)

        return out

