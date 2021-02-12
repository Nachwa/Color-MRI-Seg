import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class FFunction(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=(1,3)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(0, 1))

    def forward(self,x):
        return self.conv(x)

def linear_reconstruction(minibatch, device, max_rotation=12):
    ''' minibatch: 12 projections, width, height
        out: the 3D reconstrauction of the 12 projections

        minibatch (12, 256, 256)
        for i in range(155):
            for j in range(12):
                out[i] += minibatch[j]
        out (155, 256, 256)
    '''
    degrees = 180.0 // max_rotation

    out = torch.zeros([155, 256, 256]).float().to(device)

    '''for i in range(max_rotation):
        out += minibatch[i].repeat([155, 1, 1]).to(device)
        out = rotate_torch(out,degrees,device)
    
    out = rotate_torch(out,degrees*max_rotation,device)'''
    for i in range(max_rotation):
        proj = minibatch[i].repeat([155, 1, 1]).to(device)
        out += rotate_torch(proj,360-i*degrees,device)

    return out.unsqueeze(0)

def rotate_torch(img3d, rotate_angle=0, device=None):
    '''img3d: three dimensional image that is going to be rotated on its x axis
       rotate_angle: the angle of rotation in radians

       out: im_rot: the rotated 3d image
    '''
    DEG_TO_RAD = math.pi / 180.0
    angle_rad  = rotate_angle * DEG_TO_RAD

    image_size = img3d.shape
    rot = torch.tensor([[1,0,0,0],
                        [0,math.cos(angle_rad), -math.sin(angle_rad), 0],
                        [0,math.sin(angle_rad), math.cos(angle_rad), 0]]).unsqueeze(0).to(device)
    
    img3d  = img3d.reshape(1,1,image_size[0],image_size[1],image_size[2])
    result = torch.zeros(img3d.shape).float().to(device)

    grid   = F.affine_grid(rot, size=result.size())
    result = F.grid_sample(img3d, grid,mode='bilinear').reshape(image_size)

    return result

class TFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=(2),stride=1, padding=(1, 1, 1))
        self.norm = nn.BatchNorm3d(num_features=1)


    def forward(self,x):
        poolL = self.pool(x)
        poolL = poolL[:,1:,1:,1:]
        shift = self.norm(poolL.reshape(1,poolL.shape[0],poolL.shape[1],poolL.shape[2],poolL.shape[3]))

        return torch.sigmoid(shift)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, device, bilinear=True):
        super(UNet, self).__init__()
        self.device  = device
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.drop1 = nn.Dropout(p=0.2)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.drop2 = nn.Dropout(p=0.5)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64 * factor, bilinear)
        self.outc = OutConv(64, n_classes)
        self.fFunct = FFunction(n_classes,1,(1,3))
        self.tFunct = TFunction()

    def forward(self, x):
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        d1 = self.drop1(x4)
        x5 = self.down4(d1)
        d2 = self.drop2(x5)
        x = self.up1(d2, d1)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        f = self.fFunct(logits)
        r = linear_reconstruction(f, self.device)
        t = self.tFunct(r)

        return t
