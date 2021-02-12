import time
import torch
import torch.nn as nn
from brats_dataset import Brats_db_angle
from PIL import Image
from normalization import GaussianNormalization,PyTMinMaxScalerVectorized
import SimpleITK as sitk
import numpy as np
import matplotlib.image as mpimg
from skimage.color import lab2rgb
import torch.nn.functional as F
import math
import os
from scipy import ndimage

num_classes = 2

root_dir = 'data/'
db_name  = 'test'
imgs_whole = os.listdir(root_dir + db_name + '/whole_img/')
num_Projections = 12
degrees = 180.0 // num_Projections
img_size = (155, 256, 256)
colorC = 3
#device = device

def loadImg(idx,device=None):
    batch = torch.zeros((num_Projections, colorC, img_size[1], img_size[2]), dtype=torch.float) #each batch would have 12 images which are the projections of each image

    sitk_whole = sitk.ReadImage(root_dir+ db_name + '/whole_img/'+ imgs_whole[idx])
    np_whole   = F.pad(torch.from_numpy(sitk.GetArrayFromImage(sitk_whole)),(8, 8, 8, 8), mode='constant', value=0)
    np_whole = np_whole.numpy()
        
    for i in range(num_Projections):
        flair = np.amax(ndimage.rotate(np_whole[0], i*degrees, reshape=False),axis=0)
        t1gd  = np.amax(ndimage.rotate(np_whole[2], i*degrees, reshape=False),axis=0)
        t2    = np.amax(ndimage.rotate(np_whole[3], i*degrees, reshape=False),axis=0)

        flair = GaussianNormalization(torch.from_numpy(flair),np_whole[0].std())
        t1gd = GaussianNormalization(torch.from_numpy(t1gd),np_whole[2].std())
        t2 = GaussianNormalization(torch.from_numpy(t2),np_whole[3].std())

        colored_img = torch.stack([ flair,
                                    t1gd,
                                    t2], dim=0)
        batch[i, :, :, :] = colored_img

    return batch.numpy()

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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    since = time.time()
    for i in range(len(imgs_whole)):
        img = loadImg(i)
        n = 'data/'
        n+=f'{num_Projections}/'
        np.save(n+f'{imgs_whole[i][:-7]}.npy',img)
        print(f'{i}: done with {imgs_whole[i]}, time since start: {time.time() - since}')

    root_dir = 'data/'
    db_name  = 'train'
    imgs_whole = os.listdir(root_dir + db_name + '/whole_img/')
    for i in range(len(imgs_whole)):
        img = loadImg(i)
        n = 'data/'
        n+=f'{num_Projections}/'
        np.save(n+f'{imgs_whole[i][:-7]}.npy',img)
        print(f'{i}: done with {imgs_whole[i]}, time since start: {time.time() - since}')

    root_dir = 'data/'
    db_name  = 'validation'
    imgs_whole = os.listdir(root_dir + db_name + '/whole_img/')
    for i in range(len(imgs_whole)):
        img = loadImg(i)
        n = 'data/'
        n+=f'{num_Projections}/'
        np.save(n+f'{imgs_whole[i][:-7]}.npy',img)
        print(f'{i}: done with {imgs_whole[i]}, time since start: {time.time() - since}')