import math
import numpy as np
import os
import SimpleITK as sitk
from normalization import GaussianNormalization
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from skimage.color import lab2rgb

class Brats_db_angle(Dataset):
    def __init__(self, subset_name = 'train', root_dir = './', num_Projections = 12):
        self.root_dir = root_dir + 'data/'
        self.db_name  = subset_name
        self.imgs_whole = os.listdir(self.root_dir + self.db_name + '/whole_img/')
        self.num_Projections = num_Projections
        self.degrees = 180.0 // self.num_Projections
        self.img_size = (155, 256, 256)
        self.colorC = 3
        #self.device = device

    def __len__(self):
        return len(self.imgs_whole)

    def __getitem__(self, idx):
        #batch = torch.zeros((self.num_Projections, self.colorC, self.img_size[1], self.img_size[2]), dtype=torch.float) #each batch would have 12 images which are the projections of each image

        #sitk_whole = sitk.ReadImage(self.root_dir+ self.db_name + '/whole_img/'+ self.imgs_whole[idx])
        #np_whole   = F.pad(torch.from_numpy(sitk.GetArrayFromImage(sitk_whole)),(8, 8, 8, 8), mode='constant', value=0)

        sitk_label = sitk.ReadImage(self.root_dir + self.db_name + '/label/' + self.imgs_whole[idx])
        labels     = F.pad(torch.from_numpy(sitk.GetArrayFromImage(sitk_label)),(8, 8, 8, 8), mode='constant', value=0)

        labels_binary = torch.zeros_like(labels)
        labels_binary[labels>1] = 1
        batch = torch.from_numpy(np.load(self.root_dir + f'{self.num_Projections}/{self.imgs_whole[idx][:-7]}.npy')).float()
        '''
        for i in range(self.num_Projections):
            flair = self.rotate_torch(np_whole[0],self.degrees*i).max(dim=0,keepdim=True).values.squeeze()
            t1gd  = self.rotate_torch(np_whole[2],self.degrees*i).max(dim=0,keepdim=True).values.squeeze()
            t2    = self.rotate_torch(np_whole[3],self.degrees*i).max(dim=0,keepdim=True).values.squeeze()

            flair = GaussianNormalization(flair,np_whole[0].std())
            t1gd = GaussianNormalization(t1gd,np_whole[2].std())
            t2 = GaussianNormalization(t2,np_whole[3].std())

            colored_img = torch.stack([ flair,
                                        t1gd,
                                        t2], dim=0)
            batch[i, :, :, :] = colored_img'''
        return batch, labels_binary.float()

    #def rotate_torch(self,img3d, rotate_angle=0):
        '''img3d: three dimensional image that is going to be rotated on its x axis
        rotate_angle: the angle of rotation in radians

        out: im_rot: the rotated 3d image
        '''
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #img3d = img3d.clone().to(device)
        '''
        DEG_TO_RAD = math.pi / 180.0
        angle_rad  = rotate_angle * DEG_TO_RAD

        image_size = img3d.shape
        rot = torch.tensor([[1,0,0,0],
                            [0,math.cos(angle_rad), -math.sin(angle_rad), 0],
                            [0,math.sin(angle_rad), math.cos(angle_rad), 0]]).unsqueeze(0)
    
        img3d  = img3d.reshape(1,1,image_size[0],image_size[1],image_size[2])
        result = torch.zeros(img3d.shape).float()

        grid   = F.affine_grid(rot, size=result.size())
        result = F.grid_sample(img3d, grid,mode='bilinear').reshape(image_size)

        return result.cpu()'''

class Brats_slice(Dataset):
    def __init__(self, subset_name = 'train', root_dir = './'):
        self.root_dir = root_dir + 'data/'
        self.db_name  = subset_name
        self.imgs_whole = os.listdir(self.root_dir + self.db_name + '/whole_img/')
        self.img_size = (155, 256, 256)
        self.colorC = 3

    def __len__(self):
        return len(self.imgs_whole)*self.img_size[0]

    def __getitem__(self, idx):
        img_idx = idx // self.img_size[0]
        slice_idx = idx % self.img_size[0]

        batch = torch.zeros((1, self.colorC, self.img_size[1], self.img_size[2]), dtype=torch.float)

        sitk_whole = sitk.ReadImage(self.root_dir+ self.db_name + '/whole_img/'+ self.imgs_whole[img_idx])
        np_whole   = sitk.GetArrayFromImage(sitk_whole)

        sitk_label = sitk.ReadImage(self.root_dir + self.db_name + '/label/' + self.imgs_whole[img_idx])
        labels     = torch.from_numpy(sitk.GetArrayFromImage(sitk_label)[slice_idx])
        #if binary classification
        labels_binary = torch.zeros_like(labels)
        labels_binary[labels>1] = 1 
        labels_binary     = F.pad(labels_binary,(8, 8, 8, 8), mode='constant', value=0)
        #labels_binary2 = torch.zeros_like(labels_binary)
        #labels_binary2[labels_binary<1]=1
        #labels = torch.stack([labels_binary2,labels_binary],dim = 0)

        flair = torch.from_numpy(np_whole[0,slice_idx,:,:]).float() #self.imgs_flair[idx*self.num_Projections+i])
        t1gd  = torch.from_numpy(np_whole[2,slice_idx,:,:]).float()
        t2    = torch.from_numpy(np_whole[3,slice_idx,:,:]).float()

        flair = F.pad(GaussianNormalization(flair,np_whole[0].std()),
                        (8,8,8,8), mode='constant', value=0)
        t1gd =  F.pad(GaussianNormalization(t1gd, np_whole[2].std()),
                        (8,8,8,8), mode='constant', value=0)
        t2 =    F.pad(GaussianNormalization(t2,   np_whole[3].std()),
                        (8,8,8,8), mode='constant', value=0)

        colored_img = torch.stack([ flair,
                                    t1gd,
                                    t2], dim=0)
        batch[0, :, :, :] = colored_img
        return batch, labels_binary.float()


class Brats_db(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, subset_name = 'train', root_dir = './'):
        self.root_dir = root_dir + 'data/'
        self.db_name  = subset_name
        self.imgs_whole = os.listdir(self.root_dir + self.db_name + '/whole_img/')
        self.imgs_flair = os.listdir(self.root_dir + self.db_name + '/flair/')
        self.num_Projections = 12#int(len(self.imgs_flair)/len(self.imgs_whole))
        self.degrees  = 15
        self.img_size = (155, 256, 256)
        self.colorC = 3

    def __len__(self):
        return len(self.imgs_whole)

    def __getitem__(self, idx):
        batch = torch.zeros((self.num_Projections, self.colorC, self.img_size[1], self.img_size[2]), dtype=torch.float) #each batch would have 12 images which are the projections of each image

        sitk_whole = sitk.ReadImage(self.root_dir+ self.db_name + '/whole_img/'+ self.imgs_whole[idx])
        np_whole   = sitk.GetArrayFromImage(sitk_whole)

        sitk_label = sitk.ReadImage(self.root_dir + self.db_name + '/label/' + self.imgs_whole[idx])
        labels     = torch.from_numpy(sitk.GetArrayFromImage(sitk_label))
        #if binary classification
        labels_binary = torch.zeros_like(labels)
        labels_binary[labels>1] = 1 
        labels_binary     = F.pad(labels_binary,(8, 8, 8, 8), mode='constant', value=0)      
        

        for i in range(self.num_Projections):

            flair = torch.from_numpy(np.load(self.root_dir + self.db_name + f'/flair/{self.imgs_whole[idx][:-7]}_{i*self.degrees}.npy')).float() #self.imgs_flair[idx*self.num_Projections+i])
            t1gd  = torch.from_numpy(np.load(self.root_dir + self.db_name + f'/t1gd/{self.imgs_whole[idx][:-7]}_{i*self.degrees}.npy')).float()
            t2    = torch.from_numpy(np.load(self.root_dir + self.db_name + f'/t2/{self.imgs_whole[idx][:-7]}_{i*self.degrees}.npy')).float()

            # TODO for gausian and z-score normalization look into using tensors fror the imgs and the padding

            flair = F.pad(GaussianNormalization(flair,np_whole[0].std()),
                        (8,8,8,8), mode='constant', value=0)
            t1gd =  F.pad(GaussianNormalization(t1gd, np_whole[2].std()),
                        (8,8,8,8), mode='constant', value=0)
            t2 =    F.pad(GaussianNormalization(t2,   np_whole[3].std()),
                        (8,8,8,8), mode='constant', value=0)

            colored_img = torch.stack([ flair,
                                        t1gd,
                                        t2], dim=0)

            #colored_img = torch.from_numpy(lab2rgb(colored_img.numpy().transpose(1,2,0)).transpose(2,0,1))
            batch[i, :, :, :] = colored_img
        return batch, labels_binary.float()

class Brats_db_one_modality(Dataset):
    def __init__(self, modality='flair', subset_name = 'train', root_dir = './'):
        self.root_dir = root_dir + 'data/'
        self.db_name  = subset_name
        self.imgs_whole = os.listdir(self.root_dir + self.db_name + '/whole_img/')
        self.modality = modality
        self.num_Projections = 12#int(len(self.imgs_flair)/len(self.imgs_whole))
        self.degrees  = 15
        self.img_size = (155, 256, 256)
        self.colorC = 1

    def __len__(self):
        return len(self.imgs_whole)

    def decideModality(self,modality):
        mod = {
            'flair':0,
            't1':1,
            't1gd':2,
            't2':3,
        }
        return mod.get(modality)

    def __getitem__(self, idx):
        batch = torch.zeros((self.num_Projections, self.colorC, self.img_size[1], self.img_size[2]), dtype=torch.float) #each batch would have 12 images which are the projections of each image

        sitk_whole = sitk.ReadImage(self.root_dir+ self.db_name + '/whole_img/'+ self.imgs_whole[idx])
        np_whole   = sitk.GetArrayFromImage(sitk_whole)

        sitk_label = sitk.ReadImage(self.root_dir + self.db_name + '/label/' + self.imgs_whole[idx])
        labels     = torch.from_numpy(sitk.GetArrayFromImage(sitk_label))
        #if binary classification
        labels_binary = torch.zeros_like(labels)
        labels_binary[labels>1] = 1 
        labels_binary     = F.pad(labels_binary,(8, 8, 8, 8), mode='constant', value=0)
        for i in range(self.num_Projections):
            img = torch.from_numpy(np.load(self.root_dir + self.db_name + f'/{self.modality}/{self.imgs_whole[idx][:-7]}_{i*self.degrees}.npy')).float()
            img = F.pad(GaussianNormalization(img,np_whole[self.decideModality(self.modality)].std()),(8,8,8,8), mode='constant', value=0)
            #colored_img = torch.stack([ img,
            #                            img,
            #                            img], dim=0)
            batch[i, :, :, :] = img
        return batch, labels_binary.float()


