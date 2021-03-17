from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from glob import glob 
from utils import rotate_torch, compute_rotation_matrices, GaussianNormalization
from os import path
#from skimage.color import lab2rgb

class Brats_db(Dataset):
    def __init__(self, subset_name = 'train', root_dir = './', num_Projections = 12, device='cuda', save_batches=True):
        self.root_dir = root_dir 
        self.db_name  = subset_name
        self.samples = glob(self.root_dir+ self.db_name+ '/whole_img/'+'*.nii.gz')
        self.num_Projections = num_Projections
        self.img_size = (155, 256, 256)
        self.device = device
        self.rot_mat = compute_rotation_matrices(self.num_Projections)
        self.saved_samples_dir = path.join(self.root_dir, self.db_name, 'saved/')
        self.save_batches = save_batches
        self.padding = (22, 22, 22, 22) #(padding_left,padding_right, padding_top,padding_bottom,padding_front,padding_back) 

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        sample_name = self.samples[idx].split('/')[-1]
        if self.save_batches:
            if path.exists(self.saved_samples_dir+sample_name+'.pt'):
                return torch.load(self.saved_samples_dir+sample_name+'.pt')

        degrees = 180.0 // self.num_Projections
        
        # load label img
        label_img = sitk.ReadImage(self.root_dir + self.db_name + '/label/' + sample_name)
        label_img = torch.from_numpy(sitk.GetArrayFromImage(label_img))
        
        # Use binary labels
        labels_binary = torch.zeros_like(label_img)
        labels_binary[label_img>1] = 1

        # load MRI image 
        mri_whole = sitk.ReadImage(self.samples[idx])
        mri_whole = torch.from_numpy(sitk.GetArrayFromImage(mri_whole)).to(self.device)
        mri_whole = F.pad(mri_whole, self.padding, mode='constant', value=0)

        batch = []
        for i in range(self.num_Projections):
            
            # rotate the volume of all modalities, rotated_img : (1, 4, d, h, w)
            rotated_img = rotate_torch(mri_whole.unsqueeze(0), self.rot_mat[i], device=self.device)

            # Get Maximum Intensity Projection image: (4, h, w)
            mip_img = torch.amax(rotated_img, axis=2).squeeze() 

            flair = GaussianNormalization(mip_img[0], mri_whole[0].std())
            t1gd  = GaussianNormalization(mip_img[2], mri_whole[2].std())
            t2    = GaussianNormalization(mip_img[3], mri_whole[3].std())
            
            multimodal_img = torch.stack([flair, t1gd, t2], dim=0)
            batch.append(multimodal_img)

        one_batch = torch.stack(batch), labels_binary.float()
        if self.save_batches:
            torch.save(one_batch, self.saved_samples_dir+sample_name+'.pt')
            
        return one_batch
