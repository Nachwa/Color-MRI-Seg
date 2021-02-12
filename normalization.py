import torch
''' Different techniques for normalization
'''

class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [0, 1].
    used for Gaussian Normalization and Z-Score
    """
    def __call__(self, tensor):
        dist = (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0])
        dist[torch.isclose(dist,torch.zeros(dist.shape))] = 1.
        scale = 1.0 /  dist
        tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
        return tensor

def HistogramNormalization(img, whole_min, whole_max):
    '''whole: MRI 3D image, img: one projected image
    '''
    g_lir = 0
    g_hir = 1

    #whole_min, whole_max = np.min(whole), np.max(whole)
    norm = ((g_hir - g_lir)/(whole_max-whole_min))*(img-whole_min)+g_lir
    return norm


def HistogramStretching(img, whole_min, whole_max):
    '''whole: MRI 3D image, img: one projected image
    '''
    norm = (img - whole_min)/(whole_max-whole_min)

    return norm

def GaussianNormalization(img,whole_std):
    '''whole: MRI 3D image, img: one projected image
    '''
    norm = img/whole_std
    scaler = PyTMinMaxScalerVectorized()
    norm = scaler(norm)
    
    return norm

def zScore(img,whole_std,whole_mean):
    '''whole: MRI 3D image, img: one projected image
    '''
    
    norm = (img-whole_mean)/whole_std
    scaler = PyTMinMaxScalerVectorized()
    norm = scaler(norm)
    return norm