import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BinaryDiceWithLogitsLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceWithLogitsLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = torch.sigmoid(predict)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class DiceWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
   
    def forward(self, inputs, targets):      
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1).to(device)
        targets = targets.view(-1).to(device)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.epsilon) / (inputs.sum() + targets.sum() + self.epsilon)  
        
        return 1 - dice


def GaussianNormalization(img, whole_std):
    '''whole: MRI 3D image, img: one projected image
    '''
    norm = img/whole_std
    scaler = PyTMinMaxScalerVectorized()
    norm = scaler(norm, device='cuda')
    return norm

class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [0, 1].
    used for Gaussian Normalization and Z-Score
    """
    def __call__(self, tensor, device='cuda'):
        dist = (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0])
        dist[torch.isclose(dist,torch.zeros(dist.shape, device=device))] = 1.
        scale = 1.0 /  dist
        tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
        return tensor



def compute_rotation_matrices(num_Projections):
    degrees = 180.0 // num_Projections
    
    rot_mat = torch.zeros((num_Projections, 3, 4), dtype=torch.float)
    for i in range(num_Projections):
        angle_rad = i*degrees
        rot_mat[i] = get_rot_mat(angle_rad)
    return rot_mat

def get_rot_mat(rotate_angle, device=None):
    '''rotate_angle: the angle of rotation in radians

       rot_mat: the rotated 3d image
    '''
    DEG_TO_RAD = math.pi / 180.0
    angle_rad  = rotate_angle * DEG_TO_RAD

    rot = torch.tensor([[1,0,0,0],
                        [0,math.cos(angle_rad), -math.sin(angle_rad), 0],
                        [0,math.sin(angle_rad), math.cos(angle_rad), 0]]).to(device)
    
    return rot

def rotate_torch(img3d, rot_mat, device=None):
    '''img3d: three dimensional image that is going to be rotated on its x axis
       rot_mat: the rotation matrix 
       out: result: the rotated 3d image
    '''
    rot = rot_mat.unsqueeze(0).to(device)
    
    result_size = img3d.shape #(n, c, d, h, w)
    
    grid   = F.affine_grid(rot, size=result_size, align_corners=True)
    result = F.grid_sample(img3d, grid, mode='bilinear', align_corners=True)

    return result


def compute_metric(inputs, targets, device, metric_phase, smooth=1, num_classes=2):
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

    return metrics