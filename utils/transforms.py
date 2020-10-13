import torch
from torchvision import transforms
from skimage import transform
import numpy as np

def func_rf(img, p):
    if p==0:
        img=img
    elif p==1:
        img=np.flip(img, axis=1)
    elif p==2:
        img=np.rot90(img, k=1, axes=(0,1))
    elif p==3:
        img=np.flip(img, axis=1)
        img=np.rot90(img, k=1, axes=(0,1))
    elif p==4:
        img=np.rot90(img, k=2, axes=(0,1))
    elif p==5:
        img=np.flip(img, axis=1)
        img=np.rot90(img, k=2, axes=(0,1))
    elif p==6:
        img=np.rot90(img, k=3, axes=(0,1))
    elif p==7:
        img=np.flip(img, axis=1)
        img=np.rot90(img, k=3, axes=(0,1))

    # The flip is realized by giving -1 to the slice of the variable,
    # and torch does not support slices with negative numbers
    img=img.copy()

    return img 


class RotateFlip(object):
    def __call__(self, sample):
        p = torch.randint(0,8,(1,))
        return {k: func_rf(sample[k], p) for k in sample}


def to_tensor(img):
    img = img.transpose((2, 0, 1))
    return torch.from_numpy(img).float()


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        return {k: to_tensor(sample[k]) for k in sample}
