import numpy as np
import cv2

import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def image_crop(image, size):
    org_h, org_w = np.shape(image)[0], np.shape(image)[1]
    aug = transforms.Compose(
           [
            transforms.RandomCrop((size, size)),
            transforms.Resize((org_h, org_w))
           ])
    return aug(image)

def image_random_affine(image, degree):
    aug = transforms.RandomAffine(degree)
    return aug(image)

def image_horizontal_flip(image):
    aug = transforms.RandomHorizontalFlip(p=1)
    return aug(image)

def image_perspective(image):
    aug = transforms.RandomPerspective()
    return aug(image)

def image_gray(image):
    aug = transforms.RandomGrayscale(p=1)
    return aug(image)

def image_jitter(image):
    aug = transforms.ColorJitter(brightness=(0.2, 3))
    return aug(image)

def image_contrast(image):
    aug = transforms.ColorJitter(contrast=(0.2, 3))
    return aug(image)
    
def image_hue(image):
    org_h, org_w = np.shape(image)[0], np.shape(image)[1]
    aug = transforms.Compose(
           [
            transforms.ToTensor(),
            transforms.ColorJitter(hue=(-0.1, 0.1))
           ])
    res = aug(image)
    res = res.permute(1, 2, 0).cpu().detach().numpy()
    return res

def image_noise(image):    
    org_h, org_w = np.shape(image)[0], np.shape(image)[1]

    aug = transforms.Compose(
           [
            transforms.ToTensor()
           ])
    res = aug(image)

    noise = Variable(torch.zeros(org_h, org_w).cuda())
    noise.data.normal_(0, std=0.3)

    res = res.to(device) + noise.to(device)
    res = res.permute(1, 2, 0).cpu().detach().numpy()
    return res[:,:,0]
        
