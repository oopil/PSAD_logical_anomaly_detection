import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize
import SimpleITK as sitk
import math
from PIL import Image, ImageOps, ImageFilter, ImageChops
# from torchvision.transforms import Compose
from glob import glob
from pdb import set_trace
import albumentations as A

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

class DataSet2D(data.Dataset):
    def __init__(self, \
        root,
        seg_dir, 
        obj_name, 
        label=255, 
        size=(256,256),
        rot_angle=(0,360),
        transform=None):
        print("root:", root)

        self.root = root
        self.seg_dir = seg_dir
        self.size = size
        self.label = label
        self.rot_angle = rot_angle
        self.obj_name = obj_name
        self.transform = transform
        self.is_mirror = True
        self.files_pre =  glob(f"{root}/orig_512/{obj_name}/train/good/*.png")
        self.files_pre.sort()
        self.files = []
        for e in self.files_pre:
            fname = os.path.basename(e)
            lpath = f"{self.root}/{self.seg_dir}/{self.obj_name}/pred_{fname}"
            if os.path.exists(lpath):
                self.files.append(e)
        print('{} images are loaded!'.format(len(self.files)))

        self.aug = A.Compose([
            A.ToGray(p=0.2),
            A.Posterize(p=0.2),
            A.Equalize(p=0.2),
            A.Sharpen(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Solarize(p=0.2),
            A.ColorJitter(p=0.2)
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fpath = self.files[index]
        fname = os.path.basename(fpath)
        lpath = f"{self.root}/{self.seg_dir}/{self.obj_name}/pred_{fname}"
        random_angle = random.choice(list(range(self.rot_angle[0],self.rot_angle[1])))
        # random_angle = random.choice(list(range(360)))
        try:
            im = Image.open(fpath)
        except:
            im = Image.open(fpath.replace(".png",".jpg"))
        # im = im.resize(self.size)
        im = im.convert("RGB")#.filter(ImageFilter.BLUR)
        # im = ImageChops.offset(im, -10, 0)
        im = im.rotate(random_angle)
        image = np.array(im)
        image = self.aug(image=image)['image']
        image = np.array(image)/255

        ## coordinate vector
        w,h,_ = image.shape
        x = np.linspace(0,1,w)
        y = np.linspace(0,1,h)
        xx,yy = np.meshgrid(x,y)
        coord = np.stack([xx,yy,np.zeros_like(yy)],axis=2)*255
        coord_im = Image.fromarray(coord.astype(np.uint8))
        coord = np.array(coord_im)[:,:,:2]/255
        coord = np.swapaxes(coord, 1, 2)
        coord = np.swapaxes(coord, 0, 1) # [2,w,h]
        coord_orig = torch.tensor(coord).float()*30
        coord_im = coord_im.rotate(random_angle)
        coord = np.array(coord_im)[:,:,:2]/255
        coord = np.swapaxes(coord, 1, 2)
        coord = np.swapaxes(coord, 0, 1) # [2,w,h]
        coord_rot = torch.tensor(coord).float()*30

        # normalize
        image = image - imagenet_mean
        image = image / imagenet_std

        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 0, 1) # [3,w,h]

        im_label = Image.open(lpath)
        # im_label = im_label.resize(self.size, Image.NEAREST)
        im_label = im_label.rotate(random_angle, Image.NEAREST)

        label = np.array(im_label)

        if np.random.rand(1) <= 0.5:  # flip W
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=0).copy()
        if np.random.rand(1) <= 0.5:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=1).copy()

        image = torch.tensor(image).float()
        label = torch.tensor(label).long()

        sample = {
            "image": image,
            "label": label,
            "coord_rot": coord_rot,
            "coord_orig": coord_orig,
            "original_size": image.shape[1:],

        }
        return sample


class ValDataSet2D(data.Dataset):
    def __init__(self, \
        root, 
        obj_name, 
        label=255, 
        ref_name="001", 
        n_shot=1, 
        n_zero=3, 
        size=(256,256),
        transform=None,
        save_dir="orig_512_seg"):

        self.size = size
        self.root = root
        self.label = label
        self.obj_name = obj_name
        self.save_dir = save_dir
        self.transform = transform
        self.files = glob(f"{root}/orig_512/{obj_name}/*/*/*.png")
        self.files.sort()
        print('{} images are loaded!'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fpath = self.files[index]
        lpath = self.files[index].replace("Images","Annotations")

        # read nii file
        try:
            im = Image.open(fpath).convert("RGB")#.filter(ImageFilter.BLUR)
        except:
            im = Image.open(fpath.replace(".png",".jpg")).convert("RGB")

        # im = im.resize(self.size)
        image = np.array(im)/255
        ## coordinate vector
        w,h,_ = image.shape
        x = np.linspace(0,1,w)
        y = np.linspace(0,1,h)
        xx,yy = np.meshgrid(x,y)
        coord = np.stack([xx,yy,np.zeros_like(yy)],axis=2)*255
        coord_im = Image.fromarray(coord.astype(np.uint8))
        coord = np.array(coord_im)[:,:,:2]/255
        coord = np.swapaxes(coord, 1, 2)
        coord = np.swapaxes(coord, 0, 1) # [2,w,h]
        coord_orig = torch.tensor(coord).float()*30


        image = image - imagenet_mean
        image = image / imagenet_std
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 0, 1) # [3,w,h]
        opath = fpath.replace("orig_512",self.save_dir) 

        image = torch.tensor(image).float()
        sample = {
            "image": image,
            "opath": opath,
            "coord_orig": coord_orig,
            "original_size": image.shape[1:],

        }
        return sample
        # return image, np.zeros_like(image)#, coord_orig

def my_collate(batch):
    return batch