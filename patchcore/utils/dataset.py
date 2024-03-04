import os
import numpy as np
import faiss
import glob
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, Dataset

class MVTecLOCODataset(Dataset):
    def __init__(self, root, transform, phase, args, anomal_type = None):
        self.transform = transform
        self.phase = phase
        self.args = args
        self.resize = transforms.Resize((int(args.size/8), int(args.size/8)), Image.NEAREST)

        self.img_paths = glob.glob(os.path.join(root, 'orig_512', args.category, phase, '*', '*.png'))
        self.seg_paths = glob.glob(os.path.join(root, 'orig_512_seg_scratch_3level_unet', args.category, phase, '*', '*.png'))

        if anomal_type != None:
            self.img_paths = [i for i in self.img_paths if anomal_type in i or 'good' in i]
            self.seg_paths = [i for i in self.seg_paths if anomal_type in i or 'good' in i]

        self.labels = []
        for i in self.img_paths:
            if 'good' in i:
                self.labels.append(0)
            else:
                self.labels.append(1)

        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)
        seg = Image.open(self.seg_paths[idx]).convert('RGB')
        seg = self.resize(seg)
        seg = self.totensor(seg)
        label = self.labels[idx]
        name = self.img_paths[idx].split('/')[-3:]
        name = name[0] + '/' + name[1] + '/' + name[2].replace('.png', '')

        return img, seg, label, name

class MVTecADDataset(Dataset):
    def __init__(self, root, transform, phase, args, anomal_type = None):
        self.transform = transform
        self.phase = phase
        self.args = args
        self.resize = transforms.Resize((int(args.size/8), int(args.size/8)), Image.NEAREST)

        self.img_paths = glob.glob(os.path.join(root, args.category, phase, '*', '*.png'))

        self.labels = []
        for i in self.img_paths:
            if 'good' in i:
                self.labels.append(0)
            else:
                self.labels.append(1)

        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        name = self.img_paths[idx].split('/')[-3:]
        name = name[0] + '/' + name[1] + '/' + name[2].replace('.png', '')

        return img, img, label, name

class VisADataset(Dataset):
    def __init__(self, root, transform, phase, args, anomal_type = None):
        self.transform = transform
        self.phase = phase
        self.args = args
        self.resize = transforms.Resize((int(args.size/8), int(args.size/8)), Image.NEAREST)

        self.img_paths = glob.glob(os.path.join(root, args.category, phase, '*', '*.JPG'))

        self.labels = []
        for i in self.img_paths:
            if 'good' in i:
                self.labels.append(0)
            else:
                self.labels.append(1)

        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        name = self.img_paths[idx].split('/')[-3:]
        name = name[0] + '/' + name[1] + '/' + name[2].replace('.JPG', '')

        return img, img, label, name