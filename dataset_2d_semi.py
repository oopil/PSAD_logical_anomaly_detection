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
from PIL import Image, ImageOps, ImageFilter
# from torchvision.transforms import Compose
from glob import glob
from pdb import set_trace
import albumentations as A

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

sc_types = {0: ['000.png', '002.png', '005.png', '006.png', '008.png', '017.png', '021.png', '022.png', '023.png', '026.png', '039.png', '041.png', '049.png', '051.png', '052.png', '054.png', '061.png', '063.png', '065.png', '066.png', '069.png', '070.png', '071.png', '072.png', '074.png', '076.png', '078.png', '082.png', '083.png', '087.png', '095.png', '096.png', '097.png', '101.png', '104.png', '106.png', '107.png', '108.png', '109.png', '110.png', '116.png', '121.png', '122.png', '127.png', '128.png', '129.png', '130.png', '137.png', '140.png', '141.png', '143.png', '150.png', '151.png', '153.png', '154.png', '158.png', '160.png', '165.png', '167.png', '171.png', '179.png', '181.png', '182.png', '183.png', '189.png', '194.png', '199.png', '200.png', '204.png', '205.png', '212.png', '216.png', '217.png', '220.png', '221.png', '227.png', '231.png', '234.png', '235.png', '236.png', '238.png', '239.png', '241.png', '242.png', '246.png', '250.png', '253.png', '254.png', '259.png', '261.png', '267.png', '276.png', '278.png', '279.png', '280.png', '282.png', '283.png', '285.png', '296.png', '297.png', '298.png', '299.png', '300.png', '307.png', '308.png', '310.png', '312.png', '313.png', '314.png', '319.png', '326.png', '328.png', '332.png', '338.png', '340.png', '342.png', '345.png', '348.png', '355.png', '357.png'], 1: ['001.png', '004.png', '009.png', '010.png', '012.png', '013.png', '014.png', '018.png', '019.png', '025.png', '028.png', '029.png', '031.png', '032.png', '035.png', '038.png', '043.png', '045.png', '046.png', '047.png', '048.png', '050.png', '053.png', '055.png', '056.png', '057.png', '058.png', '060.png', '062.png', '064.png', '067.png', '068.png', '081.png', '084.png', '088.png', '092.png', '094.png', '103.png', '105.png', '111.png', '113.png', '115.png', '117.png', '123.png', '124.png', '126.png', '131.png', '133.png', '134.png', '136.png', '144.png', '145.png', '146.png', '148.png', '149.png', '152.png', '157.png', '159.png', '161.png', \
     '162.png', '164.png', '173.png', '174.png', '180.png', '184.png', '185.png', '186.png', '188.png', '191.png', '195.png', '202.png', '206.png', '210.png', '213.png', '214.png', '223.png', '226.png', '229.png', '230.png', '233.png', '240.png', '244.png', '247.png', '248.png', '249.png', '260.png', '262.png', '263.png', '264.png', '265.png', '268.png', '269.png', '271.png', '272.png', '273.png', '275.png', '287.png', '289.png', '290.png', '293.png', '295.png', '302.png', '303.png', '316.png', '317.png', '318.png', '321.png', '322.png', '323.png', '327.png', '329.png', '330.png', '333.png', '335.png', '339.png', '346.png', '349.png', '350.png', '358.png', '359.png'], 2: ['003.png', '007.png', '011.png', '015.png', '016.png', '020.png', '024.png', '027.png', '030.png', '033.png', '034.png', '036.png', '037.png', '040.png', '042.png', '044.png', '059.png', '073.png', '075.png', '077.png', '079.png', '080.png', '085.png', '086.png', '089.png', '090.png', '091.png', '093.png', '098.png', '099.png', '100.png', '102.png', '112.png', '114.png', '118.png', '119.png', '120.png', '125.png', '132.png', '135.png', '138.png', '139.png', '142.png', '147.png', '155.png', '156.png', '163.png', '166.png', '168.png', '169.png', '170.png', '172.png', '175.png', '176.png', '177.png', '178.png', '187.png', '190.png', '192.png', '193.png', '196.png', '197.png', '198.png', '201.png', '203.png', '207.png', '208.png', '209.png', '211.png', '215.png', '218.png', '219.png', '222.png', '224.png', '225.png', '228.png', '232.png', '237.png', '243.png', '245.png', '251.png', '252.png', '255.png', '256.png', '257.png', '258.png', '266.png', '270.png', '274.png', '277.png', '281.png', '284.png', '286.png', '288.png', '291.png', '292.png', '294.png', '301.png', '304.png', '305.png', '306.png', '309.png', '311.png', '315.png', '320.png', '324.png', '325.png', '331.png', '334.png', '336.png', '337.png', '341.png', '343.png', '344.png', '347.png', '351.png', '352.png', '353.png', '354.png', '356.png']}
jb_types = {0: ['000.png', '001.png', '005.png', '009.png', '012.png', '013.png', '014.png', '017.png', '020.png', '023.png', '025.png', '035.png', '036.png', '037.png', '039.png', '042.png', '045.png', '046.png', '048.png', '051.png', '052.png', '053.png', '056.png', '058.png', '059.png', '068.png', '069.png', '070.png', '071.png', '080.png', '081.png', '087.png', '091.png', '096.png', '099.png', '100.png', '102.png', '106.png', '115.png', '116.png', '117.png', '118.png', '119.png', '121.png', '130.png', '131.png', '132.png', '137.png', '139.png', '146.png', '147.png', '148.png', '156.png', '159.png', '162.png', '163.png', '164.png', '166.png', '168.png', '172.png', '177.png', '180.png', '183.png', '184.png', '185.png', '188.png', '192.png', '193.png', '195.png', '206.png', '207.png', '211.png', '213.png', '215.png', '216.png', '217.png', '218.png', '221.png', '232.png', '234.png', '239.png', '240.png', '241.png', '247.png', '248.png', '249.png', '250.png', '253.png', '255.png', '257.png', '258.png', '263.png', '264.png', '266.png', '268.png', '269.png', '275.png', '278.png', '280.png', '290.png', '293.png', '295.png', '297.png', '300.png', '305.png', '308.png', '315.png', '321.png', '323.png', '324.png', '325.png', '333.png'], 1: ['002.png', '003.png', '006.png', '007.png', '011.png', '018.png', '019.png', '021.png', '022.png', '024.png', '028.png', '031.png', '032.png', '034.png', '047.png', '050.png', '060.png', '063.png', '064.png', '065.png', '066.png', '067.png', '072.png', '073.png', '075.png', '076.png', '078.png', '082.png', '083.png', '084.png', '088.png', '092.png', '093.png', '097.png', '101.png', '103.png', '105.png', '109.png', '110.png', '111.png', '112.png', '113.png', '125.png', '127.png', '135.png', '140.png', '141.png', '142.png', '143.png', '145.png', '150.png', '157.png', '158.png', '165.png', '169.png', \
    '170.png', '171.png', '173.png', '174.png', '176.png', '178.png', '182.png', '187.png', '191.png', '198.png', '199.png', '200.png', '202.png', '208.png', '209.png', '210.png', '214.png', '219.png', '228.png', '230.png', '236.png', '237.png', '238.png', '243.png', '245.png', '256.png', '260.png', '261.png', '262.png', '265.png', '267.png', '271.png', '274.png', '276.png', '279.png', '281.png', '284.png', '286.png', '287.png', '289.png', '292.png', '294.png', '298.png', '306.png', '310.png', '312.png', '313.png', '316.png', '317.png', '318.png', '320.png', '322.png', '326.png', '327.png', '329.png', '330.png', '331.png'], 2: ['004.png', '008.png', '010.png', '015.png', '016.png', '026.png', '027.png', '029.png', '030.png', '033.png', 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         '038.png', '040.png', '041.png', '043.png', '044.png', '049.png', '054.png', '055.png', '057.png', '061.png', '062.png', '074.png', '077.png', '079.png', '085.png', '086.png', '089.png', '090.png', '094.png', '095.png', '098.png', '104.png', '107.png', '108.png', '114.png', '120.png', '122.png', '123.png', '124.png', '126.png', '128.png', '129.png', '133.png', '134.png', '136.png', '138.png', '144.png', '149.png', '151.png', '152.png', '153.png', '154.png', '155.png', '160.png', '161.png', '167.png', '175.png', '179.png', '181.png', '186.png', '189.png', '190.png', '194.png', '196.png', '197.png', '201.png', '203.png', '204.png', '205.png', '212.png', '220.png', '222.png', '223.png', '224.png', '225.png', '226.png', '227.png', '229.png', '231.png', '233.png', '235.png', '242.png', '244.png', '246.png', '251.png', '252.png', '254.png', '259.png', '270.png', '272.png', '273.png', '277.png', '282.png', '283.png', '285.png', '288.png', '291.png', '296.png', '299.png', '301.png', '302.png', '303.png', '304.png', '307.png', '309.png', '311.png', '314.png', '319.png', '328.png', '332.png', '334.png']}


class DataSet2D(data.Dataset):
    def __init__(self,
                 root,
                 obj_name,
                 label=255,
                 ref_name="001",
                 n_shot=5,
                 n_zero=3,
                 size=(256, 256),
                 transform=None):
        print("root:", root)

        self.root = root
        self.size = size
        self.label = label
        self.obj_name = obj_name
        self.transform = transform
        self.is_mirror = True
        self.support = glob(
            f"{root}/Images_fewshot_512/{obj_name}/*.png")[:n_shot]  # [:3]
        # self.support =  glob(f"{root}/Images_fewshot_512/{obj_name}/*.png")
        self.files = glob(f"{root}/Images_512/{obj_name}/*.png")

        # if n_shot > 1 :
        #     random.seed(1234)
        #     all_files = glob(f"{root}/Images_fewshot_512/{obj_name}/*.png")
        #     self.files = random.sample(all_files,n_shot)
        #     print(self.files)
        #     # self.files = [f"{root}/Images/{obj_name}/{str(i).zfill(n_zero)}.png" for i in range(n_shot)]
        print('{} unlabeled images are loaded!'.format(len(self.files)))
        print('{} support images are loaded!'.format(len(self.support)))

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
        fpath = random.choice(self.support)
        lpath = fpath.replace("Images_fewshot_512", "Annotations_fewshot_512")
        # lpath = fpath.replace("Images_fewshot_512","Annotations_fewshot_512")
        un_fpath = self.files[index]
        if self.obj_name == "juice_bottle":
            s_idx = self.support.index(fpath)
            new_fname = random.choice(jb_types[s_idx])

            un_fpath = os.path.join(os.path.dirname(
                os.path.abspath(un_fpath)), new_fname)
        elif self.obj_name == "splicing_connectors":
            s_idx = self.support.index(fpath)
            new_fname = random.choice(sc_types[s_idx])
            un_fpath = os.path.join(os.path.dirname(
                os.path.abspath(un_fpath)), new_fname)
        else:
            pass

        if self.obj_name in ["screw_bag", "breakfast_box"]:
            random_angle = random.choice(list(range(360)))
        else:
            random_angle = random.choice(list(range(-20, 20)))

        try:
            im = Image.open(fpath)
        except:
            im = Image.open(fpath.replace(".png", ".jpg"))
        # im = im.resize(self.size)
        im = im.convert("RGB")  # .filter(ImageFilter.BLUR)
        im = im.rotate(random_angle)
        image = np.array(im)
        image = self.aug(image=image)['image']
        image = np.array(image)/255

        # coordinate vector
        w, h, _ = image.shape
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        coord = np.stack([xx, yy, np.zeros_like(yy)], axis=2)*255
        coord_im = Image.fromarray(coord.astype(np.uint8))
        coord = np.array(coord_im)[:, :, :2]/255
        coord = np.swapaxes(coord, 1, 2)
        coord = np.swapaxes(coord, 0, 1)  # [2,w,h]
        coord_orig = torch.tensor(coord).float()*1
        coord_im = coord_im.rotate(random_angle)
        coord = np.array(coord_im)[:, :, :2]/255
        coord = np.swapaxes(coord, 1, 2)
        coord = np.swapaxes(coord, 0, 1)  # [2,w,h]
        coord_rot = torch.tensor(coord).float()*1

        image = image - imagenet_mean
        image = image / imagenet_std

        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 0, 1)  # [3,w,h]

        im_label = Image.open(lpath)
        # im_label = im_label.resize(self.size, Image.NEAREST)
        im_label = im_label.rotate(random_angle, Image.NEAREST)

        # if self.transform is not None:
        #     im = self.tranform(im)

        label = np.array(im_label)
        # print(np.unique(label))

        # if np.random.rand(1) <= 0.5:  # flip W
        #     image = np.flip(image, axis=1).copy()
        #     label = np.flip(label, axis=0).copy()
        # if np.random.rand(1) <= 0.5:
        #     image = np.flip(image, axis=2).copy()
        #     label = np.flip(label, axis=1).copy()

        image = torch.tensor(image).float()
        label = torch.tensor(label).long()

        # read unlabeled images
        un_im = Image.open(un_fpath)
        # un_im = un_im.resize(self.size)

        # random_angle = random.choice(list(range(360)))
        # un_im = un_im.rotate(random_angle)
        un_image = np.array(un_im)/255
        un_image = un_image - imagenet_mean
        un_image = un_image / imagenet_std

        un_image = np.swapaxes(un_image, 1, 2)
        un_image = np.swapaxes(un_image, 0, 1)  # [3,w,h]
        un_image = torch.tensor(un_image).float()

        sample = {
            "image": image,
            "label": label,
            "un_image": un_image,
            "coord_rot": coord_rot,
            "coord_orig": coord_orig,
            "original_size": image.shape[1:],

        }
        return sample


class ValDataSet2D(data.Dataset):
    def __init__(self,
                 root,
                 obj_name,
                 label=255,
                 ref_name="001",
                 n_shot=1,
                 n_zero=3,
                 size=(256, 256),
                 transform=None):

        self.size = size
        self.root = root
        self.label = label
        self.obj_name = obj_name
        self.transform = transform
        # self.files = [f"{root}/Images/{obj_name}/{str(e).zfill(n_zero)}.png" for e in range(1,6)]
        self.files = glob(f"{root}/Images_512/{obj_name}/*.png")
        self.files.sort()
        print('{} images are loaded!'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fpath = self.files[index]
        lpath = self.files[index].replace("Images", "Annotations")

        # read nii file
        try:
            im = Image.open(fpath).convert("RGB")  # .filter(ImageFilter.BLUR)
        except:
            im = Image.open(fpath.replace(".png", ".jpg")).convert("RGB")

        # im = im.resize(self.size)
        image = np.array(im)/255
        # coordinate vector
        w, h, _ = image.shape
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        coord = np.stack([xx, yy, np.zeros_like(yy)], axis=2)*255
        coord_im = Image.fromarray(coord.astype(np.uint8))
        coord = np.array(coord_im)[:, :, :2]/255
        coord = np.swapaxes(coord, 1, 2)
        coord = np.swapaxes(coord, 0, 1)  # [2,w,h]
        coord_orig = torch.tensor(coord).float()*1

        image = image - imagenet_mean
        image = image / imagenet_std
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 0, 1)  # [3,w,h]

        image = torch.tensor(image).float()
        return image, np.zeros_like(image), coord_orig


def my_collate(batch):
    return batch
