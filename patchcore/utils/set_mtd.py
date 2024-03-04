import glob
import os
from shutil import copyfile
import random

category = os.listdir('../')
category.remove('MT')

for c in category:
    imgs = glob.glob(os.path.join('..', c, '*', '*.jpg'))
    random.shuffle(imgs)
    gts = glob.glob(os.path.join('..', c, '*', '*.png'))

    if 'Free' in c:
        trainset = int(len(imgs)*0.8)
        train_imgs = imgs[:trainset]
        test_imgs = imgs[trainset:]
        if not os.path.exists(os.path.join('.', 'train', 'good')):
            os.makedirs(os.path.join('.', 'train', 'good'))

        for i in train_imgs:
            copyfile(i, os.path.join('.', 'train', 'good', i.split('/')[-1]))

        if not os.path.exists(os.path.join('.', 'test', 'good')):
            os.makedirs(os.path.join('.', 'test', 'good'))

        for i in test_imgs:
            copyfile(i, os.path.join('.', 'test', 'good', i.split('/')[-1]))

    else:
        if not os.path.exists(os.path.join('.', 'test', c)):
            os.makedirs(os.path.join('.', 'test', c))

        test_imgs = imgs
        for i in test_imgs:
            copyfile(i, os.path.join('.', 'test', c, i.split('/')[-1]))

        if not os.path.exists(os.path.join('.', 'ground_truth', c)):
            os.makedirs(os.path.join('.', 'ground_truth', c))

        gt_imgs = gts
        for i in gt_imgs:
            copyfile(i, os.path.join('.', 'ground_truth', c, i.split('/')[-1]))
