import os
import cv2
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from glob import glob
from pdb import set_trace
from PIL import Image, ImageDraw
import json

def crop(obj_name, img, is_mask=False):
    w,h = img.size
    arr = np.array(img)
    mg = 0
    if obj_name == "screw_bag":
        mg = 150

    if is_mask:
        arr = arr[mg:h-mg, mg:w-mg]
    else:
        arr = arr[mg:h-mg, mg:w-mg]
    return Image.fromarray(arr)


def zero_pad(img, size, is_mask=False):
    if is_mask:
        new_arr = np.zeros((size, size),  dtype=np.uint8)
        arr = np.array(img)
        print(arr.shape)
        w, h = arr.shape
        new_arr[:w, :h] = arr
    else:
        new_arr = np.zeros((size, size, 3), dtype=np.uint8)
        arr = np.array(img)
        w, h, _ = arr.shape
        new_arr[:w, :h] = arr
    return Image.fromarray(new_arr)


def resize_im(trg_size, im, is_mask=False):
    w, h = im.size
    d_max = np.amax([w, h])
    scale = trg_size/d_max
    new_size = [int(w*scale), int(h*scale)]
    if is_mask:
        im = im.resize(new_size, Image.NEAREST)
    else:
        im = im.resize(new_size)
    return im


obj_names = ["breakfast_box", "juice_bottle",
            "pushpins", "screw_bag", "splicing_connectors"]

def main(obj_name):
    trg_size = 448
    shot = "fewshot"
    print(obj_name)
    dpath = f"./orig/{obj_name}/train/good"
    os.makedirs(f"./Annotations_{trg_size}/{obj_name}", exist_ok=True)
    os.makedirs(f"./Images_{trg_size}/{obj_name}", exist_ok=True)
    fpaths = glob(f"{dpath}/*.png")
    # print(fpaths)
    fpaths.sort()
    for i, fpath in enumerate(fpaths):
        # read nii file
        im = Image.open(fpath).convert("RGB")
        im = crop(obj_name, im, is_mask=False)
        im = resize_im(trg_size, im, is_mask=False)
        im = zero_pad(im, trg_size, is_mask=False)
        im.save(f"./Images_{trg_size}/{obj_name}/{str(i).zfill(3)}.png")
        print(f"{i}/{len(fpaths)}", end="\r")

    return
    dpath = f"./{shot}"
    fpaths = glob(f"{dpath}/{obj_name}*.png")
    fpaths.sort()
    os.makedirs(f"./Annotations_{shot}_{trg_size}/{obj_name}", exist_ok=True)
    os.makedirs(f"./Images_{shot}_{trg_size}/{obj_name}", exist_ok=True)
    # print(fpaths)
    for i, fpath in enumerate(fpaths):
        # read nii file
        im = Image.open(fpath).convert("RGB")
        im = crop(obj_name, im, is_mask=False)
        im = resize_im(trg_size, im, is_mask=False)
        # print(im.size)
        # break
        im = zero_pad(im, trg_size, is_mask=False)
        im.save(f"./Images_{shot}_{trg_size}/{obj_name}/{str(i).zfill(3)}.png")

        lpath = fpath.replace(f"{shot}",f"{shot}_mask")
        im = Image.open(lpath)
        # print(lpath)
        im = crop(obj_name, im, is_mask=True)
        im = resize_im(trg_size, im, is_mask=True)
        im = zero_pad(im, trg_size, is_mask=True)
        im.save(f"./Annotations_{shot}_{trg_size}/{obj_name}/{str(i).zfill(3)}.png")

        print(f"{i}/{len(fpaths)}", end="\r")


if __name__ == '__main__':
    for obj_name in obj_names:
        main(obj_name)
