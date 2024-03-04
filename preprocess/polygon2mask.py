import os
import sys
import pdb
import json
import shutil
import numpy as np
from glob import glob
from PIL import Image,ImageDraw

# root_dir = '/media/NAS/nas_187/soopil/project/battery/sample_crack/data/imgs_train'
# src_dir = f"{root_dir}/imgs"
shot = "fewshot"
trg_dir = f"./{shot}_mask"
# trg_dir = f"./{shot}_mask_vis"

if not os.path.exists(trg_dir):
    os.makedirs(trg_dir)

label_fpath = f"./{shot}/ano_vgg_v6.json"
# label_fpath = f"{root_dir}/{label_fname}"
print(label_fpath)
with open(label_fpath, "r") as json_file:
    data = json.load(json_file)


palette = [0, 0, 0, 204, 241, 227, 112, 142, 18, 254, 8, 23, 207, 149, 84, 202, 24, 214,
           230, 192, 37, 241, 80, 68, 74, 127, 0, 2, 81, 216, 24, 240, 129, 20, 215, 125, 161, 31, 204,
           254, 52, 116, 117, 198, 203, 4, 41, 68, 127, 252, 61, 21, 3, 142, 40, 10, 159, 241, 61, 36,
           14, 175, 77, 144, 61, 115, 131, 79, 97, 109, 177, 163, 58, 198, 140, 17, 235, 168, 47, 128, 91,
           238, 103, 45, 124, 35, 228, 101, 48, 232, 74, 124, 114, 78, 49, 30, 35, 167, 27, 137, 231, 47,
           235, 32, 39, 56, 112, 32, 62, 173, 79, 86, 44, 201, 77, 47, 217, 246, 223, 57, ]
# Pad with zeroes to 768 values, i.e. 256 RGB colours
palette = palette + [0]*(768-len(palette))

# pdb.set_trace()
imgs = list(data.keys())
for img in imgs:
    print(img)
    img_path = f"./{shot}/{img}"
    im = Image.open(img_path).convert("RGB")
    (w,h) = im.size

    d = data[img]
    num_objs = len(d['regions'])
    mask = Image.new('L', (w,h), 0)
    for idx_obj in reversed(range(num_objs)):
        x_coords = d['regions'][str(idx_obj)]['shape_attributes']['all_points_x']
        y_coords = d['regions'][str(idx_obj)]['shape_attributes']['all_points_y']
        coords = [(x,y) for x,y in zip(x_coords, y_coords)]
        label = int(d['regions'][str(idx_obj)]['region_attributes']['label'])
        # print(len(coords), label)
        ImageDraw.Draw(mask).polygon(coords, fill=(label)*1)
        print(idx_obj,label)
        # ImageDraw.Draw(mask).polygon(coords, outline=idx_obj+1, fill=idx_obj+1)
    mask = np.array(mask)
    print(mask.shape, np.unique(mask))

    opath = f"{trg_dir}/{img}"
    # new_im = Image.fromarray(mask)
    # new_im.save(opath)
    pi = Image.fromarray(mask,'P')
    pi.putpalette(palette)
    # pi.show()
    pi.save(opath)

    