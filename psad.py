import os
import csv
import cv2
import sys
import pickle
import argparse
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import time
from pdb import set_trace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


class Encoder(nn.Module):
    def __init__(self, avgpool_size=5):
        super().__init__()

        # pretrained CNN feature extractor
        self.init_features()

        def hook_t(module, input, output):
            self.features.append(output)

        # self.model = models.wide_resnet50_2(pretrained=True)
        # self.model = models.wide_resnet101_2(pretrained=True)
        self.model = models.resnet101(pretrained=True)
        # self.model = models.resnet50(pretrained=True)
        # self.model = models.resnet18(pretrained=True)
        # self.model = models.resnet34(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer1[-1].register_forward_hook(hook_t)
        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)
        self.model.layer4[-1].register_forward_hook(hook_t)

        self.ks = avgpool_size
        self.ps = self.ks//2

    def init_features(self):
        self.features = []

    def extract_ft(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

    def forward(self, x):
        features = self.extract_ft(x)
        features[0] = F.avg_pool2d(
            features[0], kernel_size=self.ks, padding=self.ps, stride=1)
        features[1] = F.avg_pool2d(
            features[1], kernel_size=self.ks, padding=self.ps, stride=1)
        features[2] = F.avg_pool2d(
            features[2], kernel_size=self.ks, padding=self.ps, stride=1)

        f0 = F.interpolate(
            features[0], align_corners=True, mode="bilinear", size=x.shape[-2:])
        f1 = F.interpolate(
            features[1], align_corners=True, mode="bilinear", size=x.shape[-2:])
        f2 = F.interpolate(
            features[2], align_corners=True, mode="bilinear", size=x.shape[-2:])
        # print(f0.shape, f1.shape, f2.shape)
        f3 = F.interpolate(
            features[3], align_corners=True, mode="bilinear", size=x.shape[-2:])
        ft = torch.cat([f0, f1, f2], dim=1)

        return ft


def get_args_parser():
    parser = argparse.ArgumentParser(
        'MAE Anomaly Detection Test', add_help=False)
    # Model parameters
    parser.add_argument('--memory_type', default='hcp', type=str, help='')
    parser.add_argument('--scale_type', default='max', type=str, help='')
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--avgpool_size', default=5, type=int)
    parser.add_argument('--standardize', default=1, type=int)
    parser.add_argument('--less_data', default=1, type=int)
    parser.add_argument('--save_csv', default=0, type=int)
    parser.add_argument('--save_img', default=0, type=int)

    # Dataset parameters
    parser.add_argument('--data_path', default='./LOCO_MVTec_AD', type=str,
                        help='dataset path')
    parser.add_argument('--patchcore_path', default='./patchcore_score', type=str,
                        help='dataset path')
    parser.add_argument('--obj_name', default='breakfast_box', type=str,
                        help='dataset path')
    parser.add_argument('--seg_dir', default="orig_512_seg", type=str,
                        help='')
    parser.add_argument('--type', default='logical', type=str,
                        help='type of anomaly')
    parser.add_argument('--ckpt', default='./',
                        help='')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def read_img(args, img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    img = np.array(img) / 255.
    img = img - imagenet_mean
    img = img / imagenet_std
    return img


def read_mask(args, img_path, num_cls):
    img = Image.open(img_path)
    img = torch.tensor(np.array(img))
    img = img.unsqueeze(0)
    onehot_img = torch.zeros_like(img).repeat(num_cls, 1, 1)
    # print(img.shape, onehot_img.shape)
    onehot_img = onehot_img.scatter(0, img.long(), 1)
    return onehot_img.float().cuda()


def get_cls_idx(mask):
    num_cls = mask.shape[0]
    cnt = mask.sum((1, 2))
    indices = []
    for idx in range(1, num_cls):
        if cnt[idx] > 100:
            indices.append(idx)
    return indices


def get_feature(args, encoder, img, mask, cls_list):
    # make it a batch-like
    x = torch.tensor(img).float().cuda()
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    fts = encoder(x)
    global_ft = fts.mean((0, 2, 3))

    cnt = mask.sum((1, 2))
    num_cls = mask.shape[0]
    ft_list = []
    for idx in range(1, num_cls):
        # if cnt[idx] > 100:
        mask_cls = mask[[idx]].unsqueeze(0)
        ft_cls = (fts*mask_cls).sum((0, 2, 3))/(mask_cls.sum()+1)
        ft_list.append(ft_cls)

    ad_ft = torch.cat(ft_list, dim=0)
    return ad_ft


def main(args):
    vis_dir = f"./sample/{args.obj_name}"
    os.makedirs(vis_dir, exist_ok=True)

    ''' number of classes for each component '''
    if args.obj_name == "screw_bag":
        num_cls = 7
    elif args.obj_name == "breakfast_box":
        num_cls = 7
    elif args.obj_name == "juice_bottle":
        num_cls = 9
    elif args.obj_name == "pushpins":
        num_cls = 26  # 16
    elif args.obj_name == "splicing_connectors":
        num_cls = 10  # 7
    else:
        assert False

    ''' model definition '''
    model = Encoder()
    encoder = model.cuda()
    encoder.eval()

    seg_dir = args.seg_dir
    print(seg_dir, f"Num cls : {num_cls}")
    train_mpaths = glob(
        f'{args.data_path}/{seg_dir}/{args.obj_name}/train/good/*.png')
    train_mpaths.sort()
    n_train = len(train_mpaths)//args.less_data
    train_mpaths = train_mpaths[:n_train]
    print(f"# of training samples : {len(train_mpaths)}")

    if args.type == "all":
        abnormal_fpaths = glob(
            f'{args.data_path}/{seg_dir}/{args.obj_name}/test/*_anomalies/*.png')
    else:
        abnormal_fpaths = glob(
            f'{args.data_path}/{seg_dir}/{args.obj_name}/test/{args.type}_anomalies/*.png')

    normal_fpaths = glob(
        f'{args.data_path}/{seg_dir}/{args.obj_name}/test/good/*.png')
    test_mpaths = normal_fpaths+abnormal_fpaths
    print(f"# of testing samples : {len(test_mpaths)}")

    ''' load structural anomaly scores predicted by PatchCore '''
    sa_dpath = f"{args.patchcore_path}/{args.obj_name}"
    sa_score_path = f"{args.patchcore_path}/{args.obj_name}/ADscore.txt"
    fd = open(sa_score_path)
    sa_scores = fd.readlines()
    sa_scores = [e[:-1].split(",")[:2] for e in sa_scores[1:]]
    sa_scores = [[e[0]+".png", float(e[1])] for e in sa_scores]

    with torch.no_grad():
        ''' Update 3 memory banks '''
        memory = []
        cls_indices = []
        dist_patch_list = []
        for i, mpath in enumerate(train_mpaths):
            mask = read_mask(args, mpath, num_cls)
            ad_mft = mask.sum((1, 2))
            cls_index = get_cls_idx(mask)
            cls_indices.append(cls_index)

            ipath = mpath.replace(seg_dir, "orig_512")
            img = read_img(args, ipath)
            ad_ift = get_feature(args, encoder, img, mask, cls_index)
            ad_ft = torch.cat([ad_mft, ad_ift], dim=0)

            img_name = mpath.split("/")[-1]
            memory.append(ad_ft)

            # patchcore scores
            fpath = "/".join(mpath.split("/")[-3:])
            fpath = fpath.split(".")[0]+".pt"
            sa_fpath = f"{sa_dpath}/{fpath}"
            pc_data = torch.load(sa_fpath)
            dist_patch = pc_data["anomaly_scores"]
            dist_patch_list.append(dist_patch)

            print(f"{i}/{len(train_mpaths)}", end='\r')

        memory = torch.stack(memory, dim=0).cuda()  # [N, ft_dim]
        mean = torch.mean(memory, dim=0)
        std = torch.std(memory, dim=0)
        if args.standardize:
            memory = (memory-mean)/(std+1e-10)

        dist_patch_list = torch.stack(dist_patch_list, dim=0)

        mem_seg = memory[:, :num_cls]
        mem_ft = memory[:, num_cls:]

        ''' compute training statistics for adaptive scaling'''
        dist_seg_list = []
        dist_ft_list = []
        for i in range(memory.shape[0]):
            ft = mem_seg[[i]]
            dist = torch.sqrt(((mem_seg-ft)**2).sum(1))
            indices = dist.argsort()
            dist_seg_list.append(dist[indices == 1])
        for i in range(memory.shape[0]):
            ft = mem_ft[[i]]
            dist = torch.sqrt(((mem_ft-ft)**2).sum(1))
            indices = dist.argsort()
            dist_ft_list.append(dist[indices == 1])

        dist_seg_list = torch.cat(dist_seg_list, dim=0)
        dist_ft_list = torch.cat(dist_ft_list, dim=0)

        all_dists_train = torch.stack(
            [dist_seg_list.cpu(), dist_ft_list.cpu(), dist_patch_list], dim=1)
        all_dists_train = np.array(all_dists_train)

        dist_seg_mean = dist_seg_list.mean()
        dist_ft_mean = dist_ft_list.mean()
        dist_patch_mean = dist_patch_list.mean()
        dist_seg_std = torch.std(dist_seg_list, dim=0)
        dist_ft_std = torch.std(dist_ft_list, dim=0)
        dist_patch_std = torch.std(dist_patch_list, dim=0)
        dist_seg_min = dist_seg_list.min()
        dist_ft_min = dist_ft_list.min()
        dist_patch_min = dist_patch_list.min()
        dist_seg_max = dist_seg_list.max()
        dist_ft_max = dist_ft_list.max()
        dist_patch_max = dist_patch_list.max()

        ''' test anomaly deteciton '''
        min_dists = []
        gt_list = []
        all_dists = []
        for i, mpath in enumerate(test_mpaths):
            ''' loading image and mask features'''
            mask = read_mask(args, mpath, num_cls)
            ad_mft = mask.sum((1, 2))
            ipath = mpath.replace(seg_dir, "orig_512")
            img = read_img(args, ipath)
            ad_ift = get_feature(args, encoder, img, mask, cls_index)
            ad_ft = torch.cat([ad_mft, ad_ift], dim=0)
            if args.standardize:
                ad_ft = (ad_ft-mean)/(std+1e-10)

            dist = (memory-ad_ft)**2  # 1e-2, 1

            '''  loading patchcore scores '''
            fpath = "/".join(mpath.split("/")[-3:])
            fpath = fpath.split(".")[0]+".pt"
            sa_fpath = f"{sa_dpath}/{fpath}"
            pc_data = torch.load(sa_fpath)
            dist_patch = pc_data["anomaly_scores"].item()

            if args.scale_type == "none":
                dist_seg = (torch.sqrt(dist[:, :num_cls].sum(1)))
                dist_ft = (torch.sqrt(dist[:, num_cls:].sum(1)))
                dist_patch = torch.tensor(dist_patch)
            elif args.scale_type == "std":
                dist_seg = (torch.sqrt(dist[:, :num_cls].sum(
                    1))-dist_seg_mean)/(dist_seg_std+1e-10)
                dist_ft = (torch.sqrt(dist[:, num_cls:].sum(
                    1))-dist_ft_mean)/(dist_ft_std+1e-10)
                dist_patch = (dist_patch-dist_patch_mean) / \
                    (dist_patch_std+1e-10)
            elif args.scale_type == "max":
                dist_seg = (torch.sqrt(
                    dist[:, :num_cls].sum(1)))/(dist_seg_max + 1e-10)
                dist_ft = (torch.sqrt(
                    dist[:, num_cls:].sum(1)))/(dist_ft_max + 1e-10)
                dist_patch = (dist_patch)/(dist_patch_max + 1e-10)
            else:
                assert False

            min_idx = torch.argmin(dist_seg, dim=0)
            min_dist_seg = dist_seg[min_idx]
            min_idx = torch.argmin(dist_ft, dim=0)
            min_dist_ft = dist_ft[min_idx]
            min_dist = 0
            if "h" in args.memory_type:
                min_dist += min_dist_seg
            if "c" in args.memory_type:
                min_dist += min_dist_ft
            if "p" in args.memory_type:
                min_dist += dist_patch

            min_dists.append(min_dist.cpu())

            if "good" in mpath:
                gt = 0
            elif "anomalies" in mpath:
                gt = 1
            else:
                assert False
            gt_list.append(gt)

            all_dists.append([min_dist_seg.cpu().numpy(
            ), min_dist_ft.cpu().numpy(), dist_patch.cpu().numpy(), gt])
            print(f"{i}/{len(test_mpaths)}", end='\r')

        all_dists = np.array(all_dists)
        min_dists = torch.tensor(min_dists)
        ''' Normalization '''
        max_score = min_dists.max()
        min_score = min_dists.min()
        scores = (min_dists - min_score) / (max_score - min_score)

        ''' calculate image-level ROC AUC score '''
        img_scores = scores.view(scores.size(0), -1).max(dim=1)[0]
        gt_list = np.array(gt_list)
        fpr, tpr, thresholds = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        optimal_thresh = thresholds[np.argmax(tpr-fpr)]
        return img_roc_auc


if __name__ == '__main__':
    obj_names = ["breakfast_box", "pushpins",
                 "splicing_connectors", "juice_bottle", "screw_bag"]
    args = get_args_parser()
    args = args.parse_args()
    scores = []
    for obj_name in obj_names:
        args.obj_name = obj_name
        score = main(args)
        scores.append(str(score))

    print(args.type, args.seg_dir, args.memory_type,
          args.scale_type, args.less_data, ",".join(scores))
    # main("screw_bag")

"""
CUDA_VISIBLE_DEVICES=2 python semantic_ad_v5_fusion_std.py --type logical --obj_name breakfast_box &
CUDA_VISIBLE_DEVICES=4 python semantic_ad_v5_fusion_std.py --type logical --obj_name pushpins &
CUDA_VISIBLE_DEVICES=5 python semantic_ad_v5_fusion_std.py --type logical --obj_name splicing_connectors &
CUDA_VISIBLE_DEVICES=6 python semantic_ad_v5_fusion_std.py --type logical --obj_name juice_bottle &
CUDA_VISIBLE_DEVICES=7 python semantic_ad_v5_fusion_std.py --type logical --obj_name screw_bag &

CUDA_VISIBLE_DEVICES=4 python semantic_ad_v5_fusion_std.py --type structural --obj_name breakfast_box &
CUDA_VISIBLE_DEVICES=5 python semantic_ad_v5_fusion_std.py --type structural --obj_name pushpins &
CUDA_VISIBLE_DEVICES=6 python semantic_ad_v5_fusion_std.py --type structural --obj_name splicing_connectors &
CUDA_VISIBLE_DEVICES=7 python semantic_ad_v5_fusion_std.py --type structural --obj_name juice_bottle &
CUDA_VISIBLE_DEVICES=3 python semantic_ad_v5_fusion_std.py --type structural --obj_name screw_bag &

CUDA_VISIBLE_DEVICES=4 python semantic_ad_v5_fusion_std.py --type all --obj_name breakfast_box
CUDA_VISIBLE_DEVICES=5 python semantic_ad_v5_fusion_std.py --type all --obj_name pushpins
CUDA_VISIBLE_DEVICES=6 python semantic_ad_v5_fusion_std.py --type all --obj_name splicing_connectors
CUDA_VISIBLE_DEVICES=7 python semantic_ad_v5_fusion_std.py --type all --obj_name juice_bottle
CUDA_VISIBLE_DEVICES=3 python semantic_ad_v5_fusion_std.py --type all --obj_name screw_bag

"""
