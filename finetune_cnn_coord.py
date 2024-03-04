from pdb import set_trace
from PIL import Image
from statistics import mean
import os
import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.nn.functional import threshold, normalize
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os.path as osp
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from dataset_2d_semi import DataSet2D, my_collate, ValDataSet2D
from torch.utils.data import DataLoader
import random
import timeit
from tensorboardX import SummaryWriter
from sklearn import metrics
from math import ceil
from apex import amp
from apex.parallel import convert_syncbn_model
import sys
sys.path.append("..")


start = timeit.default_timer()


class Encoder(nn.Module):
    def __init__(self, num_cls=15+1, level=3):
        # def __init__(self, num_cls=6+1):
        super().__init__()

        # pretrained CNN feature extractor
        self.init_features()

        def hook_t(module, input, output):
            self.features.append(output)

        self.model = models.wide_resnet101_2(pretrained=True)
        self.level = level
        # in_ch = 256+512+2
        if level == 3:
            in_ch = 256+512+1024+2
        elif level == 2:
            in_ch = 256+512+2
            
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//2, num_cls, 1),

        )

        self.model.layer1[-1].register_forward_hook(hook_t)
        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)
        # self.model.layer4[-1].register_forward_hook(hook_t)

    def init_features(self):
        self.features = []

    def extract_ft(self, x):
        self.init_features()
        _ = self.model(x)
        f0 = self.features[0]
        f1 = F.interpolate(
            self.features[1], align_corners=True, mode="bilinear", size=f0.shape[-2:])
        f2 = F.interpolate(
            self.features[2], align_corners=True, mode="bilinear", size=f0.shape[-2:])
        # f3 = F.interpolate(self.features[3], align_corners=True, mode="bilinear", size=f0.shape[-2:])
        # print(f0.shape, f1.shape, f2.shape)
        # return torch.cat([f0, f1], dim=1)
        return torch.cat([f0, f1, f2], dim=1)
        # return self.features

    def forward(self, x, coord):
        fts = self.extract_ft(x)
        coord = F.interpolate(coord, align_corners=True,
                              mode="bilinear", size=fts.shape[-2:])
        fts = torch.cat([fts, coord], dim=1)
        out = self.conv(fts)
        out = F.interpolate(out, align_corners=True,
                            mode="bilinear", size=x.shape[-2:])
        return out


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():

    parser = argparse.ArgumentParser(description="SAM for Medical Image")

    parser.add_argument("--data_dir", type=str,
                        default="/media/NAS/nas_187/soopil/data/stanford/LOCO_AD_pre")
    parser.add_argument("--obj_name", type=str, default='screw_bag')
    parser.add_argument("--label", type=int, default=255)
    parser.add_argument("--ref_name", type=str, default='001')
    parser.add_argument("--snapshot_dir", type=str, default='./output/results')
    parser.add_argument("--n_shot", type=int, default=1)
    parser.add_argument("--n_zero", type=int, default=3)
    parser.add_argument("--input_size", type=str, default='256,256')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--gpu", type=str, default='None')
    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def get_dice(pred, trg):  # multi chprednnel
    # set_trace()

    eps = 0.0001
    trg = trg.unsqueeze(1)
    new_trg = torch.zeros_like(trg).repeat(1, pred.shape[1], 1, 1).long()
    new_trg = new_trg.scatter(1, trg, 1)

    numer = (pred * new_trg).sum((2, 3))
    denom = pred.sum((2, 3)) + new_trg.sum((2, 3)) + eps
    dsc = (numer*2)/denom
    dsc_score = dsc.mean((0, 1))
    return dsc_score


def proportion_loss(pred, trg, obj_name):
    # set_trace()
    eps = 0.0001
    trg = trg.unsqueeze(1)
    new_trg = torch.zeros_like(trg).repeat(1, pred.shape[1], 1, 1).long()
    new_trg = new_trg.scatter(1, trg, 1).float()
    diff = torch.abs(new_trg.mean((2, 3)) - pred.mean((2, 3)))
    if obj_name == "breakfast_box":
        loss = diff[:, 1:4].sum() + diff[:, 6].sum()  # exclude BG
    else:
        loss = diff[:, 1:].sum()  # exclude BG
    return loss


def save_sample(pred, epoch):
    pred = torch.sigmoid(pred).round()*255
    pred = np.array(pred.detach().cpu(), dtype=np.uint8)
    # set_trace()
    for k in range(pred.shape[0]):
        slice = pred[k, 0]
        slice = np.expand_dims(slice, axis=2)
        slice = np.repeat(slice, 3, axis=2)
        im = Image.fromarray(slice)
        im.save(f"./sample/{epoch}_{k}.png")


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)

    args = parser.parse_args()
    if args.num_gpus > 1:
        torch.cuda.set_device(args.local_rank)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.benchmark = True
    seed = args.random_seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if args.obj_name == "screw_bag":
        num_cls = 7
    elif args.obj_name == "breakfast_box":
        num_cls = 7
    elif args.obj_name == "juice_bottle":
        num_cls = 9
    elif args.obj_name == "pushpins":
        num_cls = 26  # 16 # 20
    elif args.obj_name == "splicing_connectors":
        num_cls = 10  # 7
    else:
        assert False

    model = Encoder(num_cls=num_cls)
    model = model.cuda()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate)

    save_dir = f"{args.snapshot_dir}/{args.obj_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_dataset = DataSet2D(root=args.data_dir,
                              obj_name=args.obj_name,
                              label=args.label,
                              ref_name=args.ref_name,
                              n_shot=args.n_shot,
                              n_zero=args.n_zero,
                              size=input_size,
                              transform=None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=5,
        shuffle=True,
        num_workers=8, )  # , collate_fn=my_collate)

    loss_fn = torch.nn.CrossEntropyLoss()

    all_tr_loss = []
    val_best_loss = 999999
    losses = []

    for epoch in range(args.num_epochs):
        epoch_losses = []
        epoch_dice = []
        epoch_ce = []
        epoch_H = []
        epoch_prop = []
        # adjust_learning_rate(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)

        for iter, batch in enumerate(train_loader):
            supp = batch['image'].cuda()
            un = batch['un_image'].cuda()
            coord_orig = batch['coord_orig'].cuda()
            coord_rot = batch['coord_rot'].cuda()
            labels = batch['label'].cuda()

            pred_supp = model(supp, coord_rot)
            pred_un = model(un, coord_orig)
            # set_trace()
            ce_loss = loss_fn(pred_supp, labels) * 10
            prob = F.softmax(pred_supp, dim=1)
            prob_un = F.softmax(pred_un, dim=1)
            dice_loss = 1 - get_dice(prob, labels)
            entropy_loss = (-1*prob_un*((prob_un+1e-5).log())).mean() * 10
            prop_loss = proportion_loss(prob_un, labels, args.obj_name) * 1
            # prop_loss = prob_un.sum((2,3))
            # print(entropy_loss.shape)
            # assert False
            loss = ce_loss + dice_loss
            if epoch >= 50:
                loss += prop_loss
                loss += entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_dice.append(dice_loss.item())
            epoch_ce.append(ce_loss.item())
            epoch_H.append(entropy_loss.item())
            epoch_prop.append(prop_loss.item())
            # print(f"Iter:{iter}, Loss:{loss:.4f}, Dice:{1-dice_loss:.2f}", end="\r")

        # save_sample(upscaled_masks, epoch)
        losses.append(epoch_losses)
        # print()
        print(
            f'EPOCH: {epoch}, CE: {mean(epoch_ce):.3f}, Dice: {1-mean(epoch_dice):.3f}, H: {mean(epoch_H):.3f}, Prop: {mean(epoch_prop):.3f}, {args.obj_name}')

    print('save model ...')
    torch.save(model.state_dict(), osp.join(
        save_dir, f'{args.obj_name}_{args.num_epochs}.pth'))

    end = timeit.default_timer()
    print(end - start, 'seconds')

    val_dataset = ValDataSet2D(root=args.data_dir,
                               obj_name=args.obj_name,
                               label=args.label,
                               ref_name=args.ref_name,
                               n_shot=args.n_shot,
                               n_zero=args.n_zero,
                               size=input_size,
                               transform=None)

    valloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4)  # , collate_fn=my_collate)

    palette = [0, 0, 0, 204, 241, 227, 112, 142, 18, 254, 8, 23, 207, 149, 84, 202, 24, 214,
               230, 192, 37, 241, 80, 68, 74, 127, 0, 2, 81, 216, 24, 240, 129, 20, 215, 125, 161, 31, 204,
               254, 52, 116, 117, 198, 203, 4, 41, 68, 127, 252, 61, 21, 3, 142, 40, 10, 159, 241, 61, 36,
               14, 175, 77, 144, 61, 115, 131, 79, 97, 109, 177, 163, 58, 198, 140, 17, 235, 168, 47, 128, 91,
               238, 103, 45, 124, 35, 228, 101, 48, 232, 74, 124, 114, 78, 49, 30, 35, 167, 27, 137, 231, 47,
               235, 32, 39, 56, 112, 32, 62, 173, 79, 86, 44, 201, 77, 47, 217, 246, 223, 57, ]
    # Pad with zeroes to 768 values, i.e. 256 RGB colours
    palette = palette + [0]*(768-len(palette))

    for index, batch in enumerate(valloader):
        image, label, coord = batch
        index_str = str(index).zfill(3)

        image = image.cuda()
        coord = coord.cuda()
        with torch.no_grad():
            pred = model(image, coord)
            pred = torch.argmax(pred, dim=1)[0]
            pred = np.array(pred.cpu(), np.uint8)
            pi = Image.fromarray(pred, 'P')
            pi.putpalette(palette)
            pi.show()
            pi.save(f"{save_dir}/pred_{index_str}.png")
            print(f"{index}/{len(valloader)}", end='\r')


if __name__ == '__main__':
    main()

"""
CUDA_VISIBLE_DEVICES=3 python finetune_cnn_coord.py --n_shot 5 --num_epochs 100 --obj_name juice_bottle --snapshot_dir ./output/onetype_5shot_2level_selective &
CUDA_VISIBLE_DEVICES=4 python finetune_cnn_coord.py --n_shot 5 --num_epochs 100 --obj_name splicing_connectors --snapshot_dir ./output/onetype_5shot_2level_selective &
CUDA_VISIBLE_DEVICES=5 python finetune_cnn_coord.py --n_shot 5 --num_epochs 100 --obj_name pushpins --snapshot_dir ./output/onetype_5shot_2level_selective &
CUDA_VISIBLE_DEVICES=6 python finetune_cnn_coord.py --n_shot 5 --num_epochs 100 --obj_name screw_bag --snapshot_dir ./output/onetype_5shot_3level &
CUDA_VISIBLE_DEVICES=7 python finetune_cnn_coord.py --n_shot 5 --num_epochs 100 --obj_name breakfast_box --snapshot_dir ./output/onetype_5shot_2level_selective &
"""


# def get_dice(a, b):
#     dsc_list = []
#     for i in range(a.shape[0]):
#         numer = (a[i]*b[i]).sum()
#         denom = a[i].sum()+b[i].sum()
#         dsc = (numer*2)/denom
#         dsc_list.append(dsc)
#     dsc_all = torch.tensor(dsc_list)
#     return dsc_all.mean()
