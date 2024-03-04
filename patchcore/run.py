from argparse import ArgumentParser, Namespace
import os

parser = ArgumentParser()
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--dataset", type=str, default = 'mvtec_loco', choices=['mvtec_ad', 'mpdd', 'mtd', 'mvtec_loco', 'visa'])
parser.add_argument("--mode", type=str, default='train', choices=['train', 'test', 'standardization'])
parser.add_argument("--category", type=str)
parser.add_argument("--size", type=int, default=256) #512
parser.add_argument("--coreset_sampling_ratio", type=float, default=1.0)

parser.add_argument("--datapath", type=str, default = '/media/NAS/nas_187/datasets/MVTec_AD') #'/media/NAS/nas_187/soopil/data/stanford/LOCO_AD_pre'
parser.add_argument("--result_path", type=str, default='./result')
parser.add_argument("--backbone", type=str, default = 'wide_resnet101_2', choices = ['resnet18', 'resnet50', 'wide_resnet50_2', 'wide_resnet101_2'])

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
from lightning_model import Patchcore
from torchvision import transforms

from PIL import Image
import warnings
import random
import numpy as np

from utils.dataset import MVTecLOCODataset, MVTecADDataset, VisADataset
from utils.utils import compute_anomaly_score, compute_anomaly_score_standardization, feature_extraction

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

result_path = os.path.join(args.result_path, args.dataset, args.backbone, args.category)
if args.size != 256:
    result_path = result_path.replace(args.category, args.category+'_'+str(args.size))
args.result_path = result_path
print(result_path)

if not os.path.exists(result_path):
    os.makedirs(result_path)

'''Load Dataset'''
args.mean_train = [0.485, 0.456, 0.406]
args.std_train = [0.229, 0.224, 0.225]
inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                     std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
data_transforms = transforms.Compose([
    transforms.Resize((args.size, args.size), Image.ANTIALIAS),
    transforms.ToTensor(),
    transforms.CenterCrop(args.size),
    transforms.Normalize(mean=args.mean_train, std=args.std_train)])
gt_transforms = transforms.Compose([
    transforms.Resize((args.size, args.size)),
    transforms.ToTensor(),
    transforms.CenterCrop(args.size)])

if args.dataset == 'mvtec_loco':
    args.datapath = '/media/NAS/nas_187/soopil/data/stanford/LOCO_AD_pre'
    train_datasets = MVTecLOCODataset(root=args.datapath,
                                      transform=data_transforms,
                                      phase='train',
                                      args=args)
    test_datasets = MVTecLOCODataset(root=args.datapath,
                                     transform=data_transforms,
                                     phase='test',
                                     args=args,
                                     anomal_type = None)
elif args.dataset == 'mvtec_ad':
    args.datapath = '/media/NAS/nas_187/datasets/MVTec_AD'
    train_datasets = MVTecADDataset(root=args.datapath,
                                      transform=data_transforms,
                                      phase='train',
                                      args=args)
    test_datasets = MVTecADDataset(root=args.datapath,
                                     transform=data_transforms,
                                     phase='test',
                                     args=args,
                                     anomal_type = None)

elif args.dataset == 'visa':
    args.datapath = '/media/NAS/nas_187/datasets/VisA/split/1cls'
    train_datasets = VisADataset(root=args.datapath,
                                      transform=data_transforms,
                                      phase='train',
                                      args=args)
    test_datasets = VisADataset(root=args.datapath,
                                     transform=data_transforms,
                                     phase='test',
                                     args=args,
                                     anomal_type = None)

train_loader = DataLoader(train_datasets, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=4)

'''Load pre-trained model'''
if args.backbone == 'resnet18':
    args.indim = 384
    args.fm = 1
elif args.backbone == 'resnet50':
    args.indim = 1536
    args.fm = 4
elif args.backbone == 'wide_resnet50_2':
    args.indim = 1536
    args.fm = 4
elif args.backbone == 'wide_resnet101_2':
    args.indim = 1536
    args.fm = 4

model = Patchcore(input_size=(args.size, args.size), backbone=args.backbone, layers=['layer2', 'layer3']).cuda()
model.model.feature_extractor.eval()

if args.mode == 'train':
    if args.coreset_sampling_ratio != 1 and os.path.exists(os.path.join(args.result_path, args.backbone+'_'+str(args.coreset_sampling_ratio)+'.pt')):
        extracted_features = torch.load(os.path.join(args.result_path, args.backbone+'_'+str(args.coreset_sampling_ratio)+'.pt'))
        model.model.memory_bank = extracted_features
    else:
        '''Training'''
        extracted_features, segmentation, labels = feature_extraction(model, train_datasets, args)

        '''Build memory bank'''
        model.model.subsample_embedding(extracted_features.reshape(-1, extracted_features.shape[-1]), args.coreset_sampling_ratio)
        torch.save(model.model.memory_bank, os.path.join(args.result_path, args.backbone+'_'+str(args.coreset_sampling_ratio)+'.pt'))

    print(f'Memory bank : {model.model.memory_bank.shape}')

    # if not os.path.exists(os.path.join(result_path, 'prediction')):
    #     os.makedirs(os.path.join(result_path, 'prediction'))

    '''Compute anomaly score'''
    model.model.feature_extractor.eval()
    with torch.no_grad():
        model.model.training = False
        compute_anomaly_score(test_datasets, model, args)

elif args.mode == 'standardization':
    '''Training'''
    extracted_features, segmentation, labels = feature_extraction(model, train_datasets, args)

    '''Build memory bank'''
    model.model.subsample_embedding(extracted_features.reshape(-1, extracted_features.shape[-1]),
                                    args.coreset_sampling_ratio)
    torch.save(model.model.memory_bank,
               os.path.join(args.result_path, 'memory_bank_' + str(args.coreset_sampling_ratio) + '.pt'))

    '''Standardization'''
    memory_bank = model.model.memory_bank # N, 1536
    mean = torch.mean(memory_bank, dim = 0)
    std = torch.std(memory_bank, dim = 0)

    '''For original patchcore'''
    mean = torch.zeros_like(mean)
    std = torch.ones_like(std)

    memory_bank = (memory_bank - mean) / std
    # torch.save(model.model.memory_bank, os.path.join(args.result_path, 'memory_bank_std' + str(args.coreset_sampling_ratio) + '.pt'))

    model.model.memory_bank = memory_bank
    print(f'Memory bank : {model.model.memory_bank.shape}')

    '''Compute distance for train set'''
    model.model.feature_extractor.eval()
    with torch.no_grad():
        model.model.training = False
        compute_anomaly_score_standardization(train_datasets, model, mean, std, True, args)

    '''Compute anomaly score'''
    model.model.feature_extractor.eval()
    with torch.no_grad():
        model.model.training = False
        compute_anomaly_score_standardization(test_datasets, model, mean, std, False, args)