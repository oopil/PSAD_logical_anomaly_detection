import numpy as np
import os
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
import csv

def feature_extraction(model, train_datasets, args):
    train_datasets.unlabeled = False
    train_loader = DataLoader(train_datasets, batch_size=1, shuffle=False, num_workers=4)
    extracted_features = []
    segmentation = []
    labels = []
    for idx, batch in enumerate(train_loader):
        print(f'Extract features {idx+1}/{len(train_loader)}', end='\r')
        with torch.no_grad():
            img = batch[0].cuda()
            embed = model.model(img)
            embed = embed.reshape(img.shape[0], int(args.size/8), int(args.size/8), -1)
            extracted_features.append(embed.detach().cpu())
            segmentation.append(batch[1])
            labels.append(batch[2])
    print('')

    extracted_features = torch.cat(extracted_features, dim=0)
    segmentation = torch.cat(segmentation, dim=0)
    labels = torch.cat(labels, dim=0)

    return extracted_features, segmentation, labels

def compute_anomaly_score(test_datasets, model, args):
    test_datasets.unlabeled = False
    test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=4)
    outputs = []
    for idx, batch in enumerate(test_loader):
        print(f'Extract features from testset - {idx + 1}/{len(test_loader)}', end='\r')
        img = batch[0].cuda()
        anomaly_map, anomaly_score, _ = model.model(img)
        output = {}
        output['anomaly_scores'] = anomaly_score.detach().cpu()
        output['label'] = batch[2]
        outputs.append(output)

        '''Visualize'''
        # fig = plt.figure(figsize=(20, 5))
        # invnorm_img = (img[0].detach().cpu() * torch.Tensor(args.std_train).reshape(3, 1, 1)) + torch.Tensor(args.mean_train).reshape(3, 1, 1)
        # plt.subplot(1, 4, 1)
        # plt.title(str(anomaly_score.item()))
        # plt.imshow(to_pil_image(invnorm_img))
        # plt.axis('off')
        #
        # plt.subplot(1, 4, 2)
        # anomaly_map = anomaly_map.clone().squeeze().detach().cpu().numpy()
        # anomaly_map_ = anomaly_map - np.min(anomaly_map)
        # anomaly_map = anomaly_map_ / np.max(anomaly_map)
        # anomaly_map = np.uint8(255 * anomaly_map)
        # heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        # plt.imshow(heatmap)
        # plt.axis('off')
        #
        # plt.subplot(1, 4, 3)
        # plt.imshow(np.uint8(heatmap * 0.3 + np.transpose(invnorm_img.numpy(), (1, 2, 0)) * 255 * 0.5))
        # plt.axis('off')
        #
        # plt.subplot(1, 4, 4)
        # mask = batch[1].clone().squeeze().numpy()
        # mask = np.uint8(255 * mask)
        # mask = Image.fromarray(mask)
        # plt.imshow(mask)
        # plt.axis('off')
        #
        # plt.savefig(os.path.join(args.result_path, 'prediction', 'result-' + batch[4][0] + '-' + batch[3][0] + '.png'))
        # plt.clf()
        # plt.close()

    print('')

    anomaly_scores = []
    labels = []
    for output in outputs:
        anomaly_scores.append(output['anomaly_scores'].item())
        labels.append(output['label'].item())

    anomaly_scores = np.array(anomaly_scores)
    labels = np.array(labels)
    img_auc = roc_auc_score(labels, anomaly_scores)
    print("Image AUROC: ", img_auc)

    f = open(os.path.join(args.result_path, 'result.txt'), 'w')
    f.write(f'Image AUROC: {img_auc}\n')
    f.close()

def compute_anomaly_score_standardization(test_datasets, model, mean, std, is_training, args):
    test_datasets.unlabeled = False
    test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=4)
    outputs = []
    original_memory_bank = model.model.memory_bank
    for idx, batch in enumerate(test_loader):
        if is_training:
            model.model.memory_bank = torch.cat([original_memory_bank[:idx * int(args.size/8) * int(args.size/8)],
                                                 original_memory_bank[(idx+1) * int(args.size/8) * int(args.size/8):]],
                                                dim = 0)
        print(f'Extract features from testset - {idx + 1}/{len(test_loader)}', end='\r')
        img = batch[0].cuda()
        anomaly_map, anomaly_score, _ = model.model(img, mean, std)
        output = {}
        output['distance'] = model.model.patch_scores
        output['anomaly_maps_interpolate'] = anomaly_map.detach().cpu()
        output['anomaly_maps'] = _.detach().cpu()
        output['anomaly_scores'] = anomaly_score.detach().cpu()
        output['label'] = batch[2]
        output['name'] = batch[3][0]

        os.makedirs(os.path.join(args.result_path, batch[3][0].split('/')[0], batch[3][0].split('/')[1]), exist_ok=True)
        torch.save(output, os.path.join(args.result_path, batch[3][0] + '.pt'))
        del output['anomaly_maps']
        del output['anomaly_maps_interpolate']
        outputs.append(output)
    print('')

    if not is_training:
        if args.dataset == 'mvtec_loco':
            anomal_type = ['both', 'logical', 'structural']
        else:
            anomal_type = ['both']
        for i in anomal_type:
            if i == 'both':
                anomaly_scores = []
                labels = []
                f = open(os.path.join(args.result_path, 'ADscore'+'_all.txt'), 'w')
                wr = csv.writer(f)
                wr.writerow(['Name', 'Score', 'Label'])
                for output in outputs:
                    anomaly_scores.append(output['anomaly_scores'].item())
                    labels.append(output['label'].item())
                    wr.writerow([output['name'], output['anomaly_scores'].item(), output['label'].item()])
                f.close()

                anomaly_scores = np.array(anomaly_scores)
                labels = np.array(labels)
                img_auc = roc_auc_score(labels, anomaly_scores)
                print("Image AUROC: ", img_auc)

                f = open(os.path.join(args.result_path, 'result'+'_all.txt'), 'w')
                f.write(f'Image AUROC: {img_auc}\n')
                f.close()
            else:
                anomaly_scores = []
                labels = []
                f = open(os.path.join(args.result_path, 'ADscore_'+i+ '.txt'), 'w')
                wr = csv.writer(f)
                wr.writerow(['Name', 'Score', 'Label'])
                for output in outputs:
                    if 'good' in output['name'] or i in output['name']:
                        anomaly_scores.append(output['anomaly_scores'].item())
                        labels.append(output['label'].item())
                        wr.writerow([output['name'], output['anomaly_scores'].item(), output['label'].item()])
                f.close()

                anomaly_scores = np.array(anomaly_scores)
                labels = np.array(labels)
                img_auc = roc_auc_score(labels, anomaly_scores)
                print("Image AUROC: ", img_auc)

                f = open(os.path.join(args.result_path, 'result_'+i+'.txt'), 'w')
                f.write(f'Image AUROC: {img_auc}\n')
                f.close()