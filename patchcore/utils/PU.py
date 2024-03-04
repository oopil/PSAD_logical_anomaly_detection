import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import faiss
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

def generate_pseudo(labeled, unlabeled, args):
    distances_pu = torch.cdist(labeled, unlabeled, p=2.0)
    distances_up = distances_pu.transpose(1, 0)
    dist_pu, index_pu = distances_pu.topk(k=1, largest=False, dim=1)
    dist_up, index_up = distances_up.topk(k=1, largest=False, dim=1)

    index_pu_x = (index_pu / int(args.size/8)).type(torch.LongTensor)
    index_pu_y = (index_pu % int(args.size/8)).type(torch.LongTensor)
    index_pu_2d = torch.zeros((index_pu.shape[0], 2, index_pu.shape[1])).type(torch.LongTensor).cuda()
    index_pu_2d[:, 0] = index_pu_x
    index_pu_2d[:, 1] = index_pu_y

    index_sorted = torch.Tensor(list(range(index_up.shape[0]))).type(torch.LongTensor).unsqueeze(-1).cuda()
    index_up_x = (index_sorted / int(args.size/8)).type(torch.LongTensor)
    index_up_y = (index_sorted % int(args.size/8)).type(torch.LongTensor)
    index_up_2d = torch.zeros((index_sorted.shape[0], 2, index_sorted.shape[1])).type(torch.LongTensor).cuda()
    index_up_2d[:, 0] = index_up_x
    index_up_2d[:, 1] = index_up_y

    near_dist = torch.pow(index_pu_2d[index_up[:, 0]] - index_up_2d, 2).sum(1).cuda()
    pos_idx = torch.where(near_dist <= args.t1)[0]  # 2
    neg_idx = torch.where(near_dist > args.t2)[0]  # 7
    pos_diff = dist_up[pos_idx] - dist_pu[index_up[:, 0]][pos_idx]
    pos_idx = pos_idx[torch.where(pos_diff < args.t3 * args.fm)[0]]
    neg_diff = dist_up[neg_idx] - dist_pu[index_up[:, 0]][neg_idx]
    neg_idx = neg_idx[torch.where(neg_diff >= args.t4 * args.fm)[0]]

    if pos_idx.shape[0] > neg_idx.shape[0]:
        pos_idx = pos_idx[dist_pu[index_up[:, 0]][pos_idx].topk(k=neg_idx.shape[0], dim=0)[1][:, 0]]
    else:
        neg_idx = neg_idx[dist_up[neg_idx].topk(k=pos_idx.shape[0], largest=False, dim=0)[1][:, 0]]

    pos_idx = pos_idx.detach().cpu()
    neg_idx = neg_idx.detach().cpu()

    pos = torch.cat((unlabeled[pos_idx], labeled[index_up[:, 0]][pos_idx]), dim=1)
    neg = torch.cat((unlabeled[neg_idx], labeled[index_up[:, 0]][neg_idx]), dim=1)

    return pos, neg

def co_occurrence(train_datasets, unlabeled_datasets, model, args):
    train_loader = DataLoader(train_datasets, batch_size=1, shuffle=True, num_workers=4)
    unlabeled_loader = DataLoader(unlabeled_datasets, batch_size=1, shuffle=True, num_workers=4)
    unlabeled_features = []
    unlabeled_masks = []
    unlabeled_ids = []
    unlabeled_count = 0

    train_datasets.unlabeled = False
    for idx, batch in enumerate(train_loader):
        with torch.no_grad():
            img = batch[0]
            mask = batch[1]
            angle = list(range(0, 360, 10))
            if args.few >= 50 or args.dataset == 'mpdd' or args.dataset == 'mtd':
                angle = [0]
            rotate_count = 0
            for rotate in angle:
                img_ = transforms.functional.rotate(img.clone(), rotate, interpolation=transforms.InterpolationMode.BILINEAR).cuda()
                mask_ = transforms.functional.rotate(mask.clone(), rotate)
                feature = model.model(img_).detach().cpu().numpy()
                unlabeled_features.append(feature)
                mask_ = F.interpolate(mask_, size=(int(args.size/8), int(args.size/8)))
                mask_[mask_ >= 0.5] = 1
                mask_[mask_ < 0.5] = 0
                unlabeled_masks.append(mask_.reshape(-1))
                unlabeled_ids += [unlabeled_count] * feature.shape[0]
                unlabeled_count += 1
                rotate_count += 1
        labeled_count = unlabeled_count

    train_datasets.unlabeled = True
    unlabeled_datasets.transforms = False
    for idx, batch in enumerate(unlabeled_loader):
        with torch.no_grad():
            img = batch[0].cuda()
            feature = model.model(img).detach().cpu().numpy()
            unlabeled_features.append(feature)
            mask = F.interpolate(batch[1], size=(int(args.size/8), int(args.size/8)))
            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
            unlabeled_masks.append(mask.reshape(-1))
            unlabeled_ids += [unlabeled_count]*feature.shape[0]
            unlabeled_count += 1
    unlabeled_datasets.transforms = True

    unlabeled_ids = np.array(unlabeled_ids, dtype=np.int16)
    unlabeled_features = np.ascontiguousarray(np.concatenate(unlabeled_features, axis=0), dtype=np.float32)
    kmeans = faiss.Kmeans(d=args.indim, k=1024, gpu=True)
    kmeans.train(unlabeled_features)
    dists, ids = kmeans.index.search(unlabeled_features, 1)
    ids_, ids_count = np.unique(ids, return_counts=True)
    pos_ids = [i for i in range(ids_.shape[0])
               if np.unique(unlabeled_ids[labeled_count*int(args.size/8)*int(args.size/8):][np.where(ids[labeled_count*int(args.size/8)*int(args.size/8):][:, 0] == i)[0]]).shape[0] >= ((unlabeled_count-labeled_count)*args.gamma)]

    return kmeans, pos_ids

class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])

    def forward(self, x):
        x_ = x.detach()
        for idx, i in enumerate(self.layers):
            if idx != (len(self.layers)-1):
                x_ = i(x_)
                x_ = F.relu(x_)
            else:
                x_ = i(x_)

        return x_

def cycle_consistency(extracted_features, unlabeled_datasets, model, args):
    unlabeled_loader = DataLoader(unlabeled_datasets, batch_size=1, shuffle=True, num_workers=4)
    classifier = MLP([args.indim * 2, 512, 64, 2]).cuda()
    optim = torch.optim.Adam(classifier.parameters(), lr=0.00001, weight_decay=1e-7)
    ce_loss = nn.CrossEntropyLoss()

    epoch = 0
    while True:
        acc = 0
        loss = torch.zeros(1).cuda()
        count = 0
        for idx, batch in enumerate(unlabeled_loader):
            optim.zero_grad()
            img = batch[0].cuda()
            with torch.no_grad():
                embed = model.model(img)

            miniloss = 0
            minicount = 0
            for idx_, extracted_feature in enumerate(extracted_features):
                extracted_feature = extracted_feature.cuda()
                pos, neg = generate_pseudo(extracted_feature, embed, args)
                x = torch.cat((pos, neg))
                y = torch.cat((torch.zeros(pos.shape[0]), torch.ones(neg.shape[0])))
                pred = classifier(x)

                if pred.shape[0] != 0:
                    miniloss += ce_loss(pred, y.type(torch.LongTensor).cuda()) * x.shape[0]
                    minicount += x.shape[0]
                    acc += (pred.detach().cpu().argmax(1) == y.detach().cpu()).sum().item()

            if minicount != 0:
                miniloss /= minicount
                miniloss.backward()
                optim.step()

                loss += miniloss * minicount
                count += minicount

        loss /= count
        acc /= count
        print(f'{epoch + 1} epoch Acc: {acc} Loss: {loss.item()}', end='\r')
        epoch += 1
        if loss.item() <= args.alpha or np.isnan(loss.item()):  # epoch+1 >= args.epoch:
            torch.save(classifier.state_dict(), os.path.join(args.result_path, 'classifier.pth'))
            print('')
            break

    return classifier

def pu_pred(train_datasets, test_datasets, model, classifier, kmeans, pos_ids, args):
    train_loader = DataLoader(train_datasets, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=4)
    with torch.no_grad():
        embeddings = []
        '''Set memory bank from labeled set -> M'''
        train_datasets.unlabeled = False
        for idx, batch in enumerate(train_loader):
            img = batch[0].cuda()
            embed = model.model(img)
            embeddings.append(embed)
        embeddings_ = torch.vstack(embeddings.copy())
        model.model.subsample_embedding(embeddings_, args.coreset_sampling_ratio)

        '''Set memory bank from unlabeled set -> M_U'''
        test_datasets.unlabeled = True
        selected_features = []
        classifier.eval()
        noisy_count = 0
        update_count = 0

        for idx, batch in enumerate(test_loader):
            for aug in range(batch[0].shape[1]):
                with torch.no_grad():
                    img = batch[0][:, aug].cuda()
                    mask = F.interpolate(batch[1][:, aug], size=(int(args.size/8), int(args.size/8)))
                    mask[mask >= 0.5] = 1
                    mask[mask < 0.5] = 0
                    embed = model.model(img)

                    '''Prediction by CC'''
                    distances = torch.cdist(embed, model.model.memory_bank, p=2.0)  # euclidean norm
                    dist, index = distances.topk(k=1, largest=False, dim=1)
                    pos = model.model.memory_bank[index[:, 0]]
                    x = torch.cat((embed, pos), dim=1)
                    pred = classifier(x)
                    pred_p = pred.softmax(1)[:, 0]
                    select_cc = torch.where(pred_p >= args.beta)[0].detach().cpu().numpy()
                    select_cc = np.ndarray.tolist(select_cc)

                    '''Prediction by CO'''
                    dists, ids = kmeans.index.search(np.ascontiguousarray(embed.detach().cpu().numpy()), 1)
                    select_co = [i for i in range(embed.shape[0]) if ids[i, 0] in pos_ids]

                    '''Merge'''
                    select_idx = [i for i in range(embed.shape[0]) if i in select_co or i in select_cc]
                    select_idx = np.array(select_idx, dtype=np.int32)
                    update_count += select_idx.shape[0]
                    selected_features.append(embed[select_idx])

                    '''Visualize'''
                    invnorm_img = (img[0].detach().cpu() * torch.Tensor(args.std_train).reshape(3, 1, 1)) \
                                  + torch.Tensor(args.mean_train).reshape(3, 1, 1)
                    pred_mask = pred_p.detach().cpu().numpy().reshape(int(args.size/8), int(args.size/8))
                    pred_mask_ = pred_mask - np.min(pred_mask)
                    pred_mask = pred_mask_ / np.max(pred_mask)
                    pred_mask = np.uint8(255 * pred_mask)
                    pred_mask = Image.fromarray(pred_mask)
                    pred_mask = np.array(pred_mask.resize((args.size, args.size)))
                    heatmap = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                    plt.figure(figsize=(20, 5))
                    '''Test sample'''
                    plt.subplot(1, 4, 1)
                    plt.imshow(to_pil_image(invnorm_img))
                    plt.axis('off')

                    '''Cycle consistency'''
                    plt.subplot(1, 4, 2)
                    select_mask = torch.zeros((3, 1024))
                    mask_ = mask.reshape(-1)
                    select_mask[:, np.array(select_cc)] = 1
                    non_select_mask = torch.abs(1 - select_mask)
                    select_mask[0:2] *= mask_
                    select_mask[1:3] *= torch.abs(1 - mask_)
                    non_select_mask[0] *= mask_
                    non_select_mask[2] *= torch.abs(1 - mask_)
                    select_mask_ = select_mask + non_select_mask
                    select_mask_ = to_pil_image(select_mask_.reshape(3, int(args.size/8), int(args.size/8))).resize((args.size, args.size))
                    plt.imshow(select_mask_)
                    plt.axis('off')

                    '''Co-occurrence'''
                    plt.subplot(1, 4, 3)
                    select_mask = torch.zeros((3, 1024))
                    mask_ = mask.reshape(-1)
                    select_mask[:, select_co] = 1
                    non_select_mask = torch.abs(1 - select_mask)
                    select_mask[0:2] *= mask_
                    select_mask[1:3] *= torch.abs(1 - mask_)
                    non_select_mask[0] *= mask_
                    non_select_mask[2] *= torch.abs(1 - mask_)
                    select_mask_ = select_mask + non_select_mask
                    select_mask_ = to_pil_image(select_mask_.reshape(3, int(args.size/8), int(args.size/8))).resize((args.size, args.size))
                    plt.imshow(select_mask_)
                    plt.axis('off')

                    '''Merge'''
                    plt.subplot(1, 4, 4)
                    select_mask = torch.zeros((3, 1024))
                    mask_ = mask.reshape(-1)
                    select_mask[:, select_idx] = 1
                    non_select_mask = torch.abs(1 - select_mask)
                    select_mask[0:2] *= mask_
                    select_mask[1:3] *= torch.abs(1 - mask_)
                    non_select_mask[0] *= mask_
                    non_select_mask[2] *= torch.abs(1 - mask_)
                    select_mask_ = select_mask + non_select_mask
                    select_mask_ = to_pil_image(select_mask_.reshape(3, int(args.size/8), int(args.size/8))).resize((args.size, args.size))
                    plt.imshow(select_mask_)
                    plt.axis('off')

                    if not os.path.exists(os.path.join(args.result_path, 'PU')):
                        os.makedirs(os.path.join(args.result_path, 'PU'))
                    plt.savefig(os.path.join(args.result_path, 'PU', batch[4][0] + '-' + str(aug) + '-' + batch[3][0]))
                    plt.clf()
                    plt.close()

        train_datasets.unlabeled = True
        for idx, batch in enumerate(train_loader):
            for aug in range(batch[0].shape[1]):
                with torch.no_grad():
                    img = batch[0][:, aug].cuda()
                    mask = F.interpolate(batch[1][:, aug], size=(int(args.size/8), int(args.size/8)))
                    mask[mask >= 0.5] = 1
                    mask[mask < 0.5] = 0
                    embed = model.model(img)

                    '''Prediction by CC'''
                    distances = torch.cdist(embed, model.model.memory_bank, p=2.0)  # euclidean norm
                    dist, index = distances.topk(k=1, largest=False, dim=1)
                    pos = model.model.memory_bank[index[:, 0]]
                    x = torch.cat((embed, pos), dim=1)
                    pred = classifier(x)
                    pred_p = pred.softmax(1)[:, 0]
                    select_cc = torch.where(pred_p >= args.beta)[0].detach().cpu().numpy()
                    select_cc = np.ndarray.tolist(select_cc)

                    '''Prediction by CO'''
                    dists, ids = kmeans.index.search(np.ascontiguousarray(embed.detach().cpu().numpy()), 1)
                    select_co = [i for i in range(embed.shape[0]) if ids[i, 0] in pos_ids]

                    '''Merge'''
                    select_idx = [i for i in range(embed.shape[0]) if i in select_co or i in select_cc]
                    select_idx = np.array(select_idx, dtype=np.int32)
                    update_count += select_idx.shape[0]
                    selected_features.append(embed[select_idx])

                    '''Visualize'''
                    invnorm_img = (img[0].detach().cpu() * torch.Tensor(args.std_train).reshape(3, 1, 1)) \
                                  + torch.Tensor(args.mean_train).reshape(3, 1, 1)
                    pred_mask = pred_p.detach().cpu().numpy().reshape(int(args.size/8), int(args.size/8))
                    pred_mask_ = pred_mask - np.min(pred_mask)
                    pred_mask = pred_mask_ / np.max(pred_mask)
                    pred_mask = np.uint8(255 * pred_mask)
                    pred_mask = Image.fromarray(pred_mask)
                    pred_mask = np.array(pred_mask.resize((args.size, args.size)))
                    heatmap = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                    plt.figure(figsize=(20, 5))
                    '''Test sample'''
                    plt.subplot(1, 4, 1)
                    plt.imshow(to_pil_image(invnorm_img))
                    plt.axis('off')

                    '''Cycle consistency'''
                    plt.subplot(1, 4, 2)
                    select_mask = torch.zeros((3, 1024))
                    mask_ = mask.reshape(-1)
                    select_mask[:, np.array(select_cc)] = 1
                    non_select_mask = torch.abs(1 - select_mask)
                    select_mask[0:2] *= mask_
                    select_mask[1:3] *= torch.abs(1 - mask_)
                    non_select_mask[0] *= mask_
                    non_select_mask[2] *= torch.abs(1 - mask_)
                    select_mask_ = select_mask + non_select_mask
                    select_mask_ = to_pil_image(select_mask_.reshape(3, int(args.size/8), int(args.size/8))).resize((args.size, args.size))
                    plt.imshow(select_mask_)
                    plt.axis('off')

                    '''Co-occurrence'''
                    plt.subplot(1, 4, 3)
                    select_mask = torch.zeros((3, 1024))
                    mask_ = mask.reshape(-1)
                    select_mask[:, select_co] = 1
                    non_select_mask = torch.abs(1 - select_mask)
                    select_mask[0:2] *= mask_
                    select_mask[1:3] *= torch.abs(1 - mask_)
                    non_select_mask[0] *= mask_
                    non_select_mask[2] *= torch.abs(1 - mask_)
                    select_mask_ = select_mask + non_select_mask
                    select_mask_ = to_pil_image(select_mask_.reshape(3, int(args.size/8), int(args.size/8))).resize((args.size, args.size))
                    plt.imshow(select_mask_)
                    plt.axis('off')

                    '''Merge'''
                    plt.subplot(1, 4, 4)
                    select_mask = torch.zeros((3, 1024))
                    mask_ = mask.reshape(-1)
                    select_mask[:, select_idx] = 1
                    non_select_mask = torch.abs(1 - select_mask)
                    select_mask[0:2] *= mask_
                    select_mask[1:3] *= torch.abs(1 - mask_)
                    non_select_mask[0] *= mask_
                    non_select_mask[2] *= torch.abs(1 - mask_)
                    select_mask_ = select_mask + non_select_mask
                    select_mask_ = to_pil_image(select_mask_.reshape(3, int(args.size/8), int(args.size/8))).resize((args.size, args.size))
                    plt.imshow(select_mask_)
                    plt.axis('off')

                    if not os.path.exists(os.path.join(args.result_path, 'PU')):
                        os.makedirs(os.path.join(args.result_path, 'PU'))
                    plt.savefig(os.path.join(args.result_path, 'PU', batch[4][0] + '-' + str(aug) + '-' + batch[3][0]))
                    plt.clf()
                    plt.close()

            train_datasets.unlabeled = False

        if update_count != 0:
            selected_features = torch.cat(selected_features, dim=0)
            model.model.memory_bank = torch.cat((model.model.memory_bank, selected_features), dim=0)
