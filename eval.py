import os
import sys
from tqdm import tqdm
import pprint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('..')
from semilearn.core.utils import get_net_builder, get_dataset, over_write_args_from_file
from semilearn.algorithms.openmatch.openmatch import OpenMatchNet
from semilearn.algorithms.iomatch.iomatch import IOMatchNet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--c', type=str, default='')

def load_model_at(step='best'):
    args.step = step
    if step == 'best':
        args.load_path = '/'.join(args.load_path.split('/')[1:-1]) + "/model_best.pth"
    else:
        args.load_path = '/'.join(args.load_path.split('/')[:-1]) + "/model_at_{args.step}_step.pth"
    print(args.load_path)
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['ema_model']
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    if step == 'best':
        args.save_dir = os.path.join(save_dir, f"model_best")
    else:
        args.save_dir = os.path.join(save_dir, f"step_{args.step}")
    os.makedirs(args.save_dir, exist_ok=True)
    _net_builder = get_net_builder(args.net, args.net_from_name)
    net = _net_builder(num_classes=args.num_classes)
    if args.algorithm == 'openmatch':
        net = OpenMatchNet(net, args.num_classes)
    elif args.algorithm == 'iomatch':
        net = IOMatchNet(net, args.num_classes)
    else:
        raise NotImplementedError
    keys = net.load_state_dict(load_state_dict)
    print(f'Model at step {args.step} loaded!')
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    return net


def evaluate_io(args, net, dataset_dict, extended_test=True):
    """
    evaluation function for open-set SSL setting
    """
    #unlabel
    full_loader = DataLoader(dataset_dict['train_ulb'], batch_size=256, drop_last=False, shuffle=False, num_workers=4)
    #label
    # print(len(full_loader.dataset))
    # full_loader = DataLoader(dataset_dict['train_lb'], batch_size=256, drop_last=False, shuffle=False, num_workers=4)
    # print(len(full_loader.dataset))
    # exit()
    total_num = 0.0
    y_true_list = []
    p_list = []
    pred_p_list = []
    pred_hat_q_list = []
    pred_q_list = []
    pred_hat_p_list = []

    all_feats = torch.tensor([]).cuda()
    all_targets = torch.tensor([]).cuda()
    with torch.no_grad():
        for data in tqdm(full_loader):
            #IOMatch
            # x_w = data['x_ulb_w']
            # x_s = data['x_ulb_s']
            #
            # y = data['y_ulb']
            # x_w = data['x_lb']
            # y = data['y_lb']

            #OpenMatch
            x_w = data['x_ulb_w_0']
            x_s = data['x_ulb_w_1']

            y = data['y_ulb']

            id_index = (y < 6)
            if isinstance(x_w, dict):
                x_w = {k: v.cuda() for k, v in x_w.items()}
                # x_s = {k: v.cuda() for k, v in x_s.items()}
            else:
                x_w = x_w.cuda()
                # x_s = x_s.cuda()

            y = y.cuda()
            y_true_list.extend(y.cpu().tolist())

            num_batch = y.shape[0]
            total_num += num_batch

            outputs = net(x_w)
            logits = (outputs['logits'])[id_index]
            # logits_mb = outputs['logits_mb']
            # logits_open = outputs['logits_open']
            feats = (outputs['feat'])[id_index]
            # feats = (outputs['feat_proj'])[id_index]
            # feats = F.normalize(feats, dim=1)
            weight = F.normalize(net.backbone.fc.weight, dim=1)
            _, predicted = torch.max(logits, 1)
            predicted_hot = F.one_hot(predicted, logits.size()[1]).float().cuda()

            all_feats = torch.cat([all_feats, feats], dim=0)
            all_targets = torch.cat([all_targets, predicted_hot], dim=0)


    feats = all_feats
    all_targets = all_targets

    num_classes = all_targets.size(1)  # 总类别数

    # 1. 计算每个类别的正样本特征中心
    positive_centers = []
    for i in range(num_classes):
        # 获取当前类别的正样本特征向量
        positive_feats = feats[all_targets[:, i] == 1]
        # 计算正样本特征向量的平均值作为特征中心
        positive_center = torch.mean(positive_feats, dim=0)
        positive_centers.append(positive_center)

    # 2. 计算每个类别的负样本特征中心
    negative_centers = []
    for i in range(num_classes):
        # 获取除了当前类别的所有负样本特征向量
        negative_feats = torch.cat([feats[all_targets[:, j] == 0] for j in range(num_classes) if j != i], dim=0)
        # 计算负样本特征向量的平均值作为特征中心
        negative_center = torch.mean(negative_feats, dim=0)
        negative_centers.append(negative_center)
    # print(negative_feats.shape)
    # print(negative_centers[0].shape)
    # weight = F.normalize(net.backbone.fc.weight, dim=1)
    # print(weight.shape)
    # exit()
    # 3. 计算角度
    angles_positive = []
    angles_negative = []
    for i in range(num_classes):
        # 获取当前类别的正样本特征向量
        positive_feats = feats[all_targets[:, i] == 1]
        # 计算每个样本与正类的特征中心之间的角度
        positive_center = positive_centers[i].unsqueeze(0).expand(feats.size(0), -1)
        cosine_similarity_positive = torch.matmul(F.normalize(positive_feats), (F.normalize(positive_center)).transpose(0, 1))
        angle_positive = torch.clamp(cosine_similarity_positive, -1.0, 1.0)
        angle_positive = torch.acos(angle_positive)
        angles_positive.append(angle_positive)

        # 获取除了当前类别的所有负样本特征向量
        negative_feats = torch.cat([feats[all_targets[:, j] == 0] for j in range(num_classes) if j != i], dim=0)
        # 计算每个样本与负类的特征中心之间的角度
        negative_center = negative_centers[i].unsqueeze(0).expand(feats.size(0), -1)
        cosine_similarity_negative = torch.matmul(F.normalize(negative_feats), (F.normalize(negative_center)).transpose(0, 1))
        angle_negative = torch.clamp(cosine_similarity_negative, -1.0, 1.0)
        angle_negative = torch.acos(angle_negative)
        angles_negative.append(angle_negative)

    # 4. 计算角度的方差
    angle_variances_positive = [torch.var(angle * (180.0 / 3.141592653589793)) for angle in angles_positive]
    angle_variances_negative = [torch.var(angle * (180.0 / 3.141592653589793)) for angle in angles_negative]

    # 打印结果
    for i in range(num_classes):
        print("Class {}: Positive Angle Variance: {}, Negative Angle Variance: {}".format(i, angle_variances_positive[i], angle_variances_negative[i]))

    exit()
    K = torch.sum(all_targets, dim=0)
    print("predict cls num: {}".format(K))
    K_var = K - 1.0
    K_var[K_var <= 0.0] = 1.0

    # weight = F.normalize(net.backbone.fc.weight, dim=1)

    weight = torch.matmul(all_targets.transpose(0, 1),
                                       all_feats) / (K.view(-1, 1) + 1e-5)

    cosine_beta = torch.matmul(F.normalize(all_feats),
                               F.normalize(weight).transpose(0, 1))
    cosine_beta = torch.clamp(cosine_beta, -1.0, 1.0)
    beta = torch.acos(cosine_beta)
    beta = beta * (180.0 / 3.141592653589793)

    current_means = torch.sum(beta * all_targets, dim=0) / (K + 1e-5)


    current_variances = torch.sum(
        (beta - current_means)**2 * all_targets, dim=0) / K_var

    print("variances : {}".format(current_variances))
    exit()

    return eval_dict

def evaluate_open(net, dataset_dict, num_classes, extended_test=True):
    full_loader = DataLoader(dataset_dict['test']['full'], batch_size=256, drop_last=False, shuffle=False, num_workers=4)
    if extended_test:
        extended_loader = DataLoader(dataset_dict['test']['extended'], batch_size=1024, drop_last=False, shuffle=False, num_workers=4)

    total_num = 0.0
    y_true_list = []
    y_pred_closed_list = []
    y_pred_ova_list = []

    results = {}

    with torch.no_grad():
        for data in tqdm(full_loader):
            x = data['x_lb']
            y = data['y_lb']

            if isinstance(x, dict):
                x = {k: v.cuda() for k, v in x.items()}
            else:
                x = x.cuda()
            y = y.cuda()

            num_batch = y.shape[0]
            total_num += num_batch

            out = net(x)
            logits, logits_open = out['logits'], out['logits_open']
            pred_closed = logits.data.max(1)[1]

            probs = F.softmax(logits, 1)
            probs_open = F.softmax(logits_open.view(logits_open.size(0), 2, -1), 1)
            tmp_range = torch.arange(0, logits_open.size(0)).long().cuda()
            unk_score = probs_open[tmp_range, 0, pred_closed]
            pred_open = pred_closed.clone()
            pred_open[unk_score > 0.5] = num_classes

            y_true_list.extend(y.cpu().tolist())
            y_pred_closed_list.extend(pred_closed.cpu().tolist())
            y_pred_ova_list.extend(pred_open.cpu().tolist())

    y_true = np.array(y_true_list)

    closed_mask = y_true < num_classes
    open_mask = y_true >= num_classes
    y_true[open_mask] = num_classes

    y_pred_closed = np.array(y_pred_closed_list)
    y_pred_ova = np.array(y_pred_ova_list)

    # Closed Accuracy on Closed Test Data
    y_true_closed = y_true[closed_mask]
    y_pred_closed = y_pred_closed[closed_mask]
    closed_acc = accuracy_score(y_true_closed, y_pred_closed)
    closed_cfmat = confusion_matrix(y_true_closed, y_pred_closed, normalize='true')
    results['c_acc_c_p'] = closed_acc
    results['c_cfmat_c_p'] = closed_cfmat

    # Open Accuracy on Full Test Data
    open_acc = balanced_accuracy_score(y_true, y_pred_ova)
    open_cfmat = confusion_matrix(y_true, y_pred_ova, normalize='true')
    results['o_acc_f_hq'] = open_acc
    results['o_cfmat_f_hq'] = open_cfmat

    if extended_test:
        with torch.no_grad():
            for data in tqdm(extended_loader):
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda() for k, v in x.items()}
                else:
                    x = x.cuda()
                y = y.cuda()

                num_batch = y.shape[0]
                total_num += num_batch

                out = net(x)
                logits, logits_open = out['logits'], out['logits_open']
                pred_closed = logits.data.max(1)[1]

                probs = F.softmax(logits, 1)
                probs_open = F.softmax(logits_open.view(logits_open.size(0), 2, -1), 1)
                tmp_range = torch.arange(0, logits_open.size(0)).long().cuda()
                unk_score = probs_open[tmp_range, 0, pred_closed]
                pred_open = pred_closed.clone()
                pred_open[unk_score > 0.5] = num_classes

                y_true_list.extend(y.cpu().tolist())
                y_pred_closed_list.extend(pred_closed.cpu().tolist())
                y_pred_ova_list.extend(pred_open.cpu().tolist())

        y_true = np.array(y_true_list)

        open_mask = y_true >= num_classes
        y_true[open_mask] = num_classes
        y_pred_ova = np.array(y_pred_ova_list)

        # Open Accuracy on Extended Test Data
        open_acc = balanced_accuracy_score(y_true, y_pred_ova)
        open_cfmat = confusion_matrix(y_true, y_pred_ova, normalize='true')
        results['o_acc_e_hq'] = open_acc
        results['o_cfmat_e_hq'] = open_cfmat

    print(f"#############################################################\n"
              f" Closed Accuracy on Closed Test Data: {results['c_acc_c_p'] * 100:.2f}\n"
              f" Open Accuracy on Full Test Data:     {results['o_acc_f_hq'] * 100:.2f}\n"
              f" Open Accuracy on Extended Test Data: {results['o_acc_e_hq'] * 100:.2f}\n"
              f"#############################################################\n"
        )

    return results

args = parser.parse_args(args=['--c', 'config/openset_cv/openmatch/openmatch_cifar10_150_0.yaml'])
over_write_args_from_file(args, args.c)
args.data_dir = 'data'
dataset_dict = get_dataset(args, args.algorithm, args.dataset, args.num_labels, args.num_classes, args.data_dir, eval_open=False)
best_net = load_model_at('best')
eval_dict = evaluate_io(args, best_net, dataset_dict)
