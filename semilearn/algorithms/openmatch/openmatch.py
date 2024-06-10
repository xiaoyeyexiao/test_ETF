import copy
import sys
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
from semilearn.core.utils import get_data_loader
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool

from .utils import ova_loss_func, etf_ova_loss_func, em_loss_func, socr_loss_func


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class OpenMatchDataset(BasicDataset):
    def __init__(self, dset, name):
        self.data = copy.deepcopy(dset.data)
        self.targets = copy.deepcopy(dset.targets)
        super(OpenMatchDataset, self).__init__(alg='openmatch',
                                               data=self.data,
                                               targets=self.targets,
                                               num_classes=dset.num_classes,
                                               transform=dset.transform,
                                               strong_transform=dset.strong_transform)
        self.name = name
        self.data_index = None
        self.targets_index = None
        self.set_index()

    def set_index(self, indices=None):
        if indices is not None:
            self.data_index = self.data[indices]
            self.targets_index = self.targets[indices]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def __len__(self):
        return len(self.data_index)

    def __sample__(self, idx):
        if self.targets is None:
            target = None
        else:
            target = self.targets_index[idx]
        img = self.data_index[idx]

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)  # shape of img should be [H, W, C]
        if isinstance(img, str):
            img = pil_loader(img)

        return img, target

    def __getitem__(self, idx):
        img, target = self.__sample__(idx)

        img_w = self.transform(img)
        if self.name == 'train_lb':
            return {'idx_lb': idx, 'x_lb': img_w, 'x_lb_w_0': img_w, 'x_lb_w_1': self.transform(img),
                    'y_lb': target}
        elif self.name == 'train_ulb':
            return {'idx_ulb': idx, 'x_ulb_w_0': img_w, 'x_ulb_w_1': self.transform(img), 'y_ulb': target}
        elif self.name == 'train_ulb_selected':
            # Selected for FixMatch training
            return {'x_ulb_w': img_w, 'x_ulb_s': self.strong_transform(img)}


class OpenMatchNet(nn.Module):
    def __init__(self, base, num_classes, gpu):
        super(OpenMatchNet, self).__init__()
        self.backbone = base
        # 128
        self.feat_planes = base.num_features
        self.gpu = gpu
        
        # 最开始的时候，每个样本对应一个logits，不同logits并不处于同一个特征空间，彼此关联不大
        # 这里的全连接层，其实也可以看做是6个二分类器的结合
        # 将该全连接层分为6组，对每一组使用softmax函数，该组就成为了对应类的二分类器
        # self.ova_classifiers = nn.Linear(self.feat_planes, num_classes * 2, bias=False)

        # 而对于ETF分类器来说，形状是固定的，有几个类就是几分类器，
        # 不同logits处于同一个特征空间，不能随便拆开使用，因此6个二分类器要设置6个ETF
        self.ova_classifier_size = 6
        self.ova_classifiers = [ETF_Classifier(feat_in=self.feat_planes, num_classes=2, gpu=self.gpu) for _ in range(self.ova_classifier_size) ]

    def forward(self, x, **kwargs):
        # only_feat=True，不需要使用basebone中的全连接层，返回的是features，而不是logits
        # feat.shape: (256, 128)
        feat = self.backbone(x, only_feat=True)
        # only_fc=True，只需使用basebone中的全连接层，传入features，返回logits
        logits = self.backbone(feat, only_fc=True)
        
        # logits_open[:2*batchsize].shape: (128, 12)
        # logits_open = self.ova_classifiers(feat)
        
        # logits_open[:2*batchsize].shape: (128, 2, 6)
        logits_open = torch.zeros(feat.size(0), 2, self.ova_classifier_size).cuda(self.gpu)
        # 6个etf分别对每个样本的feature使用
        for i in range(self.ova_classifier_size):
            cur_M = self.ova_classifiers[i].ori_M
            
            # 使用ETF中的forwrd()将feature进行L2归一化
            feat_l2 = self.ova_classifiers[i](feat)
            # 这一步就类似于CNN中全连接层的计算过程，用特征向量乘以权重，得到logit
            l_open = torch.matmul(feat_l2, cur_M)
            
            logits_open[:, :, i] = l_open
        
        return {'logits': logits, 'logits_open': logits_open , 'feat': feat}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


class OpenMatch(AlgorithmBase):
    """
        OpenMatch algorithm (https://arxiv.org/abs/2105.14148).
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        # 在这个地方创建了一个AlgorithmBase对象，net_builder是自定义的一个创建base的函数
        super().__init__(args, net_builder, tb_log, logger)
        # openmatch specified arguments
        self.p_cutoff = args.p_cutoff
        self.lambda_em = args.lambda_em
        self.lambda_socr = args.lambda_socr
        self.start_fix = args.start_fix
        self.fix_uratio = args.fix_uratio

    def set_dataset(self):
        dataset_dict = super(OpenMatch, self).set_dataset()
        dataset_dict['train_lb'] = OpenMatchDataset(dset=dataset_dict['train_lb'], name='train_lb')
        dataset_dict['train_ulb'] = OpenMatchDataset(dset=dataset_dict['train_ulb'], name='train_ulb')
        dataset_dict['train_ulb_selected'] = OpenMatchDataset(dset=dataset_dict['train_ulb'], name='train_ulb_selected')
        return dataset_dict

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def set_model(self):
        model = super().set_model()  # backbone
        model = OpenMatchNet(model, self.num_classes, self.gpu)  # including ova classifiers
        return model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = OpenMatchNet(ema_model, self.num_classes, self.gpu)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            self.exclude_dataset()

            self.loader_dict['train_ulb_selected'] = get_data_loader(self.args,
                                                                     self.dataset_dict['train_ulb_selected'],
                                                                     self.args.batch_size * self.fix_uratio,
                                                                     data_sampler=self.args.train_sampler,
                                                                     num_iters=self.num_train_iter // self.epochs,
                                                                     num_epochs=1,
                                                                     num_workers=2 * self.args.num_workers,
                                                                     distributed=self.distributed)

            for data_lb, data_ulb, data_ulb_selected in zip(self.loader_dict['train_lb'],
                                                            self.loader_dict['train_ulb'],
                                                            self.loader_dict['train_ulb_selected']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.tb_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb, **data_ulb_selected))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def train_step(self, x_lb_w_0, x_lb_w_1, y_lb, x_ulb_w_0, x_ulb_w_1, x_ulb_w, x_ulb_s):
        #  x_ulb_w_0 and x_ulb_w_1 are all unlabeled data for training ova_classifiers
        #  x_ulb_w and x_ulb_s are selected for FixMatch training

        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb_w_0, x_lb_w_1, x_ulb_w_0, x_ulb_w_1))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb * 2]
                # outputs['logits_open'].shape: (256, 12)
                # num_lb: 64
                # logits_open_lb.shape: (128, 12)
                # inputs中后面num_lb*2个输入专门用于训练ova_classifiers
                logits_open_lb = outputs['logits_open'][:num_lb * 2]
                logits_open_ulb_0, logits_open_ulb_1 = outputs['logits_open'][num_lb * 2:].chunk(2)
            else:
                raise ValueError("Bad configuration: use_cat should be True!")

            sup_loss = ce_loss(logits_x_lb, y_lb.repeat(2), reduction='mean')
            # logits_open_lb对应x_ulb_w_0和x_ulb_w_1两者的logits
            # x_ulb_w_0和x_ulb_w_1的数据和增强方式都是一模一样的
            # 所以y_lb直接多复制一份，两份y_lb分别对应x_ulb_w_0和x_ulb_w_1
            ova_loss = etf_ova_loss_func(logits_open_lb, y_lb.repeat(2))
            em_loss = em_loss_func(logits_open_ulb_0, logits_open_ulb_1)
            socr_loss = socr_loss_func(logits_open_ulb_0, logits_open_ulb_1)

            fix_loss = torch.tensor(0).cuda(self.gpu)
            if self.epoch >= self.start_fix:
                inputs_selected = torch.cat((x_ulb_w, x_ulb_s), 0)
                outputs_selected = self.model(inputs_selected)
                logits_x_ulb_w, logits_x_ulb_s = outputs_selected['logits'].chunk(2)
                probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)

                # compute mask
                mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

                # generate unlabeled targets using pseudo label hook
                pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                              logits=probs_x_ulb_w,
                                              use_hard_label=True,
                                              softmax=False)

                fix_loss = consistency_loss(logits_x_ulb_s,
                                            pseudo_label,
                                            'ce',
                                            mask=mask)

            total_loss = sup_loss + ova_loss + self.lambda_em * em_loss + self.lambda_socr * socr_loss + fix_loss

        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {'train/sup_loss': sup_loss.item(), 'train/ova_loss': ova_loss.item(),
                   'train/em_loss': em_loss.item(), 'train/socr_loss': socr_loss.item(),
                   'train/total_loss': total_loss.item()}

        if self.epoch >= self.start_fix:
            tb_dict['fix_loss'] = fix_loss.item()
            tb_dict['mask_ratio'] = mask.float().mean().item()

        return tb_dict

    def exclude_dataset(self):
        loader = DataLoader(dataset=self.dataset_dict['train_ulb'],
                            batch_size=self.args.eval_batch_size,
                            drop_last=False,
                            shuffle=False,
                            num_workers=4)

        self.model.eval()
        self.ema.apply_shadow()
        self.print_fn(f"Selecting...")
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                x = data['x_ulb_w_0']
                y = data['y_ulb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                outputs = self.model(x)
                logits, logits_open = outputs['logits'], outputs['logits_open']
                logits = F.softmax(logits, 1)
                logits_open = F.softmax(logits_open.view(logits_open.size(0), 2, -1), 1)
                tmp_range = torch.arange(0, logits_open.size(0)).long().cuda(self.gpu)
                pred_close = logits.data.max(1)[1]
                unk_score = logits_open[tmp_range, 0, pred_close]
                select_idx = unk_score < 0.5
                gt_idx = y < self.args.num_classes
                if batch_idx == 0:
                    select_all = select_idx
                    gt_all = gt_idx
                else:
                    select_all = torch.cat([select_all, select_idx], 0)
                    gt_all = torch.cat([gt_all, gt_idx], 0)

        select_accuracy = accuracy_score(gt_all.cpu().numpy(), select_all.cpu().numpy())
        select_precision = precision_score(gt_all.cpu().numpy(), select_all.cpu().numpy())
        select_recall = recall_score(gt_all.cpu().numpy(), select_all.cpu().numpy())
        
        selected_idx = torch.arange(0, len(select_all)).cuda(self.gpu)
        selected_idx = selected_idx[select_all]
        if self.rank == 0:
            self.print_fn(f"Selected ratio = {len(selected_idx) / len(select_all)}, accuracy = {select_accuracy}, "
                          f"precision = {select_precision}, recall = {select_recall}")

        self.ema.restore()
        self.model.train()
        if self.epoch >= self.start_fix:
            if len(selected_idx) > 0:
                self.dataset_dict['train_ulb_selected'].set_index(selected_idx)

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--lambda_em', float, 0.1),
            SSL_Argument('--lambda_socr', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.0),
            SSL_Argument('--start_fix', int, 10),
            SSL_Argument('--fix_uratio', int, 7)
        ]

class ETF_Classifier(nn.Module):
    # fix_bn: 是否固定Batch Normalization层的参数
    # LWS: 是否使用学习的权重缩放
    # reg_ETF: 是否使用正则化的ETF
    def __init__(self, feat_in, num_classes, gpu, fix_bn=False, LWS=False, reg_ETF=False):
        super(ETF_Classifier, self).__init__()
        # 随机的正交矩阵P，正交矩阵与自身的转置相乘得到单位矩阵
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        # 单位矩阵I
        I = torch.eye(num_classes)
        # 全一矩阵one
        one = torch.ones(num_classes, num_classes)
        # 原文中的公式(1)，得到ETF矩阵M
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))
        self.ori_M = M.cuda(gpu)

        self.LWS = LWS
        self.reg_ETF = reg_ETF
#        if LWS:
#            self.learned_norm = nn.Parameter(torch.ones(1, num_classes))
#            self.alpha = nn.Parameter(1e-3 * torch.randn(1, num_classes).cuda())
#            self.learned_norm = (F.softmax(self.alpha, dim=-1) * num_classes)
#        else:
#            self.learned_norm = torch.ones(1, num_classes).cuda()
        # 归一化层BN_H
        self.BN_H = nn.BatchNorm1d(feat_in)
        # BN_H的参数都在cpu上，而特征向量在gpu上，这里不改的话会报错
        self.BN_H.cuda(gpu)
        if fix_bn:
            self.BN_H.weight.requires_grad = False
            self.BN_H.bias.requires_grad = False

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        # 随机矩阵a
        a = np.random.random(size=(feat_in, num_classes))
        # 通过QR分解(施密特正交化)得到正交矩阵P
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        x = self.BN_H(x)
        
        # L2范数归一化
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        return x