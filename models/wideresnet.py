import logging
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0):
        super(WideResNet, self).__init__()
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        return self.fc(out)


class WideResNet_Open(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0):
        super(WideResNet_Open, self).__init__()
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.simclr_layer = nn.Sequential(
                nn.Linear(channels[3], 128),
                nn.ReLU(),
                nn.Linear(128, 128),
        )
        self.fc = nn.Linear(channels[3], num_classes)
        out_open = 2 * num_classes
        # self.fc_open = nn.Linear(channels[3], out_open, bias=False)
        self.ova_classifier_size = 6
        # self.ova_classifiers = [ETF_Classifier(feat_in=channels[3], num_classes=2) for _ in range(self.ova_classifier_size) ]
        
        # ova_classifiers是通过列表解析直接在列表中创建的，而不是通过 nn.Module 的属性初始化的。
        # PyTorch 不能自动检测到列表中的这些模块，并将它们移到 GPU 上，
        # 即使调用 model.cuda()，这些线性层仍然留在 CPU 上
        # nn.ModuleList 是 PyTorch 提供的一个容器，它会自动将其中的 nn.Module 子模块移到与父模块相同的设备上。
        # 可以将 ova_classifiers 从 Python 列表换成 nn.ModuleList，
        # 这样就能确保这些子模块在调用 model.cuda() 时也会被正确移动到 GPU 上
        self.ova_classifiers = nn.ModuleList([nn.Linear(channels[3], 2, bias=False) for _ in range(self.ova_classifier_size) ])
        self.channels = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, feature=False, feat_only=False):
        #self.weight_norm()
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)


        if feat_only:
            return self.simclr_layer(out)
        # out_open = self.fc_open(out)
        
        # logits_open[:2*batchsize].shape: (128, 2, 6)
        logits_open = torch.zeros(out.size(0), 2, self.ova_classifier_size).cuda()
        # 6个etf分别对每个样本的feature使用
        for i in range(self.ova_classifier_size):
            # cur_M = self.ova_classifiers[i].ori_M
            
            # # 使用ETF中的forwrd()将feature进行L2归一化
            # feat_l2 = self.ova_classifiers[i](out)
            # # 这一步就类似于CNN中全连接层的计算过程，用特征向量乘以权重，得到logit
            # l_open = torch.matmul(feat_l2.half(), cur_M.half())
            l_open = self.ova_classifiers[i](out)
            
            logits_open[:, :, i] = l_open
            
        if feature:
            return self.fc(out), logits_open, out
        else:
            return self.fc(out), logits_open

    def weight_norm(self):
        w = self.fc_open.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc_open.weight.data = w.div(norm.expand_as(w))


#
#
class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

#
class ResNet_Open(nn.Module):
    def __init__(self, block, num_blocks, low_dim=128, num_classes=10):
        super(ResNet_Open, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, low_dim)
        self.simclr_layer = nn.Sequential(
            nn.Linear(512*block.expansion, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.fc1 = nn.Linear(512*block.expansion, num_classes)
        self.fc_open = nn.Linear(512*block.expansion, num_classes*2, bias=False)


        #self.l2norm = Normalize(2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out_open = self.fc_open(out)
        if feature:
            return self.fc1(out), out_open, self.simclr_layer(out)
        else:
            return self.fc1(out), out_open



def ResNet18(low_dim=128, num_classes=10):
   return ResNet_Open(ResBasicBlock, [2,2,2,2], low_dim, num_classes)


def build_wideresnet(depth, widen_factor, dropout, num_classes, open=False):
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    build_func = WideResNet_Open if open else WideResNet
    return build_func(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,
                      num_classes=num_classes)

class ETF_Classifier(nn.Module):
    # fix_bn: 是否固定Batch Normalization层的参数
    # LWS: 是否使用学习的权重缩放
    # reg_ETF: 是否使用正则化的ETF
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False):
        super(ETF_Classifier, self).__init__()
        # 随机的正交矩阵P，正交矩阵与自身的转置相乘得到单位矩阵
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        # 单位矩阵I
        I = torch.eye(num_classes)
        # 全一矩阵one
        one = torch.ones(num_classes, num_classes)
        # 原文中的公式(1)，得到ETF矩阵M
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))
        self.ori_M = M.cuda()

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
        self.BN_H.cuda()
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
        # x = self.BN_H(x)
        
        # # L2范数归一化
        # x = x / torch.clamp(
        #     torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        return x