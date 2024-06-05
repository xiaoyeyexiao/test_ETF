import torch
import torch.nn.functional as F
import sys

# 在multi-binary classifier中，全连接层有2*num_classes个结点，每个类对应两个结点
def ova_loss_func(logits_open, label):
    # Eq.(1) in the paper
    # 将 logits_open 的形状重塑为 (batch_size, 2, num_classes)
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    # 使用softmax函数计算每个类所对应的两个结点
    logits_open = F.softmax(logits_open, 1)
    # 形状为 (batch_size, num_classes) 的全零张量
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    # label_range的作用相当于 0:128
    # 每一列相当于openmatch原文中的p^j(t=0|xb)，只有第j个值为1，表示样本属于第j个类的概率
    label_s_sp[label_range, label] = 1  # one-hot labels, in the shape of (bsz, num_classes)
    # 每一列相当于openmatch原文中的p^j(t=1|xb)，只有第j个值为0，表示样本不属于第j个类的概率
    label_sp_neg = 1 - label_s_sp
    # openmatch原文公式(1)
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :] + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :] + 1e-8) * label_sp_neg, 1)[0])
    l_ova = open_loss_neg + open_loss
    return l_ova


def em_loss_func(logits_open_u1, logits_open_u2):
    # Eq.(2) in the paper
    def em(logits_open):
        logits_open = logits_open.view(logits_open.size(0), 2, -1)
        logits_open = F.softmax(logits_open, 1)
        _l_em = torch.mean(torch.mean(torch.sum(-logits_open * torch.log(logits_open + 1e-8), 1), 1))
        return _l_em

    l_em = (em(logits_open_u1) + em(logits_open_u2)) / 2

    return l_em


def socr_loss_func(logits_open_u1, logits_open_u2):
    # Eq.(3) in the paper
    logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
    logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
    logits_open_u1 = F.softmax(logits_open_u1, 1)
    logits_open_u2 = F.softmax(logits_open_u2, 1)
    l_socr = torch.mean(torch.sum(torch.sum(torch.abs(
        logits_open_u1 - logits_open_u2) ** 2, 1), 1))
    return l_socr
