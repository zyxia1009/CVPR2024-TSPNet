import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PTAL_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor_dict = {
            'prop_point_loss': 1,
            'prop_saliency_loss': 20,
        }
        self.loss_dic = {
            'prop_point_loss': self._prop_point_loss,
            'prop_saliency_loss': self._prop_saliency_loss,
        }

    def _prop_point_loss(self, inputs):
        prop_cas = inputs[0].permute(0, 2, 1)
        prop_att = torch.sigmoid(inputs[1]).permute(0, 2, 1)
        point_label = inputs[2]
        prop_cas = prop_cas * prop_att
        point_label = torch.cat((point_label, torch.zeros_like(point_label[:, :, [0]])), dim=-1)
        point_label[:, torch.where(torch.sum(point_label[0, :, :], dim=-1) == 0)[0], -1] = 1
        loss_fuse = - (point_label * F.log_softmax(prop_cas, dim=-1)).sum(dim=-1).mean()
        return loss_fuse

    def _prop_saliency_loss(self, inputs):
        center_score = torch.sigmoid(inputs[0]).squeeze()
        center_label = inputs[1].squeeze()
        loss = torch.mean(torch.square(center_score - center_label))
        return loss

    def forward(self, s_l_dict):
        losses = {}
        for type, score_label in s_l_dict.items():
            losses[type] = self.loss_dic[type](score_label)
        return losses

    def compute_total_loss(self, loss_dict):
        total_loss = 0
        for type, loss in loss_dict.items():
            total_loss += self.factor_dict[type] * loss
        return total_loss