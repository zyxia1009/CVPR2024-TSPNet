import numpy as np
import torch
import torch.nn as nn
import torchvision


class Backbone_TSPNet(torch.nn.Module):

    def __init__(self, feat_dim, n_class, dropout_ratio, roi_size):
        super().__init__()
        embed_dim = feat_dim // 2
        self.roi_size = roi_size

        self.prop_fusion = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
        )
        self.prop_classifier = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, n_class + 1, 1),
        )
        self.prop_attention = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, 1),
        )
        self.prop_completeness = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, 1),
        )

    def forward(self, feat):
        """
        Inputs:
            feat: tensor of size [B, M, roi_size, D]

        Outputs:
            prop_cas:  tensor of size [B, C, M]
            prop_attn: tensor of size [B, 1, M]
            prop_center:  tensor of size [B, 1, M]
        """
        feat1 = feat[:, :, : self.roi_size // 6, :].max(2)[0]
        feat2 = feat[:, :, self.roi_size // 6: self.roi_size // 6 * 5, :].max(2)[0]
        feat3 = feat[:, :, self.roi_size // 6 * 5:, :].max(2)[0]
        feat = torch.cat((feat2 - feat1, feat2, feat2 - feat3), dim=2)

        feat_fuse = self.prop_fusion(feat)  # [1, M, D]
        feat_fuse = feat_fuse.transpose(-1, -2)  # [1, D, M]

        prop_cas = self.prop_classifier(feat_fuse)  # [1, C, M]
        prop_attn = self.prop_attention(feat_fuse)  # [1, 1, M]
        prop_center = self.prop_completeness(feat_fuse)  # [1, 1, M]
        return prop_cas, prop_attn, prop_center, feat_fuse


class TSPNet_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_class = args.num_classes
        dropout_ratio = 0.5
        self.feat_dim = args.feature_size
        self.max_proposal = args.max_proposal
        self.roi_size = args.roi_size
        self.prop_backbone = Backbone_TSPNet(self.feat_dim, n_class, dropout_ratio, self.roi_size)
        self.refined_label = []

    def extract_roi_features(self, feature, proposal, is_training):
        """
        Surrounding contrastive feature extraction
        Extract region of interest (RoI) features from raw i3d features based on given proposals
        These codes are from <proposal-based multiple instance learning for weakly-supervised temporal action localization>
        Inputs:
            feature: [T, D] tensors
            proposal: [M, 2] tensors
            is_training: bool

        Outputs:
            prop_features:tensor of size [B, M, roi_size, D]
            prop_mask: tensor of size [B, M]
        """
        proposal = proposal[0]
        num_prop = proposal.shape[0]
        # Limit the max number of proposals during training
        if is_training:
            num_prop = min(num_prop, self.max_proposal)
        prop_features = torch.zeros((1, num_prop, self.roi_size, self.feat_dim)).to(feature.device)
        prop_mask = torch.zeros((1, num_prop)).to(feature.device)

        if proposal.shape[0] > num_prop:
            sampled_idx = torch.randperm(proposal.shape[0])[:num_prop]
            proposal = proposal[sampled_idx]

        # Extend the proposal by 25% of its length at both sides
        start, end = proposal[:, 0], proposal[:, 1]
        len_prop = end - start
        start_ext = start - 0.25 * len_prop
        end_ext = end + 0.25 * len_prop

        # Fill in blank at edge of the feature, offset 0.5, for more accurate RoI_Align results
        fill_len = torch.ceil(0.25 * len_prop.max()).long() + 1  # +1 because of offset 0.5
        fill_blank = torch.zeros(fill_len, self.feat_dim).to(feature.device)
        feature = torch.cat([fill_blank, feature[0], fill_blank], dim=0)
        start_ext = start_ext + fill_len - 0.5
        end_ext = end_ext + fill_len - 0.5
        proposal_ext = torch.stack((start_ext, end_ext), dim=1)

        # Extract RoI features using RoI Align operation
        y1, y2 = proposal_ext[:, 0], proposal_ext[:, 1]
        x1, x2 = torch.zeros_like(y1), torch.ones_like(y2)
        boxes = torch.stack((x1, y1, x2, y2), dim=1)  # [M, 4]
        feature = feature.transpose(0, 1).unsqueeze(0).unsqueeze(3)  # [1, D, T, 1]
        feat_roi = torchvision.ops.roi_align(feature, [boxes], [self.roi_size, 1])  # [M, D, roi_size, 1]
        feat_roi = feat_roi.squeeze(3).transpose(1, 2)  # [M, roi_size, D]
        prop_features[0, :proposal.shape[0], :, :] = feat_roi  # [1, M, roi_size, D]
        prop_mask[0, :proposal.shape[0]] = 1  # [1, M]
        return prop_features, prop_mask

    def forward(self, features, proposals, is_training=True):
        """
        Inputs:
            features: list of [T, D] tensors
            proposals: list of [M, 2] tensors
            is_training: bool
        Outputs:

        """
        prop_features, prop_mask = self.extract_roi_features(features, proposals, is_training)
        prop_cas, prop_attn, prop_center, feat_fuse = self.prop_backbone(prop_features)
        return prop_cas, prop_attn, prop_center, feat_fuse
