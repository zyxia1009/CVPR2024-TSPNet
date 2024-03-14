import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import json


class PTAL_Dataset(Dataset):

    def __init__(self,
                 data_path: str,
                 subset: str = 'test',
                 modality: str = 'both',
                 num_classes: int = 20,
                 feature_fps: int = 25,
                 soft_value: float = 0.4
                 ):
        self.data_path = data_path
        self.subset = subset

        self.modality = modality
        self.feature_fps = feature_fps

        self.dataset = self.data_path.split('/')[-1]
        self.cls_dict = json.load(open('./data/dataset_cls_dict.json', 'rb'))[self.dataset]
        self.num_classes = num_classes
        self.soft_value = soft_value
        # Load label files
        self.gt = json.load(open(os.path.join(self.data_path, 'gt.json'), 'rb'))
        self.p_label = pd.read_csv(os.path.join(self.data_path, 'train_df_ts_in_gt.csv')).groupby('video_id')
        self.fps_dict = json.load(open(os.path.join(self.data_path, 'fps.json'), 'rb'))
        self.delta_dict = {}
        # Get video names
        self.vid_names = self._get_vidname()

        # Get proposals
        self.proposals, \
        self.proposals_point, \
        self.proposals_center_label, \
        self.proposals_multi_flag, \
        self.proposals_point_id = self._get_proposals()

    def __getitem__(self, item):
        video_name = self.vid_names[item]
        feature = torch.as_tensor(self._get_feature(video_name), dtype=torch.float)  # [T, N]
        video_label = torch.as_tensor(self._get_video_label(video_name), dtype=torch.float)  # [C]
        proposal = torch.as_tensor(self.proposals[video_name], dtype=torch.float)
        proposals_point = torch.as_tensor(self.proposals_point[video_name], dtype=torch.float)
        proposals_center_label = torch.as_tensor(self.proposals_center_label[video_name], dtype=torch.float)
        proposals_multi_flag = torch.as_tensor(self.proposals_multi_flag[video_name], dtype=torch.float)
        return feature, video_label, video_name, proposal, proposals_point, proposals_center_label, proposals_multi_flag

    def __len__(self):
        return len(self.vid_names)

    def _get_vidname(self):
        """
        Get video name list
        """
        vid_names = []
        for name, label in self.gt['database'].items():
            if label['subset'] == self.subset:
                vid_names.append(name)
        return vid_names

    def _get_feature(self, vid_name):
        """
        Get video feature
        """
        feature_path = os.path.join(self.data_path, 'features', self.subset)
        if self.modality == 'rgb':
            feature = np.load(os.path.join(feature_path, 'rgb', f'{vid_name}.npy'))
        elif self.modality == 'flow':
            feature = np.load(os.path.join(feature_path, 'flow', f'{vid_name}.npy'))
        else:
            r_feature = np.load(os.path.join(feature_path, 'rgb', f'{vid_name}.npy'))
            f_feature = np.load(os.path.join(feature_path, 'flow', f'{vid_name}.npy'))
            feature = np.concatenate((r_feature, f_feature), axis=1)
        return feature

    def _get_video_label(self, vid_name):
        """
        Get video-level label
        """
        video_label = np.zeros(self.num_classes, dtype=np.float32)
        annotations = self.gt['database'][vid_name]['annotations']
        for annotation in annotations:
            cls_id = self.cls_dict[annotation['label']]
            video_label[cls_id] = 1
        return video_label

    def _get_proposals(self, delta_point_dict=None):
        """
        get proposals and generate the center labels from the original points or the updated saliency points
        """
        history_points = []
        proposals_file = json.load(open(f'{self.data_path}/LAC_proposal_{self.dataset}_{self.subset}.json'))[
            'results']
        proposals = {}
        proposals_point = {}
        proposals_center_label = {}
        proposals_multi_flag = {}
        proposals_point_id = {}
        proposals_mask = {}
        t_factor = self.feature_fps / 16.0

        act, bg, multi = 0, 0, 0
        for idx, name in enumerate(self.vid_names):
            video_proposals = proposals_file[name]
            if delta_point_dict is not None:
                delta_point = delta_point_dict[name]
            else:
                delta_point = None

            segments = []
            for i in range(len(video_proposals)):
                segment = video_proposals[i]['segment']
                t_start = round(segment[0] * t_factor, 2)
                t_end = round(segment[1] * t_factor, 2)
                segments.append([t_start, t_end])

            if len(segments) == 0:
                segments.append([0, 0])

            segments = np.array(segments)  # M, 2
            proposals[name] = segments
            proposals_multi_flag[name] = np.zeros((len(video_proposals), 200))
            proposals_point_id[name] = np.ones(len(video_proposals)) * -1
            proposals_mask[name] = np.zeros(len(video_proposals))

            t_factor_point = self.feature_fps / (self.fps_dict[name] * 16)
            labels_point = np.zeros((len(video_proposals), self.num_classes))  # point
            weight_point = np.ones(len(video_proposals)) * self.soft_value  # [M] for roi pooling

            if self.subset == 'test':
                proposals_point[name] = labels_point
                proposals_center_label[name] = weight_point
                continue

            for p_label_idx, annotation in self.p_label.get_group(name).iterrows():
                # get point annotation information
                ann_point = int(annotation['point'])
                ann_class = annotation['class']
                time_point = float(ann_point * t_factor_point)

                if p_label_idx not in self.delta_dict.keys():
                    self.delta_dict[p_label_idx] = 0

                if delta_point is not None:
                    if str(p_label_idx) in delta_point.keys():
                        time_point += delta_point[str(p_label_idx)]

                feature_clsidx = self.cls_dict[ann_class]

                history_points.append({})
                history_points[-1]['video'] = name
                history_points[-1]['start'] = float(int(annotation['start_frame']) * t_factor_point)
                history_points[-1]['end'] = float(int(annotation['stop_frame']) * t_factor_point)
                history_points[-1]['point'] = time_point

                # filter the proposals that contain the point label
                label_point_mask = np.zeros(len(video_proposals))
                label_point_mask[np.where(time_point - segments[:, 0] > 0)[0]] += 1
                label_point_mask[np.where(time_point - segments[:, 1] < 0)[0]] += 1
                vaild_idx = np.where(label_point_mask == 2)[0]

                for id in vaild_idx:
                    # generate proposal-level point label
                    labels_point[id, feature_clsidx] = 1  # Mv
                    # generate proposal-level center label
                    if proposals_mask[name][id] == 0:
                        weight = (time_point - segments[id, 0]) / (segments[id, 1] - segments[id, 0])  # Mv
                        weight_point[id] = 1 - np.abs(weight - 0.5) * 2
                        proposals_point_id[name][id] = p_label_idx

                    proposals_multi_flag[name][id, int(proposals_mask[name][id])] = time_point
                    proposals_mask[name][id] += 1

            proposals_point[name] = labels_point
            proposals_center_label[name] = weight_point
            act += np.where(np.sum(labels_point, axis=1) > 0)[0].shape[0]
            bg += np.where(np.sum(labels_point, axis=1) == 0)[0].shape[0]
            multi += np.where(proposals_mask[name] > 1)[0].shape[0]

        print(f'act:{act}, bg:{bg}, multi:{multi}')
        return proposals, proposals_point, proposals_center_label, proposals_multi_flag, proposals_point_id

    def updata_label(self, delta_point_dict):
        self.proposals, \
        self.proposals_point, \
        self.proposals_center_label, \
        self.proposals_multi_flag, \
        self.proposals_point_id = self._get_proposals(
            delta_point_dict)
