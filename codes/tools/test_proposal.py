import json
import time
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from utils import res2json, log_metrics, soft_nms, filter_segments, boundary_adaption
from utils.eval_detectionpmil import ANETdetection


@torch.no_grad()
def test_proposal(config,
                  model: nn.Module,
                  device: torch.device,
                  dataloader: DataLoader,
                  itr: int):
    model.eval()
    dataset = config.data_path.split('/')[-1]

    final_res = {}
    final_res['results'] = {}

    predictions = []

    for idx, [feature, _, vd_name, proposals, _, _, _] in enumerate(
            dataloader):

        feature = feature.to(device)
        proposals = proposals.to(device)
        vid_name = vd_name[0]

        prop_cas, prop_attn, prop_center, sim = model(feature, proposals, is_training=False)
        prop_cas = F.softmax(prop_cas.permute(0, 2, 1), dim=-1).squeeze()
        prop_attn = torch.sigmoid(prop_attn.permute(0, 2, 1))[0]
        prop_center = torch.sigmoid(prop_center.permute(0, 2, 1))[0]

        pred_vid_score = (prop_cas * prop_attn).sum(0) / (prop_attn.sum(0) + 1e-6)
        pred_vid_score = pred_vid_score[:-1]
        pred_vid_score = pred_vid_score.cpu().numpy()
        prop_score = (prop_cas * prop_attn * prop_center).cpu().numpy()

        pred = np.where(pred_vid_score >= 0.1)[0]

        if len(pred) == 0:
            pred = np.array([np.argmax(pred_vid_score)])
        proposal_dict = {}
        proposals = proposals[0].cpu().numpy()

        for c in pred:
            c_temp = []
            c_proposals = boundary_adaption(proposals, prop_score[:, c],
                                            config.refine_threshold)
            for i in range(proposals.shape[0]):
                c_score = prop_score[i, c]
                c_temp.append([c_proposals[i, 0], c_proposals[i, 1], c, c_score])
            proposal_dict[c] = c_temp

        final_proposals = []
        for class_id in proposal_dict.keys():
            temp_proposal = soft_nms(proposal_dict[class_id])
            final_proposals += temp_proposal

        final_proposals = np.array(final_proposals)
        if dataset == 'THUMOS14':
            final_proposals = filter_segments(final_proposals, vid_name)
        #

        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for i in range(np.shape(final_proposals)[0]):
            video_lst.append(vid_name)
            t_start_lst.append(final_proposals[i, 0])
            t_end_lst.append(final_proposals[i, 1])
            label_lst.append(final_proposals[i, 2])
            score_lst.append(final_proposals[i, 3])
        prediction = pd.DataFrame({"video-id": video_lst,
                                   "t-start": t_start_lst,
                                   "t-end": t_end_lst,
                                   "label": label_lst,
                                   "score": score_lst, })
        predictions.append(prediction)

        final_res['results'][vid_name] = res2json(final_proposals, dataset)

    iou = np.linspace(0.1, 0.7, 7)
    dmap_detect = ANETdetection(f'./data/{dataset}/Annotations', tiou_thresholds=iou,
                                subset="test", verbose=True)

    dmap_detect.prediction = pd.concat(predictions).reset_index(drop=True)
    mAP, dmap_class = dmap_detect.evaluate()

    best_flag = log_metrics(map_iou=mAP, step=itr, config=config)

    model_file = os.path.join(config.output_folder, "last_model.pkl")
    torch.save(model.state_dict(), model_file)

    json_path = os.path.join(config.output_folder, 'last_proposals.json')
    with open(json_path, 'w') as f:
        json.dump(final_res, f)
        f.close()

    best_proposal_file = os.path.join(config.output_folder, 'best_proposals.json')
    if not os.path.exists(best_proposal_file) or best_flag:
        os.system(f'cp {json_path} {best_proposal_file}')

    best_model_file = os.path.join(config.output_folder, 'best_model.pkl')
    if not os.path.exists(best_proposal_file) or best_flag:
        os.system(f'cp {model_file} {best_model_file}')
