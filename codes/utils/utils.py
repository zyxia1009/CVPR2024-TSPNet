import json
import numpy as np
import os
import torch

class_dict_thumos = {0: 'BaseballPitch',
                     1: 'BasketballDunk',
                     2: 'Billiards',
                     3: 'CleanAndJerk',
                     4: 'CliffDiving',
                     5: 'CricketBowling',
                     6: 'CricketShot',
                     7: 'Diving',
                     8: 'FrisbeeCatch',
                     9: 'GolfSwing',
                     10: 'HammerThrow',
                     11: 'HighJump',
                     12: 'JavelinThrow',
                     13: 'LongJump',
                     14: 'PoleVault',
                     15: 'Shotput',
                     16: 'SoccerPenalty',
                     17: 'TennisSwing',
                     18: 'ThrowDiscus',
                     19: 'VolleyballSpiking'}

class_dict_beoid = {0: 'rinse_cup',
                    1: 'move_rest',
                    2: 'take_cup',
                    3: 'open_door',
                    4: 'move_seat',
                    5: 'pull_drawer',
                    6: 'insert_wire',
                    7: 'place_tape',
                    8: 'plug_plug',
                    9: 'pour_spoon',
                    10: 'pull-out_weight-pin',
                    11: 'switch-on_socket',
                    12: 'fill_cup',
                    13: 'push_rowing-machine',
                    14: 'press_button',
                    15: 'pick-up_cup',
                    16: 'insert_weight-pin',
                    17: 'insert_foot',
                    18: 'scoop_spoon',
                    19: 'take_spoon',
                    20: 'turn_tap',
                    21: 'pick-up_plug',
                    22: 'hold-down_button',
                    23: 'rotate_weight-setting',
                    24: 'open_jar',
                    25: 'let-go_rowing-machine',
                    26: 'put_jar',
                    27: 'pull_rowing-machine',
                    28: 'stir_spoon',
                    29: 'put_cup',
                    30: 'scan_card-reader',
                    31: 'push_drawer',
                    32: 'pick-up_jar',
                    33: 'pick-up_tape'}

class_dict_gtea = {
    0: 'stir', 1: 'open', 2: 'put', 3: 'close', 4: 'take', 5: 'pour', 6: 'scoop'
}


def t_nms(pro_list, threshold=0.7):
    proposals = np.array(pro_list)
    x1 = proposals[:, 2]
    x2 = proposals[:, 3]
    scores = proposals[:, 1]

    areas = x2 - x1 + 1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]

        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < threshold)[0]
        order = order[inds + 1]
    return keep


def soft_nms(pro_list, iou_thr=0.7, method='gaussian', sigma=0.3):
    dets = np.array(pro_list)
    areas = dets[:, 1] - dets[:, 0] + 1
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 3], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1].tolist())

        xx1 = np.maximum(dets[0, 0], dets[1:, 0])
        xx2 = np.minimum(dets[0, 1], dets[1:, 1])
        inter = np.maximum(xx2 - xx1 + 1, 0.0)
        iou = inter / (dets[0, -1] + dets[1:, -1] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 3] *= weight
        dets = dets[1:, :]

    return retained_box


def filter_segments(segment_predict, vn):
    ambilist = list(open('./data/THUMOS14/Ambiguous_test.txt', 'r'))
    ambilist = [a.strip("\n").split(" ") for a in ambilist]
    num_segment = len(segment_predict)
    ind = np.zeros(num_segment)
    for i in range(num_segment):
        for a in ambilist:
            if a[0] == vn:
                gt = range(int(round(float(a[2]) * 25 / 16)), int(round(float(a[3]) * 25 / 16)))
                pd = range(int(segment_predict[i][0]), int(segment_predict[i][1]))
                IoU = float(len(set(gt).intersection(set(pd)))) / float(len(set(gt).union(set(pd))))
                if IoU > 0:
                    ind[i] = 1
    s = [segment_predict[i, :] for i in range(num_segment) if ind[i] == 0]
    return np.array(s)


def res2json(result, dataset='THUMOS14'):
    if dataset == 'THUMOS14':
        class_dict = class_dict_thumos
    elif dataset == 'GTEA':
        class_dict = class_dict_gtea
    else:
        class_dict = class_dict_beoid
    result_json = []
    for proposal in result:
        res = {}
        res["label"] = class_dict[proposal[2]]
        res["score"] = proposal[3]
        res["segment"] = [proposal[0], proposal[1]]
        result_json.append(res)
    return result_json


def log_metrics(map_iou, step, config):
    best_flag = False
    text = {}
    text['step'] = step
    for i in range(len(map_iou)):
        text[f'mAP@0.{i + 1}:'] = round(map_iou[i], 4)
    text['Average mAP 0.1_0.7:'] = round(map_iou[:7].mean(), 4)
    text['Average mAP 0.1_0.5:'] = round(map_iou[:5].mean(), 4)
    text['Average mAP 0.3_0.7:'] = round(map_iou[2:7].mean(), 4)

    map_file = os.path.join(config.output_folder, 'mAP.json')
    if not os.path.exists(map_file):
        record = {}
        record['mAP_record'] = []
    else:
        record = json.load(open(map_file, 'r'))
    record['mAP_record'].append(text)
    json.dump(record, open(map_file, 'w'), indent=1)

    best_file = os.path.join(config.output_folder, 'best_mAP.json')
    if not os.path.exists(best_file):
        os.system(f'cp {map_file} {best_file}')
        best_flag = True
    else:
        best_map = json.load(open(best_file, 'r'))
        if float(best_map['mAP_record'][0]["Average mAP 0.1_0.7:"]) < text["Average mAP 0.1_0.7:"]:
            best_record = {}
            best_record['mAP_record'] = []
            best_record['mAP_record'].append(text)
            json.dump(best_record, open(best_file, 'w'), indent=1)
            best_flag = True
    return best_flag


def boundary_adaption(proposals, proposal_score, refine_threshold=0.4):
    """
    alignment-based boundary adaption
    """
    lock_flag = np.ones(proposals.shape[0]) * -1
    new_proposal = np.copy(proposals)
    descend_idx = np.argsort(proposal_score)[::-1]
    for idx in descend_idx:
        if lock_flag[idx] == 1:
            continue
        tar_proposal = proposals[idx]

        x1 = np.maximum(tar_proposal[0], proposals[:, 0])
        x2 = np.minimum(tar_proposal[1], proposals[:, 1])
        segments_intersection = (x2 - x1).clip(0).astype(float)
        segments_union = (proposals[:, 1] - proposals[:, 0]) + (
                tar_proposal[1] - tar_proposal[0]) - segments_intersection
        iou = segments_intersection / segments_union

        sim_idx = np.where(iou > refine_threshold)
        iou = iou[sim_idx] / np.sum(iou[sim_idx])
        norm_iou_score = (proposal_score[sim_idx] * iou) / np.sum(proposal_score[sim_idx] * iou)  # [M1]
        new_proposal[idx] = np.sum(proposals[sim_idx] * norm_iou_score[:, np.newaxis], axis=0)

        lock_flag[sim_idx] = 1
    return new_proposal


@torch.no_grad()
def update_label(dataset, dataloader, model: torch.nn.Module, device: torch.device, up_threshold=0.8):
    """
    saliency point updating
    """
    delta_point_dict = {}
    model.eval()
    for idx, [feature, _, vd_name, proposals, _, _, proposals_multi_flag] in enumerate(
            dataloader):
        feature = feature.to(device)
        proposals = proposals.to(device)
        video_name = vd_name[0]

        _, _, prop_iou, _ = model(feature, proposals, is_training=False)
        comp_score = torch.sigmoid(prop_iou).squeeze(0).squeeze(0).cpu().numpy()
        proposals = proposals.squeeze(0).cpu().numpy()
        proposals_point_id = dataset.proposals_point_id[video_name]
        proposals_multi_flag = proposals_multi_flag[0].cpu().numpy()

        delta_point_dict[video_name] = {}
        update_idx = np.where(comp_score > up_threshold)[0]
        comp_score = comp_score[update_idx]
        proposals = proposals[update_idx]
        proposals_point_id = proposals_point_id[update_idx]
        proposals_multi_flag = proposals_multi_flag[update_idx]

        point_set = set(proposals_point_id.tolist())
        for point_id in point_set:
            id = np.where(proposals_point_id == point_id)[0]
            score = comp_score[id] / np.sum(comp_score[id])
            center = (proposals[id, 1] + proposals[id, 0]) / 2
            point = proposals_multi_flag[id[0], 0]
            delat_point = np.sum((center - point) * score)
            delta_point_dict[video_name][str(int(point_id))] = delat_point
    dataset.updata_label(delta_point_dict)
