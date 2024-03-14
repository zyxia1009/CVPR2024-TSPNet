import numpy as np
import torch.nn as nn
import torch.optim

def train_one_proposal_batch(model: nn.Module,
                             device: torch.device,
                             dataloader: iter,
                             criterion: nn.Module,
                             optimier: torch.optim.Optimizer,
                             batch_size: int):
    model.train()
    total_loss = []
    log_losses = {}
    for b in range(1, batch_size + 1):
        feature, v_label, vid_name, proposal, proposals_point, proposals_center_label, proposals_multi_flag = next(dataloader)
        feature = feature.to(device)
        proposal = proposal.to(device)
        proposals_point = proposals_point.to(device)
        proposals_center_label = proposals_center_label.to(device)

        prop_cas, prop_attn, prop_center_score, _ = model(feature, proposal, is_training=True)

        # compute loss
        s_l_dict = {
            'prop_point_loss': [prop_cas, prop_attn, proposals_point],
            'prop_saliency_loss': [prop_center_score, proposals_center_label],
        }
        loss_dict = criterion(s_l_dict)

        for k in loss_dict.keys():
            if k not in log_losses:
                log_losses[k] = []
            else:
                log_losses[k].append(loss_dict[k].detach().cpu().item())
        total_loss.append(criterion.compute_total_loss(loss_dict))

    total_loss = sum(total_loss) / batch_size
    optimier.zero_grad()
    total_loss.backward()
    optimier.step()
