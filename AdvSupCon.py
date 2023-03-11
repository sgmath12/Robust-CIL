import torch



class AsymAdvSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1.0, contrast_mode='one',
                 base_temperature=1.0):
        super(AsymAdvSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, cur_task_idx = None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        iteration_steps = 1
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        past_task_idx = ~cur_task_idx
     

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits

        feature_cur = features[cur_task_idx].squeeze()
        feature_past = features[past_task_idx].squeeze()

        
        feature_adv = feature_past.clone().detach()
        for i in range(iteration_steps):
            feature_adv.requires_grad = True
            dist = torch.matmul(feature_adv, feature_past.T)
            dist = dist.sum(axis = 1).mean()
            grad = torch.autograd.grad(dist, feature_adv, retain_graph=False, create_graph=False)[0]
            feature_adv = feature_adv.detach() - 4/255 * grad.sign()
            

        # 
        anchor_feature[past_task_idx] = feature_adv
        contrast_feature[past_task_idx] = feature_adv


        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask

        logits_mask = mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        # mask = mask * logits_mask

        # compute log_prob
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask = mask * cur_task_idx.reshape([-1,1])
        
        log_prob = log_prob[cur_task_idx]
        mask_ = mask
        mask = mask[cur_task_idx]
        # anchor_count = anchor_count[cur_task_idx]
        batch_size = cur_task_idx.sum()

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # mean_log_prob_pos = mean_log_prob_pos * cur_task_idx.reshape([-1,1])
        # mean_log_prob_pos[mean_log_prob_pos!=mean_log_prob_pos]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
