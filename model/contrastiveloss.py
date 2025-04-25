import torch.nn.functional as F
import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, label):
        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(anchor, positive, dim=1, eps=1e-8)

        # 对比损失：对于正样本，尽量最大化相似度；对于负样本，尽量最小化相似度
        loss = 0.5 * (label.float() * torch.pow((1 - cosine_similarity), 2) +
                      (1 - label.float()) * torch.pow(F.relu(cosine_similarity - self.margin), 2))
        # print('LOSS:', loss)
        return loss.mean()