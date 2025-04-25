import numpy as np
import torch.nn.functional as F
import torch

from model.load_model import load_model
from model.load_data import load_csv


class Classifier():
    def __init__(self, model_path, anchor_path):
        self.model = load_model(model_path)
        self.anchor = load_csv(anchor_path)

    def classify(self, input, device='cuda'):
        # 将输入数据转换为张量，并移动到指定的设备（默认是GPU）
        input_tensor = torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(device)

        # 随机从 anchor 中选择100个样本
        random_anchors = np.random.choice(self.anchor.shape[0], 100, replace=False)
        random_anchors = self.anchor[random_anchors]
        anchor_tensors = [torch.tensor(anchor, dtype=torch.float32).unsqueeze(0).to(device) for anchor in
                          random_anchors]

        # 计算输入和每个 anchor 的嵌入
        with torch.no_grad():
            input_embedding = self.model(input_tensor)
            anchor_embeddings = [self.model(anchor_tensor) for anchor_tensor in anchor_tensors]

        # 计算余弦相似度，并统计相似度大于 0.5 的个数
        count_above_threshold = sum(F.cosine_similarity(input_embedding, anchor_embedding, dim=1, eps=1e-8) > 0.5
                                    for anchor_embedding in anchor_embeddings)

        # 如果大于 90 个返回 True，否则返回 False
        return count_above_threshold > 90