# model/load_model.py
import torch
from torch import nn

from model import config
from model import TransformerModel


def load_model(model_path, device='cuda'):
    """
    加载指定路径的模型
    :param model_path: 模型文件的路径
    :param device: 模型加载设备（'cuda' 或 'cpu'）
    :return: 加载的模型
    """
    try:
        model = TransformerModel(input_size=config.MODEL_CONFIG["input_size"], hidden_dim=config.MODEL_CONFIG["hidden_dim"])
        model.apply(init_weights)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)), strict=False)
        model.to(device)
        model.eval()  # 设置模型为评估模式
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)