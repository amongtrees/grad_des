# 模型参数
MODEL_CONFIG = {
    "input_size": 29,  # 输入特征的维度
    "hidden_dim": 64,  # 隐藏层维度
    "num_layers": 6,  # Transformer 编码器层数
    "nhead": 8,  # Transformer 注意力头的数量
}

# 数据加载器参数
DATA_LOADER_CONFIG = {
    "batch_size": 64,  # 每个批次的样本数量
    "shuffle": True,  # 是否打乱数据
}

# 训练参数
TRAIN_CONFIG = {
    "num_epochs": 10,  # 训练的轮数
    "learning_rate": 1e-4,  # 学习率
}

# 损失函数参数
LOSS_CONFIG = {
    "margin": 0.0,  # 对比损失的边际
}