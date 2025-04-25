# import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchviz import make_dot

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_dim=128, num_layers=6, nhead=8):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_dim)

        # Transformer encoder
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                                    dropout=0.1, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, 3)  # 隐藏层到最终嵌入向量
        self.norm = nn.LayerNorm(3)

    def forward(self, x):
        # 先通过线性层将输入转换为隐藏维度
        x = self.embedding(x)
        x = x.unsqueeze(0)  # Transformer要求输入维度为 (seq_len, batch_size, features)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # 对每个样本的所有词向量求均值
        x = self.fc(x)
        x = self.norm(x)  # 对最终输出使用LayerNorm
        x = F.normalize(x, p=2, dim=-1)
        return x

# if __name__ == '__main__':
#     model = TransformerModel(input_size=29)
#
#     # 创建一个随机输入张量
#     x = torch.randn(1, 29)  # 输入维度为 (batch_size, input_size)
#
#     # 前向传播
#     output = model(x)
#
#     # 使用 torchviz 绘制网络结构
#     dot = make_dot(output, params=dict(model.named_parameters()))
#
#     # 保存为图片
#     dot.format = 'png'
#     dot.render('transformer_model')
#
#     print("网络结构图已保存为 transformer_model.png")