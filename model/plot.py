import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from model.load_model import load_model
from model.load_data import load_csv
if __name__ == '__main__':
    model = load_model(r"E:\Graduate_Design\project\model\is_non_model.pth")
    all_features = []
    csv_paths = ["E:\Graduate_Design\project\CSVs\Total_CSVs\l1-doh.csv", "E:\Graduate_Design\project\CSVs\Total_CSVs\l1-nondoh.csv"]
    first_file_features = load_csv(csv_paths[0])
    num_samples = 1000
    for path in csv_paths:
        if path == csv_paths[0]:
            # 第一个文件，加载全部样本
            features = load_csv(path, num_samples=num_samples)
        else:
            # 其他文件，加载与第一个文件相同数量的样本
            features = load_csv(path, num_samples=num_samples)
        all_features.append(features)
    all_features = np.vstack(all_features)
    embeddings = []
    for sample in all_features:
        # 将样本输入模型，生成嵌入向量
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to('cuda')
        with torch.no_grad():
            embedding = model(sample_tensor)
        embeddings.append(embedding.cpu().numpy())

    # 合并所有嵌入向量
    embeddings = np.vstack(embeddings)
    # embeddings = F.normalize(torch.tensor(embeddings), p=2, dim=-1).numpy()
    print(embeddings.shape)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 获取嵌入向量的最小值和最大值
    x_min, x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
    y_min, y_max = embeddings[:, 1].min(), embeddings[:, 1].max()
    z_min, z_max = embeddings[:, 2].min(), embeddings[:, 2].max()

    # 设置坐标轴范围
    ax.set_xlim(x_min - 0.1, x_max + 0.1)  # X 轴范围
    ax.set_ylim(y_min - 0.1, y_max + 0.1)  # Y 轴范围
    ax.set_zlim(z_min - 0.1, z_max + 0.1)  # Z 轴范围
    # 前 num_samples 个点用蓝色表示
    ax.scatter(embeddings[:num_samples, 0], embeddings[:num_samples, 1], embeddings[:num_samples, 2], c='b', marker='o',
               label='First File',alpha=0.1)

    # 后 num_samples 个点用红色表示
    ax.scatter(embeddings[num_samples:, 0], embeddings[num_samples:, 1], embeddings[num_samples:, 2], c='r', marker='o',
               label='Second File',alpha=0.1)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(elev=20, azim=45)
    ax.legend()  # 显示图例
    plt.show()