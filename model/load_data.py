import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

def load_csv(path, num_samples=None):
    # 加载CSV文件，从第6列到倒数第2列
    df = pd.read_csv(path)
    if num_samples is not None:
        df = df.head(num_samples)
    df = df.round(5)
    df = df.dropna()
    features = df.iloc[:, 5:-1].values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled

class ContrastiveDataset(Dataset):
    def __init__(self, doh_data, nondoh_data):
        self.doh_data = doh_data
        if len(nondoh_data) > len(doh_data):
            nondoh_data = nondoh_data[:len(doh_data)]
        self.nondoh_data = nondoh_data

    def __len__(self):
        return len(self.doh_data) + len(self.nondoh_data)

    def __getitem__(self, idx, device='cuda'):
        if idx < len(self.doh_data):
            # 正样本对
            anchor = self.doh_data[idx]
            positive = self.doh_data[(idx + 1) % len(self.doh_data)]  # 从DoH中选择一个正样本
            label = 1
        else:
            # 负样本对
            anchor = self.doh_data[idx % len(self.doh_data)]
            positive = self.nondoh_data[idx % len(self.nondoh_data)]  # 从Non-DoH中选择一个负样本
            label = 0
        label_tensor = torch.tensor(label, dtype=torch.long)
        # 将anchor和positive转换为torch张量
        return torch.tensor(anchor, dtype=torch.float32).to(device), torch.tensor(positive, dtype=torch.float32).to(device), label_tensor.to(device)


if __name__ == '__main__':
    print(torch.cuda.is_available())
