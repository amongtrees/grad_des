import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from model.eval import evaluate
import model.config as config
from model.train import train
from model.transformer import TransformerModel
from model.load_data import load_csv, ContrastiveDataset
from model.load_model import load_model
if __name__ == '__main__':
    # 加载数据
    doh_data = load_csv('../CSVs/Total_CSVs/l1-doh.csv')
    nondoh_data = load_csv('../CSVs/Total_CSVs/l1-nondoh.csv')

    # 划分 DoH 和 Non-DoH 数据集
    doh_train, doh_test = train_test_split(doh_data, test_size=0.2, random_state=42)
    nondoh_train, nondoh_test = train_test_split(nondoh_data, test_size=0.2, random_state=42)

    # 创建训练集和测试集的 ContrastiveDataset
    train_dataset = ContrastiveDataset(doh_train, nondoh_train)
    test_dataset = ContrastiveDataset(doh_test, nondoh_test)

    # 创建数据加载器
    train_data_loader = DataLoader(train_dataset, batch_size=config.DATA_LOADER_CONFIG["batch_size"], shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=config.DATA_LOADER_CONFIG["batch_size"], shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = TransformerModel(input_size=doh_data.shape[1], hidden_dim=config.MODEL_CONFIG["hidden_dim"],
    #                      num_layers=config.MODEL_CONFIG["num_layers"],
    #                      nhead=config.MODEL_CONFIG["nhead"]).to(device)

    # def init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)
    #
    #
    # model.apply(init_weights)
    # model.to(device)
    # train(model,device,train_data_loader,test_data_loader)
    model = load_model(r"E:\Graduate_Design\project\model\is_non_model.pth")
    evaluate(model, test_data_loader, device)
    # torch.save(model.state_dict(), "is_non_model.pth")