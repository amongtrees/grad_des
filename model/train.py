import torch

from model import config
from model.contrastiveloss import ContrastiveLoss
from model.eval import evaluate


def train(model, device, train_data_loader, test_data_loader):

    # 损失函数和优化器
    criterion = ContrastiveLoss(margin=config.LOSS_CONFIG["margin"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN_CONFIG["learning_rate"])
    max_grad_norm = 1.0
    # 训练循环
    num_epochs = config.TRAIN_CONFIG["num_epochs"]

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for step, (anchor, positive, label) in enumerate(train_data_loader):
            anchor, positive, label = anchor.to(device), positive.to(device), label.to(device)
            # 清空梯度
            optimizer.zero_grad()

            # 获取模型输出
            anchor_output = model(anchor)
            positive_output = model(positive)

            # 计算对比损失
            loss = criterion(anchor_output, positive_output, label)
            running_loss += loss.item()

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # 更新模型参数
            optimizer.step()
            if step % 100 == 0:  # 每100步打印一次
                print(f"Epoch [{epoch + 1}/{config.TRAIN_CONFIG['num_epochs']}], Step [{step}/{len(train_data_loader)}], Loss:{loss.item():.4f}")
                # evaluate(model, test_data_loader, device)
            if step and step % 1000 == 0:
                evaluate(model, test_data_loader, device)
        avg_loss = running_loss / len(train_data_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        evaluate(model, test_data_loader, device)