import torch
import matplotlib.pyplot as plt
from model import config
from model.contrastiveloss import ContrastiveLoss
from model.eval import evaluate


def train(model, device, train_data_loader, test_data_loader):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot([], [], 'b-', label='Training Loss')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Real-time Training Loss')
    ax.grid(True)
    ax.legend()
    # 训练记录
    step_losses = []
    x_steps = []
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

            # 记录loss
            global_step = epoch * len(train_data_loader) + step
            step_losses.append(loss.item())
            x_steps.append(global_step)

            # 更新曲线（每10步更新一次）
            if global_step % 10 == 0:
                line.set_data(x_steps, step_losses)
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.01)
            if step % 100 == 0:  # 每100步打印一次
                print(f"Epoch [{epoch + 1}/{config.TRAIN_CONFIG['num_epochs']}], Step [{step}/{len(train_data_loader)}], Loss:{loss.item():.4f}")
                # evaluate(model, test_data_loader, device)
            # if step and step % 1000 == 0:
            #     evaluate(model, test_data_loader, device)
        global_step = (epoch + 1) * len(train_data_loader) - 1

        # 绘制epoch边界竖线（红色虚线）
        ax.axvline(x=global_step, color='r', linestyle='--', alpha=0.2, linewidth=0.8)

        # 添加epoch标签文本（可选）
        ax.text(global_step, ax.get_ylim()[1] * 0.9,
                f'E{epoch + 1}', color='r', ha='center', va='top', fontsize=8)

        # 立即更新图形
        plt.draw()
        plt.pause(0.01)
        avg_loss = running_loss / len(train_data_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        # evaluate(model, test_data_loader, device)
    plt.ioff()
    plt.show()