import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
def evaluate(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for anchor, positive, label in data_loader:
            anchor, positive, label = anchor.to(device), positive.to(device), label.to(device)
            # print(label)
            anchor_output = model(anchor)
            positive_output = model(positive)
            anchor_output = F.normalize(anchor_output, p=2, dim=-1)
            positive_output = F.normalize(positive_output, p=2, dim=-1)
            # 计算余弦相似度
            cosine_similarity = F.cosine_similarity(anchor_output, positive_output, dim=1, eps=1e-8)
            # print(cosine_similarity)

            predictions = (cosine_similarity > 0.5).long()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")