import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import precision_recall_curve, f1_score
from torch_geometric.data import DataLoader


class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        # 定义三层GCN结构
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        # self.conv3 = GCNConv(32, 16)
        # 定义全连接层
        self.linear = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, batch):
        # 图卷积层
        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0., training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, p=0., training=self.training)
        # x = F.relu(self.conv3(x, edge_index))

        # 全局池化
        x = global_mean_pool(x, batch)

        # 全连接层输出
        x = self.linear(x)
        return torch.sigmoid(x)  # 输出0-1之间的预测值


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        # 使用MSE损失
        loss = F.mse_loss(out.view(-1), data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            y_true.append(data.y.cpu().numpy())
            y_pred.append(out.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred).reshape(-1)


    # 计算precision-recall曲线和最佳F1分数
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1 = np.max(f1_scores)
    best_index = np.argmax(f1_scores)
    print(f1_scores[best_index], precisions[best_index], recalls[best_index])

    return best_f1

save_path = "data"
train_set = torch.load(save_path + "/New_ATSD_train_dataset_cor0.2.pt", weights_only=False)
val_set = torch.load(save_path + "/New_ATSD_val_dataset_cor0.2.pt", weights_only=False)
test_set = torch.load(save_path + "/New_ATSD_test_dataset_cor0.2.pt", weights_only=False)

train_loader = DataLoader(train_set, 100, shuffle=True)
val_loader = DataLoader(val_set, 100, shuffle=False)
test_loader = DataLoader(test_set, 100, shuffle=False)


def main():
    device = torch.device('cpu')

    # 初始化模型
    model = GCN(num_node_features=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_f1 = 0
    test_f1 = 0

    for epoch in range(100):
        # 训练
        loss = train(model, train_loader, optimizer, device)

        # 评估
        val_f1 = evaluate(model, val_loader, device)
        tmp_test_f1 = evaluate(model, test_loader, device)

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            test_f1 = tmp_test_f1

        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, '
              f'Test F1: {tmp_test_f1:.4f}')

    print(f'Best validation F1: {best_val_f1:.4f}')
    print(f'Test F1: {test_f1:.4f}')

if __name__ == '__main__':
    main()



