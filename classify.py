import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 1. 加载数据
X = np.load("iq_dataset/X.npy")  # [N, 1024, 2]
y = np.load("iq_dataset/y.npy")  # [N]

# 2. 转换为 PyTorch Tensor，调整维度为 [N, 2, 1024]
X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
y_tensor = torch.tensor(y, dtype=torch.long)

# 3. 划分训练 / 测试集
train_ratio = 0.8
N = X_tensor.shape[0]
split = int(N * train_ratio)
X_train, X_test = X_tensor[:split], X_tensor[split:]
y_train, y_test = y_tensor[:split], y_tensor[split:]

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. 定义模型
class InterferenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(64 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)

# 5. 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InterferenceClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 6. 训练
EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# 7. 测试准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        pred_labels = preds.argmax(dim=1)
        correct += (pred_labels == yb).sum().item()
        total += yb.size(0)

acc = correct / total * 100
print(f"测试准确率：{acc:.2f}%")
