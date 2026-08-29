"""
train_cnn_pytorch.py
====================
基于 PyTorch 的 DeepCNN 多标签分类模型训练脚本。
输入：理论协方差矩阵的实部、虚部、相位 [batch, 3, 10, 10]
输出：多热标签（网格点数自动从数据读取）
模型结构严格参照原仓库 DeepCNN（已修复卷积层权重共享 Bug）。
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- 配置 ----------
SEED = 42
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-5
VAL_SPLIT = 0.1
EARLY_STOP_PATIENCE = 20
DATA_PATH = 'CNN_train.h5'          # 训练数据
MODEL_SAVE = 'cnn_model.pth'     # 模型保存路径
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)

ANGLE_GRID = None


def normalize_covariance_batch(covariance: np.ndarray) -> np.ndarray:
    def with_phase(two_channel: np.ndarray) -> np.ndarray:
        complex_cov = two_channel[:, 0] + 1j * two_channel[:, 1]
        phase = np.angle(complex_cov).astype(np.float32)
        return np.concatenate([two_channel.astype(np.float32), phase[:, None]], axis=1)

    if covariance.ndim == 4:
        if covariance.shape[1] == 2:
            return with_phase(covariance)
        if covariance.shape[1] == 3:
            return covariance.astype(np.float32)
        if covariance.shape[2] == 2:
            return with_phase(covariance.transpose(3, 2, 0, 1))
        if covariance.shape[-1] == 2:
            return with_phase(covariance.transpose(0, 3, 1, 2))
        raise ValueError(f'Cannot locate channel axis in covariance array, shape={covariance.shape}')

    if covariance.ndim == 5:
        if covariance.shape[2] == 2:
            return with_phase(covariance.reshape(-1, covariance.shape[2], covariance.shape[3], covariance.shape[4]))
        if covariance.shape[-1] == 2:
            reordered = covariance.transpose(0, 1, 4, 2, 3)
            return with_phase(reordered.reshape(-1, reordered.shape[2], reordered.shape[3], reordered.shape[4]))
        raise ValueError(f'Cannot locate channel axis in covariance array, shape={covariance.shape}')

    raise ValueError(f'Expected 4D or 5D covariance array, got shape={covariance.shape}')


def build_labels_from_angles(angles: np.ndarray, angle_grid: np.ndarray) -> np.ndarray:
    if angles.ndim != 2:
        raise ValueError(f'Expected 2D angle array, got shape={angles.shape}')

    if angles.shape[0] == angle_grid.size and angles.shape[1] != angle_grid.size:
        angles = angles.T

    labels = np.zeros((angles.shape[0], angle_grid.size), dtype=np.float32)
    for row_idx, row in enumerate(angles):
        for value in np.asarray(row).ravel():
            if np.isnan(value):
                continue
            grid_idx = int(np.argmin(np.abs(angle_grid - value)))
            labels[row_idx, grid_idx] = 1.0
    return labels

# ---------- 数据集 ----------
class CovarianceDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            if 'X' in f and 'Y' in f:
                self.X = f['X'][:]   # (N, 2, 10, 10)  float32
                self.Y = f['Y'][:]   # (N, grid_size)   float32
            elif 'theor' in f and 'angles' in f:
                covariance = f['theor'][:]
                angle_grid = f['angle_grid'][:] if 'angle_grid' in f else np.arange(-60, 60.5, 0.5, dtype=np.float32)
                self.X = normalize_covariance_batch(covariance)
                self.Y = build_labels_from_angles(f['angles'][:], angle_grid)
            else:
                raise KeyError("HDF5 file must contain either X/Y or theor/angles datasets")
        print(f'Loaded {len(self.X)} samples from {h5_path}')
        print(f'X shape: {self.X.shape}, Y shape: {self.Y.shape}')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.Y[idx]).float()
        return x, y

# ---------- 模型（与原仓库 DeepCNN 一致，已修复） ----------
class DeepCNN(nn.Module):
    def __init__(self, grid_size=241):
        super(DeepCNN, self).__init__()
        self.grid_size = grid_size
        # 输入 2 通道 -> 256 通道，卷积核 3x3，第一层 stride=2（与上游一致）
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        # 三个独立的 2x2 卷积层，不再共享权重
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # 经过 conv1(stride=2)->conv2->conv3->conv4 后，针对 10x10 输入尺寸结果为 256×1×1
        self.fc1 = nn.Linear(256 * 1 * 1, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, grid_size)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        # 全连接层使用 xavier uniform（与原仓库一致）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [batch, 2, 10, 10]
        # Conv 1: 3x3, stride=2, no padding => [batch, 256, 4, 4]
        x = self.relu(self.bn1(self.conv1(x)))
        # Conv 2: 2x2, no padding => [batch, 256, 3, 3]
        x = self.relu(self.bn2(self.conv2(x)))
        # Conv 3: 2x2, no padding => [batch, 256, 2, 2]
        x = self.relu(self.bn3(self.conv3(x)))
        # Conv 4: 2x2, no padding => [batch, 256, 1, 1]
        x = self.relu(self.bn4(self.conv4(x)))
        # 展平并进入全连接层
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

# ---------- 训练函数 ----------
def train():
    dataset = CovarianceDataset(DATA_PATH)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DeepCNN(grid_size=dataset.Y.shape[1]).to(DEVICE)
    print(model)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, verbose=True)

    # history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    def calc_accuracy(loader):
        total_correct = 0
        total_count = 0
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred_bin = (output >= 0.5).float()
            total_correct += (pred_bin == target).sum().item()
            total_count += target.numel()
        return total_correct / total_count if total_count > 0 else 0.0

    best_val_loss = float('inf')
    epochs_since_improve = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        epoch_start = time.perf_counter()
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch+1:3d}/{EPOCHS}', unit='batch', leave=False):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # compute element-wise binary accuracy over the full split
        model.eval()
        with torch.no_grad():
            train_acc = calc_accuracy(train_loader)
            val_acc = calc_accuracy(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        print(f'Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {time.perf_counter() - epoch_start:.0f}s')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f'  -> Best model saved at epoch {epoch+1}')
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= EARLY_STOP_PATIENCE:
            print(f'Early stopping triggered at epoch {epoch+1} (no val loss improvement for {EARLY_STOP_PATIENCE} epochs)')
            break

    print(f'Training finished. Best Val Loss: {best_val_loss:.6f}')

    # save history
    np.savez('train_history.npz', train_losses=train_losses, val_losses=val_losses, train_accs=train_accs, val_accs=val_accs)

    # plot
    epochs_range = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8,4))
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Accuracy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('training_accuracy.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=200)
    plt.close()
    print('Training history saved to train_history.npz, plots saved to training_accuracy.png and training_loss.png')

# ---------- 主入口 ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=DATA_PATH, help='训练数据 HDF5')
    parser.add_argument('--output', default=MODEL_SAVE, help='模型保存路径')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='批次大小')
    args = parser.parse_args()

    DATA_PATH = args.data
    MODEL_SAVE = args.output
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    train()
