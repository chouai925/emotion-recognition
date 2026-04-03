import torch
import torch.nn as nn
import torch.nn.functional as F

'''
更新的改進點說明：
CBAM 3D 整合：在 _Path3D 的每一層卷積後都加入了 CBAM。

Channel Attention：學習不同頻帶或過濾器之間的關聯，自動抑制無關的特徵通道。

Spatial Attention：針對你的 7x5 拓譜網格，強化對情緒敏感腦區（如前額葉）的關注度。

抑制過擬合 (Overfitting)：針對你提供的 loss_curve_fold1.png 中 Val Loss 劇烈震盪的問題，我將 classifier 的輸出層改為 256 維，並將 Dropout 提升至 0.6。

t-SNE 優化：透過注意力機制過濾雜訊，預計能讓 t-SNE 圖中的標籤 0 與 1 形成更明顯的聚類，減少重疊現象。
'''

# ==========================================
# 1. CBAM 3D 注意力模組
# ==========================================
class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1)).view(x.size(0), x.size(1), 1, 1, 1)
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1)).view(x.size(0), x.size(1), 1, 1, 1)
        out = self.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return x * out

class CBAM3D(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM3D, self).__init__()
        self.ca = ChannelAttention3D(in_planes, ratio)
        self.sa = SpatialAttention3D(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ==========================================
# 2. 整合 CBAM 的 3D CNN 分支
# ==========================================
class _Path3D(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=(3, 3, 7), padding=(1, 1, 3))
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.cbam1 = CBAM3D(base_channels) # 加入注意力機制

        # Block 2
        self.conv2 = nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 3, 7), padding=(1, 1, 3))
        self.bn2 = nn.BatchNorm3d(base_channels * 2)
        self.cbam2 = CBAM3D(base_channels * 2)

        # Block 3
        self.conv3 = nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(3, 3, 7), padding=(1, 1, 3))
        self.bn3 = nn.BatchNorm3d(base_channels * 4)
        self.cbam3 = CBAM3D(base_channels * 4)

        self.pool_time = nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self._out_dim = base_channels * 4

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, x):
        # x: (B, C, H, W, T)
        x = self.pool_time(self.cbam1(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool_time(self.cbam2(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool_time(self.cbam3(F.relu(self.bn3(self.conv3(x)))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x

# ==========================================
# 3. 最終整合模型 (Local-Global + CBAM)
# ==========================================
class EEGTopoLocalGlobal3D(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.local_path = _Path3D(in_channels=in_channels, base_channels=16)
        self.global_path = _Path3D(in_channels=in_channels, base_channels=16)
        self._out_dim = self.local_path.out_dim * 2 + self.global_path.out_dim

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, x):
        # x: (B, 1, 7, 5, 128)
        x_left = x[:, :, :, 0:2, :]   # 左側拓譜
        x_right = x[:, :, :, 3:5, :]  # 右側拓譜
        x_global = x                  # 全腦拓譜

        feat_left = self.local_path(x_left)
        feat_right = self.local_path(x_right)
        feat_global = self.global_path(x_global)

        return torch.cat([feat_left, feat_right, feat_global], dim=1)

class EEGEmotionNetTopoLG3D(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.eeg_branch = EEGTopoLocalGlobal3D(in_channels=1)
        feat_dim = self.eeg_branch.out_dim

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256), # 增加維度處理更複雜的注意力特徵
            nn.ReLU(),
            nn.Dropout(0.6),          # 調高 Dropout 以減少過擬合
            nn.Linear(256, n_classes),
        )

    def forward(self, x, return_feat=False):
        feat = self.eeg_branch(x)
        logits = self.classifier(feat)
        if return_feat:
            return logits, feat
        return logits