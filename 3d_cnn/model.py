# eeg_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG3DCNN(nn.Module):
    """
    3D CNN EEG branch
    Input:  (B, 1, 7, 5, 128)  = (batch, channel, H, W, T)
    Output: feature vector (B, F)
    """

    def __init__(self,
                 in_channels: int = 1,
                 time_len: int = 128):
        super().__init__()

        self.in_channels = in_channels
        self.time_len = time_len

        # Block 1
        self.conv1 = nn.Conv3d(
            in_channels, 16,
            kernel_size=(3, 3, 7),
            stride=(1, 1, 1),
            padding=(1, 1, 3),
        )
        self.bn1 = nn.BatchNorm3d(16)

        # Block 2
        self.conv2 = nn.Conv3d(
            16, 32,
            kernel_size=(3, 3, 7),
            stride=(1, 1, 1),
            padding=(1, 1, 3),
        )
        self.bn2 = nn.BatchNorm3d(32)

        # Block 3
        self.conv3 = nn.Conv3d(
            32, 64,
            kernel_size=(3, 3, 7),
            stride=(1, 1, 1),
            padding=(1, 1, 3),
        )
        self.bn3 = nn.BatchNorm3d(64)

        # 時間軸只在 T 維度做 pooling，空間不縮小（H=7, W=5 保留）
        self.pool_time = nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))

        # 最後對 (H, W, T) 做 Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

        self._out_dim = 64  # global_pool 後剩下的 channel 數

    def forward(self, x):
        # x: (B, 1, 7, 5, 128)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool_time(x)   # (B, 16, 7, 5, 64)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool_time(x)   # (B, 32, 7, 5, 32)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool_time(x)   # (B, 64, 7, 5, 16)

        # Global Average Pooling → (B, 64, 1, 1, 1)
        x = self.global_pool(x)

        # Flatten → (B, 64)
        x = x.view(x.size(0), -1)
        return x

    @property
    def out_dim(self):
        return self._out_dim


class EEGEmotionNet3D(nn.Module):
    """
    使用 3D CNN 的 valence 二分類
    Input:  (B, 1, 7, 5, 128)
    Output: logits (B, 2)
    """

    def __init__(self, n_classes: int = 2):
        super().__init__()

        self.eeg_branch = EEG3DCNN(in_channels=1, time_len=128)
        feat_dim = self.eeg_branch.out_dim

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        feat = self.eeg_branch(x)
        logits = self.classifier(feat)
        return logits

class EEGLocalGlobalCNN(nn.Module):
    """
    EEG branch:
    Input:  (B, 1, 32, 128)
    Output: feature vector (B, F)
    """

    def __init__(
        self,
        n_channels: int = 32,
        segment_len: int = 128,
        n_kernels: int = 20,
        kernel_time: int = 64,
        pool_time: int = 4,
    ):
        super().__init__()

        assert n_channels % 2 == 0
        self.n_channels = n_channels
        self.segment_len = segment_len

        # Local: 左右半球共用 conv
        self.local_conv = nn.Conv2d(
            in_channels=1,
            out_channels=n_kernels,
            kernel_size=(n_channels // 2, kernel_time),
            stride=(1, 1),
            padding=(0, kernel_time // 2),
        )

        # Global: 全腦
        self.global_conv = nn.Conv2d(
            in_channels=1,
            out_channels=n_kernels,
            kernel_size=(n_channels, kernel_time),
            stride=(1, 1),
            padding=(0, kernel_time // 2),
        )

        self.time_pool = nn.MaxPool2d(
            kernel_size=(1, pool_time),
            stride=(1, pool_time),
        )

        self._out_dim = None

    def forward(self, x):
        # x: (B, 1, 32, 128)
        B, _, C, L = x.shape
        assert C == self.n_channels

        left = x[:, :, : C // 2, :]    # (B, 1, 16, L)
        right = x[:, :, C // 2 :, :]   # (B, 1, 16, L)

        left_feat = self.time_pool(F.relu(self.local_conv(left)))    # (B, K, 1, L')
        right_feat = self.time_pool(F.relu(self.local_conv(right)))  # (B, K, 1, L')

        local_feat = torch.cat([left_feat, right_feat], dim=1)       # (B, 2K, 1, L')

        global_feat = self.time_pool(F.relu(self.global_conv(x)))    # (B, K, 1, L')

        feat = torch.cat([local_feat, global_feat], dim=1)           # (B, 3K, 1, L')
        feat = feat.view(B, -1)                                      # (B, F)

        if self._out_dim is None:
            self._out_dim = feat.size(1)

        return feat

    @property
    def out_dim(self):
        if self._out_dim is None:
            dummy = torch.zeros(1, 1, self.n_channels, self.segment_len)
            with torch.no_grad():
                _ = self.forward(dummy)
        return self._out_dim


class EEGEmotionNet(nn.Module):
    """
    簡單 valence 二分類（0/1）
    使用 CrossEntropyLoss -> 輸出 logits (B, 2)
    """

    def __init__(self,
                 n_channels: int = 32,
                 segment_len: int = 128,
                 n_classes: int = 2):
        super().__init__()

        self.eeg_branch = EEGLocalGlobalCNN(
            n_channels=n_channels,
            segment_len=segment_len,
            n_kernels=20,
            kernel_time=64,
            pool_time=4,
        )

        feat_dim = self.eeg_branch.out_dim

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        feat = self.eeg_branch(x)
        logits = self.classifier(feat)
        return logits

class _Path3D(nn.Module):
    """
    一條 3D CNN path：Conv3D ×3 + 時間軸 max-pool + global avg-pool
    可拿來當作：左 local / 右 local / global 的基本 block
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, base_channels,
            kernel_size=(3, 3, 7),
            stride=(1, 1, 1),
            padding=(1, 1, 3),
        )
        self.bn1 = nn.BatchNorm3d(base_channels)

        self.conv2 = nn.Conv3d(
            base_channels, base_channels * 2,
            kernel_size=(3, 3, 7),
            stride=(1, 1, 1),
            padding=(1, 1, 3),
        )
        self.bn2 = nn.BatchNorm3d(base_channels * 2)

        self.conv3 = nn.Conv3d(
            base_channels * 2, base_channels * 4,
            kernel_size=(3, 3, 7),
            stride=(1, 1, 1),
            padding=(1, 1, 3),
        )
        self.bn3 = nn.BatchNorm3d(base_channels * 4)

        # 只在時間軸做 pooling，空間 H×W (7×5 或 7×2) 不縮小
        self.pool_time = nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))

        # 最後對 H×W×T 做 global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

        self._out_dim = base_channels * 4

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, x):
        # x: (B, C, H, W, T)
        x = self.pool_time(F.relu(self.bn1(self.conv1(x))))
        x = self.pool_time(F.relu(self.bn2(self.conv2(x))))
        x = self.pool_time(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)   # (B, C_out, 1, 1, 1)
        x = x.view(x.size(0), -1) # (B, C_out)
        return x


class EEGTopoLocalGlobal3D(nn.Module):
    """
    EEG branch，結合 7×5 拓譜 + Local-Global：
      - Local-L : 左半 (col=0,1)
      - Local-R : 右半 (col=3,4)
      - Global  : 全腦 (col=0..4)
    Input:  (B, 1, 7, 5, 128)
    Output: feature vector (B, F)
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # 左 / 右 共用同一個 local path（權重共享）
        self.local_path = _Path3D(in_channels=in_channels, base_channels=16)

        # 全腦 global path 用另一組權重
        self.global_path = _Path3D(in_channels=in_channels, base_channels=16)

        self._out_dim = self.local_path.out_dim * 2 + self.global_path.out_dim

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, x):
        # x: (B, 1, 7, 5, 128)
        B, C, H, W, T = x.shape
        assert H == 7 and W == 5, f"Expect (7,5) topo, got {(H,W)}"

        # 左：最左兩欄 (col 0,1)
        x_left = x[:, :, :, 0:2, :]   # (B, 1, 7, 2, 128)

        # 右：最右兩欄 (col 3,4)
        x_right = x[:, :, :, 3:5, :]  # (B, 1, 7, 2, 128)

        # 中間那一欄 (col 2) 只給 global 用
        x_global = x                   # (B, 1, 7, 5, 128)

        feat_left = self.local_path(x_left)
        feat_right = self.local_path(x_right)
        feat_global = self.global_path(x_global)

        # 拼成 [Left_local, Right_local, Global_all]
        feat = torch.cat([feat_left, feat_right, feat_global], dim=1)   # (B, F)
        return feat


class EEGEmotionNetTopoLG3D(nn.Module):
    """
    最外層 classifier：
      EEGTopoLocalGlobal3D → FC → logits (valence 二分類)
    """

    def __init__(self, n_classes: int = 2):
        super().__init__()

        self.eeg_branch = EEGTopoLocalGlobal3D(in_channels=1)
        feat_dim = self.eeg_branch.out_dim

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, return_feat=False):
        feat = self.eeg_branch(x)
        logits = self.classifier(feat)
        if return_feat:
            return logits, feat
        return logits
