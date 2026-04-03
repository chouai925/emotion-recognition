# eeg_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


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
