# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ---------------------------
# 1. 梯度翻轉層 (保持不變)
# ---------------------------
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# ---------------------------
# 2. 修正後的 3DCNN 基礎模組
# ---------------------------
class EEG3DCNN(nn.Module):
    """
    3D CNN EEG branch
    針對 DE 特徵 (4個頻段) 進行優化
    """
    def __init__(self, in_channels=1, time_len=4): # 修改：time_len 預設改為 4
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv3d(
            in_channels, 16,
            kernel_size=(3, 3, 3), # 修改：kernel 從 7 縮小到 3，適應深度為 4 的頻段
            stride=(1, 1, 1),
            padding=(1, 1, 1),    # 保持空間維度
        )
        self.bn1 = nn.BatchNorm3d(16)

        # Block 2
        self.conv2 = nn.Conv3d(
            16, 32,
            kernel_size=(3, 3, 3), # 修改：kernel 縮小
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        self.bn2 = nn.BatchNorm3d(32)

        # 全域平均池化，將 (7, 5, 4) 壓縮為 (1, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        # x shape: (B, 1, H, W, 4)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 輸出維度: (B, 32)
        return x

# ---------------------------
# 3. 修正後的 LG3D 融合結構
# ---------------------------
class EEGEmotionNetTopoLG3D(nn.Module):
    def __init__(self, time_len=4):
        super().__init__()
        self.local_path = EEG3DCNN(time_len=time_len)
        self.global_path = EEG3DCNN(time_len=time_len)
        self.embed_dim = 32
        
        # 1. 局部領域辨識器 (Local DANN，消除受試者差異)
        self.local_d_left = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
        self.local_d_right = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
        self.local_d_global = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))

        # 🚀 2. 新增：局部情緒分類器 (給特徵穿防彈衣，保留情緒)
        self.local_c_left = nn.Linear(32, 2)
        self.local_c_right = nn.Linear(32, 2)
        self.local_c_global = nn.Linear(32, 2)

        self.self_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)
        self.layernorm = nn.LayerNorm(self.embed_dim)
        self._out_dim = 32 * 3 

    def get_out_dim(self):
        return self._out_dim

    def forward(self, x, alpha=1.0): 
        x_left = x[:, :, :, 0:2, :]   
        x_right = x[:, :, :, 3:5, :]  
        x_global = x                  

        feat_left = self.local_path(x_left)   
        feat_right = self.local_path(x_right) 
        feat_global = self.global_path(x_global) 

        # (A) 局部領域對抗 (經過 ReverseLayerF 洗掉受試者特徵)
        d_l = self.local_d_left(ReverseLayerF.apply(feat_left, alpha))
        d_r = self.local_d_right(ReverseLayerF.apply(feat_right, alpha))
        d_g = self.local_d_global(ReverseLayerF.apply(feat_global, alpha))

        # 🚀 (B) 局部情緒預測 (直接預測，不翻轉梯度)
        c_l = self.local_c_left(feat_left)
        c_r = self.local_c_right(feat_right)
        c_g = self.local_c_global(feat_global)

        # (C) 注意力融合 
        feat_seq = torch.stack([feat_left, feat_right, feat_global], dim=1)
        attn_output, attn_weights = self.self_attention(feat_seq, feat_seq, feat_seq)
        feat_seq = self.layernorm(feat_seq + attn_output)
        final_feat = feat_seq.reshape(feat_seq.size(0), -1) 
        
        # ⚠️ 回傳：最終特徵、領域預測(三個)、情緒預測(三個)
        return final_feat, (d_l, d_r, d_g), (c_l, c_r, c_g)
# ---------------------------
# 4. 對抗性封裝 (DANN-EEGNet)
# ---------------------------
class DANNEEGNet(nn.Module):
    def __init__(self, num_classes=2, num_subjects=32, time_len=4): # 新增：傳遞 time_len
        super(DANNEEGNet, self).__init__()
        # 使用修正後的特徵提取器
        self.feature_extractor = EEGEmotionNetTopoLG3D(time_len=time_len)
        feature_dim = self.feature_extractor.get_out_dim()

        self.label_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_subjects)
        )

    def forward(self, x, alpha=1.0):
        feature = self.feature_extractor(x)
        label_output = self.label_classifier(feature)

        # 梯度翻轉對抗
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature)

        return label_output, domain_output