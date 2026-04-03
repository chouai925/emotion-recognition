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
'''
class EEGEmotionNetTopoLG3D(nn.Module):
    def __init__(self, time_len=4): # 新增：傳遞 time_len
        super().__init__()
        # 確保子模組也使用新的 time_len
        self.local_path = EEG3DCNN(time_len=time_len)
        self.global_path = EEG3DCNN(time_len=time_len)
        self._out_dim = 32 * 3 # [Left_local, Right_local, Global_all]

    def get_out_dim(self):
        return self._out_dim

    def forward(self, x):
        # x shape: (B, 1, 7, 5, 4)
        # 左右腦切割邏輯保持不變
        x_left = x[:, :, :, 0:2, :]   # (B, 1, 7, 2, 4)
        x_right = x[:, :, :, 3:5, :]  # (B, 1, 7, 2, 4)
        x_global = x                  # (B, 1, 7, 5, 4)

        feat_left = self.local_path(x_left)
        feat_right = self.local_path(x_right)
        feat_global = self.global_path(x_global)

        # 拼接特徵
        feat = torch.cat([feat_left, feat_right, feat_global], dim=1) 
        return feat
'''

class EEGEmotionNetTopoLG3D(nn.Module):
    def __init__(self, time_len=4):
        super().__init__()
        self.local_path = EEG3DCNN(time_len=time_len)
        self.global_path = EEG3DCNN(time_len=time_len)
        
        # 定義特徵維度
        self.embed_dim = 32
        
        # --- 新增：Self-Attention 層 ---
        # embed_dim 是每個特徵向量的長度 (32)
        # num_heads 必須能被 embed_dim 整除 (例如 4 或 8)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim, 
            num_heads=4, 
            batch_first=True
        )
        
        # 層標準化，有助於跨受試者的訓練穩定性
        self.layernorm = nn.LayerNorm(self.embed_dim)
        
        # 最終輸出的維度依然保持 32 * 3 = 96，以相容你後續的 Classifier
        self._out_dim = 32 * 3 

    def get_out_dim(self):
        return self._out_dim

    def forward(self, x):
        # 1. 提取三路特徵 (各為 B, 32)
        x_left = x[:, :, :, 0:2, :]   
        x_right = x[:, :, :, 3:5, :]  
        x_global = x                  

        feat_left = self.local_path(x_left)   # (B, 32)
        feat_right = self.local_path(x_right) # (B, 32)
        feat_global = self.global_path(x_global) # (B, 32)

        # 2. 將特徵包裝成 Token 序列格式: (B, 序列長度=3, 特徵維度=32)
        # feat_seq shape: (B, 3, 32)
        feat_seq = torch.stack([feat_left, feat_right, feat_global], dim=1)

        # 3. 執行 Self-Attention
        # attn_output shape: (B, 3, 32)
        # 它會計算：左腦應該看多少比例的右腦資訊？全腦資訊對左腦的幫助是多少？
        attn_output, attn_weights = self.self_attention(feat_seq, feat_seq, feat_seq)
        
        # 4. 殘差連接與層標準化 (Residual Connection + LayerNorm)
        # 這有助於保留原始特徵，同時注入注意力學到的交互資訊
        feat_seq = self.layernorm(feat_seq + attn_output)

        # 5. 展平回一維特徵向量 (B, 96)
        # 這樣就不需要改動你後續的 DANNEEGNet 或 Label Classifier
        feat = feat_seq.reshape(feat_seq.size(0), -1) 
        
        return feat
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