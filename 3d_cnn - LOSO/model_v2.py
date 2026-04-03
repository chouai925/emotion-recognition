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
'''class EEG3DCNN(nn.Module):
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
        return x'''
class EEG3DCNN(nn.Module):
    """
    3D CNN EEG branch
    針對 DE 特徵 (4個頻段) 進行優化
    """
    def __init__(self, in_channels=1, time_len=4):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(16)

        # Block 2
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)
        
        # --- 補上 Dropout3d ---
        self.dropout = nn.Dropout3d(p=0.2)

        # 全域平均池化
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)  # --- 加入 Dropout ---
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)  # --- 加入 Dropout ---
        x = self.avgpool(x)
        x = torch.flatten(x, 1) 
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
        self.embed_dim = 32
        
        # --- 借鑒論文概念：新增三個局部領域辨識器 (Local Discriminators) ---
        # 分別針對左腦、右腦、全腦路徑 [cite: 378, 385, 387, 388]
        self.local_d_left = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
        self.local_d_right = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
        self.local_d_global = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))

        self.self_attention = nn.MultiheadAttention(self.embed_dim, 4, batch_first=True)
        self.layernorm = nn.LayerNorm(self.embed_dim)
        self._out_dim = 32 * 3

    def get_out_dim(self):
        return self._out_dim

    def forward(self, x, alpha=1.0):
        # 提取特徵 [cite: 23, 248, 308]
        feat_l = self.local_path(x[:, :, :, 0:2, :])
        feat_r = self.local_path(x[:, :, :, 3:5, :])
        feat_g = self.global_path(x)

        # --- 局部對抗：在融合前先模糊受試者身份 [cite: 97, 152] ---
        # 利用 GRL (梯度翻轉層) 進行對抗訓練 [cite: 129, 401, 469]
        d_l = self.local_d_left(ReverseLayerF.apply(feat_l, alpha))
        d_r = self.local_d_right(ReverseLayerF.apply(feat_r, alpha))
        d_g = self.local_d_global(ReverseLayerF.apply(feat_g, alpha))

        # 注意力融合 [cite: 24, 94, 339, 342, 830]
        feat_seq = torch.stack([feat_l, feat_r, feat_g], dim=1)
        attn_out, _ = self.self_attention(feat_seq, feat_seq, feat_seq)
        feat_seq = self.layernorm(feat_seq + attn_out)
        
        final_feat = feat_seq.reshape(feat_seq.size(0), -1) 
        
        # 回傳最終特徵以及三個局部辨識器的預測結果
        return final_feat, (d_l, d_r, d_g)
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

    '''def forward(self, x, alpha=1.0):
        feature = self.feature_extractor(x)
        label_output = self.label_classifier(feature)

        # 梯度翻轉對抗
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature)

        return label_output, domain_output'''
    def forward(self, x, alpha=1.0):
        # 修正：因為特徵提取器現在回傳 (final_feat, 局部判別器結果)
        # 我們只取第 0 個元素 (final_feat) 餵給分類器
        feature, local_preds = self.feature_extractor(x, alpha)
        label_output = self.label_classifier(feature)

        # 梯度翻轉對抗
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature)

        return label_output, domain_output