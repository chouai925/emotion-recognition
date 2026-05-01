import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import models

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
# 2. EEG 3DCNN 模組 (維持你原本的 LG3D)
# ---------------------------
class EEG3DCNN(nn.Module):
    def __init__(self, in_channels=1, time_len=4):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        return torch.flatten(x, 1)

class EEGEmotionNetTopoLG3D(nn.Module):
    def __init__(self, time_len=4, use_asymmetry=True):
        super().__init__()
        self.use_asymmetry = use_asymmetry
        self.local_path = EEG3DCNN(time_len=time_len)
        self.global_path = EEG3DCNN(time_len=time_len)
        self._out_dim = 32 * 4 if use_asymmetry else 32 * 3

    def forward(self, x):
        x_left = x[:, :, :, 0:2, :]
        x_right = x[:, :, :, 3:5, :]
        feat_left = self.local_path(x_left)
        feat_right = self.local_path(x_right)
        feat_global = self.global_path(x)
        if self.use_asymmetry:
            feat_diff = feat_left - feat_right
            return torch.cat([feat_left, feat_right, feat_diff, feat_global], dim=1)
        return torch.cat([feat_left, feat_right, feat_global], dim=1)

# ---------------------------
# 3. 臉部影像分支 (新增：ResNet + Temporal Attention)
# ---------------------------
class FaceBranch(nn.Module):
    def __init__(self):
        super(FaceBranch, self). __init__()
        # 使用預訓練的 ResNet18
        res18 = models.resnet18(pretrained=True)
        # 移除最後的 FC 層，輸出維度為 512
        self.backbone = nn.Sequential(*list(res18.children())[:-1])
        
        # 時間注意力機制：學習 5 幀影像的權重
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: (B, 5, 3, 64, 64)
        batch_size, seq_len, c, h, w = x.size()
        
        # 1. 展平 Batch 和 Sequence，一次餵給 CNN
        x = x.view(-1, c, h, w) # (B*5, 3, 64, 64)
        feat = self.backbone(x) # (B*5, 512, 1, 1)
        feat = feat.view(batch_size, seq_len, 512) # (B, 5, 512)
        
        # 2. 計算 Attention 權重
        attn_weights = self.attention(feat) # (B, 5, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 3. 加權總和：5 幀變 1 個特徵
        weighted_feat = torch.sum(feat * attn_weights, dim=1) # (B, 512)
        return weighted_feat

# ---------------------------
# 4. 終極多模態模型 (EEG + Face)
# ---------------------------
class MultimodalDANNEEGNet(nn.Module):
    def __init__(self, num_subjects=22, use_asymmetry=True):
        super(MultimodalDANNEEGNet, self).__init__()
        # EEG 分支
        self.eeg_module = EEGEmotionNetTopoLG3D(use_asymmetry=use_asymmetry)
        # Face 分支
        self.face_module = FaceBranch()
        
        # 融合後的維度：128 (EEG) + 512 (Face) = 640
        combined_dim = self.eeg_module._out_dim + 512
        
        # 情緒預測 (回歸)
        self.reg_classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

        # 分類輔助頭
        self.cls_classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

        # 領域對抗 (Subject Recognition)
        self.domain_classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_subjects)
        )

    def forward(self, eeg_x, face_x, alpha=1.0):
        # 1. 提取兩路特徵
        eeg_feat = self.eeg_module(eeg_x)
        face_feat = self.face_module(face_x)
        
        # 2. 特徵拼接 (Fusion)
        combined_feat = torch.cat([eeg_feat, face_feat], dim=1)
        
        # 3. 輸出結果
        reg_output = self.reg_classifier(combined_feat)
        cls_output = self.cls_classifier(combined_feat)
        
        # 4. 領域對抗
        rev_feat = ReverseLayerF.apply(combined_feat, alpha)
        domain_output = self.domain_classifier(rev_feat)
        
        return combined_feat, reg_output, cls_output, domain_output