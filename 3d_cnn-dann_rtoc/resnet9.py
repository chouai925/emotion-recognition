import torch
import torch.nn as nn

# ==========================================
# 1. 輔助函數：卷積區塊
# ==========================================
def conv_block(in_channels, out_channels, pool=False, dropout_rate=0.0):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool: 
        layers.append(nn.MaxPool2d(2))
    if dropout_rate > 0: 
        layers.append(nn.Dropout2d(dropout_rate))
    return nn.Sequential(*layers)

# ==========================================
# 2. 局部特徵提取器：輕量極速版 ResNet-9
# ==========================================
class ResNet9_Extractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True, dropout_rate=0.2)
        self.conv4 = conv_block(256, 512, pool=True, dropout_rate=0.2)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)

# ==========================================
# 3. 終極模型：多分支微表情主網路
# ==========================================
import torch
import torch.nn as nn

# ... (前面的 conv_block 和 ResNet9_Extractor 保持不變) ...

class MultiBranchMicroNet(nn.Module):
    def __init__(self, num_branches=9, num_classes=2): 
        super().__init__()
        self.num_branches = num_branches
        
        self.extractors = nn.ModuleList([
            ResNet9_Extractor(feature_dim=256) for _ in range(num_branches)
        ])
        
        merged_dim = num_branches * 256
        
        # 🚀 任務一：回歸頭 (Regressor) -> 輸出 0~1 的連續分數
        self.regressor = nn.Sequential(
            nn.Linear(merged_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 🚀 任務二：分類頭 (Classifier) -> 輸出 High/Low 分類機率
        self.classifier = nn.Sequential(
            nn.Linear(merged_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        region_features = []
        for i in range(self.num_branches):
            single_region_img = x[:, i, :, :, :] 
            feat = self.extractors[i](single_region_img) 
            region_features.append(feat)
            
        fused_features = torch.cat(region_features, dim=1)
        
        # 輸出兩個預測結果
        reg_out = self.regressor(fused_features)
        cls_out = self.classifier(fused_features)
        return reg_out, cls_out
        
# ====== 測試區塊 (直接執行此檔會驗證模型維度) ======
if __name__ == "__main__":
    print("🔧 正在測試模型架構維度...")
    dummy_input = torch.randn(16, 9, 3, 64, 64) # Batch=16, 9分支, 64x64
    model = MultiBranchMicroNet(num_branches=9, num_classes=2)
    out = model(dummy_input)
    print(f"✅ 模型測試成功！輸出維度: {out.shape} (預期為 [16, 2])")