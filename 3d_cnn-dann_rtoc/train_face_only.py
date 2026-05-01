import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
import matplotlib
matplotlib.use('Agg') # 強制後端，避免 GUI 報錯
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 匯入你原本寫好的資料載入與模型架構
from multimodal_dataLoader import MultimodalDEAPDataset, load_multimodal_loso_data
# 我們需要定義一個純影像的網路 (借用你 multimodal_model 裡的組件)
from multimodal_model import FaceBranch

# ===== 參數設定 =====
WANDB_PROJECT = "FaceOnly-Arousal-LOSO-224px"
EEG_NPZ_PATH = r"D:\Deap_eeg\3d_cnn-dann_rtoc\deap_a_minmax.npz"
FACE_INDEX_PATH = r"D:\Deap_eeg\3d_cnn-dann_rtoc\face_index_resize_224.npz"
OUT_DIR = r"D:\Deap_eeg\3d_cnn-dann_rtoc\Results_LOSO\arousal_face_only_224px"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 🚀 防禦性降級 1：因為 224x224 佔用很大，先從 64 開始，如果顯存夠再調回 128
BATCH_SIZE = 64  
EPOCHS = 30
LR = 1e-4
FIXED_THRESHOLD = 0.4375 
# 🚀 防禦性降級 2：先設為 2 測試 HDD/SSD 的讀取能力，若順利可改為 4
NUM_WORKERS = 2  

os.makedirs(OUT_DIR, exist_ok=True)

# ===== 1. 定義純影像模型 (FaceOnlyNet) =====
# ===== 1. 定義純影像模型 (FaceOnlyNet) =====
class FaceOnlyNet(nn.Module):
    def __init__(self):
        super(FaceOnlyNet, self).__init__()
        # 🚀 直接使用你寫好的 FaceBranch (裡面已包辦 ResNet18 與 Temporal Attention)
        self.face_branch = FaceBranch()
        
        # 回歸頭 (輸出 Valence/Arousal 分數)
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        
        # 分類頭 (輸出 High/Low)
        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, face_seq):
        # face_seq shape: (Batch, 5, 3, 224, 224)
        
        # 🚀 呼叫 FaceBranch，它會自動處理 5 幀展開與 Attention 融合
        face_feat = self.face_branch(face_seq) # 輸出 (B, 512)
        
        # 預測
        reg_out = self.regressor(face_feat)
        cls_out = self.classifier(face_feat)
        
        return face_feat, reg_out, cls_out

# ===== t-SNE 視覺化 =====
def visualize_tsne(model, source_loader, target_loader, device, save_path, epoch):
    model.eval()
    s_features, s_labels = [], []
    t_features, t_labels = [], []

    with torch.no_grad():
        for i, (_, f_x, y, _) in enumerate(source_loader): # 注意：我們只取 f_x (影像)
            if i > 10: break 
            with autocast(): # 加上 AMP 避免 OOM
                feat, _, _ = model(f_x.to(device))
            s_features.append(feat.cpu().numpy())
            s_labels.append((y >= FIXED_THRESHOLD).long().numpy())

        for _, f_x, y, _ in target_loader:
            with autocast():
                feat, _, _ = model(f_x.to(device))
            t_features.append(feat.cpu().numpy())
            t_labels.append((y >= FIXED_THRESHOLD).long().numpy())

    s_feat = np.concatenate(s_features, axis=0)
    s_label = np.concatenate(s_labels, axis=0)
    t_feat = np.concatenate(t_features, axis=0)
    t_label = np.concatenate(t_labels, axis=0)

    tsne = TSNE(n_components=2, random_state=42)
    all_feat = np.concatenate([s_feat, t_feat], axis=0)
    embedded = tsne.fit_transform(all_feat)
    
    s_emb = embedded[:len(s_feat)]
    t_emb = embedded[len(s_feat):]

    plt.figure(figsize=(10, 8))
    plt.scatter(s_emb[s_label==0, 0], s_emb[s_label==0, 1], c='skyblue', marker='o', alpha=0.3, label='Source Low')
    plt.scatter(s_emb[s_label==1, 0], s_emb[s_label==1, 1], c='salmon', marker='o', alpha=0.3, label='Source High')
    plt.scatter(t_emb[t_label==0, 0], t_emb[t_label==0, 1], c='blue', marker='*', s=100, label='Target Low')
    plt.scatter(t_emb[t_label==1, 0], t_emb[t_label==1, 1], c='red', marker='*', s=100, label='Target High')
    plt.legend()
    plt.title(f"FaceOnly t-SNE at Epoch {epoch}")
    plt.savefig(save_path)
    plt.close('all')

# ===== 訓練主程式 =====
def train_loso_face_only(test_sub_id):
    sub_name = f"S{test_sub_id+1:02d}"
    sub_dir = os.path.join(OUT_DIR, sub_name)
    os.makedirs(sub_dir, exist_ok=True)
    
    wandb.init(project=WANDB_PROJECT, name=sub_name, group="FaceOnly_224px", reinit=True)
    
    # 載入資料 (這裡雖然會載入 EEG，但我們訓練時只會拿 f_s)
    # 載入資料
    (x_s, f_s, y_s, id_s), (x_t, f_t, y_t, id_t) = load_multimodal_loso_data(
        EEG_NPZ_PATH, FACE_INDEX_PATH, test_sub_id
    )
    
    train_loader = DataLoader(
        MultimodalDEAPDataset(x_s, f_s, y_s, id_s), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=False,
        persistent_workers=True # 保持 Worker 活著，加速 224x224 讀取
    )
    
    val_loader = DataLoader(
        MultimodalDEAPDataset(x_t, f_t, y_t, id_t), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=False
    )

    model = FaceOnlyNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    scaler = GradScaler() # 🚀 必須使用 AMP 來節省 224x224 帶來的龐大 VRAM

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_mse = 0
        pbar = tqdm(train_loader, desc=f"{sub_name} Ep {epoch+1}")
        
        # 注意：這裡我們忽略了 eeg_x (第一個回傳值) 和 subject_id (最後一個)
        for _, face_x, y, _ in pbar:
            face_x, y = face_x.to(DEVICE), y.to(DEVICE)
            y_cls = (y >= FIXED_THRESHOLD).long()
            
            optimizer.zero_grad()
            
            with autocast():
                _, reg_out, cls_out = model(face_x)
                loss_reg = criterion_reg(reg_out.squeeze(), y.float())
                loss_cls = criterion_cls(cls_out, y_cls)
                # 影像任務因為預訓練權重強大，不用像腦波那樣把 Cls 壓得那麼低，可以給 0.1
                loss = loss_reg + 0.1 * loss_cls 

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_mse += loss_reg.item()
            pbar.set_postfix({"MSE": f"{loss_reg.item():.4f}"})

        # --- Validation ---
        model.eval()
        all_preds, all_trues = [], []
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            tsne_path = os.path.join(sub_dir, f"tsne_ep{epoch+1}.png")
            visualize_tsne(model, train_loader, val_loader, DEVICE, tsne_path, epoch+1)

        with torch.no_grad():
            for _, face_x, y, _ in val_loader:
                face_x = face_x.to(DEVICE)
                with autocast():
                    _, reg_out, _ = model(face_x)
                all_preds.extend(reg_out.squeeze().cpu().numpy())
                all_trues.extend((y >= FIXED_THRESHOLD).float().numpy())
                
        all_preds, all_trues = np.array(all_preds), np.array(all_trues)
        val_acc = np.mean((all_preds >= FIXED_THRESHOLD).astype(float) == all_trues)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(sub_dir, f"best_face_model_{sub_name}.pth"))

        wandb.log({"epoch": epoch+1, "val_acc": val_acc, "train_mse": total_mse/len(train_loader)})
        scheduler.step(val_acc)
        print(f"   => Val Acc: {val_acc:.4f} (Best: {best_acc:.4f})")

    wandb.finish()
    return best_acc

if __name__ == "__main__":
    results = []
    for sub_id in range(9,22): 
        acc = train_loso_face_only(sub_id)
        results.append(acc)
    print(f"\n🎯 Face-Only 平均準確率: {np.mean(results):.4f}")