import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.cuda.amp import autocast, GradScaler
# 匯入自定義組件
from multimodal_dataLoader import MultimodalDEAPDataset, load_multimodal_loso_data
from multimodal_model import MultimodalDANNEEGNet
from sklearn.metrics import roc_auc_score

# ===== 參數設定 =====
WANDB_PROJECT = "Valence-Multimodal-LOSO-Baseline"
EEG_PATH = r"D:\Deap_eeg\3d_cnn-dann_rtoc\deap_v_minmax.npz"
FACE_PATH = r"D:\Deap_eeg\3d_cnn-dann_rtoc\face_index_resize.npz"
OUT_DIR = r"D:\Deap_eeg\3d_cnn-dann_rtoc\Valence_Multimodal_Baseline"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128 
EPOCHS = 50
LR = 2e-4
FIXED_THRESHOLD = 0.4375
NUM_WORKERS = 4

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# 🚀 t-SNE 視覺化函數
# ---------------------------
def visualize_tsne(model, source_loader, target_loader, device, save_path, epoch):
    model.eval()
    s_features, s_labels = [], []
    t_features, t_labels = [], []

    with torch.no_grad():
        # 收集源域特徵 (取部分樣本避免繪圖過擠)
        for i, (e, f, y, _) in enumerate(source_loader):
            if i > 10: break 
            feat, _, _, _ = model(e.to(device), f.to(device), alpha=0.0)
            s_features.append(feat.cpu().numpy())
            s_labels.append((y >= FIXED_THRESHOLD).long().numpy())

        # 收集目標域特徵 (該位受試者)
        for e, f, y, _ in target_loader:
            feat, _, _, _ = model(e.to(device), f.to(device), alpha=0.0)
            t_features.append(feat.cpu().numpy())
            t_labels.append((y >= FIXED_THRESHOLD).long().numpy())

    s_feat = np.concatenate(s_features, axis=0)
    s_label = np.concatenate(s_labels, axis=0)
    t_feat = np.concatenate(t_features, axis=0)
    t_label = np.concatenate(t_labels, axis=0)

    # 執行 t-SNE 降維
    tsne = TSNE(n_components=2, random_state=42)
    all_feat = np.concatenate([s_feat, t_feat], axis=0)
    embedded = tsne.fit_transform(all_feat)
    
    s_emb = embedded[:len(s_feat)]
    t_emb = embedded[len(s_feat):]

    plt.figure(figsize=(10, 8))
    # 繪製源域 (圓點)
    plt.scatter(s_emb[s_label==0, 0], s_emb[s_label==0, 1], c='skyblue', marker='o', alpha=0.2, label='Source Low')
    plt.scatter(s_emb[s_label==1, 0], s_emb[s_label==1, 1], c='salmon', marker='o', alpha=0.2, label='Source High')
    
    # 繪製目標域 (星號)
    plt.scatter(t_emb[t_label==0, 0], t_emb[t_label==0, 1], c='blue', marker='*', s=100, edgecolors='white', label='Target Low')
    plt.scatter(t_emb[t_label==1, 0], t_emb[t_label==1, 1], c='red', marker='*', s=100, edgecolors='white', label='Target High')
    
    plt.legend()
    plt.title(f"t-SNE Visualization at Epoch {epoch}")
    plt.savefig(save_path)
    plt.close()

def train_loso_multimodal(test_sub_id):
    sub_name = f"S{test_sub_id+1:02d}"
    sub_dir = os.path.join(OUT_DIR, sub_name); os.makedirs(sub_dir, exist_ok=True)
    wandb.init(project=WANDB_PROJECT, name=sub_name, group="AMP_Opt_128Batch", reinit=True)
    
    (x_s, f_s, y_s, id_s), (x_t, f_t, y_t, id_t) = load_multimodal_loso_data(EEG_PATH, FACE_PATH, test_sub_id)
    
    # 🚀 優化後的 DataLoader 設定
    # 🚀 Windows 安全多執行緒模式
    train_loader = DataLoader(
        MultimodalDEAPDataset(x_s, f_s, y_s, id_s), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,          # 🚀 先嘗試 4，如果沒當機，下一次可以試試 8
        pin_memory=True,
        persistent_workers=True # 🚀 只保留這個，"不要" 加 prefetch_factor
    )
    val_loader = DataLoader(
        MultimodalDEAPDataset(x_t, f_t, y_t, id_t), 
        batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )

    model = MultimodalDANNEEGNet(num_subjects=32).to(DEVICE) 
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler() # 🚀 建立 AMP 縮放器
    
    criterion_reg = nn.MSELoss(); criterion_cls = nn.CrossEntropyLoss(); criterion_dom = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_mse, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"{sub_name} Ep {epoch+1}")
        
        for i, (e_in, f_in, labels, subjects) in enumerate(pbar):
            e_in, f_in, labels, subjects = e_in.to(DEVICE), f_in.to(DEVICE), labels.to(DEVICE), subjects.to(DEVICE)
            
            p = float(i + epoch * len(train_loader)) / (EPOCHS * len(train_loader))
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            optimizer.zero_grad()
            
            # 🚀 關鍵：使用 autocast 開啟半精度
            with autocast():
                _, reg_out, cls_out, dom_out = model(e_in, f_in, alpha=alpha)
                loss_reg = criterion_reg(reg_out.squeeze(), labels.float())
                loss_dom = criterion_dom(dom_out, subjects)
                y_cls = (labels >= FIXED_THRESHOLD).long()
                loss_cls = criterion_cls(cls_out, y_cls)
                loss = loss_reg + loss_dom + (0.01 * loss_cls)
            
            # 🚀 關鍵：使用 scaler 進行更新
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_mse += loss_reg.item()
            correct += cls_out.argmax(1).eq(y_cls).sum().item()
            total += labels.size(0)
            pbar.set_postfix({"MSE": f"{loss_reg.item():.4f}", "Acc": f"{100.*correct/total:.2f}%"})

        # 3. 驗證與 t-SNE
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for e, f, y, _ in val_loader:
                _, reg_out, _, _ = model(e.to(DEVICE), f.to(DEVICE), alpha=0)
                all_preds.extend(reg_out.squeeze().cpu().numpy())
                all_trues.extend((y >= FIXED_THRESHOLD).float().numpy())
        
        val_acc = np.mean((np.array(all_preds) >= FIXED_THRESHOLD) == np.array(all_trues))
        
        # 每 10 個 Epoch 繪製一次 t-SNE
        if (epoch + 1) % 10 == 0 or epoch == 0:
            tsne_path = os.path.join(sub_dir, f"tsne_ep{epoch+1}.png")
            visualize_tsne(model, train_loader, val_loader, DEVICE, tsne_path, epoch+1)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(sub_dir, f"best_multimodal_{sub_name}.pth"))

        wandb.log({"epoch": epoch+1, "val_acc": val_acc, "train_mse": total_mse/len(train_loader)})
        scheduler.step(val_acc)
        print(f"   => Val Acc: {val_acc:.4f} (Best: {best_acc:.4f})")

    wandb.finish()
    return best_acc

if __name__ == "__main__":
    for sub_id in range(22):
        train_loso_multimodal(sub_id)