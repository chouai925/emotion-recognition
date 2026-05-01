import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from dataloader import load_subject_independent_data
from model_Asymmetry import DANNEEGNet
from cmmd import cmmd 

import matplotlib
# 🚀 強制使用 'Agg' 後端，這個後端專門用來存檔，不會啟動任何 GUI 執行緒
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ===== 參數設定 =====
#ver.2 : 將cMMD的權重改成動態調整、信心度門檻降到0.5、t-SNE 圖
#v3 : 信心度門檻調到0.6
WANDB_PROJECT = "EEG-Valence-LOSO-cMMD-v3"
NPZ_PATH = r"D:\Deap_eeg\3d_cnn-dann_rtoc\deap_v_minmax.npz" 
OUT_DIR = r"D:\Deap_eeg\3d_cnn-dann_rtoc\Results_LOSO\valence_cmmd_v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 50
LR = 5e-4
FIXED_THRESHOLD = 0.4375 

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# 🚀 新增：t-SNE 視覺化函數
# ---------------------------
def visualize_tsne(model, source_loader, target_loader, device, save_path, epoch):
    model.eval()
    s_features, s_labels = [], []
    t_features, t_labels = [], []

    with torch.no_grad():
        # 收集源域特徵 (取前 10 個 Batch)
        for i, (x, y, _) in enumerate(source_loader):
            if i > 10: break 
            feat, _, _, _ = model(x.to(device), alpha=0.0)
            s_features.append(feat.cpu().numpy())
            s_labels.append((y >= FIXED_THRESHOLD).long().numpy())

        # 收集目標域特徵 (目前受試者)
        for x, y, _ in target_loader:
            feat, _, _, _ = model(x.to(device), alpha=0.0)
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
    plt.title(f"t-SNE at Epoch {epoch}")
    plt.savefig(save_path)
    plt.close()

def train_one_loso_fold_with_cmmd_v2(test_sub_id):
    sub_name = f"S{test_sub_id+1:02d}"
    sub_dir = os.path.join(OUT_DIR, sub_name)
    os.makedirs(sub_dir, exist_ok=True)
    
    wandb.init(project=WANDB_PROJECT, name=sub_name, group="Dynamic_cMMD", reinit=True)
    
    train_dataset, test_dataset = load_subject_independent_data(NPZ_PATH, test_subject_id=test_sub_id)
    
    # 🚀 注意：這裡將 num_workers 設為 4，以匹配你先前優化的硬體設定
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True, num_workers=0)
    target_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

    model = DANNEEGNet(num_subjects=32, use_asymmetry=True).to(DEVICE) 
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_mse, total_cmmd = 0, 0
        
        # 🚀 修改：將信心度門檻降到 0.5 (前 10 個 Epoch 不過濾)
        conf_thresh = 0.5 if epoch >= 10 else 0.0

        target_iter = iter(target_loader)
        pbar = tqdm(train_loader, desc=f"{sub_name} Ep {epoch+1}")
        
        for i, (x_s, y_s, s_s) in enumerate(pbar):
            try:
                x_t, _, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                x_t, _, _ = next(target_iter)
            
            x_s, y_s, s_s, x_t = x_s.to(DEVICE), y_s.to(DEVICE), s_s.to(DEVICE), x_t.to(DEVICE)

            # 🚀 修改：動態計算 cMMD 對齊權重 lambd
            p = float(i + epoch * len(train_loader)) / (EPOCHS * len(train_loader))
            lambd = (2. / (1. + np.exp(-10 * p)) - 1) 
            
            # DANN 的 alpha 也使用相同的動態排程
            feat_s, reg_s, cls_s, dom_s = model(x_s, alpha=lambd)
            feat_t, _, cls_t, _ = model(x_t, alpha=0.0)

            loss_reg = criterion_reg(reg_s.squeeze(), y_s.float())
            loss_dom = criterion_domain(dom_s, s_s)
            y_s_cls = (y_s >= FIXED_THRESHOLD).long()
            loss_cls = criterion_cls(cls_s, y_s_cls)

            # cMMD 計算
            t_probs = torch.softmax(cls_t, dim=1)
            conf_t, y_t_pseudo = torch.max(t_probs, dim=1)
            mask = conf_t > conf_thresh
            
            if mask.sum() > 1:
                loss_cmmd = cmmd(feat_s, feat_t[mask], y_s_cls, y_t_pseudo[mask], num_classes=2)
            else:
                loss_cmmd = torch.tensor(0.0).to(DEVICE)

            # 🚀 修改：將動態權重 lambd 乘上 loss_cmmd
            # 這裡給 cMMD 一個基礎係數 0.1，再乘上 lambd 確保平滑增長
            loss = loss_reg + loss_dom + (0.01 * loss_cls) + (0.1 * lambd * loss_cmmd)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_mse += loss_reg.item()
            total_cmmd += loss_cmmd.item()
            pbar.set_postfix({"MSE": f"{loss_reg.item():.4f}", "cMMD": f"{loss_cmmd.item():.4f}"})

        # --- 驗證與繪圖階段 ---
        model.eval()
        
        # 🚀 新增：每 10 個 Epoch 畫一次 t-SNE
        if (epoch + 1) % 10 == 0 or epoch == 0:
            tsne_path = os.path.join(sub_dir, f"tsne_ep{epoch+1}.png")
            visualize_tsne(model, train_loader, val_loader, DEVICE, tsne_path, epoch+1)

        all_preds, all_trues = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(DEVICE)
                _, reg_out, _, _ = model(x, alpha=0)
                all_preds.extend(reg_out.squeeze().cpu().numpy())
                all_trues.extend((y >= FIXED_THRESHOLD).float().numpy())
        
        all_preds, all_trues = np.array(all_preds), np.array(all_trues)
        val_acc = np.mean((all_preds >= FIXED_THRESHOLD).astype(float) == all_trues)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(sub_dir, f"best_model_cmmd_{sub_name}.pth"))

        wandb.log({"epoch": epoch+1, "val_acc": val_acc, "train_mse": total_mse/len(train_loader), "cmmd_loss": total_cmmd/len(train_loader)})
        scheduler.step(val_acc)
        print(f"   => Val Acc: {val_acc:.4f} (Best: {best_acc:.4f})")

    wandb.finish()
    return best_acc

if __name__ == "__main__":
    results = []
    for sub_id in range(0,32):
        acc = train_one_loso_fold_with_cmmd_v2(sub_id)
        results.append(acc)
    print(f"\n🎯 cMMD 動態權重實驗最終平均準確率: {np.mean(results):.4f} ± {np.std(results):.4f}")