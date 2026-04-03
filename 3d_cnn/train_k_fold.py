import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torchnet as tnt
from sklearn.model_selection import KFold
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, 
                             recall_score, matthews_corrcoef, roc_auc_score)
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

# 匯入你自定義的模組
from dataloader import DEAPEEGDataset
from model import EEGEmotionNetTopoLG3D

# ===== 參數與路徑設定 =====
DATA_PATH = r"C:\chou\Deap_eeg\3d_cnn\deap_eeg_segments_3dtopo.npz"
OUT_DIR = r"C:\chou\Deap_eeg\3d_cnn\Results_10Fold"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 30 
LR = 5e-4
K_FOLDS = 10

os.makedirs(OUT_DIR, exist_ok=True)

def calculate_metrics(y_true, y_pred, y_probs):
    """計算學術論文所需的完整指標"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    gmean = np.sqrt(rec * spec)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs[:, 1])
    
    return prec, rec, spec, f1, gmean, mcc, auc, cm

def plot_tsne(features, labels, fold):
    """ 繪製 t-SNE 降維圖 (隨機抽樣 1000 點以防卡死) """
    print(f"  [Visual] 正在計算 Fold {fold} 的 t-SNE...")
    if len(features) > 1000:
        idx = np.random.choice(len(features), 1000, replace=False)
        features, labels = features[idx], labels[idx]
    
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    embed = tsne.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embed[:,0], y=embed[:,1], hue=labels, palette='coolwarm', alpha=0.7)
    plt.title(f't-SNE Feature Space - Fold {fold}')
    plt.savefig(os.path.join(OUT_DIR, f'tsne_fold{fold}.png'))
    plt.close()

def plot_topomap(model, fold):
    """ 視覺化 3D 卷積核的空間權重 (7x5 拓譜熱圖) """
    # 提取 Global Path 第一層卷積權重 (out, in, H, W, T)
    weights = model.eeg_branch.global_path.conv1.weight.detach().cpu().numpy()
    # 對輸出通道與時間維度取絕對值平均，得到 7x5 空間強度
    spatial_map = np.mean(np.abs(weights), axis=(0, 1, 4))
    
    plt.figure(figsize=(6, 7))
    sns.heatmap(spatial_map, annot=True, cmap='YlOrRd', fmt=".4f")
    plt.title(f"Spatial Importance Map - Fold {fold}")
    plt.savefig(os.path.join(OUT_DIR, f'topomap_fold{fold}.png'))
    plt.close()

def main():
    print(f"Using Device: {DEVICE}")
    npz = np.load(DATA_PATH)
    full_dataset = DEAPEEGDataset(npz["X"], npz["y"])
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    all_fold_metrics = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset)):
        print(f"\n>>> Fold {fold+1}/{K_FOLDS} ...")
        train_loader = DataLoader(Subset(full_dataset, train_ids), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(Subset(full_dataset, test_ids), batch_size=BATCH_SIZE, shuffle=False)

        model = EEGEmotionNetTopoLG3D(n_classes=2).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        # [Torchnet] 初始化 Meters
        tr_loss_meter = tnt.meter.AverageValueMeter()
        tr_acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        
        history = {'train_loss': [], 'val_loss': []}
        best_acc = 0.0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            tr_loss_meter.reset()
            tr_acc_meter.reset()
            
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch}/{EPOCHS}", leave=False)
            for X, y in pbar:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                logits = model(X) # 僅執行一次 forward
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                tr_loss_meter.add(loss.item())
                tr_acc_meter.add(logits.detach(), y.detach())
                
                # 及時更新進度條指標
                pbar.set_postfix({
                    'Loss': f"{tr_loss_meter.value()[0]:.4f}",
                    'Acc': f"{tr_acc_meter.value()[0]:.2f}%"
                })

            # --- Validation ---
            model.eval()
            val_loss_meter = tnt.meter.AverageValueMeter()
            val_true, val_pred, val_probs, val_feats = [], [], [], []
            
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    logits, feat = model(X, return_feat=True)
                    val_loss_meter.add(criterion(logits, y).item())
                    
                    probs = torch.softmax(logits, dim=1)
                    val_true.extend(y.cpu().numpy())
                    val_pred.extend(logits.argmax(dim=1).cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())
                    val_feats.extend(feat.cpu().numpy())

            # 紀錄歷史供繪圖
            history['train_loss'].append(tr_loss_meter.value()[0])
            history['val_loss'].append(val_loss_meter.value()[0])
            
            val_acc = (np.array(val_true) == np.array(val_pred)).mean()
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(OUT_DIR, f"best_fold{fold+1}.pth"))
                best_metrics = calculate_metrics(np.array(val_true), np.array(val_pred), np.array(val_probs))
                best_feats = (np.array(val_feats), np.array(val_true))

        # --- Fold 結束後的產出 ---
        # 1. 繪製 Loss Curve
        plt.figure(figsize=(8, 5))
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.title(f'Loss Curve - Fold {fold+1}')
        plt.legend()
        plt.savefig(os.path.join(OUT_DIR, f'loss_curve_fold{fold+1}.png'))
        plt.close()

        # 2. 繪製 t-SNE (僅限第一個 Fold 以節省時間)
        if fold == 0:
            plot_tsne(best_feats[0], best_feats[1], fold+1)
            plot_topomap(model, fold+1)
            
        all_fold_metrics.append(best_metrics)
        print(f"  Fold {fold+1} 完畢。Best Val Acc: {best_acc:.4f}")

    # --- 最終統計與報表 ---
    res = np.array([m[:-1] for m in all_fold_metrics])
    means = np.mean(res, axis=0)
    stds = np.std(res, axis=0)
    names = ["Prec", "Recall", "Spec", "F1", "GMean", "MCC", "AUC"]
    
    # 定義報告內容
    report_content = "="*50 + "\n10-Fold 最終平均指標 (Valence):\n" + "-"*50 + "\n"
    for i, name in enumerate(names):
        line = f"{name:10s}: {means[i]:.4f} ± {stds[i]:.4f}\n"
        report_content += line
    report_content += "="*50
    
    # 1. 印在螢幕上
    print(report_content)
    
    # 2. 存成 txt 檔
    with open(os.path.join(OUT_DIR, "final_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n報告已儲存至: {os.path.join(OUT_DIR, 'final_report.txt')}")

if __name__ == "__main__":
    main()