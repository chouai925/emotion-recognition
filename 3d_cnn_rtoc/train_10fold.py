import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import wandb
import time

# 匯入自定義模組
from dataloader import load_single_subject_data, DEAPEEGDataset
from model_Asymmetry import DANNEEGNet
from sklearn.metrics import roc_auc_score

# 🚀 專注於 Subject-Dependent 10-Fold (固定閾值版)
WANDB_PROJECT = "EEG-Valence-10Fold-Baseline"

# ===== 參數設定 =====
NPZ_PATH = r"D:\Deap_eeg\3d_cnn-dann_rtoc\deap_v_minmax.npz" 
OUT_DIR = r"D:\Deap_eeg\3d_cnn-dann_rtoc\Results_10Fold\valence_0421_baseline"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 40  # 受試者內資料較單純，通常收斂較快，先設 40
LR = 5e-4
NUM_SUBJECTS = 32
FIXED_THRESHOLD = 0.4375 

os.makedirs(OUT_DIR, exist_ok=True)

def train_subject_10fold(subject_id):
    sub_name = f"S{subject_id+1:02d}"
    sub_dir = os.path.join(OUT_DIR, sub_name)
    os.makedirs(sub_dir, exist_ok=True)
    
    log_file = open(os.path.join(sub_dir, f"train_log_{sub_name}.txt"), "a", encoding="utf-8")
    log_file.write(f"\n========== 開始 {sub_name} 10-Fold 實驗 | {time.ctime()} ==========\n")
    print(f"\n========== 正在進行 Subject-Dependent 10-Fold: 受試者 {sub_name} ==========")

    # 1. 載入該受試者的所有資料 (通常為 2400 筆)
    X_sub, y_sub, s_sub = load_single_subject_data(NPZ_PATH, subject_id)
    
    # 初始化 KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    fold_results = [] # 記錄這 10 次的最佳準確率

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_sub)):
        print(f"\n--- {sub_name} | Fold {fold+1}/10 ---")
        
        # 為了避免 W&B 開太多 run，我們將實驗分組
        wandb.init(project=WANDB_PROJECT, name=f"{sub_name}_Fold{fold+1}", group=f"10Fold_{sub_name}", reinit=True, settings=wandb.Settings(init_timeout=300))

        # 切割資料集
        train_dataset = DEAPEEGDataset(X_sub[train_idx], y_sub[train_idx], s_sub[train_idx])
        test_dataset = DEAPEEGDataset(X_sub[test_idx], y_sub[test_idx], s_sub[test_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        # 每次 Fold 都要「重新初始化模型」，確保沒有資料外洩
        model = DANNEEGNet(num_subjects=NUM_SUBJECTS, use_asymmetry=True).to(DEVICE) 
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion_reg = nn.MSELoss()
        criterion_cls = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        best_val_acc = 0.0

        for epoch in range(EPOCHS):
            model.train()
            total_mse, correct_label, total_samples = 0, 0, 0
            
            lambda_cls = 0.01  

            for inputs, labels, subjects in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                # 🚀 重點：Subject-Dependent 不需要領域對抗，alpha 設為 0
                _, reg_outputs, cls_outputs, _ = model(inputs, alpha=0.0)
                
                # 計算 Loss (拔除 loss_domain)
                loss_reg = criterion_reg(reg_outputs.squeeze(), labels.float())
                labels_cls = (labels >= FIXED_THRESHOLD).long()
                loss_cls = criterion_cls(cls_outputs, labels_cls)

                loss = loss_reg + (lambda_cls * loss_cls)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_mse += loss_reg.item()
                pred_class = cls_outputs.argmax(dim=1)
                correct_label += pred_class.eq(labels_cls).sum().item()
                total_samples += labels.size(0)

            # ==========================================
            # 驗證階段 (Validation) - 固定閾值盲測
            # ==========================================
            model.eval()
            all_preds, all_trues = [], []
            
            with torch.no_grad():
                for inputs, labels, _ in test_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    _, reg_outputs, _, _ = model(inputs, alpha=0.0)
                    
                    reg_probs = reg_outputs.squeeze()
                    all_preds.extend(reg_probs.cpu().numpy())
                    true_bin = (labels >= FIXED_THRESHOLD).float().cpu().numpy()
                    all_trues.extend(true_bin)
                    
            all_preds, all_trues = np.array(all_preds), np.array(all_trues)
            
            # 使用固定閾值算 Acc
            final_pred_class = (all_preds >= FIXED_THRESHOLD).astype(float)
            val_acc = np.mean(final_pred_class == all_trues)
            
            try:
                val_auc = roc_auc_score(all_trues, all_preds)
            except ValueError:
                val_auc = 0.5
                
            avg_train_mse = total_mse / len(train_loader)
            
            # 記錄最佳
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            wandb.log({
                "epoch": epoch + 1,
                "train_mse": avg_train_mse,
                "val_acc": val_acc,
                "val_auc": val_auc,
            })
            
            scheduler.step(val_acc)
            
        # 一個 Fold 結束
        print(f"✅ {sub_name} Fold {fold+1} 最佳準確率: {best_val_acc:.4f}")
        log_file.write(f"Fold {fold+1} Best Acc: {best_val_acc:.4f}\n")
        fold_results.append(best_val_acc)
        wandb.finish()

    # 10 個 Fold 跑完，計算平均
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    summary_str = f"\n🎯 {sub_name} 10-Fold 最終結果: {mean_acc:.4f} ± {std_acc:.4f}\n"
    print(summary_str)
    log_file.write(summary_str)
    log_file.close()
    
    return mean_acc

if __name__ == "__main__":
    # 建議：先只跑第一個受試者 (S01) 來驗證程式運作正常！
    # 確認跑得通且 Accuracy 合理後，再把 range 改成 range(32)
    subjects_to_test = range(1) # 先測 S01
    
    overall_results = []
    for sub_id in subjects_to_test:
        acc = train_subject_10fold(sub_id)
        overall_results.append(acc)