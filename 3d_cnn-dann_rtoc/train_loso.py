import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import time

from dataloader import load_subject_independent_data
from model_Asymmetry import DANNEEGNet
from sklearn.metrics import roc_curve, roc_auc_score

# 🚀 專注於 Valence 與微量分類輔助的實驗
WANDB_PROJECT = "EEG-Arousal-Ablation-loss_function_alpha0.0.1"

# ===== 參數設定 =====
NPZ_PATH = r"D:\Deap_eeg\3d_cnn-dann_rtoc\deap_a_minmax.npz" # 確保這裡是 valence 的檔案
OUT_DIR = r"D:\Deap_eeg\3d_cnn-dann_rtoc\Results_LOSO\arousal_0416_loss_function_alpha0.01"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 50
LR = 5e-4
NUM_SUBJECTS = 32

os.makedirs(OUT_DIR, exist_ok=True)

def save_checkpoint(state, sub_dir, filename="checkpoint.pth"):
    torch.save(state, os.path.join(sub_dir, filename))

def train_one_loso_fold(test_sub_id):
    sub_name = f"S{test_sub_id+1:02d}"
    sub_dir = os.path.join(OUT_DIR, sub_name)
    os.makedirs(sub_dir, exist_ok=True)
    
    wandb.init(project=WANDB_PROJECT, name=sub_name, group="LOSO_Ablation", reinit=True)
    
    log_file = open(os.path.join(sub_dir, f"train_log_{sub_name}.txt"), "a", encoding="utf-8")
    log_file.write(f"\n--- 訓練啟動 {sub_name} | {time.ctime()} ---\n")

    print(f"\n========== 正在進行 LOSO: 測試受試者 {sub_name} ==========")
    
    # 1. 載入數據
    train_dataset, test_dataset = load_subject_independent_data(NPZ_PATH, test_subject_id=test_sub_id)
    wandb.log({f"Label_Dist/S{test_sub_id+1:02d}": wandb.Histogram(test_dataset.y.numpy())})
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 2. 初始化模型 (Valence 任務，開啟左右腦不對稱分支)
    model = DANNEEGNet(num_subjects=NUM_SUBJECTS, use_asymmetry=True).to(DEVICE) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    start_epoch = 0
    best_acc = 0.0

    # 斷點續傳
    ckpt_path = os.path.join(sub_dir, "checkpoint.pth")
    if os.path.exists(ckpt_path):
        print(f"發現斷點檔案，正在從 {ckpt_path} 恢復...")
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss, total_mse, correct_label, total_samples = 0, 0, 0, 0
        
        # 🚀 核心修改：固定極小的分類輔助權重，不讓它干擾 MSE
        lambda_cls = 0.01  

        pbar = tqdm(train_loader, desc=f"{sub_name} Ep {epoch+1}/{EPOCHS} [λc={lambda_cls}]")
        
        for i, (inputs, labels, subjects) in enumerate(pbar):
            inputs, labels, subjects = inputs.to(DEVICE), labels.to(DEVICE), subjects.to(DEVICE)
            
            # 計算 DANN 動態權重 alpha
            p = float(i + epoch * len(train_loader)) / (EPOCHS * len(train_loader))
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            # 前向傳播
            _, reg_outputs, cls_outputs, domain_outputs = model(inputs, alpha=alpha)
            
            # 計算個別 Loss
            loss_reg = criterion_reg(reg_outputs.squeeze(), labels.float())
            loss_domain = criterion_domain(domain_outputs, subjects)
            
            labels_cls = (labels >= 0.5).long()
            loss_cls = criterion_cls(cls_outputs, labels_cls)

            # 🚀 終極 Loss 融合：MSE (定海神針 1.0) + CE_Domain (動態 Alpha) + CE_Task (微量 0.01)
            loss = loss_reg + loss_domain + (lambda_cls * loss_cls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 計算分類準確率 (僅供訓練期觀察)
            pred_class = cls_outputs.argmax(dim=1)
            correct_label += pred_class.eq(labels_cls).sum().item()
            total_samples += labels.size(0)
            
            total_mse += loss_reg.item()
            total_loss += loss.item()

            pbar.set_postfix({
                "MSE": f"{loss_reg.item():.4f}", 
                "CE": f"{loss_cls.item():.4f}", 
                "Acc": f"{100.*correct_label/total_samples:.2f}%"
            })

        # ==========================================
        # 3. 驗證階段 (Validation)
        # ==========================================
        model.eval()
        all_preds = []
        all_trues = []
        printed_batch = False 
        
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                # 測試時關閉領域對齊 (alpha=0)
                _, reg_outputs, cls_outputs, _ = model(inputs, alpha=0)
                
                # 🚀 退回：維持使用「分類機率 (cls_outputs)」來計算 Opt Threshold
                cls_probs = torch.softmax(cls_outputs, dim=1)[:, 1]
                
                if not printed_batch:
                    gt_print = labels[:10].cpu().numpy()
                    pred_print = cls_probs[:10].cpu().numpy() 
                    monitor_str = f"   👉 [分類監控] GT: {np.round(gt_print, 2)} | Pred(Prob): {np.round(pred_print, 2)}\n"
                    print(monitor_str.strip())
                    log_file.write(monitor_str)
                    printed_batch = True
                
                # 收集分類機率預測值算 AUC 與找尋 Threshold
                all_preds.extend(cls_probs.cpu().numpy())
                true_bin = (labels >= 0.5).float().cpu().numpy()
                all_trues.extend(true_bin)
                
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        
        try:
            val_auc = roc_auc_score(all_trues, all_preds)
            fpr, tpr, thresholds = roc_curve(all_trues, all_preds)
            
            # 使用 Youden's J statistic 尋找最佳閾值
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
        except ValueError:
            val_auc = 0.5
            optimal_threshold = 0.5
            
        # 🚀 用找出的「最佳分類閾值」來切分 High/Low
        final_pred_class = (all_preds >= optimal_threshold).astype(float)
        val_acc = np.mean(final_pred_class == all_trues)
        
        avg_train_mse = total_mse / len(train_loader)
        
        log_str = f"Epoch {epoch+1:02d} | Train MSE: {avg_train_mse:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | Opt Thresh: {optimal_threshold:.3f}\n"
        print(log_str.strip())
        log_file.write(log_str)
        log_file.flush()

        wandb.log({
            "epoch": epoch + 1,
            "train_mse": avg_train_mse,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "optimal_threshold": optimal_threshold,
            "lr": optimizer.param_groups[0]['lr'],
        })
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(sub_dir, f"best_model_{sub_name}.pth"))
            print(f"🔥 新的最佳模型！Acc: {best_acc:.4f}")

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, sub_dir)

    log_file.close()
    wandb.finish()
    return best_acc

if __name__ == "__main__":
    all_subject_results = []
    subjects_to_test = range(32) 

    for sub_id in subjects_to_test:
        acc = train_one_loso_fold(sub_id)
        all_subject_results.append(acc)
        
    mean_acc = np.mean(all_subject_results)
    std_acc = np.std(all_subject_results)

    print("\n" + "="*40)
    print(f"🎉 實驗結束！共測試了 {len(all_subject_results)} 位受試者")
    print(f"LOSO 平均準確率 (Mean): {mean_acc:.4f}")
    print(f"LOSO 準確率標準差 (Std):  {std_acc:.4f}")
    print(f"論文回報格式: {mean_acc:.4f} ± {std_acc:.4f}")
    print("="*40 + "\n")