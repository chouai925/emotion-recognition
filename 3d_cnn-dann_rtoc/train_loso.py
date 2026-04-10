import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb
import time  # 🚀 修正：補上遺漏的 time 模組

# 匯入你修改後的模組
from dataloader import load_subject_independent_data
#from model import DANNEEGNet
from model_Asymmetry import DANNEEGNet
from sklearn.metrics import roc_curve, roc_auc_score

WANDB_PROJECT = "EEG-Arousal-Regression-0410_alpha_dynamic"

# ===== 參數設定 =====
NPZ_PATH = r"D:\Deap_eeg\3d_cnn-dann_rtoc\deap_a_minmax.npz"
OUT_DIR = r"D:\Deap_eeg\3d_cnn-dann_rtoc\Results_LOSO\arousal_0410_alpha_dynamic"
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
    
    # 🚀 初始化 W&B
    wandb.init(project=WANDB_PROJECT, name=sub_name, group="LOSO_Regression", reinit=True)
    
    # 🚀 初始化本地 Log (append 模式)
    log_file = open(os.path.join(sub_dir, f"train_log_{sub_name}.txt"), "a", encoding="utf-8")
    log_file.write(f"\n--- 訓練啟動 {sub_name} | {time.ctime()} ---\n")

    print(f"\n========== 正在進行 LOSO: 測試受試者 {sub_name} ==========")
    
    # 1. 載入數據
    train_dataset, test_dataset = load_subject_independent_data(NPZ_PATH, test_subject_id=test_sub_id)
    wandb.log({f"Label_Dist/S{test_sub_id+1:02d}": wandb.Histogram(test_dataset.y.numpy())})
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 2. 初始化模型
    #model = DANNEEGNet(num_subjects=NUM_SUBJECTS).to(DEVICE)
    model = DANNEEGNet(num_subjects=NUM_SUBJECTS, use_asymmetry=False).to(DEVICE) #當你要跑 Arousal 時，關閉 Asymmetry 分支，傳入 use_asymmetry=False
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    start_epoch = 0
    best_acc = 0.0

    # 🚀 斷點續傳
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
        pbar = tqdm(train_loader, desc=f"{sub_name} Ep {epoch+1}/{EPOCHS}")
        
        for i, (inputs, labels, subjects) in enumerate(pbar):
            inputs, labels, subjects = inputs.to(DEVICE), labels.to(DEVICE), subjects.to(DEVICE)
            
            warmup_epochs = 10  # 前 10 個 Epoch 專注回歸
            start_lambda = 0.1  # 初始分類權重極低
            end_lambda = 2.0    # 最終分類權重 (讓分類主導)

            if epoch < warmup_epochs:
                lambda_cls = start_lambda
            else:
            # 從第 10 到第 50 個 Epoch，權重從 0.1 線性爬升到 2.0
                progress = (epoch - warmup_epochs) / (EPOCHS - warmup_epochs - 1)
                lambda_cls = start_lambda + (end_lambda - start_lambda) * progress

            # 稍微修改進度條標題，讓你一眼看出現在的 lambda 是多少
            pbar = tqdm(train_loader, desc=f"{sub_name} Ep {epoch+1}/{EPOCHS} [λ={lambda_cls:.2f}]")

            p = float(i + epoch * len(train_loader)) / (EPOCHS * len(train_loader))
            # 🚀 修正 1：改為標準的 DANN 公式 (-10 * p)，讓 alpha 平滑且完整地從 0 升到接近 1
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            reg_outputs, cls_outputs, domain_outputs = model(inputs, alpha=alpha)
            
            loss_reg = criterion_reg(reg_outputs.squeeze(), labels.float())

            # 將連續標籤轉為類別標籤 (0 或 1)
            labels_cls = (labels >= 0.5).long()
            loss_cls = criterion_cls(cls_outputs, labels_cls)

            loss_domain = criterion_domain(domain_outputs, subjects)
            
            # 🚀 修正 2：拔除 hardcode 的 0.1！
            # 因為 alpha 已經在 GRL (ReverseLayerF) 裡面作用了，這裡直接相加即可
            # 總損失：可以調整 lambda_cls 權重
            #lambda_cls = 1.0
            #loss = loss_reg + (lambda_cls * loss_cls) + loss_domain
            loss = loss_reg + (lambda_cls * loss_cls) + loss_domain
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 準確率 (迴歸轉分類)
            # 計算準確率 (使用分類頭的輸出)
            pred_class = cls_outputs.argmax(dim=1)
            correct_label += pred_class.eq(labels_cls).sum().item()
            #pred_class = (reg_outputs.squeeze() >= 0.5).float()
            #true_class = (labels >= 0.5).float()
            #correct_label += pred_class.eq(true_class).sum().item()
            total_samples += labels.size(0)
            total_mse += loss_reg.item()
            total_loss += loss.item()

            #pbar.set_postfix({"MSE": f"{loss_reg.item():.4f}", "Acc": f"{100.*correct_label/total_samples:.2f}%"})
            # 🚀 完整進度條：同時監控回歸 (MSE)、分類 (CE) 與準確率 (Acc)
            pbar.set_postfix({
                "MSE": f"{loss_reg.item():.4f}", 
                "CE": f"{loss_cls.item():.4f}", 
                "Acc": f"{100.*correct_label/total_samples:.2f}%"
            })

        # 3. 驗證階段
        # 3. 驗證階段
        model.eval()
        
        # 🚀 新增：用來收集整個 Test Set 的結果
        all_preds = []
        all_trues = []
        
        printed_batch = False 
        
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                # 🚀 接收三個輸出
                reg_outputs, cls_outputs, _ = model(inputs, alpha=0)
                
                # 🚀 先計算出分類機率 (提早算，這樣下面才能印)
                cls_probs = torch.softmax(cls_outputs, dim=1)[:, 1]
                
                # 🚀 印出監控 (把 scores 換成 cls_probs)
                if not printed_batch:
                    gt_print = labels[:10].cpu().numpy()
                    pred_print = cls_probs[:10].cpu().numpy() # ✅ 修正這裡
                    monitor_str = f"   👉 [預測監控] GT: {np.round(gt_print, 2)} | Pred(Prob): {np.round(pred_print, 2)}\n"
                    print(monitor_str.strip())
                    log_file.write(monitor_str)
                    printed_batch = True
                
                # 收集資料算 AUC
                all_preds.extend(cls_probs.cpu().numpy())
                
                true_bin = (labels >= 0.5).float().cpu().numpy()
                all_trues.extend(true_bin)
                
        # 🚀 驗證集全數跑完後，進行 ROC/AUC 分析
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        
        try:
            # 1. 計算 AUC
            val_auc = roc_auc_score(all_trues, all_preds)
            
            # 2. 計算 ROC 曲線的所有可能閾值
            fpr, tpr, thresholds = roc_curve(all_trues, all_preds)
            
            # 3. 使用 Youden's J statistic 尋找最佳閾值 (TPR - FPR 的最大值)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
        except ValueError:
            # 防呆機制：如果該受試者的測試集剛好只有單一類別 (例如全都是 High)
            val_auc = 0.5
            optimal_threshold = 0.5
            
        # 4. 用找出的「最佳閾值」來計算最終的 Accuracy
        final_pred_class = (all_preds >= optimal_threshold).astype(float)
        val_acc = np.mean(final_pred_class == all_trues)
        
        avg_train_mse = total_mse / len(train_loader)
        
        # 🚀 記錄 Log (加入 AUC 和 Optimal Threshold 資訊)
        log_str = f"Epoch {epoch+1:02d} | Train MSE: {avg_train_mse:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | Opt Thresh: {optimal_threshold:.3f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Lambda: {lambda_cls:.2f}\n"
        print(log_str.strip())
        log_file.write(log_str)
        log_file.flush()

        # 🚀 W&B 記錄更新
        wandb.log({
            "epoch": epoch + 1,
            "train_mse": avg_train_mse,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "optimal_threshold": optimal_threshold,
            "lr": optimizer.param_groups[0]['lr'],
            "lambda_cls": lambda_cls  # 🚀 把權重也傳上去
        })
        
        # Scheduler 可以選擇監控 AUC 還是 Acc，這裡我們維持監控 Acc
        scheduler.step(val_acc)
        
        # 🚀 儲存最佳模型 (存入子目錄)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(sub_dir, f"best_model_{sub_name}.pth"))
            print(f"🔥 新的最佳模型！Acc: {best_acc:.4f}")

        # 🚀 儲存斷點
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
    
    # 🚀 當你要跑完整實驗時，使用這行 (跑 32 人)
    subjects_to_test = range(32) 
    
    # 🚀 當你還在 Debug 時，用這行 (例如只跑剛才的 S28)
    #subjects_to_test = [27] 

    for sub_id in subjects_to_test:
        acc = train_one_loso_fold(sub_id)
        all_subject_results.append(acc)
        
    # === 計算平均值與標準差 ===
    mean_acc = np.mean(all_subject_results)
    std_acc = np.std(all_subject_results)

    print("\n" + "="*40)
    print(f"🎉 實驗結束！共測試了 {len(all_subject_results)} 位受試者")
    print(f"LOSO 平均準確率 (Mean): {mean_acc:.4f}")
    print(f"LOSO 準確率標準差 (Std):  {std_acc:.4f}")
    print(f"論文回報格式: {mean_acc:.4f} ± {std_acc:.4f}")
    print("="*40 + "\n")