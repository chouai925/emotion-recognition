import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import wandb
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg') # 強制後端，避免伺服器無 GUI 時報錯
import matplotlib.pyplot as plt
import seaborn as sns

from resnet9 import MultiBranchMicroNet

# ==========================================
# 1. 🌟 核心實驗設定 (論文參數調整區)
# ==========================================
TARGET_TASK = "Arousal"   # 可切換 "Valence"
FIXED_THRESHOLD = 0.4375  # 與 EEG 完美對齊的 1~9 分正規化中線

# 檔案路徑設定 (⚠️ 請確認你的電腦路徑是否正確)
LABEL_CSV_PATH = r"D:\Deap_eeg\participant_ratings.csv" 
NPY_DATA_DIR = r"D:\Deap_eeg\3d_cnn-dann_rtoc\roi_npy_data_64"

BATCH_SIZE = 64  
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_BRANCHES = 9  
NUM_CLASSES = 2 

PROJECT_NAME = f"MicroExpression_LOSO_{TARGET_TASK}"
LOG_FILE = f"train_log_{TARGET_TASK}.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================
# 2. 日誌系統設定
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)

# ==========================================
# 3. 資料載入器 (Dataset) - 支援 Multi-Task
# ==========================================
class MicroExpressionNpyDataset(Dataset):
    def __init__(self, npy_paths, labels, selected_indices=None):
        self.npy_paths = npy_paths
        self.labels = labels
        self.selected_indices = selected_indices if selected_indices is not None else list(range(9))

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        npy_data = np.load(self.npy_paths[idx])
        selected_data = npy_data[self.selected_indices]
        # 正規化圖片 (0~255 -> 0~1)
        tensor_data = torch.from_numpy(selected_data).permute(0, 3, 1, 2).float() / 255.0
        
        # 🚀 拆解出 Score (用於 MSE) 與 Label (用於 CrossEntropy)
        score_val, label_val = self.labels[idx]
        score_tensor = torch.tensor(score_val, dtype=torch.float32)
        label_tensor = torch.tensor(label_val, dtype=torch.long)
        
        return tensor_data, score_tensor, label_tensor

# ==========================================
# 4. 🌟 LOSO 資料讀取與 Min-Max 正規化
# ==========================================
def get_loso_data(val_subject, task):
    logging.info(f"🔍 準備 LOSO 資料 | 任務: {task} | 驗證受試者: {val_subject}...")
    df_labels = pd.read_csv(LABEL_CSV_PATH)
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    subject_folders = sorted([f for f in os.listdir(NPY_DATA_DIR) if f.startswith('s')])
    
    for sub in subject_folders:
        sub_dir = os.path.join(NPY_DATA_DIR, sub)
        if not os.path.isdir(sub_dir): continue
        sub_npy_paths = glob.glob(os.path.join(sub_dir, "*", "*.npy"))
        
        subject_id = int(sub.replace('s', ''))
        sub_data_list = []
        
        for npy_path in sub_npy_paths:
            # 解析 trial 編號
            trial_id = int(os.path.normpath(npy_path).split(os.sep)[-2].replace('trial', ''))
            row = df_labels[(df_labels['Participant_id'] == subject_id) & (df_labels['Trial'] == trial_id)]
            
            raw_score = row[task].values[0]

            # 🚀 執行 Min-Max 正規化 (將 1~9 分映射到 0~1 之間)
            norm_score = (raw_score - 1.0) / (9.0 - 1.0)
            
            # 🚀 使用 0.4375 決定高低情緒
            label = 1 if norm_score >= FIXED_THRESHOLD else 0
            
            sub_data_list.append((norm_score, label))

        if sub == val_subject:
            val_paths.extend(sub_npy_paths)
            val_labels.extend(sub_data_list)
        else:
            train_paths.extend(sub_npy_paths)
            train_labels.extend(sub_data_list)
            
    if len(val_paths) == 0:
        raise ValueError(f"找不到受試者 {val_subject} 的資料！")
        
    return train_paths, train_labels, val_paths, val_labels

# ==========================================
# 5. 繪製混淆矩陣
# ==========================================
def plot_and_log_confusion_matrix(y_true, y_pred, epoch, subject, task):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix ({task}) - {subject} (Epoch {epoch})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    img_path = f'temp_cm_{subject}.png'
    plt.savefig(img_path)
    plt.close()
    return wandb.Image(img_path)

# ==========================================
# 6. 主訓練迴圈 (AMP + Scheduler + Dual Loss)
# ==========================================
# ==========================================
# 6. 單一受試者訓練管線 (處理一個 Fold)
# ==========================================
def train_loso_fold(val_subject, task):
    logging.info(f"\n{'='*50}\n🚀 啟動 LOSO Fold: 驗證受試者 {val_subject} ({task})\n{'='*50}")
    
    train_paths, train_labels, val_paths, val_labels = get_loso_data(val_subject, task)
    
    # 確保每次 fold 都有清空記憶體，避免 OOM
    torch.cuda.empty_cache()

    train_loader = DataLoader(MicroExpressionNpyDataset(train_paths, train_labels), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(MicroExpressionNpyDataset(val_paths, val_labels), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = MultiBranchMicroNet(num_branches=NUM_BRANCHES, num_classes=NUM_CLASSES).to(device)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    scaler = GradScaler()

    best_acc = 0.0
    best_f1 = 0.0  # 紀錄最高 Acc 時對應的 F1
    
    # 🚀 Wandb 群組化：讓 22 個受試者收納在同一個 Group 裡面，方便雲端觀看
    wandb.init(
        project=PROJECT_NAME, 
        name=f"Fold_{val_subject}", 
        group=f"Full_LOSO_{task}", # 加上 Group
        reinit=True,               # 允許同一個腳本重複啟動 Wandb
        config={"lr": LEARNING_RATE, "batch_size": BATCH_SIZE, "task": task, "val_subject": val_subject}
    )

    for epoch in range(1, EPOCHS + 1):
        # ---------- 訓練階段 ----------
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS} [{val_subject}]", leave=False)
        for inputs, scores, targets in pbar:
            inputs, scores, targets = inputs.to(device), scores.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with autocast():
                reg_out, cls_out = model(inputs)
                loss_reg = criterion_reg(reg_out.squeeze(), scores)
                loss_cls = criterion_cls(cls_out, targets)
                loss = loss_reg + 0.1 * loss_cls 

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---------- 驗證階段 ----------
        model.eval()
        val_loss = 0.0
        all_preds_cls, all_targets = [], []
        
        with torch.no_grad():
            for inputs, scores, targets in val_loader:
                inputs, scores, targets = inputs.to(device), scores.to(device), targets.to(device)
                
                reg_out, _ = model(inputs)
                loss_reg = criterion_reg(reg_out.squeeze(), scores)
                val_loss += loss_reg.item()
                
                # 回歸分數 >= 0.4375 判定分類
                pred_scores = reg_out.squeeze()
                predicted = (pred_scores >= FIXED_THRESHOLD).long()
                
                if predicted.dim() == 0: predicted = predicted.unsqueeze(0)
                    
                all_preds_cls.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_acc = accuracy_score(all_targets, all_preds_cls) * 100
        avg_val_loss = val_loss / len(val_loader)
        
        # 計算 F1
        cm = confusion_matrix(all_targets, all_preds_cls)
        if cm.shape == (2, 2):
            f1 = f1_score(all_targets, all_preds_cls)
        else:
            f1 = 0.0 

        wandb.log({
            "Epoch": epoch,
            "Train Mix Loss": avg_train_loss,
            "Val MSE Loss": avg_val_loss, 
            "Val Acc (%)": val_acc,
            "Val F1 Score": f1
        })

        scheduler.step(val_acc)

        # 儲存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = f1
            # 自動根據受試者命名儲存權重
            torch.save(model.state_dict(), f"best_model_{val_subject}_{task}.pth")

    logging.info(f"🎉 {val_subject} 訓練結束！最高驗證準確率: {best_acc:.2f}% | F1: {best_f1:.4f}")
    wandb.finish()
    
    return best_acc, best_f1

# ==========================================
# 7. 🚀 全自動化總指揮 (Auto LOSO Loop)
# ==========================================
if __name__ == "__main__":
    # 自動產生 s01 到 s22 的清單
    all_subjects = [f"s{i:02d}" for i in range(1, 23)]
    
    results_acc = []
    results_f1 = []
    
    logging.info(f"🔥 開始執行全自動 LOSO ({TARGET_TASK})，共計 {len(all_subjects)} 名受試者...")
    
    for subject in all_subjects:
        try:
            # 依序跑每一個受試者，並把最佳成績存下來
            fold_acc, fold_f1 = train_loso_fold(subject, TARGET_TASK)
            results_acc.append(fold_acc)
            results_f1.append(fold_f1)
        except Exception as e:
            logging.error(f"❌ 處理 {subject} 時發生錯誤: {e}")
            continue # 如果某個受試者報錯（例如缺圖），就跳過繼續跑下一個
            
    # ---------------------------------------------------------
    # 🏆 論文數據結算
    # ---------------------------------------------------------
    mean_acc = np.mean(results_acc)
    std_acc = np.std(results_acc)
    mean_f1 = np.mean(results_f1)
    std_f1 = np.std(results_f1)
    
    logging.info("\n" + "="*50)
    logging.info(f"🏆 【{TARGET_TASK} 任務】22 人全自動 LOSO 最終報告 🏆")
    logging.info("="*50)
    logging.info(f"👉 平均準確率 (Mean Acc) : {mean_acc:.2f}% ± {std_acc:.2f}%")
    logging.info(f"👉 平均 F1分數 (Mean F1) : {mean_f1:.4f} ± {std_f1:.4f}")
    logging.info("="*50)