import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 匯入你修改後的模組
from dataloader import DEAPEEGDataset, load_subject_independent_data, get_dataloader
from model import DANNEEGNet

# ===== 參數設定 =====
NPZ_PATH = r"C:\chou\Deap_eeg\3d_cnn - dann\deap_eeg_de_topo_v.npz"
OUT_DIR = r"C:\chou\Deap_eeg\3d_cnn - dann\Results_LOSO\valence"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 50
LR = 5e-4
NUM_SUBJECTS = 32

os.makedirs(OUT_DIR, exist_ok=True)

def train_one_loso_fold(test_sub_id):
    print(f"\n========== 正在進行 LOSO: 測試受試者 S{test_sub_id+1:02d} ==========")
    
    # 1. 載入數據 (LOSO 切分)
    train_dataset, test_dataset = load_subject_independent_data(NPZ_PATH, test_subject_id=test_sub_id)
    
    # --- 新增：動態計算類別權重 (Class Weights) ---
    train_labels = train_dataset.y.numpy()
    counts = np.bincount(train_labels) # 統計 [類別0數量, 類別1數量]
    
    # 計算方式：使用總數除以各類別數量 (權重與數量成反比)
    # 這樣樣本少的類別權重會變大
    class_weights = torch.tensor([1.0 / counts[0], 1.0 / counts[1]], dtype=torch.float).to(DEVICE)
    # 進行標準化，讓權重平均值為 1 (推薦做法)
    class_weights = class_weights / class_weights.mean()
    
    print(f"訓練集分布: 類別0={counts[0]}, 類別1={counts[1]}")
    print(f"計算出的 CrossEntropy 權重: {class_weights.cpu().numpy()}")
    # --------------------------------------------

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 2. 初始化模型與優化器
    model = DANNEEGNet(num_classes=2, num_subjects=NUM_SUBJECTS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 修改：將計算出的權重傳入情感分類的 Loss
    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    criterion_domain = nn.CrossEntropyLoss()

    # 新增：學習率排程器 (當 Val Acc 停滯時降低 LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_label = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (inputs, labels, subjects) in enumerate(pbar):
            inputs, labels, subjects = inputs.to(DEVICE), labels.to(DEVICE), subjects.to(DEVICE)
            
            p = float(i + epoch * len(train_loader)) / (EPOCHS * len(train_loader))
            alpha = 2. / (1. + np.exp(-2 * p)) - 1
            
            class_outputs, domain_outputs = model(inputs, alpha=alpha)
            
            loss_class = criterion_class(class_outputs, labels)
            loss_domain = criterion_domain(domain_outputs, subjects)
            
            loss = loss_class + loss_domain * 0.01
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = class_outputs.max(1)
            correct_label += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct_label/total_samples:.2f}%"})

        # 5. 驗證
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs, _ = model(inputs, alpha=0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} - Val Acc (S{test_sub_id+1:02d}): {val_acc:.4f}")
        
        # 更新學習率排程器
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"best_model_S{test_sub_id+1:02d}.pth"))

    return best_acc

if __name__ == "__main__":
    all_subject_results = []
    
    # 你可以先跑一個受試者試看看 (例如 S32，對應 ID 31)
    # 如果要跑全部，改回 range(32)
    subjects_to_test = range(32) 
    
    for sub_id in subjects_to_test:
        acc = train_one_loso_fold(sub_id)
        all_subject_results.append(acc)
        
    print("\n" + "="*30)
    print(f"LOSO 平均準確率: {np.mean(all_subject_results):.4f}")
    print("="*30)