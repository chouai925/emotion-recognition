# train_eeg_5fold.py
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from dataloader import DEAPEEGDataset  # 確保從 dataloader 引入 Dataset 類別
from model import EEGEmotionNet
import matplotlib.pyplot as plt

# ===== 參數設定 =====
DATA_PATH = r"C:\chou\Deap_eeg\1DCNN\deap_eeg_segments_baseline.npz"
OUT_DIR = r"C:\chou\Deap_eeg\1DCNN\Results_5Fold"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
K_FOLDS = 5  # 改為論文使用的 5-Fold

os.makedirs(OUT_DIR, exist_ok=True)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += X.size(0)
    return total_loss / total_samples, total_correct / total_samples

def eval_model(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += X.size(0)
    return total_loss / total_samples, total_correct / total_samples

def main():
    # 1. 載入完整數據
    print(f"Loading data from {DATA_PATH}...")
    npz = np.load(DATA_PATH)
    full_dataset = DEAPEEGDataset(npz["X"], npz["y"])
    
    # 2. 設定 K-Fold
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset)):
        print(f"\n--- Starting Fold {fold + 1}/{K_FOLDS} ---")
        
        # 建立當前 Fold 的 DataLoader
        train_sub = Subset(full_dataset, train_ids)
        test_sub = Subset(full_dataset, test_ids)
        
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # 初始化模型、損失函數與優化器
        model = EEGEmotionNet(n_channels=32, segment_len=128, n_classes=2).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        best_fold_acc = 0.0
        
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            test_loss, test_acc = eval_model(model, test_loader, criterion)
            
            if test_acc > best_fold_acc:
                best_fold_acc = test_acc
                # 儲存該 Fold 的最佳模型
                save_path = os.path.join(OUT_DIR, f"best_model_fold{fold+1}.pth")
                torch.save(model.state_dict(), save_path)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f} (Best={best_fold_acc:.4f})")
        
        fold_results.append(best_fold_acc)
        print(f"Fold {fold+1} Completed. Best Test Acc: {best_fold_acc:.4f}")

    # 3. 輸出最終統計結果
    avg_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    print("\n==============================")
    print(f"5-Fold Cross-Validation Results:")
    for i, res in enumerate(fold_results):
        print(f"  Fold {i+1}: {res:.4f}")
    print(f"Average Accuracy: {avg_acc:.4f} (+/- {std_acc:.4f})")
    print("==============================")

if __name__ == "__main__":
    main()