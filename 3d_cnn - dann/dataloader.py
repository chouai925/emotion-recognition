import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DEAPEEGDataset(Dataset):
    def __init__(self, X, y, s):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.s = torch.from_numpy(s).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.s[idx]

def load_subject_independent_data(npz_path, test_subject_id=31):
    print(f"Loading data from {npz_path}...")
    npz = np.load(npz_path)
    X_all = npz["X"].astype(np.float32) # (76800, 1, 7, 5, 4)
    y_all = npz["y"].astype(np.int64)
    s_all = npz["s"].astype(np.int64)

    # ==========================================
    # 關鍵：Subject-wise Z-Score Normalization
    # ==========================================
    print("Applying Subject-wise Z-Score...")
    for sub_id in range(32):
        mask = (s_all == sub_id)
        if np.any(mask):
            sub_X = X_all[mask] # 取得該位受試者的所有資料
            
            # 對每個頻段 (Dimension 4) 分別計算均值與標準差
            # 我們對每個 Channel (7x5) 的分佈進行統一縮放
            mean = np.mean(sub_X, axis=(0, 2, 3), keepdims=True)
            std = np.std(sub_X, axis=(0, 2, 3), keepdims=True)
            
            # 執行標準化：(x - mean) / std
            X_all[mask] = (sub_X - mean) / (std + 1e-6)
    # ==========================================

    train_mask = (s_all != test_subject_id)
    test_mask = (s_all == test_subject_id)

    X_train, y_train, s_train = X_all[train_mask], y_all[train_mask], s_all[train_mask]
    X_test, y_test, s_test = X_all[test_mask], y_all[test_mask], s_all[test_mask]

    train_dataset = DEAPEEGDataset(X_train, y_train, s_train)
    test_dataset = DEAPEEGDataset(X_test, y_test, s_test)

    return train_dataset, test_dataset

# 為了相容性，保留一個快速建立 Loader 的函式
def get_dataloader(dataset, batch_size=128, shuffle=True, num_workers=0):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )