import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DEAPEEGDataset(Dataset):
    def __init__(self, X, y, s):
        # X: (B, 1, 7, 5, 4), y: (B,), s: (B,)
        self.X = torch.from_numpy(X).float()
        # 🚀 確保標籤是 float，這對 MSELoss 迴歸至關重要
        self.y = torch.from_numpy(y).float() 
        self.s = torch.from_numpy(s).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.s[idx]

def load_subject_independent_data(npz_path, test_subject_id=31):
    print(f"Loading data from {npz_path}...")
    npz = np.load(npz_path)
    X_all = npz["X"].astype(np.float32) 
    
    # 🚀 修正：標籤絕對不能轉成 int64，必須保留 float32 精度
    y_all = npz["y"].astype(np.float32) 
    s_all = npz["s"].astype(np.int64)

    # ==========================================
    # 關鍵：Subject-wise Z-Score Normalization
    # ==========================================
    '''
    print("Applying Subject-wise Z-Score...")
    for sub_id in range(32):
        mask = (s_all == sub_id)
        if np.any(mask):
            sub_X = X_all[mask] 
            
            # 針對頻段 (最後一維 4) 進行正規化
            # 計算該位受試者在所有時間、所有空間位置上的均值與標準差
            mean = np.mean(sub_X, axis=(0, 2, 3), keepdims=True)
            std = np.std(sub_X, axis=(0, 2, 3), keepdims=True)
            
            # 標準化：(x - mean) / (std + epsilon)
            X_all[mask] = (sub_X - mean) / (std + 1e-6)
            '''
    # ==========================================

    # 執行 LOSO 切分
    train_mask = (s_all != test_subject_id)
    test_mask = (s_all == test_subject_id)

    X_train, y_train, s_train = X_all[train_mask], y_all[train_mask], s_all[train_mask]
    X_test, y_test, s_test = X_all[test_mask], y_all[test_mask], s_all[test_mask]

    train_dataset = DEAPEEGDataset(X_train, y_train, s_train)
    test_dataset = DEAPEEGDataset(X_test, y_test, s_test)

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    return train_dataset, test_dataset

def get_dataloader(dataset, batch_size=128, shuffle=True, num_workers=0):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

def load_single_subject_data(npz_path, subject_id):
    """
    專門用於 Subject-Dependent (10-Fold) 的資料載入函數。
    只提取特定一位受試者的所有資料。
    """
    print(f"Loading data for Subject {subject_id+1:02d} from {npz_path}...")
    npz = np.load(npz_path)
    X_all = npz["X"].astype(np.float32) 
    y_all = npz["y"].astype(np.float32) 
    s_all = npz["s"].astype(np.int64)

    # 找出該受試者的所有資料索引
    mask = (s_all == subject_id)
    
    X_sub = X_all[mask]
    y_sub = y_all[mask]
    s_sub = s_all[mask]

    print(f"Subject {subject_id+1:02d} - Total samples: {len(X_sub)}")
    return X_sub, y_sub, s_sub