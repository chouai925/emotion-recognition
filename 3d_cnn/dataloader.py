# dataloader.py
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader


class DEAPEEGDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, 1, 7, 5, 128), y: (N,)
        # N = 總樣本數（segments 數）；
        # 1 = EEG channel 數(因為 EEG 是單通道時間序列)，「每個 sample 有 1 個 EEG 特徵圖」；
        # 7,5 = EEG channels 數；(頭皮拓譜圖)
        # 128 = EEG segment length（1 秒 * 128Hz）
        # y[i] = 第 i 個 segment 的 label（0 或 1）
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_deap_eeg_loaders(
    npz_path: str,
    test_size: float = 0.3,      # 7:3 hold-out
    batch_size: int = 128,
    num_workers: int = 4,
    random_state: int = 42,
):
    npz = np.load(npz_path)
    X = npz["X"].astype(np.float32)   # (N, 1, 32, 128)
    y = npz["y"].astype(np.int64)     # (N,)

    print("Loaded X:", X.shape, "y:", y.shape)

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, #建立一個從 0 到 N-1 的 index list，用來記住第幾筆資料
        test_size=test_size,
        stratify=y, #保持 train/test set 的 label 分佈一致
        random_state=random_state,
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("Train:", X_train.shape, "Test:", X_test.shape)

    train_ds = DEAPEEGDataset(X_train, y_train)
    test_ds = DEAPEEGDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader
