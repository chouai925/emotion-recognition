import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os

# 1. 定義影像預處理流程 (配合論文格式)
face_transforms = transforms.Compose([
    #transforms.Resize((64, 64)), # 根據 DeepVANet 建議尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MultimodalDEAPDataset(Dataset):
    def __init__(self, X_eeg, face_paths, y, s, transform=None):
        """
        X_eeg: EEG 特徵 (N, 1, 7, 5, 4)
        face_paths: 圖片路徑陣列 (N, 5)
        y: 標籤 (N,)
        s: 受試者 ID (N,)
        """
        # 轉換 EEG 為 Tensor
        self.X_eeg = torch.from_numpy(X_eeg).float()
        self.face_paths = face_paths
        self.y = torch.from_numpy(y).float()
        self.s = torch.from_numpy(s).long()
        self.transform = transform if transform else face_transforms

    def __len__(self):
        return self.X_eeg.shape[0]

    def __getitem__(self, idx):
        # 1. 取得 EEG 資料
        #eeg_data = torch.tensor(self.X_eeg[idx], dtype=torch.float32)
        eeg_data = self.X_eeg[idx].clone().detach().float()

        # 2. 取得影像路徑並載入圖片
        frame_paths = self.face_paths[idx]
        face_images = []
        for path in frame_paths:
            # 🚀 修復：過濾空字串與無效路徑
            if not path or path == "": 
                # 如果路徑是空的，給一張全黑的 224x224 假圖片代替
                # 這樣才不會破壞 5 幀的連續性
                img = Image.new('RGB', (224, 224), (0, 0, 0))
            else:
                try:
                    img = Image.open(path).convert('RGB')
                except Exception as e:
                    # 萬一檔案損壞讀不出來，也給全黑圖片，避免整個訓練當掉
                    img = Image.new('RGB', (224, 224), (0, 0, 0))
            
            face_images.append(img)
            
        # 3. 進行 Transform 並堆疊成 5 幀的 Sequence
        if self.transform:
            face_seq = torch.stack([self.transform(img) for img in face_images])
        else:
            face_seq = torch.stack([torch.tensor(np.array(img)) for img in face_images])

        # 確保記憶體連續
        face_seq = face_seq.contiguous()

        # 4. 取得 Label
        label = self.y[idx]
        subject_id = self.s[idx]

        return eeg_data, face_seq, label, subject_id

# 2. 多模態資料載入與對齊函數
def load_multimodal_data(eeg_npz_path, face_npz_path):
    """
    將 32 人的 EEG 與 22 人的影像進行對齊，只保留重疊的部分。
    """
    print(f"📦 正在載入並對齊多模態資料...")
    eeg_data = np.load(eeg_npz_path)
    face_data = np.load(face_npz_path)

    X_eeg_all = eeg_data["X"]  # (76800, 1, 7, 5, 4)
    y_all = eeg_data["y"]
    s_all = eeg_data["s"]
    face_paths_all = face_data["paths"] # (52800, 5)

    # 🚀 關鍵對齊：DEAP 影像只有前 22 位 (s00-s21)
    # 我們利用 mask 只抓取前 22 位的腦波資料
    mask = (s_all < 22)
    X_eeg_aligned = X_eeg_all[mask]
    y_aligned = y_all[mask]
    s_aligned = s_all[mask]

    print(f"✅ 對齊完成！")
    print(f"EEG 樣本數: {len(X_eeg_aligned)}")
    print(f"影像路徑筆數: {len(face_paths_all)}")
    
    return X_eeg_aligned, face_paths_all, y_aligned, s_aligned

# 3. 獲取 DataLoader 的便利函數
def get_multimodal_dataloader(eeg_npz, face_npz, batch_size=64, shuffle=True):
    X_eeg, face_paths, y, s = load_multimodal_data(eeg_npz, face_npz)
    dataset = MultimodalDEAPDataset(X_eeg, face_paths, y, s)
    
    # 注意：num_workers 建議設為 4 以上，因為讀圖比讀 EEG 慢很多
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=4, 
        pin_memory=True
    )

def load_multimodal_loso_data(eeg_npz_path, face_npz_path, test_subject_id):
    """
    針對特定受試者執行 LOSO 切分的多模態資料載入函數。
    """
    # 1. 先呼叫現有的對齊函數載入前 22 位受試者的所有資料
    X_all, face_paths_all, y_all, s_all = load_multimodal_data(eeg_npz_path, face_npz_path)
    
    # 2. 執行 LOSO 切分邏輯
    # 訓練集：受試者 ID 不等於測試受試者
    train_mask = (s_all != test_subject_id)
    # 測試集：受試者 ID 等於測試受試者
    test_mask = (s_all == test_subject_id)

    train_data = (X_all[train_mask], face_paths_all[train_mask], y_all[train_mask], s_all[train_mask])
    test_data = (X_all[test_mask], face_paths_all[test_mask], y_all[test_mask], s_all[test_mask])

    print(f"📊 LOSO 受試者 S{test_subject_id+1:02d} 切分完成:")
    print(f"   - 訓練樣本數: {len(train_data[0])}")
    print(f"   - 測試樣本數: {len(test_data[0])}")
    
    return train_data, test_data