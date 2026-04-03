import os
import pickle
import numpy as np
from scipy.signal import butter, lfilter

# ===== 路徑與參數設定 =====
DATA_DIR = r"C:\chou\data_preprocessed_python"   # 原始 .dat 檔案路徑
SAVE_PATH = r"C:\chou\Deap_eeg\3d_cnn - dann\deap_eeg_de_topo_a.npz"

FS = 128                # DEAP 採樣率
TRIAL_SEC = 60          # 實驗長度 60 秒
N_CHANNELS = 32         # 使用前 32 通道

# 頻段定義 (Hz)
BANDS = {
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta': (14, 31),
    'gamma': (31, 45)
}

# ===== 1. 拓撲佈局設定 (與你原架構一致) =====
DEAP_CHANNELS = [
    "FP1","AF3","F3","F7","FC5","FC1","C3","T7","CP5","CP1","P3","P7","PO3",
    "O1","Oz","Pz","FP2","AF4","F4","F8","FC6","FC2","C4","T8","CP6","CP2",
    "P4","P8","PO4","O2","Fz","Cz"
]
NAME_TO_IDX = {name: i for i, name in enumerate(DEAP_CHANNELS)}

TOPO_GRID_NAMES = [
    ["AF3", "FP1",  None, "FP2", "AF4"],
    ["F7",  "F3",  "Fz",  "F4",  "F8"],
    ["FC5", "FC1", "FC2", "FC6", None],
    ["T7",  "C3",  "Cz",  "C4",  "T8"],
    ["CP5", "CP1", "CP2", "CP6", None],
    ["P7",  "P3",  "Pz",  "P4",  "P8"],
    ["PO3", "O1",  "Oz",  "O2",  "PO4"],
]

TOPO_GRID_IDX = np.array([
    [NAME_TO_IDX.get(name, -1) if name is not None else -1 for name in row]
    for row in TOPO_GRID_NAMES
])

# ===== 2. 訊號處理函數 =====

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """帶通濾波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data, axis=-1)
    return y

def compute_de(signal):
    """計算微分熵 (Differential Entropy)"""
    # 假設訊號服從高斯分佈: DE = 0.5 * log(2 * pi * e * variance)
    var = np.var(signal, ddof=1)
    return 0.5 * np.log(2 * np.pi * np.e * var + 1e-6)

def process_trial_to_de_topo(eeg_data):
    """
    將單一 Trial (32, 8064) 轉換為 DE 拓譜序列
    回傳 Shape: (60, 1, 7, 5, 4) -> (秒, Channel, H, W, 頻段)
    """
    # 移除 Baseline (前 3 秒)，只取後 60 秒
    exper_signal = eeg_data[:N_CHANNELS, 3*FS : 63*FS] # (32, 7680)
    
    n_secs = 60
    H, W = 7, 5
    D = len(BANDS)
    
    # 初始化輸出矩陣
    de_topo = np.zeros((n_secs, 1, H, W, D))
    
    # 預先對整個 Trial 進行四個頻段的濾波，增加效率
    filtered_bands = {}
    for b_name, (low, high) in BANDS.items():
        filtered_bands[b_name] = butter_bandpass_filter(exper_signal, low, high, FS)

    for s in range(n_secs):
        t_start, t_end = s * FS, (s + 1) * FS
        
        for b_idx, (b_name, _) in enumerate(BANDS.items()):
            band_sig = filtered_bands[b_name][:, t_start:t_end] # (32, 128)
            
            for r in range(H):
                for c in range(W):
                    ch_idx = TOPO_GRID_IDX[r, c]
                    if ch_idx != -1:
                        # 計算該秒、該通道、該頻段的 DE 值
                        val = compute_de(band_sig[ch_idx, :])
                        de_topo[s, 0, r, c, b_idx] = val
                        
    return de_topo

# ===== 3. 主執行邏輯 =====

if __name__ == "__main__":
    all_X, all_y, all_s = [], [], []

    for sub_idx in range(1, 33):
        file_name = f"s{sub_idx:02d}.dat"
        path = os.path.join(DATA_DIR, file_name)
        
        if not os.path.exists(path):
            print(f"Skip {file_name}: File not found.")
            continue
            
        print(f"Processing {file_name} (Differential Entropy)...")
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        eeg = data['data']    # (40, 40, 8064)
        labels = data['labels'] # (40, 4)
        
        # 選擇目標：Valence (index 0)
        target_labels = (labels[:, 1] > 5).astype(np.int64)
        
        for t in range(40):
            # 轉換為 DE 拓撲序列 (60, 1, 7, 5, 4)
            de_segs = process_trial_to_de_topo(eeg[t])
            
            all_X.append(de_segs)
            all_y.append(np.full(60, target_labels[t]))
            all_s.append(np.full(60, sub_idx - 1)) # Subject ID 0-31

    # 合併數據
    X = np.concatenate(all_X, axis=0) # (1920*40, 1, 7, 5, 4)
    y = np.concatenate(all_y, axis=0)
    s = np.concatenate(all_s, axis=0)

    print("\n" + "="*30)
    print(f"Final Shape X: {X.shape}") # 預計 (76800, 1, 7, 5, 4)
    print(f"Final Shape y: {y.shape}")
    print(f"Final Shape s: {s.shape}")
    
    # 儲存
    np.savez_compressed(SAVE_PATH, X=X, y=y, s=s)
    print(f"Saved DE Topo dataset to: {SAVE_PATH}")