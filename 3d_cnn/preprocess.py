import os
import pickle
import numpy as np



DATA_DIR = r"C:\chou\data_preprocessed_python"   # s01.dat ~ s32.dat
SAVE_PATH = r"C:\chou\Deap_eeg\3d_cnn\deap_eeg_segments_3dtopo_a.npz"

FS = 128                # 採樣率
BASE_SEC = 3            # baseline = 3 秒
TRIAL_SEC = 60          # exper = 60 秒
N_CHANNELS = 32         # 取前 32 channels（DEAP 預設是 40 channels）


# DEAP 32 通道的順序（Geneva order）
DEAP_CHANNELS = [
    "FP1","AF3","F3","F7","FC5","FC1","C3","T7","CP5","CP1","P3","P7","PO3",
    "O1","Oz","Pz","FP2","AF4","F4","F8","FC6","FC2","C4","T8","CP6","CP2",
    "P4","P8","PO4","O2","Fz","Cz"
]

NAME_TO_IDX = {name: i for i, name in enumerate(DEAP_CHANNELS)}

# 我們要的 7×5 頭皮拓撲佈局（None 表示沒有 channel）
TOPO_GRID_NAMES = [
    ["AF3", "FP1",  None, "FP2", "AF4"],
    ["F7",  "F3",  "Fz",  "F4",  "F8"],
    ["FC5", "FC1", "FC2", "FC6", None],
    ["T7",  "C3",  "Cz",  "C4",  "T8"],
    ["CP5", "CP1", "CP2", "CP6", None],
    ["P7",  "P3",  "Pz",  "P4",  "P8"],
    ["PO3", "O1",  "Oz",  "O2",  "PO4"],
]

# 把名字轉成 0-based channel index；沒有的地方設為 -1
TOPO_GRID_IDX = [
    [NAME_TO_IDX.get(name, -1) if name is not None else -1 for name in row]
    for row in TOPO_GRID_NAMES
]


# ---------------------------
# 對單一 trial 做 baseline subtraction + segmentation
# ---------------------------

def process_trial(eeg_trial):
    """
    eeg_trial: shape (40, 8064)
    return: (60, 1, 32, 128)
    """
    eeg_trial = eeg_trial[:N_CHANNELS, :]  # 取前 32 個 channel

    # ---- 切 baseline 與 exper ----
    n_base = BASE_SEC * FS              # 3*128 = 384
    n_exper = TRIAL_SEC * FS            # 60*128 = 7680

    baseline = eeg_trial[:, :n_base]    # (32, 384)
    exper = eeg_trial[:, n_base:n_base + n_exper]  # (32, 7680)

    # ---- baseline segmentation: (3, 32, 128) ----
    baseline_segments = baseline.reshape(N_CHANNELS, BASE_SEC, FS)
    baseline_segments = np.transpose(baseline_segments, (1, 0, 2))

    # ---- exper segmentation: (60, 32, 128) ----
    exper_segments = exper.reshape(TRIAL_SEC, N_CHANNELS, FS)

    # ---- baseline 平均 ----
    baseline_mean = baseline_segments.mean(axis=0)   # (32, 128)

    # ---- baseline subtraction ----
    # final_segments: (60, 32, 128) = 60 秒 × 32 channels × 128 samples
    final_segments = exper_segments - baseline_mean[None, :, :]  # (60, 32, 128)

    # ---- 建立 7×5 頭皮拓撲 grid ----
    n_segments = final_segments.shape[0]  # 一個 trial 60 個 segments
    H, W = 7, 5

    # 先建一個空的 (60, 7, 5, 128)
    topo_segments = np.zeros((n_segments, H, W, FS), dtype=final_segments.dtype)

    for r in range(H):
        for c in range(W):
            ch_idx = TOPO_GRID_IDX[r][c]
            if ch_idx == -1:
                # 這個格子沒有 channel，保持 0
                continue
            topo_segments[:, r, c, :] = final_segments[:, ch_idx, :]

    # ---- 增加 CNN 輸入維度 (B, 1, H, W, L) ----
    topo_segments = topo_segments[:, None, :, :, :]   # (60, 1, 7, 5, 128)

    return topo_segments


# ---------------------------
# 主程式：處理全部 subjects
# ---------------------------

all_X = []
all_y = []

for subject in range(1, 33):  # s01 ~ s32
    file_name = f"s{subject:02d}.dat"
    file_path = os.path.join(DATA_DIR, file_name)

    print(f"Processing {file_name} ...")

    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    eeg = data["data"]    # (40, 40, 8064)
    labels = data["labels"]  # (40, 4)  valence, arousal, dominance, liking

    # 取 valence=0，threshold 5.0 做二分類 (0/1)
    valence_raw = labels[:, 1]
    valence_binary = (valence_raw > 5).astype(np.int64)  # shape (40,)

    # 處理每一 trial
    for t in range(40):
        segs = process_trial(eeg[t])  # (60, 1, 7, 5, 128)
        all_X.append(segs)

        # 每個 trial 60 個 segment → label 複製 60 次
        seg_labels = np.repeat(valence_binary[t], 60)
        all_y.append(seg_labels)

# ---------------------------
# 合併全部資料
# ---------------------------
X_final = np.concatenate(all_X, axis=0)  # (N, 1, 7, 5, 128)
y_final = np.concatenate(all_y, axis=0)  # (同樣數量)

print("Final X shape:", X_final.shape)
print("Final y shape:", y_final.shape)

# ---------------------------
# 儲存為 npz
# ---------------------------
np.savez_compressed(SAVE_PATH, X=X_final, y=y_final)

print(f"\nSaved to: {SAVE_PATH}")
