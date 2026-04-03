import os
import pickle
import numpy as np



DATA_DIR = r"C:\chou\data_preprocessed_python"   # s01.dat ~ s32.dat
SAVE_PATH = r"C:\chou\Deap_eeg\deap_eeg_segments_baseline.npz"

FS = 128                # 採樣率
BASE_SEC = 3            # baseline = 3 秒
TRIAL_SEC = 60          # exper = 60 秒
N_CHANNELS = 32         # 取前 32 channels（DEAP 預設是 40 channels）

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
    final_segments = exper_segments - baseline_mean[None, :, :]  # (60, 32, 128)

    # ---- 增加 CNN 輸入維度 (B, 1, C, L) ----
    final_segments = final_segments[:, None, :, :]  # (60, 1, 32, 128)

    return final_segments


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

    # 取 valence，threshold 5.0 做二分類 (0/1)
    valence_raw = labels[:, 0]
    valence_binary = (valence_raw > 5).astype(np.int64)  # shape (40,)

    # 處理每一 trial
    for t in range(40):
        segs = process_trial(eeg[t])  # (60, 1, 32, 128)
        all_X.append(segs)

        # 每個 trial 60 個 segment → label 複製 60 次
        seg_labels = np.repeat(valence_binary[t], 60)
        all_y.append(seg_labels)

# ---------------------------
# 合併全部資料
# ---------------------------
X_final = np.concatenate(all_X, axis=0)  # (32*40*60, 1, 32, 128)
y_final = np.concatenate(all_y, axis=0)  # (同樣數量)

print("Final X shape:", X_final.shape)
print("Final y shape:", y_final.shape)

# ---------------------------
# 儲存為 npz
# ---------------------------
np.savez_compressed(SAVE_PATH, X=X_final, y=y_final)

print(f"\nSaved to: {SAVE_PATH}")
