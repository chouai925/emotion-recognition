import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 路徑與基礎設定 (請確認路徑無誤)
# ==========================================
SRC_DIR = r"D:\Deap_eeg\3d_cnn-dann_rtoc\frames"          # 原始全臉 Frame 目錄
DST_DIR = r"D:\Deap_eeg\3d_cnn-dann_rtoc\roi_npy_data_64"  # 🚀 新目錄：存放 64x64 的 .npy 檔案

# 確保裁切尺寸為學術黃金標準 64x64
CROP_SIZE = 64 

# ==========================================
# 2. MediaPipe 初始化與 9 個部位定義
# ==========================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, 
    refine_landmarks=True, min_detection_confidence=0.5
)

# 完美的 9 個核心部位 (左眉、眉心、右眉 / 左眼角、左鼻翼、右眼角 / 左嘴角、右鼻翼、右嘴角)
ROI_LANDMARKS = [107, 9, 336, 130, 129, 359, 61, 358, 291]

# ==========================================
# 3. 核心裁切與處理函數
# ==========================================
def crop_roi(image, center_x, center_y, box_size):
    """給定中心點，裁切出 box_size 大小的局部圖。若超出邊界則補黑邊。"""
    h, w, _ = image.shape
    half_size = box_size // 2
    y1, y2 = max(0, center_y - half_size), min(h, center_y + half_size)
    x1, x2 = max(0, center_x - half_size), min(w, center_x + half_size)
    crop = image[y1:y2, x1:x2]
    
    # Padding 處理
    if crop.shape[0] != box_size or crop.shape[1] != box_size:
        padded = np.zeros((box_size, box_size, 3), dtype=np.uint8)
        padded[0:(y2-y1), 0:(x2-x1)] = crop
        return padded
    return crop

def process_image(img_path, save_path):
    """讀取單張圖片，切出 9 個部位，並存成 .npy 陣列"""
    image = cv2.imread(img_path)
    if image is None: return False
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    results = face_mesh.process(image_rgb)
    
    # 容錯機制：如果這張 Frame 沒抓到臉，回傳全黑陣列
    if not results.multi_face_landmarks:
        black_rois = np.zeros((len(ROI_LANDMARKS), CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        np.save(save_path, black_rois)
        return False

    landmarks = results.multi_face_landmarks[0]
    rois = []
    
    # 依序裁切 9 個部位
    for lm_index in ROI_LANDMARKS:
        lm = landmarks.landmark[lm_index]
        cx, cy = int(lm.x * w), int(lm.y * h)
        rois.append(crop_roi(image, cx, cy, CROP_SIZE))
        
    # 打包成 (9, 64, 64, 3) 的 Numpy 陣列並存檔
    rois_array = np.array(rois, dtype=np.uint8)
    np.save(save_path, rois_array)
    return True

# ==========================================
# 4. 批次處理主程式
# ==========================================
def generate_dataset():
    os.makedirs(DST_DIR, exist_ok=True)
    subject_folders = sorted([f for f in os.listdir(SRC_DIR) if f.startswith('s')])
    
    for sub in subject_folders:
        sub_path = os.path.join(SRC_DIR, sub)
        dst_sub_path = os.path.join(DST_DIR, sub)
        if not os.path.isdir(sub_path): continue
        
        trial_folders = sorted(os.listdir(sub_path))
        for trial in tqdm(trial_folders, desc=f"備料中 [{sub}] -> .npy"):
            trial_path = os.path.join(sub_path, trial)
            dst_trial_path = os.path.join(dst_sub_path, trial)
            if not os.path.isdir(trial_path): continue
            
            os.makedirs(dst_trial_path, exist_ok=True)
            for img_name in os.listdir(trial_path):
                if img_name.endswith('.png'):
                    npy_name = img_name.replace('.png', '.npy')
                    # 只在 .npy 檔案不存在時才處理，方便中斷後接續執行！
                    full_save_path = os.path.join(dst_trial_path, npy_name)
                    if not os.path.exists(full_save_path):
                        process_image(
                            os.path.join(trial_path, img_name),
                            full_save_path
                        )

if __name__ == "__main__":
    print("🚀 啟動：微表情 9分支 64x64 NPY 資料集備料程序...")
    generate_dataset()
    print("✅ 備料大功告成！主廚 (GPU) 可以準備開火了！")