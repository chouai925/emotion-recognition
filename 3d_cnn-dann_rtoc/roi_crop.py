import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# ===== 路徑設定 (請確認你的原始圖路徑) =====
SRC_DIR = r"D:\Deap_eeg\3d_cnn-dann_rtoc\frames"          # 原始高畫質圖片目錄
DST_DIR = r"D:\Deap_eeg\3d_cnn-dann_rtoc\roi_frames_224"   # 🚀 新生成的九宮格存放目錄

# MediaPipe 參數
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, 
    refine_landmarks=True, min_detection_confidence=0.5
)

# 🚀 完美的 9 個微表情核心部位 (按照 3x3 排列，已排除鼻樑與嘴唇中央)
ROI_LANDMARKS = [
    ("Left Inner Eyebrow", 107), ("Glabella", 9), ("Right Inner Eyebrow", 336),   # 上排：左眉頭、眉心、右眉頭
    ("Left Eye Corner", 130), ("Left Nose Ala", 129), ("Right Eye Corner", 359),  # 中排：左眼角、左鼻翼、右眼角
    ("Left Mouth Corner", 61), ("Right Nose Ala", 358), ("Right Mouth Corner", 291) # 下排：左嘴角、右鼻翼、右嘴角
]

# 裁切尺寸：從高畫質原圖切下 74x74 (74 * 3 = 222，非常接近 224)
CROP_SIZE = 74 

def crop_roi(image, center_x, center_y, box_size):
    h, w, _ = image.shape
    half_size = box_size // 2
    y1, y2 = max(0, center_y - half_size), min(h, center_y + half_size)
    x1, x2 = max(0, center_x - half_size), min(w, center_x + half_size)
    crop = image[y1:y2, x1:x2]
    
    # 邊界 Padding 處理 (如果裁切框超出圖片邊緣)
    if crop.shape[0] != box_size or crop.shape[1] != box_size:
        padded = np.zeros((box_size, box_size, 3), dtype=np.uint8)
        padded[0:(y2-y1), 0:(x2-x1)] = crop
        return padded
    return crop

def process_image(img_path, save_path):
    image = cv2.imread(img_path)
    if image is None: return False
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    results = face_mesh.process(image_rgb)
    
    # 如果抓不到臉，直接回傳全黑圖 (確保時序連續性不斷裂)
    if not results.multi_face_landmarks:
        black_grid = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.imwrite(save_path, black_grid)
        return False

    landmarks = results.multi_face_landmarks[0]
    rois = []
    
    # 1. 直接從高畫質原圖上擷取 9 個超清晰局部部位
    for _, lm_index in ROI_LANDMARKS:
        lm = landmarks.landmark[lm_index]
        cx, cy = int(lm.x * w), int(lm.y * h)
        rois.append(crop_roi(image, cx, cy, CROP_SIZE))
        
    # 2. 拼裝 3x3 九宮格 
    row1 = np.hstack([rois[0], rois[1], rois[2]])
    row2 = np.hstack([rois[3], rois[4], rois[5]])
    row3 = np.hstack([rois[6], rois[7], rois[8]])
    grid = np.vstack([row1, row2, row3]) # 此時大小為完美的 222 x 222
    
    # 3. 微微放大至 ResNet18 需要的 224 x 224 (畫質幾乎無損)
    grid_224 = cv2.resize(grid, (224, 224))
    
    # 儲存
    cv2.imwrite(save_path, grid_224)
    return True

def generate_dataset():
    os.makedirs(DST_DIR, exist_ok=True)
    subject_folders = sorted([f for f in os.listdir(SRC_DIR) if f.startswith('s')])
    
    for sub in subject_folders:
        sub_path = os.path.join(SRC_DIR, sub)
        dst_sub_path = os.path.join(DST_DIR, sub)
        if not os.path.isdir(sub_path): continue
        
        trial_folders = sorted(os.listdir(sub_path))
        for trial in tqdm(trial_folders, desc=f"處理 {sub} 九宮格"):
            trial_path = os.path.join(sub_path, trial)
            dst_trial_path = os.path.join(dst_sub_path, trial)
            if not os.path.isdir(trial_path): continue
            
            os.makedirs(dst_trial_path, exist_ok=True)
            for img_name in os.listdir(trial_path):
                if img_name.endswith('.png'):
                    process_image(
                        os.path.join(trial_path, img_name),
                        os.path.join(dst_trial_path, img_name)
                    )

if __name__ == "__main__":
    print("開始將原始圖片轉換為純表情九宮格資料集")
    generate_dataset()
    print("全部處理完成！")