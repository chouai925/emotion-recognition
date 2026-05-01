import os
import numpy as np
from tqdm import tqdm

# ===== 路徑設定 (根據你提供的路徑修改) =====
FACE_FRAMES_DIR = r"D:\Deap_eeg\3d_cnn-dann_rtoc\face_frames_224"
INDEX_SAVE_PATH = r"D:\Deap_eeg\3d_cnn-dann_rtoc\face_index_resize_224.npz"

NUM_SUBJECTS_WITH_VIDEO = 22
NUM_TRIALS = 40
SECONDS_PER_TRIAL = 60
FRAMES_PER_SECOND = 5

def build_index():
    print(f"🔍 正在根據實際路徑格式建立索引...")
    
    all_face_paths = []
    all_info = []

    for sub_idx in range(1, NUM_SUBJECTS_WITH_VIDEO + 1):
        sub_str = f"s{sub_idx:02d}" # 例如 s01
        
        for trial_idx in range(1, NUM_TRIALS + 1):
            trial_str = f"trial{trial_idx:02d}" # 例如 trial01
            
            # 🚀 根據你提供的格式組合資料夾名稱：s01\s01_trial01
            folder_name = f"{sub_str}_{trial_str}"
            target_dir = os.path.join(FACE_FRAMES_DIR, sub_str, folder_name)
            
            for s in range(SECONDS_PER_TRIAL):
                start_frame = (s * FRAMES_PER_SECOND) + 1
                end_frame = start_frame + FRAMES_PER_SECOND
                
                current_sec_paths = []
                valid_count = 0
                
                for frame_idx in range(start_frame, end_frame):
                    # 🚀 根據你提供的格式組合檔案名稱：s01_trial01_1.png
                    img_name = f"{sub_str}_{trial_str}_{frame_idx}.png"
                    img_path = os.path.join(target_dir, img_name)
                    
                    if os.path.exists(img_path):
                        current_sec_paths.append(img_path)
                        valid_count += 1
                    else:
                        current_sec_paths.append("")
                
                all_face_paths.append(current_sec_paths)
                # 這一秒只要有圖就算有效
                all_info.append([sub_idx-1, trial_idx-1, s, 1 if valid_count > 0 else 0])

    paths_array = np.array(all_face_paths)
    info_array = np.array(all_info)

    np.savez_compressed(INDEX_SAVE_PATH, paths=paths_array, info=info_array)
    
    valid_total = np.sum(info_array[:, 3])
    print("\n" + "="*40)
    print(f"✅ 索引表建立完成！")
    print(f"總筆數: {len(info_array)}")
    print(f"有效影像筆數: {valid_total}")
    print(f"有效率: {valid_total / len(info_array) * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    build_index()