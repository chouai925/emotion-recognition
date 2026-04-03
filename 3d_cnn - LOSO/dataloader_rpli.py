# dataloader_rpli.py
import numpy as np
import torch

def load_deap_data(npz_path, test_subject_id):
    """
    載入 DEAP 數據，並嚴格執行與 PR-PL 一致的 Subject-wise Z-score 標準化
    """
    print(f"Loading data from {npz_path} for RPLI Target S{test_subject_id}...")
    npz = np.load(npz_path)
    X_all = npz["X"].astype(np.float32) # 形狀: (76800, 1, 7, 5, 4)
    y_all = npz["y"].astype(np.int64)
    s_all = npz["s"].astype(np.int64)

    # 關鍵：Subject-wise Z-Score Normalization
    for sub_id in range(32):
        mask = (s_all == sub_id)
        if np.any(mask):
            sub_X = X_all[mask] 
            # 必須改成 axis=0，確保每個 (7,5,4) 的節點獨立標準化，完美對齊 PR-PL！
            mean = np.mean(sub_X, axis=0, keepdims=True)
            std = np.std(sub_X, axis=0, keepdims=True)
            X_all[mask] = (sub_X - mean) / (std + 1e-6)
    # 針對 DEAP 轉為 0 和 1 (假設原始標籤已經是 0 和 1)
    train_mask = (s_all != test_subject_id)
    test_mask = (s_all == test_subject_id)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    return X_train, y_train, X_test, y_test


def sample_groups(X_s, Y_s, X_t, Y_t_pseudo, seed=0, num_pairs_per_group=300):
    """
    DANN-RPLI 核心：將樣本組合為 6 個 Groups (專為 DEAP 二分類設計)
    回傳格式完美對齊 main_rpli.py 中的：
    g_x1, g_x2, g_y = sample_groups(...)
    """
    np.random.seed(seed)
    
    # 將張量轉為 numpy 方便處理
    if torch.is_tensor(Y_s): Y_s = Y_s.cpu().numpy()
    if torch.is_tensor(Y_t_pseudo): Y_t_pseudo = Y_t_pseudo.cpu().numpy()
    
    # 取得各類別的 index
    s_idx_0 = np.where(Y_s == 0)[0]
    s_idx_1 = np.where(Y_s == 1)[0]
    
    t_idx_0 = np.where(Y_t_pseudo == 0)[0]
    t_idx_1 = np.where(Y_t_pseudo == 1)[0]
    
    # 防呆機制：如果偽標籤初期極度不平衡，強制給予一個隨機樣本避免程式崩潰
    if len(t_idx_0) == 0: t_idx_0 = np.array([0])
    if len(t_idx_1) == 0: t_idx_1 = np.array([0])

    # 初始化三個裝資料的大陣列
    g_x1, g_x2, g_y = [], [], []

    def pick_and_add(idx_A, idx_B, data_A, data_B, size, group_label):
        """隨機抽取特徵對並直接加入大陣列中"""
        p1_idx = np.random.choice(idx_A, size, replace=True)
        p2_idx = np.random.choice(idx_B, size, replace=True)
        
        # 直接提取矩陣，維持 (size, 1, 7, 5, 4) 形狀
        g_x1.append(data_A[p1_idx])
        g_x2.append(data_B[p2_idx])
        g_y.append(np.full(size, group_label))

    # 為了類別平衡，每個 Group 的一半來自 Class 0，一半來自 Class 1
    half_size = num_pairs_per_group // 2
    
    # ==========================================
    # 建立 6 大群組 (對齊論文定義)
    # ==========================================
    # Group 1: Source Domain, Same Class (S0-S0, S1-S1)
    pick_and_add(s_idx_0, s_idx_0, X_s, X_s, half_size, 0)
    pick_and_add(s_idx_1, s_idx_1, X_s, X_s, num_pairs_per_group - half_size, 0)
    
    # Group 2: Source Domain, Different Class (S0-S1, S1-S0)
    pick_and_add(s_idx_0, s_idx_1, X_s, X_s, half_size, 1)
    pick_and_add(s_idx_1, s_idx_0, X_s, X_s, num_pairs_per_group - half_size, 1)

    # Group 3: Cross Domain, Same Class (S0-T0, S1-T1)
    pick_and_add(s_idx_0, t_idx_0, X_s, X_t, half_size, 2)
    pick_and_add(s_idx_1, t_idx_1, X_s, X_t, num_pairs_per_group - half_size, 2)

    # Group 4: Cross Domain, Different Class (S0-T1, S1-T0)
    pick_and_add(s_idx_0, t_idx_1, X_s, X_t, half_size, 3)
    pick_and_add(s_idx_1, t_idx_0, X_s, X_t, num_pairs_per_group - half_size, 3)

    # Group 5: Target Domain, Same Class (T0-T0, T1-T1)
    pick_and_add(t_idx_0, t_idx_0, X_t, X_t, half_size, 4)
    pick_and_add(t_idx_1, t_idx_1, X_t, X_t, num_pairs_per_group - half_size, 4)

    # Group 6: Target Domain, Different Class (T0-T1, T1-T0)
    pick_and_add(t_idx_0, t_idx_1, X_t, X_t, half_size, 5)
    pick_and_add(t_idx_1, t_idx_0, X_t, X_t, num_pairs_per_group - half_size, 5)

    # 使用 np.concatenate 將列表內的 numpy arrays 垂直拼接成乾淨的大矩陣！
    return np.concatenate(g_x1), np.concatenate(g_x2), np.concatenate(g_y)