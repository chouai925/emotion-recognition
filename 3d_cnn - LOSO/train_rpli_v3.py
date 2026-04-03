# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv
import matplotlib.pyplot as plt  # 新增：用於繪製折線圖
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# 引入您之前寫好的模組 (確保 model.py 已經更新為 Self-Attention 版本)
from model_v3 import EEGEmotionNetTopoLG3D, ReverseLayerF
from dataloader_rpli import load_deap_data, sample_groups
from model_PR_PL_v3 import Domain_adaption_model

from mmd import mmd_rbf_noaccelerate
from cmmd import cmmd
import copy

# ==========================================
# 1. 定義 DANN-RPLI 輔助網路 (對接 3DCNN)
# ==========================================
class Classifier(nn.Module):
    def __init__(self, input_dim=96): # 32*3=96 (EEGEmotionNetTopoLG3D 的輸出維度)
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, 2) # DEAP 為 2 類 (High/Low)

    def forward(self, x):
        return self.fc(x)

class DCD(nn.Module):
    """領域判別器 (Domain Class Discriminator)"""
    def __init__(self, input_features=192): # 兩個樣本特徵拼接 96*2 = 192
        super(DCD, self).__init__()
        self.fc1 = nn.Linear(input_features, 64)
        self.fc2 = nn.Linear(64, 6) # 對應 6 個配對 Groups

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ==========================================
# 2. 自動繪圖函數 (雙 Y 軸：Loss & Accuracy)
# ==========================================
def plot_learning_curves(csv_file, save_dir, target_id):
    """讀取 CSV 檔並畫出訓練曲線"""
    epochs, losses, pseudo_accs, true_accs = [], [], [], []
    
    if not os.path.exists(csv_file):
        print(f"找不到 {csv_file}，無法繪圖！")
        return

    # 讀取 CSV
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # 跳過標題行
        for row in reader:
            epochs.append(int(row[0]))
            losses.append(float(row[1]))
            pseudo_accs.append(float(row[2]))
            true_accs.append(float(row[3]))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左側 Y 軸：Loss (紅色)
    color1 = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', color=color1, fontsize=12)
    line1, = ax1.plot(epochs, losses, color=color1, label='Total Loss (GH)', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)

    # 創建共用 X 軸的右側 Y 軸：Accuracy (藍色與綠色)
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    color3 = 'tab:green'
    ax2.set_ylabel('Accuracy', color='black', fontsize=12)
    line2, = ax2.plot(epochs, pseudo_accs, color=color2, linestyle='--', label='Pseudo Label Acc', linewidth=2)
    line3, = ax2.plot(epochs, true_accs, color=color3, label='True Target Acc', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='black')
    
    # 設定 y 軸範圍，讓 Accuracy 限制在 0.3 到 1.0 之間較容易觀察
    ax2.set_ylim([0.3, 1.05])

    # 合併圖例
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right')

    plt.title(f'Learning Curves for Target Subject S{target_id}', fontsize=14)
    fig.tight_layout()
    
    # 儲存圖片
    pic_path = os.path.join(save_dir, f'learning_curve_S{target_id}.jpg')
    plt.savefig(pic_path, dpi=300)
    plt.close()
    print(f"✅ 學習曲線已儲存至: {pic_path}")

# ==========================================
# 3. 核心訓練流程 (LOSO 交叉驗證)
# ==========================================
NPZ_PATH = r"C:\chou\Deap_eeg\3d_cnn - LOSO\deap_eeg_de_topo_v.npz"
PRPL_MODEL_DIR = r"C:\chou\Deap_eeg\3d_cnn - LOSO\PR-PL-main\savemodel_early" 
SAVE_DIR = r"C:\chou\Deap_eeg\3d_cnn - LOSO\result_v_exp3_0331"
os.makedirs(SAVE_DIR, exist_ok=True)

def train_rpli_for_subject(target_id):
    print(f"\n{'='*40}\n開始訓練目標受試者 S{target_id}\n{'='*40}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 初始化 CSV 紀錄檔
    csv_file_path = os.path.join(SAVE_DIR, f"training_log_S{target_id}.csv")
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Epoch_Avg_Loss", "Pseudo_Acc", "True_Acc"])

    # 1. 載入資料 (經過 Z-score 標準化)
    X_s, Y_s, X_t, Y_t_true = load_deap_data(NPZ_PATH, target_id)
    
    src_dataset = TensorDataset(torch.from_numpy(X_s).float(), torch.from_numpy(Y_s).long())
    src_loader = DataLoader(src_dataset, batch_size=64, shuffle=True)
    
    X_t_tensor = torch.from_numpy(X_t).float().to(device)
    Y_t_true_tensor = torch.from_numpy(Y_t_true).long().to(device)

    # 2. 初始化 3DCNN 架構模型 (現在內部是 Self-Attention 融合)
    encoder = EEGEmotionNetTopoLG3D(time_len=4).to(device)
    classifier = Classifier().to(device)
    discriminator = DCD().to(device)
    
    optimizer_gh = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.0005)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    
    # ✅ 替換為以下這段：動態計算類別權重與定義雙重 Loss
    class_counts = np.bincount(Y_s)
    total_samples = len(Y_s)
    weights = np.sqrt(total_samples / (len(class_counts) * class_counts)) 
    class_weights = torch.FloatTensor(weights).to(device)
    
    # 1. 給情緒分類器用的 Loss (加上權重平衡，不加標籤平滑)
    loss_fn_cls = nn.CrossEntropyLoss(weight=class_weights)
    
    # 2. 給 DANN 和 DCD 領域對抗用的 Loss
    loss_fn_domain = nn.CrossEntropyLoss()

    # --- Step 1: Warm-up ---
    print(">>> Step 1: 來源域預訓練 (Warm-up)...")
    
    best_warmup_loss = float('inf')
    best_enc_weights = None
    best_cls_weights = None

    for epoch in range(10):
        encoder.train(); classifier.train()
        pbar = tqdm(src_loader, desc=f"Warm-up Epoch {epoch+1}/10")
        
        epoch_loss = 0.0
        batch_count = 0
        
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            optimizer_gh.zero_grad()
            
            # 接收三個回傳值
            final_feat, _, local_cs = encoder(data)
            loss_main = loss_fn_cls(classifier(final_feat), labels)
            
            # 🚀 計算局部情緒 Loss 
            loss_local_cls = 0
            for c_pred in local_cs:
                loss_local_cls += loss_fn_cls(c_pred, labels)
                
            loss = loss_main + 0.5 * loss_local_cls
            loss.backward(); optimizer_gh.step()

            epoch_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # 計算這個 Epoch 的平均 Loss
        avg_loss = epoch_loss / batch_count
        
        # 如果是目前最低的 Loss，就把權重「影印」下來備用
        if avg_loss < best_warmup_loss:
            best_warmup_loss = avg_loss
            best_enc_weights = copy.deepcopy(encoder.state_dict())
            best_cls_weights = copy.deepcopy(classifier.state_dict())

    # Warm-up 結束後，把表現最好的權重載回模型中
    encoder.load_state_dict(best_enc_weights)
    classifier.load_state_dict(best_cls_weights)
    print(f"✅ 已載入 Warm-up 階段最佳權重 (平均 Loss: {best_warmup_loss:.4f})")

    # --- Step 2: 載入 PR-PL 獲取初始偽標籤 ---
    print(">>> Step 2: 載入 PR-PL 產生初始偽標籤...")
    prpl_path = os.path.join(PRPL_MODEL_DIR, f"best_network{target_id}.pth")
    if not os.path.exists(prpl_path):
        print(f"找不到 PR-PL 權重 {prpl_path}，跳過 S{target_id}！")
        return
        
    base_model = torch.load(prpl_path).to(device)
    base_model.eval()
    
    with torch.no_grad():
        X_t_flat = X_t_tensor.view(X_t_tensor.shape[0], -1)
        Y_t_pseudo = torch.tensor(base_model.predict(X_t_flat)).long().to(device)
    
    # --- Step 3: DANN-RPLI 聯合對抗與迭代 ---
    print(">>> Step 3: 進入聯合對抗與 RPLI 迭代...")
    best_acc = 0.0
    
    for epoch in range(100):
        g_x1, g_x2, g_y = sample_groups(X_s, Y_s, X_t, Y_t_pseudo.cpu().numpy(), num_pairs_per_group=300)
        group_loader = DataLoader(
            TensorDataset(torch.from_numpy(g_x1), torch.from_numpy(g_x2), torch.from_numpy(g_y)),
            batch_size=64, shuffle=True
        )

        # 3.2 訓練判別器 (DCD) 
        # 3.2 訓練判別器 (DCD) 
        encoder.eval(); discriminator.train()
        for x1, x2, gy in group_loader:
            x1, x2, gy = x1.to(device), x2.to(device), gy.long().to(device)
            optimizer_d.zero_grad()
            with torch.no_grad():
                feat1 = encoder(x1)[0]  # ⚠️ 加上 [0]
                feat2 = encoder(x2)[0]  # ⚠️ 加上 [0]
            loss_d = loss_fn_domain(discriminator(torch.cat([feat1, feat2], dim=1)), gy)
            loss_d.backward(); optimizer_d.step()

        # 3.3 訓練特徵提取器與分類器 
        encoder.train(); classifier.train(); discriminator.eval()
        src_iter = iter(src_loader) 
        
        epoch_loss_gh = 0.0 
        batch_count = 0
        
        # --- 1. 設定動態權重與置信度門檻 ---
        alpha_mmd = max(0.01, 1.0 - epoch / 100.0) * 0.5  # 放大 10 倍
        beta_cmmd = min(1.0, epoch / 100.0) * 0.5         # 放大 10 倍
        
        # 🚀 修正：DANN 的梯度反轉強度 (必須從 0 慢慢平滑增長到 1)
        p = epoch / 100.0
        dann_alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        # 動態置信度門檻
        if epoch < 10:
            conf_threshold = 0.0
        elif epoch < 40:
            conf_threshold = 0.5
        elif epoch < 85:
            conf_threshold = 0.75
        else:
            conf_threshold = 0.95
        
        for x1, x2, gy in group_loader:
            x1, x2, gy = x1.to(device), x2.to(device), gy.long().to(device)
            optimizer_gh.zero_grad()
            
            feat1 = encoder(x1)[0]
            feat2 = encoder(x2)[0]
            
            try:
                src_x, src_y = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                src_x, src_y = next(src_iter)
                
            src_x, src_y = src_x.to(device), src_y.to(device)
            
            # (A) 提取 Source 特徵與局部預測
            src_feat, src_local_ds, src_local_cs = encoder(src_x, dann_alpha)
            loss_cls = loss_fn_cls(classifier(src_feat), src_y)
            
            # 🚀 (B) 計算局部情緒分類器 Loss
            loss_local_cls = 0
            for c_pred in src_local_cs:
                loss_local_cls += loss_fn_cls(c_pred, src_y)
            
            # (C) DCD 全局群組對抗
            feat_cat = torch.cat([feat1, feat2], dim=1)
            feat_cat_rev = ReverseLayerF.apply(feat_cat, 1.0) 
            loss_adv = loss_fn_domain(discriminator(feat_cat_rev), gy) 
            
            # (D) MMD 與 CMMD
            idx = torch.randperm(X_t_tensor.size(0))[:src_x.size(0)]
            tgt_x = X_t_tensor[idx]
            # 提取 Target 特徵 (改用 dann_alpha)
            tgt_feat, tgt_local_ds, _ = encoder(tgt_x, dann_alpha)
            loss_mmd = mmd_rbf_noaccelerate(src_feat, tgt_feat)
            
            tgt_logits = classifier(tgt_feat)
            tgt_probs = F.softmax(tgt_logits, dim=1)
            max_probs, tgt_pseudo_y = torch.max(tgt_probs, dim=1)
            mask = max_probs >= conf_threshold
            confident_tgt_feat = tgt_feat[mask]
            confident_tgt_pseudo_y = tgt_pseudo_y[mask]
            if confident_tgt_feat.size(0) > 0:
                loss_cmmd = cmmd(src_feat, confident_tgt_feat, src_y, confident_tgt_pseudo_y)
            else:
                loss_cmmd = torch.tensor(0.0).to(device)

            # 🚀 (E) 計算局部領域對抗 (Local DANN) Loss 
            src_d_labels = torch.zeros(src_x.size(0), dtype=torch.long).to(device)
            tgt_d_labels = torch.ones(tgt_x.size(0), dtype=torch.long).to(device)
            
            loss_local_d = 0
            for d_pred in src_local_ds:
                loss_local_d += loss_fn_domain(d_pred, src_d_labels)
            for d_pred in tgt_local_ds:
                loss_local_d += loss_fn_domain(d_pred, tgt_d_labels)
                
            loss_local_d = loss_local_d / 6.0 # ⚠️ 必須除以 6 取平均，否則權重會暴走
            
            # 🚀 (F) 終極 Loss 組合
            loss_total = loss_cls + (0.1 * loss_local_cls) + (0.05 * loss_adv) + \
                         (0.1 * loss_local_d) + (alpha_mmd * loss_mmd) + (beta_cmmd * loss_cmmd)
            
            loss_total.backward()
            optimizer_gh.step()
            
            epoch_loss_gh += loss_total.item()
            batch_count += 1

        avg_epoch_loss = epoch_loss_gh / max(1, batch_count)

        # 3.4 偽標籤迭代與特徵擾動
        # 3.4 偽標籤迭代與特徵擾動
        if epoch > 5 and epoch % 5 == 0:
            encoder.eval(); classifier.eval()
            with torch.no_grad():
                feat_t = encoder(X_t_tensor)[0] # ⚠️ 補上 [0] 提取 final_feat
                y_pred_t = torch.max(classifier(feat_t), 1)[1]
                
                idx_0 = torch.where(y_pred_t == 0)[0]
                idx_1 = torch.where(y_pred_t == 1)[0]
                
                if len(idx_0) > 0 and len(idx_1) > 0:
                    center0_idx = torch.argmin(torch.cdist(X_t_flat[idx_0], X_t_flat[idx_0]).sum(1))
                    center1_idx = torch.argmin(torch.cdist(X_t_flat[idx_1], X_t_flat[idx_1]).sum(1))
                    
                    C0 = X_t_tensor[idx_0[center0_idx]]
                    C1 = X_t_tensor[idx_1[center1_idx]]
                    
                    def cosine_sim(a, b):
                        return F.cosine_similarity(a.view(1, -1), b.view(1, -1)).item()
                    
                    fused_X_t = []
                    for i in range(len(X_t_tensor)):
                        x = X_t_tensor[i]
                        sim0, sim1 = cosine_sim(x, C0), cosine_sim(x, C1)
                        w0, w1 = sim0 / (sim0+sim1+1e-6), sim1 / (sim0+sim1+1e-6)
                        fused_x = w0 * x + w1 * C1 if y_pred_t[i] == 0 else w0 * x + w1 * C0
                        fused_X_t.append(fused_x)
                    
                    fused_X_t = torch.stack(fused_X_t) 
                    y_pred_fused = torch.max(classifier(encoder(fused_X_t)[0]), 1)[1] # ⚠️ 補上 [0]
                    
                    reliable_mask = (y_pred_t == y_pred_fused)
                    Y_t_pseudo[reliable_mask] = y_pred_t[reliable_mask]

        # 3.5 評估真實準確率與紀錄寫入
        encoder.eval(); classifier.eval()
        with torch.no_grad():
            # ⚠️ 這裡必須加上 [0] 提取 final_feat
            final_pred = torch.max(classifier(encoder(X_t_tensor)[0]), 1)[1] 
            acc = (final_pred == Y_t_true_tensor).float().mean().item()
            pseudo_acc = (Y_t_pseudo == Y_t_true_tensor).float().mean().item()
            
        print(f"S{target_id} Ep{epoch:02d} | 偽標籤Acc: {pseudo_acc:.4f} | 真實Acc: {acc:.4f} | Avg_Loss: {avg_epoch_loss:.4f}")
        
        # 寫入 CSV (確保縮進正確)
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_epoch_loss, pseudo_acc, acc])

        # 建立完整的 Checkpoint
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'best_acc': max(acc, best_acc)
        }
        
        # 覆蓋最新進度 (斷電救援用)
        torch.save(checkpoint, os.path.join(SAVE_DIR, f"latest_checkpoint_S{target_id}.pth"))

        if acc > best_acc:
            best_acc = acc
            # 儲存表現最好的模型
            torch.save(checkpoint, os.path.join(SAVE_DIR, f"best_model_S{target_id}.pth"))
            
    print(f"S{target_id} 訓練完成！最高準確率: {best_acc:.4f}")
    
    # 訓練結束後，自動繪製並儲存學習曲線圖
    plot_learning_curves(csv_file_path, SAVE_DIR, target_id)
    print("="*40 + "\n")

if __name__ == "__main__":
    # DEAP 共有 32 個 subjects
    for i in range(32):
        train_rpli_for_subject(i)