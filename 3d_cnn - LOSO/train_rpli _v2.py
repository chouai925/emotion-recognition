# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import copy

# 引入自訂模組
from model_v2 import EEGEmotionNetTopoLG3D, ReverseLayerF
from dataloader_rpli import load_deap_data, sample_groups
from model_PR_PL import Domain_adaption_model
from mmd import mmd_rbf_noaccelerate
from cmmd import cmmd

# ==========================================
# 1. 定義 DANN-RPLI 輔助網路
# ==========================================
class Classifier(nn.Module):
    def __init__(self, input_dim=96):
        super(Classifier, self).__init__()
        # 加入 Dropout 增強分類器泛化能力
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(x)

class DCD(nn.Module):
    """領域判別器 (Domain Class Discriminator)"""
    def __init__(self, input_features=192):
        super(DCD, self).__init__()
        self.fc1 = nn.Linear(input_features, 64)
        self.fc2 = nn.Linear(64, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ==========================================
# 2. 自動繪圖函數
# ==========================================
def plot_learning_curves(csv_file, save_dir, target_id):
    epochs, losses, pseudo_accs, true_accs = [], [], [], []
    if not os.path.exists(csv_file): return

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            epochs.append(int(row[0]))
            losses.append(float(row[1]))
            pseudo_accs.append(float(row[2]))
            true_accs.append(float(row[3]))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', color=color1, fontsize=12)
    line1, = ax1.plot(epochs, losses, color=color1, label='Total Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    color3 = 'tab:green'
    ax2.set_ylabel('Accuracy', color='black', fontsize=12)
    line2, = ax2.plot(epochs, pseudo_accs, color=color2, linestyle='--', label='Pseudo Label Acc', linewidth=2)
    line3, = ax2.plot(epochs, true_accs, color=color3, label='True Target Acc', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim([0.3, 1.05])

    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right')

    plt.title(f'Learning Curves for Target Subject S{target_id}', fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f'learning_curve_S{target_id}.jpg'), dpi=300)
    plt.close()

# ==========================================
# 3. 核心訓練流程 (LOSO 交叉驗證)
# ==========================================
NPZ_PATH = r"C:\chou\Deap_eeg\3d_cnn - LOSO\deap_eeg_de_topo_v.npz"
# 修正 1：務必加上 _a，讀取帶有 BN 層的高準確率預訓練權重
PRPL_MODEL_DIR = r"C:\chou\Deap_eeg\3d_cnn - LOSO\PR-PL-main\savemodel_early_v_0330" 
SAVE_DIR = r"C:\chou\Deap_eeg\3d_cnn - LOSO\rpli_result_exp2_0330_v"
os.makedirs(SAVE_DIR, exist_ok=True)

def train_rpli_for_subject(target_id):
    print(f"\n{'='*40}\n開始訓練目標受試者 S{target_id}\n{'='*40}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    csv_file_path = os.path.join(SAVE_DIR, f"training_log_S{target_id}.csv")
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Epoch_Avg_Loss", "Pseudo_Acc", "True_Acc"])

    X_s, Y_s, X_t, Y_t_true = load_deap_data(NPZ_PATH, target_id)
    
    src_dataset = TensorDataset(torch.from_numpy(X_s).float(), torch.from_numpy(Y_s).long())
    src_loader = DataLoader(src_dataset, batch_size=64, shuffle=True)
    
    X_t_tensor = torch.from_numpy(X_t).float().to(device)
    Y_t_true_tensor = torch.from_numpy(Y_t_true).long().to(device)

    encoder = EEGEmotionNetTopoLG3D(time_len=4).to(device)
    classifier = Classifier().to(device)
    discriminator = DCD().to(device)
    
    # 初始化優化器
    optimizer_gh = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.0005)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    
    # --- 動態計算來源域的類別權重 (Class Weights) ---
    class_counts = np.bincount(Y_s)
    total_samples = len(Y_s)
    weights = np.sqrt(total_samples / (len(class_counts) * class_counts)) # 開根號平滑化
    class_weights = torch.FloatTensor(weights).to(device)
    
    # ==========================================
    # 確保以下這兩個 Loss 函數都有被定義到！
    # ==========================================
    # 1. 情緒分類器專用的 Loss (給 classifier 用)
    loss_fn_cls = nn.CrossEntropyLoss(weight=class_weights)
    
    # 2. 領域與群組判別器專用的 Loss (給 discriminator 和 local_ds1 用)
    loss_fn_domain = nn.CrossEntropyLoss()

    # --- Step 1: Warm-up ---
    print(">>> Step 1: 來源域預訓練 (Warm-up)...")
    best_warmup_loss = float('inf')
    best_enc_weights, best_cls_weights = None, None

    for epoch in range(15):
        encoder.train(); classifier.train()
        pbar = tqdm(src_loader, desc=f"Warm-up Epoch {epoch+1}/10")
        epoch_loss, batch_count = 0.0, 0
        
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            optimizer_gh.zero_grad()
            # 修正 2：encoder 現在回傳 tuple，必須取 [0]
            loss = loss_fn_cls(classifier(encoder(data)[0]), labels)
            loss.backward(); optimizer_gh.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / batch_count
        if avg_loss < best_warmup_loss:
            best_warmup_loss = avg_loss
            best_enc_weights = copy.deepcopy(encoder.state_dict())
            best_cls_weights = copy.deepcopy(classifier.state_dict())

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

        # 3.2 訓練全局判別器 (DCD) 
        encoder.eval(); discriminator.train()
        for x1, x2, gy in group_loader:
            x1, x2, gy = x1.to(device), x2.to(device), gy.long().to(device)
            optimizer_d.zero_grad()
            with torch.no_grad():
                feat1 = encoder(x1)[0] 
                feat2 = encoder(x2)[0]
            # 這裡用領域 Loss (6 類)
            loss_d = loss_fn_domain(discriminator(torch.cat([feat1, feat2], dim=1)), gy)
            loss_d.backward(); optimizer_d.step()

        # 3.3 訓練特徵提取器與分類器 
        encoder.train(); classifier.train(); discriminator.eval()
        src_iter = iter(src_loader) 
        epoch_loss_gh, batch_count = 0.0, 0
        
        # 將 scale_factor 稍微調高，增強對齊的引導力
        scale_factor = 0.1  
        
        alpha = max(0.01, 1.0 - epoch / 100.0) * scale_factor
        
        # --- 新增：延遲 CMMD 啟動 ---
        # 前 15 個 Epoch 強制 beta = 0，不進行 CMMD 對齊
        if epoch < 15:
            beta = 0.0
        else:
            # 15 回合後再緩慢增加權重
            beta = min(1.0, (epoch - 15) / 85.0) * scale_factor
        
        # 動態置信度門檻
        if epoch < 10: conf_threshold = 0.0
        elif epoch < 40: conf_threshold = 0.5
        elif epoch < 85: conf_threshold = 0.75
        else: conf_threshold = 0.95
        
        for x1, x2, gy in group_loader:
            x1, x2, gy = x1.to(device), x2.to(device), gy.long().to(device)
            # 建立假的領域標籤供局部判別器使用 (假設 Source = 0)
            gy_domain = torch.zeros(x1.size(0)).long().to(device) 

            optimizer_gh.zero_grad()
            
            # (A) 提取特徵與局部對抗
            feat1, local_ds1 = encoder(x1, alpha)
            feat2, local_ds2 = encoder(x2, alpha)
            
            try:
                src_x, src_y = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                src_x, src_y = next(src_iter)
            src_x, src_y = src_x.to(device), src_y.to(device)
            
            src_feat, _ = encoder(src_x, alpha)
            # 這裡用分類 Loss (2 類，帶權重)
            loss_cls = loss_fn_cls(classifier(src_feat), src_y)
            
            # RPLI 全局對抗
            feat_cat = torch.cat([feat1, feat2], dim=1)
            feat_cat_rev = ReverseLayerF.apply(feat_cat, 1.0)
            # 這裡用領域 Loss (6 類)
            loss_adv_rpli = loss_fn_domain(discriminator(feat_cat_rev), gy)

            # 局部對抗損失 (左腦、右腦、全腦)
            loss_local_d = 0
            for d_pred in local_ds1: 
                # 這裡用領域 Loss (2 類，區分 Source/Target)
                loss_local_d += loss_fn_domain(d_pred, gy_domain)

            # (B) MMD 與 CMMD 對齊 (修正 4：補上 Target 採樣與預測邏輯)
            idx = torch.randperm(X_t_tensor.size(0))[:src_x.size(0)]
            tgt_x = X_t_tensor[idx]
            tgt_feat, _ = encoder(tgt_x, alpha)
            
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

            # (C) 總損失 (實驗貳配置：局部對抗權重降為 0.01)
            loss_total = loss_cls + (0.05 * loss_adv_rpli) + (0.01 * loss_local_d) + (alpha * loss_mmd) + (beta * loss_cmmd)
            
            loss_total.backward()
            optimizer_gh.step()
            
            epoch_loss_gh += loss_total.item()
            batch_count += 1

        avg_epoch_loss = epoch_loss_gh / max(1, batch_count)

        # 3.4 偽標籤迭代與特徵擾動
        if epoch > 5 and epoch % 5 == 0:
            encoder.eval(); classifier.eval()
            with torch.no_grad():
                feat_t = encoder(X_t_tensor)[0] # 修正 5
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
                    y_pred_fused = torch.max(classifier(encoder(fused_X_t)[0]), 1)[1] # 修正 6
                    
                    reliable_mask = (y_pred_t == y_pred_fused)
                    Y_t_pseudo[reliable_mask] = y_pred_t[reliable_mask]

        # 3.5 評估真實準確率
        encoder.eval(); classifier.eval()
        with torch.no_grad():
            final_pred = torch.max(classifier(encoder(X_t_tensor)[0]), 1)[1] # 修正 7
            acc = (final_pred == Y_t_true_tensor).float().mean().item()
            pseudo_acc = (Y_t_pseudo == Y_t_true_tensor).float().mean().item()
            
        print(f"S{target_id} Ep{epoch:02d} | 偽標籤Acc: {pseudo_acc:.4f} | 真實Acc: {acc:.4f} | Avg_Loss: {avg_epoch_loss:.4f}")
        
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_epoch_loss, pseudo_acc, acc])

        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'best_acc': max(acc, best_acc)
        }
        
        torch.save(checkpoint, os.path.join(SAVE_DIR, f"latest_checkpoint_S{target_id}.pth"))

        if acc > best_acc:
            best_acc = acc
            torch.save(checkpoint, os.path.join(SAVE_DIR, f"best_model_S{target_id}.pth"))
            
    print(f"S{target_id} 訓練完成！最高準確率: {best_acc:.4f}")
    plot_learning_curves(csv_file_path, SAVE_DIR, target_id)
    print("="*40 + "\n")

if __name__ == "__main__":
    for i in range(32):
        train_rpli_for_subject(i)