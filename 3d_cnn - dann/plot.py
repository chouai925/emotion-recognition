def plot_global_topomap(model_path):
    """ 讀取 3D-CNN 權重並繪製 7x5 腦區重要性熱圖 """
    state_dict = torch.load(model_path)
    # 取得 Global Path 的第一層 3D 卷積權重: (out_channels, in_channels, H, W, T)
    # 根據你的 model.py，對應名稱應為 eeg_branch.global_path.conv1.weight
    weights = state_dict['eeg_branch.global_path.conv1.weight'].cpu().numpy()
    
    # 對輸出通道 (kernels)、輸入通道 (1) 與時間維度 (T) 取絕對值平均
    # 剩下的維度就是 (7, 5) 的空間權重
    spatial_map = np.mean(np.abs(weights), axis=(0, 1, 4))
    
    plt.figure(figsize=(6, 7))
    sns.heatmap(spatial_map, annot=True, cmap='YlOrRd', fmt=".4f")
    plt.title("Spatial Importance Map (7x5 Topology)")
    plt.savefig(os.path.join(OUT_DIR, 'topographic_heatmap.png'))
    plt.close()