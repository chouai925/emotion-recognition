import numpy as np
import torch.utils.data as Data
import torch
import os
import pickle
from sklearn import preprocessing

def load_data(test_id, session, BATCH_SIZE):
    # 設定 DEAP 資料路徑
    path = r'C:\chou\data_preprocessed_python'
    
    feature_list = []
    label_list = []
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    
    # DEAP 共有 32 個受試者檔 s01.dat ~ s32.dat
    filenames = [f's{i+1:02d}.dat' for i in range(32)]
    
    for file in filenames:
        file_path = os.path.join(path, file)
        with open(file_path, 'rb') as f:
            # DEAP 資料結構: data(40, 40, 8064), labels(40, 4)
            data_dict = pickle.load(f, encoding='latin1')
            
        # 這裡示範將原始訊號 reshape 為二維以符合原程式流程
        # 注意：若模型需要特定特徵(如DE)，建議在此處先進行特徵提取
        feature = data_dict['data'].reshape(40, -1) 
        raw_label = data_dict['labels'][:, 0]  # 取 Valence (維度 0)
        
        feature_list.append(min_max_scaler.fit_transform(feature).astype('float32'))
        
        # DEAP 二分類：Valence > 5 為 1 (Positive), <= 5 為 0 (Negative)
        one_hot_label_mat = np.zeros((len(raw_label), 2)) 
        for i in range(len(raw_label)):
            if raw_label[i] > 5:
                one_hot_label_mat[i, :] = [0, 1]
            else:
                one_hot_label_mat[i, :] = [1, 0]
        label_list.append(one_hot_label_mat.astype('float32'))
    
    # 留一受試者法 (Leave-One-Subject-Out)
    target_feature, target_label = feature_list[test_id], label_list[test_id]
    
    temp_features = feature_list[:test_id] + feature_list[test_id+1:]
    temp_labels = label_list[:test_id] + label_list[test_id+1:]
    
    source_data = np.vstack(temp_features)
    source_label = np.vstack(temp_labels)

    torch_dataset_source = Data.TensorDataset(torch.from_numpy(source_data), torch.from_numpy(source_label))
    torch_dataset_target_test = Data.TensorDataset(torch.from_numpy(target_feature), torch.from_numpy(target_label))

    source_loader = Data.DataLoader(dataset=torch_dataset_source, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    target_train_loader = Data.DataLoader(dataset=torch_dataset_target_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    target_test_loader = Data.DataLoader(dataset=torch_dataset_target_test, batch_size=target_feature.shape[0], shuffle=False)
    
    return source_loader, target_train_loader, target_test_loader