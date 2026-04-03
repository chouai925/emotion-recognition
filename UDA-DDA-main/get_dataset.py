import numpy as np
import torch
import os
import pickle
from sklearn import preprocessing

def get_dataset(test_id, session):
    # 修改為你的實際路徑
    path = r'C:\chou\data_preprocessed_python'
    
    feature_list = []
    label_list = []
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    
    # DEAP 32 位受試者檔案清單
    filenames = [f's{i+1:02d}.dat' for i in range(32)]
    
    for file in filenames:
        file_path = os.path.join(path, file)
        with open(file_path, 'rb') as f:
            # DEAP 格式: data(40, 40, 8064), labels(40, 4)
            data_dict = pickle.load(f, encoding='latin1')
            
        feature = data_dict['data'].reshape(40, -1) # 重塑為模型可接受的維度
        label = data_dict['labels'][:, 0] # 使用 Valence 標籤
        
        feature_list.append(min_max_scaler.fit_transform(feature).astype('float32'))
        
        # 二分類處理：Valence > 5 為 1，反之為 0
        one_hot_label_mat = np.zeros((len(label), 2))
        for i in range(len(label)):
            if label[i] > 5:
                one_hot_label_mat[i, :] = [0, 1]
            else:
                one_hot_label_mat[i, :] = [1, 0]
        label_list.append(one_hot_label_mat.astype('float32'))

    target_feature, target_label = feature_list[test_id], label_list[test_id]
    
    # 移除測試集，合併其餘為來源集 (Source Set)
    temp_features = feature_list[:test_id] + feature_list[test_id+1:]
    temp_labels = label_list[:test_id] + label_list[test_id+1:]
    
    source_feature, source_label = np.vstack(temp_features), np.vstack(temp_labels)

    target_set = {'feature': target_feature, 'label': target_label}
    source_set = {'feature': source_feature, 'label': source_label}
    return target_set, source_set