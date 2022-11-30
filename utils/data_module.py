import pandas as pd
import numpy as np
import torch
import utils.binvox_rw
import ast
import torch
import os
import itertools

def stack_padding(l):
    return np.column_stack((itertools.zip_longest(*l, fillvalue=0)))

def get_data(data_path='data/binvox_data',label_csv_path='data/preprocessed_df.csv'):
    
    data: list[np.ndarray] = []
    labels = []
    
    df_data = pd.read_csv(label_csv_path)

    
    for i in range(len(df_data)):
        binvox_file = df_data.at[i,'file_number']+'.binvox'
        try:
            with open(os.path.join(data_path,binvox_file), 'rb') as f:
                voxel_data = utils.binvox_rw.read_as_3d_array(f)
            data.append(voxel_data.data.astype(np.float32))
            labels.append(df_data.at[i,'encoded_labels'])
        except:
            continue

    for i in range(len(labels)):
        labels[i] = np.asarray(ast.literal_eval(labels[i]), dtype=np.int8)

    labels = stack_padding(labels)
    data = np.array(data)

    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(labels).float()
        
    # return np.asarray(train_x), np.asarray(test_x), np.asarray(train_y), np.asarray(test_y), np.asarray(val_x), np.asarray(val_y)
    return data, labels

if __name__=='__main__':
    get_data()
