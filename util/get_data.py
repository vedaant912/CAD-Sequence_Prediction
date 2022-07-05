import pandas as pd
import numpy as np
import torch
from pyvox.models import Vox
from pyvox.writer import VoxWriter
from pyvox.parser import VoxParser
import ast
import torch

def get_data(data_path,label_path,proportion):
    
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    
    data = pd.read_csv(label_path)
    data.drop('Unnamed: 0', inplace=True, axis=1)

    # data = data.sample(frac=proportion/100).reset_index(drop=True)

    data = data[0:1000]
    train_data = data[0:800]
    test_data = data[800:1000]
    
    for i in range(len(train_data)):
        i = int(i)
        m2 = VoxParser(data_path + str(i)+'.vox').parse()
        img = m2.to_dense()
        train_x.append(img)
        train_y.append(data.at[i,'encoded_label'])
        
    for i in range(len(test_data)):
        i = int(i)
        m2 = VoxParser(data_path + str(i)+'.vox').parse()
        img = m2.to_dense()
        test_x.append(img)
        test_y.append(data.at[i,'encoded_label'])
	
    temp = []
    for i in train_y:
    	temp.append(ast.literal_eval(i))
    train_y = np.array(temp)

    temp = []
    for i in test_y:
    	temp.append(ast.literal_eval(i))
    test_y = np.array(temp)

    test_range = int(0.5*(len(test_x))) 
    val_x = test_x[test_range:]
    val_y = test_y[test_range:]
    test_x = test_x[:test_range]
    test_y = test_y[:test_range]

    X_train, X_test, y_train, y_test, X_val, y_val = np.asarray(train_x), np.asarray(test_x), np.asarray(train_y), np.asarray(test_y), np.asarray(val_x), np.asarray(val_y)

    train_x = torch.from_numpy(X_train).float()
    train_y = torch.from_numpy(y_train).long()
    test_x = torch.from_numpy(X_test).float()
    test_y = torch.from_numpy(y_test).long()
    val_x = torch.from_numpy(X_val).float()
    val_y = torch.from_numpy(y_val).long()
        
    # return np.asarray(train_x), np.asarray(test_x), np.asarray(train_y), np.asarray(test_y), np.asarray(val_x), np.asarray(val_y)
    return train_x, train_y, test_x, test_y, val_x, val_y