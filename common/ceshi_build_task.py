import pandas as pd

import csv
import os
import random
import glob 
import numpy as np

# df_train = pd.read_csv('data/linux/test_task/train.csv')
# df_val = pd.read_csv('data/linux/test_task/val.csv')
# df_test = pd.read_csv('data/linux/test_task/test.csv')

df_train = pd.read_csv('data/linux/triton/7_1_1/train.csv')
df_val = pd.read_csv('data/linux/triton/7_1_1/val.csv')
df_test = pd.read_csv('data/linux/triton/7_1_1/test.csv')






def charge(df_train, df_val, df_test):
    all_class = ["normal", "acute CSCR", "acute RAO", "acute RVO", "acute VKH",  "dAMD", "macular-off RRD", "mCNV", "MTM", "nAMD", "nPDR", "PCV", "PDR", "PIC", "PVRL", "RP"]
    for class_name in all_class:
        train_patient= df_train[df_train['Class'] == class_name]['Patient']
        tarin_id = list(np.unique(list(train_patient)))
        
        val_patient= df_val[df_val['Class'] == class_name]['Patient']
        val_id = list(np.unique(list(val_patient)))
        
        test_patient= df_test[df_test['Class'] == class_name]['Patient']
        test_id = list(np.unique(list(test_patient)))
        

        id = tarin_id + val_id + test_id
        
        set_id = set(id)
        
        if len(id)==len(set_id):
            print('列表里的元素互不重复！')
        else:
            print('列表里有重复的元素！')  
            print(class_name)
            print(tarin_id)
            print(val_id)
            print(test_id)
        

    

    
    
    

