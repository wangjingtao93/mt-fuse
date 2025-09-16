import numpy as np
import os
import pandas as pd
import random

import pandas as pd
import numpy as np
data = {"x": 2**np.arange(5),"y": 3**np.arange(5),"z": np.array([45, 98, 24, 11, 64])}

index = ["a", "b", "c", "d", "e"]

df = pd.DataFrame(data=data, index=index)

print(df)


df_1 = df[df['x'] %4 == 0]


print(df_1)

df_1.loc[df_1.index, 'z'] = 0

print(df_1)

def change_hei_id(df_ori, df_tar):
    patient_len = len(np.unique(list(df_ori['Patient'])))
    
    tar_patient_index = list(df_tar['Patient'])
    idx_new = []
    for i in tar_patient_index:
        idx_new.append(i + patient_len)
    
    df_tar.loc[df_tar.index,'Patient'] = idx_new
    # df_tar.loc[:,'Patient'] = idx_new

    return df_tar


def test():
    project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication'
    
    df_triton_train = pd.read_csv(os.path.join(project_path, 'data/linux/triton/6_1_1/train.csv'))
    df_triton_train_pvrl = df_triton_train[df_triton_train['Class'] == 'PVRL']
          
    df_triton_val = pd.read_csv(os.path.join(project_path,'data/linux/triton/6_1_1/val.csv'))
    df_triton_val_pvrl = df_triton_val[df_triton_val['Class'] == 'PVRL']
    
    df_triton_test = pd.read_csv(os.path.join(project_path,'data/linux/triton/6_1_1/test.csv'))
    df_triton_test_pvrl = df_triton_test[df_triton_test['Class'] == 'PVRL']
    
    
    # add heidelberg
    df_hei = pd.read_csv(os.path.join(project_path,'data/linux/heidelberg/data.csv'))   
    
    df_hei_pvrl = df_hei[df_hei['Class'] == 'PVRL']
    
    df_hei_val_pvrl = df_hei_pvrl[df_hei_pvrl['Patient'] == 6]
    df_hei_test_pvrl = df_hei_pvrl[df_hei_pvrl['Patient'] == 7]
    
    df_hei_train_pvrl = df_hei_pvrl.drop(df_hei_val_pvrl.index).drop(df_hei_test_pvrl.index)
    
    
    df_hei_val_pvrl = change_hei_id(df_triton_val_pvrl, df_hei_val_pvrl) # 有病吧，总是警告
    df_hei_test_pvrl = change_hei_id(df_triton_test_pvrl, df_hei_test_pvrl)
    df_hei_train_pvrl = change_hei_id(df_triton_train_pvrl, df_hei_train_pvrl)
    
    # df_train = pd.concat([df_triton_train, df_hei_train_pvrl],ignore_index=True)  
    # df_val = pd.concat([df_triton_val, df_hei_val_pvrl],ignore_index=True) 
    # df_test = pd.concat([df_triton_test, df_hei_test_pvrl],ignore_index=True)
    
# test()