import pandas as pd 
import numpy as np
import os

# 标签变换
def classname_to_classID(pdframe):
    all_class_name = np.unique(pdframe["Class"])
    
    dict_labels = {}
    for label, class_name in enumerate(all_class_name):
        dict_labels[class_name] = label
    
    
    # # 16个标签。16类
    # label_list = []
    # for class_name in pdframe['Class']:
    #     label_list.append(dict_labels[class_name])
        
        
    # 变成两类：rare, others
    label_list = []
    for class_name in pdframe['Class']:
        if class_name == 'PIC': # PIC/others
            label_list.append(0)
        else:
            label_list.append(1)
        
    return label_list

# 从val.csv和test.csv选取部分scan
# common 类和 PIC RP 各选3个病人，每个病人20张scan,PVRL 选一个病人60张，共960张scan
def choose_scans_1(pdframe):
    all_class_name = np.unique(pdframe["Class"])
    
    index_lst = []
    for class_name in all_class_name:
        
        df_class = pdframe[pdframe['Class'] == class_name]# 索引值不会变化
        patient_arr = np.unique(df_class['Patient'])
        # sample_patient = np.random_choice(patient_arr, 5)
        
        if class_name == 'PVRL':
            sample_patient = patient_arr[:1]
            for patient_name in sample_patient:
                df_patient_index = df_class[df_class['Patient'] == patient_name].index.tolist()
                index_lst += df_patient_index[:60]
                
        # elif class_name == 'PIC':
        #     sample_patient = patient_arr[:1]
        #     for patient_name in sample_patient:
        #         df_patient_index = df_class[df_class['Patient'] == patient_name].index.tolist()
        #         index_lst += df_patient_index[:5]
        # elif class_name == 'RP':
        #     sample_patient = patient_arr[:5]
        #     for patient_name in sample_patient:
        #         df_patient_index = df_class[df_class['Patient'] == patient_name].index.tolist()
        #         index_lst += df_patient_index[:5]
        else:
            if len(patient_arr) < 3:
                print('cuole')
                exit(-1)
            sample_patient = patient_arr[:3]
            for patient_name in sample_patient:
                df_patient_index = df_class[df_class['Patient'] == patient_name].index.tolist()
                
                if len(df_patient_index) < 20:
                    print('cuole again', class_name, ' and ', patient_name)
                
                index_lst += df_patient_index[:20]
                
    df_choice = pdframe.iloc[index_lst]
    
    return df_choice
           
            
# 从val.csv和test.csv选取部分scan
# rare类选择100张scans,平均分配到每个病人，common类一共选择100张，平均分配到每个类和每个病人，一共200个scans做测试
def choose_scans_2(pdframe):
    scan_num_sum = 100
    all_class_name = np.unique(pdframe["Class"])
    index_lst = []
    for class_name in all_class_name:
        df_class = pdframe[pdframe['Class'] == class_name]# 索引值不会变化
        patient_arr = np.unique(df_class['Patient'])

        yu = 0 
        if class_name == 'PIC': # PIC / others
            # per_patient_scan = int(scan_num_sum // len(patient_arr))
            
            # yu = scan_num_sum % len(patient_arr)
            
            index_lst += df_class.index.to_list()[:100]
        
        else:
            per_patient_scan = int((scan_num_sum / 15) // len(patient_arr))
            
            yu = (scan_num_sum / 15) % len(patient_arr)
            
        
        if per_patient_scan == 0:
            tmp = int(scan_num_sum / 15)
            patient_arr = patient_arr[:tmp]
        
        i = 0
        for patient_name in patient_arr:
            df_patient = df_class[df_class['Patient'] == patient_name]# 索引值不会变化
            df_patient_index = df_patient.index.to_list()
            
           
            if i < yu:    
                index_lst += df_patient_index[:per_patient_scan+1]
            else:
                index_lst += df_patient_index[:per_patient_scan]               
            i += 1
    
    pdframe.iloc[index_lst].to_csv('tmp.csv')   
    return pdframe.iloc[index_lst]
                
             


# 选择数据，分别做验证和测试
work_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication'
def data_choose():
    # df_all_dataset = pd.read_csv(os.path.join(work_path + 'data/linux/triton/data.csv'))    
    df_train_data = pd.read_csv(os.path.join(work_path, 'data/linux/triton/6_1_1/train.csv'))
    df_val_data = pd.read_csv(os.path.join(work_path, 'data/linux/triton/6_1_1/val.csv')) 
    df_test_data = pd.read_csv(os.path.join(work_path, 'data/linux/triton/6_1_1/test.csv'))
    
    
    df_val_data = choose_scans_2(df_val_data)
    df_test_data = choose_scans_2(df_test_data)
        
   
    label_lst= classname_to_classID(df_train_data)
    df_train_data['Label'] = label_lst
      
    label_lst= classname_to_classID(df_val_data)
    df_val_data['Label'] = label_lst 
    
    label_lst= classname_to_classID(df_test_data)
    df_test_data['Label'] = label_lst
    
    return df_train_data, df_val_data, df_test_data