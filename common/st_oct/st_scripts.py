import random
import numpy as np

# 按比例随机划分list
def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

# 小涵数
# 莫当真，将shot和quer平均分配到每个patient上
def choose_sq_scans(patient_ls, total_scans, df_class):
    per_patient_scan = int(total_scans // len(patient_ls))
    yu = total_scans % len(patient_ls)
    
    
    if per_patient_scan == 0:
        patient_ls = random.sample(patient_ls, total_scans)
    
    i = 0   
    index_lst = []
    for patient_name in patient_ls:
        df_patient = df_class[df_class['Patient'] == patient_name]# 索引值不会变化
        df_patient_index = df_patient.index.to_list()
        
        
        # 健壮性，先不加
        # if per_patient_scan + 1 > len(df_patient_index):
        #     index_lst += df_patient_index
        #     continue
        
        # 这个余数很巧妙
        if i < yu:
            # index_lst += df_patient_index[:per_patient_scan+1]
            index_lst += random.sample(df_patient_index, per_patient_scan+1)
            
        else:
            # index_lst += df_patient_index[:per_patient_scan]         
            index_lst += random.sample(df_patient_index, per_patient_scan)
        i += 1
        
    return index_lst
        
        
    

# 按病人选scans
def choose_scans(pdframe,shot, query, ls_tmp):
    patient_arr = np.unique(pdframe['Patient'])
    radio = shot/(shot + query)
    patient_support_ls, patient_query_ls = data_split(list(patient_arr), radio, shuffle=True)
    
    index_ls = pdframe.index.to_list()
        
    s_index_ls = choose_sq_scans(patient_support_ls, shot, pdframe)
    q_index_ls = choose_sq_scans(patient_query_ls, query, pdframe)
    
    return s_index_ls, q_index_ls


def choose_scans_test(pdframe, scan_num_sum, store_name):
    all_class_name = np.unique(pdframe["Class"])
    index_lst = []
    for class_name in all_class_name:
        df_class = pdframe[pdframe['Class'] == class_name]# 索引值不会变化
        patient_ls = list(np.unique(df_class['Patient']))

        yu = 0 
        if class_name == 'PVRL': # PIC / others
            per_patient_scan = int(scan_num_sum // len(patient_ls))            
            yu = scan_num_sum % len(patient_ls)
            
            if per_patient_scan == 0:
                patient_ls = random.sample(patient_ls, scan_num_sum)
                    
        else:
            per_patient_scan = int((scan_num_sum / 15) // len(patient_ls))
            
            yu = (scan_num_sum / 15) % len(patient_ls)
            
            # 因为scan_num_sum是大于15的
            if per_patient_scan == 0:
                patient_ls = random.sample(patient_ls,  int((scan_num_sum / 15)))
             
        i = 0
        for patient_name in patient_ls:
            df_patient = df_class[df_class['Patient'] == patient_name]# 索引值不会变化
            df_patient_index = df_patient.index.to_list()
            
            # 健壮性
            if per_patient_scan + 1 > len(df_patient_index):
                index_lst += df_patient_index
                continue
            
            if i < yu:
                # index_lst += df_patient_index[:per_patient_scan+1]
                index_lst += random.sample(df_patient_index, per_patient_scan+1)
                
            else:
                # index_lst += df_patient_index[:per_patient_scan]         
                index_lst += random.sample(df_patient_index, per_patient_scan)           
            i += 1
    
    index_lst.sort() # 思考下顺序要不要sort. 答：没有影响，封装daloader时，参数shuffer可以控制
    pdframe.iloc[index_lst].to_csv('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/' + store_name)
    return index_lst
    