import numpy as np
import pandas as pd
import random
import os

# 小函数
# 生成label_key, key = class_name, value=id
def classname_with_classID(pdframe):
    all_class_name = np.unique(pdframe["Class"])
    
    dict_labels = {}
    ls_labels = []
    for label, class_name in enumerate(all_class_name):
        dict_labels[class_name] = label
        ls_labels.append(label)

    return dict_labels, ls_labels

# 小函数，增加label列
def add_classID(pdframe, dict_labels):

    label_list = []
    for class_name in pdframe['Class']:
        label_list.append(dict_labels[class_name])
    
    pdframe['Label'] = label_list
    return pdframe

# 随机划分类别
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



# one/others
def originze_df(project_path):

    df_triton_train = pd.read_csv(os.path.join(project_path, 'data/linux/triton/7_1_1/train.csv'))
    df_triton_train_pvrl = df_triton_train[df_triton_train['Class'] == 'PVRL']


    # PVRL, 验证集的识别难度比测试集高，所以调换以下
    df_triton_val = pd.read_csv(os.path.join(project_path,'data/linux/triton/7_1_1/test.csv'))
    df_triton_val_pvrl = df_triton_val[df_triton_val['Class'] == 'PVRL']

    df_triton_test = pd.read_csv(os.path.join(project_path,'data/linux/triton/7_1_1/val.csv'))
    df_triton_test_pvrl = df_triton_test[df_triton_test['Class'] == 'PVRL']


    # add heidelberg
    df_hei = pd.read_csv(os.path.join(project_path,'data/linux/heidelberg/data.csv'))
    df_hei_pvrl = df_hei[df_hei['Class'] == 'PVRL'].copy()

    # 修改patientID
    triton_pvrl_patient_idx = np.unique(list(df_triton_train_pvrl['Patient']) + list(df_triton_val_pvrl['Patient']) + list(df_triton_test_pvrl['Patient']))
    max_id = np.max(triton_pvrl_patient_idx)
    df_hei_pvrl['Patient'] =  df_hei_pvrl['Patient'] + max_id

    df_hei_pvrl.to_csv(os.path.join(project_path,'tmp.csv'))

    hei_pvrl_patient_idx = np.unique(list(df_hei_pvrl['Patient']))

    df_hei_val_pvrl = df_hei_pvrl[df_hei_pvrl['Patient'] == hei_pvrl_patient_idx[-2]]
    df_hei_test_pvrl = df_hei_pvrl[df_hei_pvrl['Patient'] ==hei_pvrl_patient_idx[-1]]

    df_hei_train_pvrl = df_hei_pvrl.drop(df_hei_val_pvrl.index).drop(df_hei_test_pvrl.index)

    df_train = pd.concat([df_triton_train, df_hei_train_pvrl],ignore_index=True)
    df_val = pd.concat([df_triton_val, df_hei_val_pvrl],ignore_index=True)
    df_test = pd.concat([df_triton_test, df_hei_test_pvrl],ignore_index=True)


    classid_dict, classid_ls = classname_with_classID(df_train)

    # 增加label列
    df_train = add_classID(df_train, classid_dict)
    df_val = add_classID(df_val, classid_dict)
    df_test = add_classID(df_test, classid_dict)


    # 方案一：
    # train_classes:common

    # 方案二
    # train_classes: (1-12), val_classes:common(13~15), test_class:rare+others(common 1~15)

    # rare未出现在元训练，将train_data和val_data里的rare加到test_data,可以增加病人数量，更牛逼
    # df_test = pd.concat([df_test, df_train['PVRL'], df_val['PVRL']],ignore_index=True)
    df_test_ls = [df_train, df_val, df_test]

    # 先drop,后面不drop,提出创新点，增加模块或机制，提高含PVRL类的任务的权重--》
    # 提高某一类别的分类准去率
    df_train = df_train.drop(df_train[df_train['Class'] == 'PVRL'].index)
    df_val = df_val.drop(df_val[df_val['Class'] == 'PVRL'].index)


    # 划分train_classes和val_classes
    class_names = np.unique(list(df_train['Class']))
    train_classes, val_classes = data_split(list(class_names), 0.8,shuffle=False)

    train_class_id = []
    val_class_id = []
    # for i in val_classes:
    #     df_train.drop(i)
    #     val_class_id.append(classid_dict[i])


    # for i in train_classes:
    #     df_val.drop(i)
    #     train_class_id.append(classid_dict[i])


    for id, class_name in enumerate(classid_dict):
        if class_name in val_classes:
            val_class_id.append(id)

            # 可以充分利用数据进行元训练
            # 但是考虑到，元训练的数据，基本都是common类，似乎用太多common类对结果并不会有太大提升
            df_val = pd.concat([df_val, df_train[df_train['Class'] == class_name]],ignore_index=True)

            df_train.drop(df_train[df_train['Class'] == class_name].index)

        if class_name in  train_classes:
            train_class_id.append(id)

            # 可以充分利用数据进行元训练
            # 但是考虑到，元训练的数据，基本都是common类，似乎用太多common类对结果并不
            df_train = pd.concat([df_train, df_val[df_val['Class'] == class_name]],ignore_index=True)

            df_val.drop(df_val[df_val['Class'] == class_name].index)





    # 方案三:
    # train_classes:rare + others(common:1-12), val_classes:rare+others(common:13~15), test_class:rare+others(common 1~15)   

    return df_train, df_val, df_test_ls, train_class_id, val_class_id, classid_ls

# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     modellrnew = modellr * (0.1 ** (epoch // 50))
#     print("lr:", modellrnew)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = modellrnew


# originze_df_sub, 增加label列
# 四分类或二分类
def originze_df_sub(project_path):
    label_dict = {'normal':0, 'PIC':1, 'PVRL':2, 'RP':3}

    df_train = pd.read_csv(os.path.join(project_path, 'data/linux/all_device/6_2_2_10/train.csv'))
    df_for_train = getdf(df_train, mode='train')

    df_val = pd.read_csv(os.path.join(project_path,'data/linux/all_device/6_2_2_10/val.csv'))
    df_for_val = getdf(df_val, mode='val')

    df_test = pd.read_csv(os.path.join(project_path,'data/linux/all_device/6_2_2_10/test.csv'))
    df_for_test =  getdf(df_test, mode='test')

    train_class_id = [0,1,2,3]
    val_class_id = [0,1,2,3]
    test_class_id = [0,1,2,3]

    # s,q test
    df_for_test_ls = [df_for_train, df_for_val, df_for_test]

    df_for_train.to_csv(os.path.join(project_path, 'data/linux/all_device/6_2_2_10/four_classes/train.csv'))
    df_for_val.to_csv(os.path.join(project_path, 'data/linux/all_device/6_2_2_10/four_classes/val.csv'))
    df_for_test.to_csv(os.path.join(project_path, 'data/linux/all_device/6_2_2_10/four_classes/test.csv'))

    return df_for_train, df_for_val, df_for_test_ls, train_class_id, val_class_id, test_class_id

# 小函数 common
def getdf(dframe, common, mode='train'):

    label_dict = {'PVRL':0, 'PIC':1, 'RP':2, common:3}


    df_for_train = pd.DataFrame()
    add_classID_sub(dframe, label_dict)
    df_pvrl = dframe[dframe['Class'] == 'PVRL']
    df_pic = dframe[dframe['Class'] == 'PIC']
    df_rp = dframe[dframe['Class'] == 'RP']
    df_common = dframe[dframe['Class'] == common]

    # common只选择10个病人，每个病人20张, 8个训练，2个测试，2个验证
    patient_ls = np.unique(list(df_common['Patient']))
    if mode == 'train':
        chose_patient_num = 8
    elif mode == 'val':
        chose_patient_num = 2
    else:
        chose_patient_num = 2

    choose_patien_ls =  random.sample(list(patient_ls), chose_patient_num)

    df_common_choose = df_common[df_common['Patient'].isin(choose_patien_ls)]

    df_for_train = pd.concat([df_pvrl,df_pic,df_rp,df_common_choose],ignore_index=True)

    return df_for_train

# 小函数，增加label列
def add_classID_sub(pdframe, dict_labels):

    label_list = []
    for class_name in pdframe['Class']:
        if class_name in dict_labels:
            label_list.append(dict_labels[class_name])
        else:
            label_list.append('others')

    pdframe['Label'] = label_list
    return pdframe



# 四分类，maml task 20231010
def originze_df_maml_four(project_path, source_classes, target_classes):

    df_train =  pd.read_csv(os.path.join(project_path, 'data/linux/all_device/6_2_2_10/train.csv'))
    df_val = pd.read_csv(os.path.join(project_path, 'data/linux/all_device/6_2_2_10/val.csv'))
    df_test = pd.read_csv(os.path.join(project_path, 'data/linux/all_device/6_2_2_10/test.csv'))

    classes_ls = np.unique(list(df_train['Class']))
    # souce_classes = [item for item in classes_ls if item not in target_classes]


    df_meta_train = pd.DataFrame()
    df_meta_test_S = pd.DataFrame()
    df_meta_test_Q = pd.DataFrame()
    df_final_test = pd.DataFrame()

    for class_name in classes_ls:
        df_class_train = df_train[df_train['Class'] == class_name]
        df_class_val = df_val[df_val['Class'] == class_name]
        df_class_test = df_test[df_test['Class'] == class_name]
        if class_name in source_classes:
            # ignore_index=True, 重新生成索引，防止索引冲突，元训练是只需要source_class
            df_meta_train = pd.concat([df_meta_train, df_class_train, df_class_val], ignore_index=True)

        # 测试时，需要加入13个common类
        df_meta_test_S = pd.concat([df_meta_test_S, df_class_train], ignore_index=True)
        df_meta_test_Q = pd.concat([df_meta_test_Q, df_class_val], ignore_index=True)
        df_final_test = pd.concat([df_final_test, df_class_test], ignore_index=True)

    store_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/linux/meta_tasks/final_test_task/'
    test_d = {}
    # 罕见病和其他每种常见病。13个common，构造了13个任务
    for i in source_classes:
        # final_test_task任务，common只选择10个病人，每个病人20张, 8个训练，2个测试，2个验证
        test_S = getdf(df_meta_test_S, i, mode='train')
        test_Q = getdf(df_meta_test_Q, i, mode='val')
        final_test = getdf(df_final_test,i, mode='test')
        test_d[i] = [test_S, test_Q, final_test]

        test_S.to_csv(store_path + i + '_s.csv')
        test_Q.to_csv(store_path + i + '_q.csv')
        final_test.to_csv(store_path+i + '_f.csv')



    return df_meta_train, test_d


# 四分类，maml task 20241031
def originze_df_four(args):

    others_cl = ["normal", "acute CSCR", "acute RAO", "acute RVO", "acute VKH",  "dAMD", "macular-off RRD", "mCNV", "MTM", "nAMD", "nPDR", "PCV", "PDR"]
    target_cl = ['PIC', 'PVRL',  'RP']
    all_class = others_cl + target_cl

    dict_label = {'PVRL':0, 'PIC':1, 'RP':2, "normal":3, "acute CSCR":4, "acute RAO":4, "acute RVO":4, "acute VKH":4,  "dAMD":4, "macular-off RRD":4, "mCNV":4, "MTM":4, "nAMD":4, "nPDR":4, "PCV":4, "PDR":4}

    test_d = {}
    # 五折 五个测试任务
    for i in range(5):
        test_S = pd.read_csv(os.path.join(args.project_path,f'data/st_oct/origin_data/all_device/8_2/final_test_task/t_{i}_s.csv'))
        # test_Q = pd.read_csv(os.path.join(args.project_path,f'data/st_oct/origin_data/all_device/8_2/final_test_task/t_{i}_q.csv'))
        test_Q = pd.read_csv(os.path.join(args.project_path,f'data/st_oct/origin_data/all_device/8_2/final_test_task/balance/t_{i}_q_balance.csv'))
        # final_test = pd.read_csv(os.path.join(args.project_path,'data/st_oct/origin_data/all_device/8_2/test.csv'))
        final_test = pd.read_csv(os.path.join(args.project_path,'data/st_oct/origin_data/all_device/8_2/final_test_task/balance/test_balance.csv'))
        test_S = add_classID_sub(test_S, dict_label)
        test_Q = add_classID_sub(test_Q, dict_label)
        final_test = add_classID_sub(final_test, dict_label)
        test_d[i] = [test_S, test_Q, final_test]

    # # meta train 里仅包含c1~c13
    # # meta val 里包含c14~16,也包含c1~c13, 但是c1~c13 的class_name 要归为common
    # # 需要将df_train 的c14~c16 放到meta_val里
    # df_train =  pd.read_csv(f'data/st_oct/origin_data/all_device/8_2/final_test_task/t_{i}_s.csv')
    # df_val = pd.read_csv(f'data/st_oct/origin_data/all_device/8_2/final_test_task/t_{i}_q.csv')
    # df_meta_train = pd.DataFrame()
    # df_meta_val = pd.DataFrame()
    # for i in all_class:
    #     df_train_cl  = df_train[df_train['Class'] == i]
    #     if i in target_cl:
    #         df_meta_val = pd.concat([df_meta_val, df_train_cl], ignore_index=True)
    #     else:
    #         df_meta_train = pd.concat([df_meta_train, df_train_cl], ignore_index=True)

    # df_meta_val = pd.concat([df_meta_val, df_val], ignore_index=True)


    # # 将meta_val的class重新归为5类
    # class_list = []
    # for class_name in df_meta_val['Class']:
    #     if class_name == "normal" or class_name in target_cl:
    #         class_list.append(class_name)
    #     else:
    #         class_list.append('common')
    # df_meta_val['Class'] = class_list

    # df_meta_train.to_csv(os.path.join(args.project_path,'data/st_oct/origin_data/all_device/8_2/meta_task/meta_train.csv'))
    # df_meta_val.to_csv(os.path.join(args.project_path,'data/st_oct/origin_data/all_device/8_2/meta_task/meta_val.csv'))

    # 直接读取
    df_meta_train = pd.read_csv(os.path.join(args.project_path,'data/st_oct/origin_data/all_device/8_2/meta_task/meta_train.csv'))
    df_meta_val = pd.read_csv(os.path.join(args.project_path,'data/st_oct/origin_data/all_device/8_2/meta_task/meta_val.csv'))


    return df_meta_train, df_meta_val, test_d