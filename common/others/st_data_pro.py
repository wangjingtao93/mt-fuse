import pandas as pd
import os
def get_data_list(dframe):
    label_dict = {'CNV':0, 'DME':1, 'DRUSEN':2, 'NORMAL':3}
    label_dict = {"normal":0, "acute CSCR":1, "acute RAO":2, "acute RVO":3, "acute VKH":4,  "dAMD":5, "macular-off RRD":6, "mCNV":7, "MTM":8, "nAMD":9, "nPDR":10, "PCV":11, "PDR":12, "PIC":13, "PVRL":14, "RP":15}

    Label =[]
    for index, row in dframe.iterrows():
        Label.append(label_dict[row['Class']])
    
    dframe['Label'] = Label

    return dframe[["Image_path", "Label"]].values.tolist()


def st_sub_data():
    relative_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/'
    df_train_data = pd.read_csv(os.path.join(relative_path, 'data/linux/all_device/6_2_2_10/train.csv'))
    df_val_data = pd.read_csv(os.path.join(relative_path,'data/linux/all_device/6_2_2_10/test.csv'))
    df_test_data = pd.read_csv(os.path.join(relative_path,'data/linux/all_device/6_2_2_10/test.csv'))

    train_ls = get_data_list(df_train_data)
    val_ls = get_data_list(df_val_data)
    test_ls = get_data_list(df_test_data)

    return [[train_ls], [val_ls], [test_ls]],[]