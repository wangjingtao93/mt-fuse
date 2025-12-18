import pandas as pd

def get_data_list(dframe):
    label_dict = {'CNV':0, 'DME':1, 'DRUSEN':2, 'NORMAL':3}

    Label =[]
    for index, row in dframe.iterrows():
        Label.append(label_dict[row['Class']])
    
    dframe['Label'] = Label

    return dframe[["Image_path", "Label"]].values.tolist()

def zk_data():
    df_train_data = pd.read_csv('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/pre_train/zk/train.csv')
    df_val_data = pd.read_csv('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/pre_train/zk/test.csv')
    df_test_data = pd.read_csv('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/pre_train/zk/test.csv')

    train_ls = get_data_list(df_train_data)
    val_ls = get_data_list(df_val_data)
    test_ls = get_data_list(df_test_data)

    return [[train_ls], [val_ls], [test_ls]],[]

