'''
just for raw_eye_disease
'''

import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')
import yaml
import os

""" Generate commands for test. """

def gen_args():
    project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication'
    relative_path = project_path.replace('pycharm_remote/', 'pycharm_remote/result/')
    save_path = os.path.join(relative_path, 'result/result_thyroid_fuse_20250817' )
    # save_path = os.path.join(relative_path, 'result/tmp/' )
    gpu = 1
    data_type_ls = {0:'thyroid_4x', 1:'thyroid_10x', 2:'multi_centers_ffpe_4x', 3:'multi_centers_ffpe_10x', 4:'multi_centers_bf_4x',
                    5:'multi_centers_bf_10x', 6:'camely_16_4x', 7:'camely_16_10x', 8:'camely_17_4x', 9:'camely_17_10x',
                    10:'camely_all_4x', 11:'camely_all_10x', 12:'crc-4x', 13:'crc-10x', 14:'background_10x', 15:'oct',
                    16:'mini_imagenet', 17:'thyroid_fuse_micro',18:'thyroid_fuse_macro', 19:'thyroid_fuse_ipbs',
                    20:'thyroid_fuse_itcs'}
    datatype = data_type_ls[17]

    algs = {0:'dl', 1:'pretrain', 2:'maml', 3:'imaml', 4:'reptile', 5:'predict/dl', 6:'predict/imaml', 7:'predict/maml',
            8:'predict/reptile',9:'meta_test/maml'}
    alg = algs[5]

    model_names = {0:'convnet_4', 1:'alexnet', 2:'squeezenet1_0', 3:'squeezenet1_1', 4:'densenet121', 5:'densenet169', 6:'densenet201',
                   7:'densenet201', 8:'densenet161',9:'vgg11', 10:'vgg11_bn', 11:'vgg13', 12:'vgg13_bn', 13:'vgg16',
                   14:'vgg16_bn',15:'vgg19', 16:'resnet18', 17:'resnet34', 18:'resnet50',
                   19:'resnet101',20:'resnet152', 22:'vit_base_patch16_224', 23:'vit_base_patch16_224_depth_6',
                   24:'vit_base_patch16_224_depth_3', 25:'vit_tiny_patch16_224', 26:'vit_small_patch16_224', 27:'mt_tiny_model_lymph',
                   29:'mtb', 30:'mtb_res', 31:'mtb_6b_mfc', 32:'mt_small_model_lymph', 33:'retfound', 34: 'conv_84', 35: 'meta_found',
                   36:'mt_fuse_model'}
    # 使用了 1 16 17 18 21 22 26 32
    net = model_names[2]

    ablation_ls = ['', 'ablation/fuse_c', 'ablation/meta_fuse_n','ablation/meta_fuse_c', 'ablation/meta_n_fuse_n']
    ablation_prfiex = ablation_ls[0]

    with open('configs.yaml', 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

    args_dict = {}

    # base settings
    args_dict['project_path'] = project_path
    args_dict['gpu'] = [gpu]
    args_dict['save_path'] = save_path
    args_dict['ablation'] = ablation_prfiex
    # base settings data
    args_dict['datatype'] = datatype
    args_dict['resize'] = 224

    # net
    args_dict['alg'] = alg
    args_dict['net'] = net
    args_dict['num_classes'] = 2
    # net load
    args_dict['is_load_imagenet'] = True
    args_dict['is_meta_load_imagenet'] = True
    args_dict['load'] = ''
    args_dict['is_lock_notmeta'] = True
    # net met_fuse_model
    args_dict['is_fuse'] = True
    if net != model_names[36]:
        args_dict['is_fuse'] = True
    #net mt
    args_dict['trans_depth'] = 6

    # predict
    args_dict['isheatmap'] = True

    # dl
    args_dict['n_epoch'] = result[alg]['n_epoch'] # 用于dl train
    if args_dict['is_fuse'] and args_dict['net'] == model_names[36]:
        args_dict['batch_size_train'] = 64
    else:
        args_dict['batch_size_train'] = 128
    args_dict['batch_size_val'] = 16
    args_dict['batch_size_test'] = 16
    args_dict['is_save_val_net'] = True # 最好不用用字符串，尤其是'false', 都会当做true


    # meta
    args_dict['n_meta_epoch'] = 30
    args_dict['meta_size'] = 4
    args_dict['outer_lr'] = result[alg]['outer_lr']
    args_dict['inner_lr'] = result[alg]['inner_lr']
    args_dict['n_train_tasks'] = 50
    args_dict['n_val_tasks'] = 50
    args_dict['n_test_tasks'] = 5
    args_dict['test_meta_size'] = 1
    args_dict['n_way'] = 2
    args_dict['n_inner'] = result[alg]['n_inner']
    args_dict['k_shot'] = 5
    args_dict['k_qry'] = 15
    args_dict['n_inner_meta_test'] = 20
    args_dict['is_meta_test'] = False

    # imaml
    args_dict['version'] = 'GD'
    args_dict['cg_steps'] = 5
    args_dict['outer_opt'] = 'Adam'
    args_dict['lambda'] = 2
    args_dict['lr_sched'] = True


    if 'predict' in args_dict['alg']:
        args_dict = set_predict_load(args_dict)



    args_dict['description_name'] = 'n_train_tasks_' + str(args_dict['n_train_tasks'])

    return args_dict



def set_predict_load(args_dict):
    yml_file = f'config/load_for_predict/{args_dict["datatype"]}.yaml'
    with open(yml_file, 'r', encoding='utf-8') as f:
        yaml_res = yaml.load(f.read(), Loader=yaml.FullLoader)

    args_dict['load'] = yaml_res[args_dict['alg']][args_dict['net']]['load_path']

    args_dict['load']  = os.path.join('/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result',args_dict['load'])

    args_dict['meta_epoch_for_predict'] = yaml_res[args_dict['alg']][args_dict['net']]['meta_epoch_for_predict']

    check_error(args_dict)

    return args_dict

def set_for_thyroid(args_dict):


    return args_dict

def run_command(args_dict):

    command_str = ''
    for key, values in args_dict.items():
        command_str += ' --' + key + '=' + str(values)

    the_command = 'python ../main.py --is_run_command=True' + command_str

    os.system(the_command)

def check_error(args_dict):
    if args_dict['load'] == '':
        raise ValueError('predict mode load path must been exist')
    load_values = args_dict['load'].split('/')
    # predict, predict load path check
    datatpye = load_values[10]

    thyroid_ls = ['thyroid_4x', 'thyroid_10x', 'multi_centers_ffpe_4x','multi_centers_ffpe_10x','multi_centers_bf_4x','multi_centers_bf_10x', 'camely_16_4x', 'camely_16_10x', 'camely_17_4x', 'camely_17_10x', 'camely_all_4x', 'camely_all_10x', 'crc-4x', 'crc-10x', 'background_10x']

    if 'thyroid_fuse' in args_dict['datatype']:
        if args_dict['net'] == 'mt_fuse_model':
            if args_dict['alg'] == 'predict/dl' and not args_dict['is_fuse']:
                raise ValueError('mt_fuse_model in predict/dl must set is_fuse=True')
    
    elif  args_dict['datatype'] in thyroid_ls:
        if args_dict['datatype'].split('_')[-1] != datatpye.split('_')[-1]:
                raise ValueError('Resolution is not correspond')

        if args_dict['datatype'] != datatpye:
            raise ValueError(f'datatype is not correspond and {datatpye}')

        net = load_values[12]
        if args_dict['net'] != net:
            raise ValueError(f'net is not correspond and {net}')

    # meta
    alg = load_values[11]
    if args_dict['alg'].split('/')[-1] != alg:
        raise ValueError('meta load is not correspond')

    if 'dl' in args_dict['alg']:
        if args_dict['meta_epoch_for_predict'] != 0:
            raise ValueError('predict/dl meta_epoch_for_predict must be 0')

    # # 元学习算法，args.is_load_imagenet必须为false
    # if args_dict['alg'] == 'maml' or args_dict['alg'] == 'imaml' or args_dict['alg']== 'reptile':
    #     if args_dict['is_load_imagenet'] == True:
    #         raise ValueError('元学习算法，args.is_load_imagenet必须为false')

if __name__ == '__main__':

    args_dict = run_command()

    run_command(args_dict)
