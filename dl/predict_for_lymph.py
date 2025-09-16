import torch
import os
import cv2
import math
import csv

import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import torchvision.models
from torchvision import transforms

from common.dl_comm import dl_comm
import common.utils as utils
from common.data_enter import lymph_data
from dl.dl_func import save_fpr_tpr, save_model_output


def heat_map_2024(args,meta_epoch):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 非镜下
    # csv_file = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/captions/feijingxia/choice.csv'

    # 镜下
    # csv_file = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/captions/jinxia_captions.csv'

    # whole slide
    # csv_file = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/captions/whole_slide/whole_slide.csv'

    # thyroid
    # csv_file = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/thyroid/caption/forheatmap.csv'

    # crc
    csv_file = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/CRC-LN/captions/forheatmap.csv'

    relative_path = '/data1/wangjingtao/workplace/python/data/classification/lymph'




    dframe = pd.read_csv(csv_file)

    res = args.datatype.split('_')[-1]
    class_prefix = args.datatype.split('_')[-2]

    df_res = dframe[dframe['Resolution'] == res]

    df_class_postive = df_res[df_res['Class'] == class_prefix + '_micro']
    df_class_negative = df_res[df_res['Class'] == class_prefix + '_normal']

    df_test = pd.DataFrame()
    # df_test = pd.concat([df_class_postive, df_class_negative],ignore_index=True) 
    df_test =  pd.concat([df_class_postive],ignore_index=True) 

    task_num = 5 # 多构建几个任务，结果会更准确

    # 去除背景
    bck_model = torchvision.models.resnet18(weights=None)
    num_ftrs = bck_model.fc.in_features
    bck_model.fc = nn.Linear(num_ftrs,  2)
    bck_model.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result/result_202405028/background_10x/dl/resnet18/2024-07-24-20-18-05/meta_epoch/taskid_0/best_model_for_valset_0.pth', map_location=device))
    bck_model.to(device=device)
    bck_model.eval()
    for task_idx in range(task_num):
        # task_idx = 3 # 非镜下， 4x最好哦用task3， 10x用task4

        dl_ob = dl_comm(args)
        dl_ob._init_net()
        dl_ob._init_opt()

        if args.load == '':
            exit('Predict Mode must give load path')
        else:
            # dl_ob.model.load_state_dict(torch.load(args.load))
            state_dict_path = os.path.join(args.load, 'meta_epoch', f'taskid_{task_idx}', f'best_model_for_valset_{meta_epoch}.pth')
            dl_ob.model.load_state_dict(torch.load(state_dict_path, map_location=device))

        dl_ob.model.eval()
        for index,row in df_test.iterrows():
            img_file = row['Image_path']

            caption_name =  img_file.split('/')[-1].replace('.jpg', '')

            heat_metric_store_dir = os.path.join(args.store_dir, 'heatmap', f'meta_epoch_{meta_epoch}',row['Center'], row['Resolution'], row['Class'],f'taskid_{task_idx}')
            utils.mkdir(heat_metric_store_dir)

            store_ori_image = os.path.join(args.store_dir, 'heatmap', f'meta_epoch_{meta_epoch}',row['Center'], row['Resolution'], row['Class'],f'taskid_{task_idx}', 'ori_image')

            image = cv2.imread(os.path.join(relative_path,img_file))
            if image is None:
                raise ValueError("Failed to load the image")
            # 将图像裁剪成 patches
            patches = []
            height, width = image.shape[:2]

            #  for whole slide 4x
            # height = int(height * 2/3)
            # width = int(width *2/3)
            # image = cv2.resize(image, [width,height], interpolation=cv2.INTER_CUBIC)

            # for captions
            height = math.ceil(height / 256) * 256
            width = math.ceil(width / 256) * 256
            image = cv2.resize(image, [width,height], interpolation=cv2.INTER_CUBIC)
            n_rows = 0
            for i in range(0, height, 256):
                n_rows += 1
                for j in range(0, width, 256):
                    patch = image[i:i+256, j:j+256, :]

                    patches.append(patch)
                    # patch_store =  os.path.join(heat_metric_store_dir, 'patches')
                    # cv2.imwrite(f'{patch_store}/{i}_{j}.jpg', patches)
            n_cols = int(len(patches) / n_rows)

            # 对每个 patch 进行预测，并记录预测概率
            pred_score_list = []
            predictions = []
            for patch in patches:
                # 预处理图像
                input_image = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                input_tensor = preprocess_image(input_image).to(device)
                # with torch.no_grad():
                #     check_bck = bck_model(input_tensor)
                with torch.no_grad():
                    output = dl_ob.model(input_tensor)

                # 使用softmax将输出转换为概率
                # bck_probability = torch.softmax(check_bck, dim=1)

                # 选择概率最大的类别
                # bck_classes = torch.argmax(bck_probability, dim=1)
                # if bck_classes == 0:
                #     # output = torch.zeros_like(output)
                #     output = torch.tensor([[1, 0]], dtype=torch.float32).to(device=device)

                pred_score_list.append(output.data.cpu().detach().numpy())

            y_score = np.concatenate(pred_score_list)
            predictions = softmax(y_score)


            # 将预测概率映射到原图像上
            heatmap = np.zeros_like(image[:,:,0], dtype=np.float32)
            count = 0
            for i in range(0, height, 256):
                for j in range(0, width, 256):
                    heatmap[i:i+256,j:j+256]  += predictions[count][1]  # 此处假设预测概率的第二个元素是正类的概率
                    count += 1

            # 归一化热力图
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * 255
            heatmap = heatmap.astype(np.uint8)

            # 可视化彩色图像
            color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


            # 放在同一张画布上
            concat_horizontal = np.hstack((color_map, image))
            # 将彩色图像保存为图像文件
            heatmap_store_path =  os.path.join(heat_metric_store_dir, f'{caption_name}.jpg')
            cv2.imwrite(heatmap_store_path, concat_horizontal)

            score_file = os.path.join(heat_metric_store_dir, f'{caption_name}_probabity_score.csv')
            with open(str(score_file), 'w') as f:
                fields = list(range(1, n_rows+1, 1))
                datawrite = csv.writer(f, delimiter=',')
                datawrite.writerow(fields)

                for i in range(n_rows):
                    index_start= i * n_cols
                    index_end = (i+1) * n_cols
                    pro_rows = predictions[index_start:index_end, 1].tolist()
                    datawrite.writerow([round(value, 6) for value in pro_rows])

                datawrite.writerow(['score','++', '++'])
                datawrite.writerow(['score','++', '++'])
                datawrite.writerow(['score','++', '++'])
                datawrite.writerow(['score','++', '++'])

                for i in range(n_rows):
                    index_start= i * n_cols
                    index_end = (i+1) * n_cols
                    socres_rows = y_score[index_start:index_end,1].tolist()
                    datawrite.writerow([round(value, 6) for value in socres_rows])
        # break

# for mt_fuse
def heat_map_fuse_2025(args,meta_epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 遍历视野图像
    dir_caption = '/data1/wangjingtao/workplace/python/data/classification/lymph/thyroid_fuse/for_heatmap/node/'
    df_test = os.listdir(dir_caption)

     # 去除背景
    is_bck = True
    bck_model = torchvision.models.resnet18(weights=None)
    num_ftrs = bck_model.fc.in_features
    bck_model.fc = nn.Linear(num_ftrs,  2)
    bck_model.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result/result_20240528/background_10x/dl/resnet18/2024-07-24-20-18-05/meta_epoch/taskid_0/best_model_for_valset_0.pth', map_location=device))
    bck_model.to(device=device)
    bck_model.eval()


    for task_idx in [1,4]:
        dl_ob = dl_comm(args)
        dl_ob._init_net()
        dl_ob._init_opt()

        relative_path = '/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result'
        if args.load == '':
            exit('Predict Mode must give load path')
        else:
            # dl_ob.model.load_state_dict(torch.load(args.load))
            state_dict_path = os.path.join(relative_path, args.load, 'meta_epoch', f'taskid_{task_idx}', f'best_model_for_valset_{meta_epoch}.pth')
            dl_ob.model.load_state_dict(torch.load(state_dict_path, map_location=device))
            print(f'predict_load_path: {state_dict_path}')

        dl_ob.model.eval()

        for img_file in df_test:
            caption_name =  img_file.split('/')[-1].replace('.jpg', '')
            heat_metric_store_dir = os.path.join(args.store_dir, f'meta_epoch_{meta_epoch}',f'taskid_{task_idx}')

            utils.mkdir(heat_metric_store_dir)

            store_ori_image = os.path.join(args.store_dir, f'meta_epoch_{meta_epoch}',f'taskid_{task_idx}', 'ori_image')

            image = cv2.imread(os.path.join(dir_caption,img_file))
            if image is None:
                raise ValueError("Failed to load the image")
            # 将图像裁剪成 patches
            patches = []
            height, width = image.shape[:2]

            # 同心圆
            patches_concentric = []

            #  for whole slide 4x
            # height = int(height * 2/3)
            # width = int(width *2/3)
            # image = cv2.resize(image, [width,height], interpolation=cv2.INTER_CUBIC)

            # for captions
            height = math.ceil(height / 256) * 256
            width = math.ceil(width / 256) * 256
            image = cv2.resize(image, [width,height], interpolation=cv2.INTER_CUBIC)


            n_rows = 0
            for i in range(0, height, 256):
                n_rows += 1
                for j in range(0, width, 256):
                    patch = image[i:i+256, j:j+256, :]

                    # patch在边缘
                    if i==0 or j==0 or i==height-256 or j == width-256:
                        patches_concentric.append(patch)
                    else:
                        # 获取同心圆区域
                        concentric_circle = image[i-256:i+256, j-256:j+256, :]
                        # concentric_circle = cv2.resize(concentric_circle, [256,256], interpolation=cv2.INTER_CUBIC)
                        patches_concentric.append(concentric_circle)

                    patches.append(patch)
                    # patch_store =  os.path.join(heat_metric_store_dir, 'patches')
                    # cv2.imwrite(f'{patch_store}/{i}_{j}.jpg', patches)
            n_cols = int(len(patches) / n_rows)

            # 对每个 patch 进行预测，并记录预测概率
            pred_score_list = []
            predictions = []
            for idx, patch in enumerate(patches):
                # 预处理图像
                input_image = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                input_tensor = preprocess_image(input_image).to(device)

                with torch.no_grad():
                    if args.net == 'mt_fuse_model':
                        cir_image = Image.fromarray(cv2.cvtColor(patches_concentric[idx], cv2.COLOR_BGR2RGB))
                        concertric_tensor = preprocess_image(cir_image).to(device)
                        output = dl_ob.model(torch.cat((input_tensor, concertric_tensor), dim=1))
                    else:
                        output = dl_ob.model(input_tensor)

                if is_bck:
                    with torch.no_grad():
                        check_bck = bck_model(input_tensor)
                    # 使用softmax将输出转换为概率
                    bck_probability = torch.softmax(check_bck, dim=1)

                    # 选择概率最大的类别
                    bck_classes = torch.argmax(bck_probability, dim=1)
                    if bck_classes == 0:
                        # output = torch.zeros_like(output)
                        # output = torch.zeros(1, 1000, dtype=torch.float32)
                        # output[0, 0] = 1.0
                        output = torch.tensor([[1, 0]], dtype=torch.float32).to(device=device)


                pred_score_list.append(output.data.cpu().detach().numpy())

            y_score = np.concatenate(pred_score_list)
            predictions = softmax(y_score)

            # 将预测概率映射到原图像上
            heatmap = np.zeros_like(image[:,:,0], dtype=np.float32)
            count = 0
            for i in range(0, height, 256):
                for j in range(0, width, 256):
                    heatmap[i:i+256,j:j+256]  += predictions[count][1]  # 此处假设预测概率的第二个元素是正类的概率
                    count += 1

            # 归一化热力图
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * 255
            heatmap = heatmap.astype(np.uint8)

            # 可视化彩色图像
            color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


            # 放在同一张画布上
            concat_horizontal = np.hstack((color_map, image))
            # 将彩色图像保存为图像文件
            heatmap_store_path =  os.path.join(heat_metric_store_dir, f'{caption_name}.jpg')
            cv2.imwrite(heatmap_store_path, concat_horizontal)

            score_file = os.path.join(heat_metric_store_dir, f'{caption_name}_probabity_score.csv')
            with open(str(score_file), 'w') as f:
                fields = list(range(1, n_rows+1, 1))
                datawrite = csv.writer(f, delimiter=',')
                datawrite.writerow(fields)

                for i in range(n_rows):
                    index_start= i * n_cols
                    index_end = (i+1) * n_cols
                    pro_rows = predictions[index_start:index_end, 1].tolist()
                    datawrite.writerow([round(value, 6) for value in pro_rows])

                datawrite.writerow(['score','++', '++'])
                datawrite.writerow(['score','++', '++'])
                datawrite.writerow(['score','++', '++'])
                datawrite.writerow(['score','++', '++'])

                for i in range(n_rows):
                    index_start= i * n_cols
                    index_end = (i+1) * n_cols
                    socres_rows = y_score[index_start:index_end,1].tolist()
                    datawrite.writerow([round(value, 6) for value in socres_rows])
        # break



# 定义图像预处理和裁剪函数
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# 获取同心圆图像
def capture_concentric_circles(image, height, width):
    patches = []
    patches_concentric = []
    n_rows = 0
    for i in range(0, height, 256):
        n_rows += 1
        for j in range(0, width, 256):
            patch = image[i:i+256, j:j+256, :]

            # patch在边缘
            if i==0 or j==0 or i==height-256 or j == width-256:
                patches_concentric.append(patch)
            else:
                # 获取同心圆区域
                patches_concentric = image[i-256:i+256, j-256:j+256, :]

            patches.append(patch)
            # patch_store =  os.path.join(heat_metric_store_dir, 'patches')
            # cv2.imwrite(f'{patch_store}/{i}_{j}.jpg', patches)
    n_cols = int(len(patches) / n_rows)

    return patches, patches_concentric

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def test_bf_enter(args):
    record_datatype = args.datatpye
    args.datatpye = args.test_bf
    test_data_ls, _ = lymph_data(args)

    value = predict()

    args.datatpye = record_datatype

# 当前使用20250823
def predict(args, loader_ls, meta_epoch):
    relative_path = '/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result'
    # 创建一个csv文件记录测试任务的指标
    dir_meta_epoch = os.path.join(args.store_dir,'meta_epoch')
    utils.mkdir(dir_meta_epoch)

    metric_dir = os.path.join(dir_meta_epoch,'metric_' + str(meta_epoch) + '.csv')
    with open(str(metric_dir), 'w') as f:
        fields = ['task_idx','acc', 'auc', 'precision','recall','f1','ka','sensi', 'spec']
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)

    # 创建一个txt文件记录混淆矩阵
    test_cm_res_dir = os.path.join(dir_meta_epoch, 'cm_test_' + str(meta_epoch) + '.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    task_num = len(loader_ls)
    all_tasks_value = []

    # for task_idx in range(task_num):
    for task_idx in loader_ls.keys():
        test_loader = loader_ls[task_idx]

        dl_ob = dl_comm(args)
        dl_ob._init_net()
        dl_ob._init_opt()

        if args.load == '':
            exit('Predict Mode must give load path')
        else:
            # dl_ob.model.load_state_dict(torch.load(args.load))
            state_dict_path = os.path.join(relative_path, args.load, 'meta_epoch', f'taskid_{task_idx}', f'best_model_for_valset_{meta_epoch}.pth')
            dl_ob.model.load_state_dict(torch.load(state_dict_path, map_location=device))
            print(f'predict_load_path: {state_dict_path}')

        res_test = dl_ob.val(test_loader)

        all_tasks_value.append([res_test['acc'],  res_test['auc'], res_test['prec'], res_test['recall'], res_test['f1'],res_test['ka'], res_test['sens'], res_test['spec']])

        with open(str(metric_dir), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            data_row = [task_idx] + [res_test['acc'],  res_test['auc'], res_test['prec'], res_test['recall'], res_test['f1'],res_test['ka'], res_test['sens'], res_test['spec']]
            csv_write.writerow(data_row)

        with open(test_cm_res_dir, 'a+') as file:
            file.write(f"Task_ID: {task_idx}, best_acc_epoch: {0}, best_sens_epoch:{0}\n")
            file.write("Acc Confusion Matrix:\n")
            file.write(np.array2string(res_test['cm'], separator=', ') + "\n\n")  # 将矩阵转换为字符串
            file.write("ACC Classification Report:\n")
            file.write(res_test['report'].to_string())
            file.write("\n+++++++++++++++++++++++++++\n")

        # 保存y_true和y_score, fpr和tpr
        save_model_output(res_test['y_true'], res_test['y_score'], dir_meta_epoch,task_idx, meta_epoch)
        save_fpr_tpr(res_test['fpr'], res_test['tpr'],dir_meta_epoch, task_idx, meta_epoch)

