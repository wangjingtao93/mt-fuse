import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')
import sys
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import model.timm_register_models
import timm
import os
import matplotlib.pyplot as plt


# 指定GPU设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型并移动到GPU
load_path = '/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result/result_202405028/multi_centers_ffpe_4x/dl/mt_small_model_lymph/2024-06-07-12-36-51/meta_epoch/taskid_0/best_model_for_valset_0.pth'
metanet = timm.create_model('mt_small_model_lymph', pretrained=False, num_classes=2, depth=6).to(device)
metanet.load_state_dict(torch.load(load_path, map_location=device))
metanet.eval()

# 定义图像预处理和裁剪函数
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# 加载原始图像
image = cv2.imread('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/pytorch_scripts/analysis/heat_map/1-X200142-4X.jpg')
if image is None:
    raise ValueError("Failed to load the image")

# 将图像裁剪成 patches
patches = []

height, width = image.shape[:2]
patch_rows = 0
for i in range(0, height - 256, 256):
    patch_rows += 1
    for j in range(0, width - 256, 256):
        patch = image[i:i+256, j:j+256, :]
        patches.append(patch)
        cv2.imwrite(f'/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/pytorch_scripts/analysis/heat_map/patches/{i}_{j}.jpg', patch)

patch_col = len(patches) / patch_rows

# 对每个 patch 进行预测，并记录预测概率
predictions = []
conut = 0
for patch in patches:
    # 预处理图像
    input_image = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess_image(input_image).to(device)

    # 使用 MetaNet 进行预测
    with torch.no_grad():
        output = metanet(input_tensor)

    # 当且仅当白色/黑色像素点加起来不超过/少于patch截图范围的xx时，模型不予判定
    rgb_s = (abs(patch[:, :, 0] - 107) >= 93) & (abs(patch[:, :, 1] - 107) >= 93) & (
                            abs(patch[:, :, 2] - 107) >= 93)

    # 计算预测概率  
    probability = torch.sigmoid(output[0][:2]).cpu().numpy()
    

    if np.sum(rgb_s) <= 256 ** 2 * 0.15:
        probability = np.array([1,0])
        print(conut)
        conut += 1
    
    predictions.append(probability)

# 将预测概率映射到原图像上
count = 0
heatmap = np.zeros_like(image[:, :, 0], dtype=np.float32)
for i in range(0, height - 256, 256):
    for j in range(0, width - 256, 256):
        heatmap[i:i+256,j:j+256]  += predictions[count][1]  # 此处假设预测概率的第二个元素是正类的概率
        count += 1
# heatmap = np.zeros((new_h, new_w), dtype=np.float32)
# for i in range(0, new_h - 256, 256):
#     for j in range(0, new_w - 256, 256):
#         heatmap[i:i+256,j:j+256]  += predictions[count][1]  # 此处假设预测概率的第二个元素是正类的概率
#         count += 1

# 归一化热力图
heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * 255
heatmap = heatmap.astype(np.uint8)

# 可视化彩色图像
color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 将彩色图像放回原图大小
# color_map_resized = cv2.resize(color_map, (width, height))

# 将彩色图像保存为图像文件
cv2.imwrite('pytorch_scripts/analysis/heat_map/color_map.jpg', color_map)

# 显示可视化图像
# plt.imshow(cv2.cvtColor(color_map_resized, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()