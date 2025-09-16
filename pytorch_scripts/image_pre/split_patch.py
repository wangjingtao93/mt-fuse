from PIL import Image

# 打开原始图片
original_image = Image.open('GYX19460319_20180628_104105_3DOCT00_L_000.jpg')

# 获取原始图片的宽度和高度
width, height = original_image.size

# 计算每个小图像的宽度和高度
small_width = width // 3  # 除以3，分割成3列
small_height = height // 3  # 除以3，分割成3行

# 循环遍历行和列，切割图片并保存
for i in range(3):
    for j in range(3):
        # 计算切割框的坐标
        left = j * small_width
        upper = i * small_height
        right = left + small_width
        lower = upper + small_height

        # 切割图片
        small_image = original_image.crop((left, upper, right, lower))

        # 保存切割后的图片
        small_image.save(f'small_image_{i}_{j}.jpg')

# 关闭原始图片
original_image.close()
