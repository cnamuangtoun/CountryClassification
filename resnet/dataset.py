import os
import shutil
import random
metadata_path = 'metadata/'
data_path = 'data/'
train_ratio = 0.8  

os.makedirs(data_path + 'train', exist_ok=True)
os.makedirs(data_path + 'test', exist_ok=True)

for class_folder in os.listdir(metadata_path):
    if class_folder not in ['train', 'test', '.DS_Store']:
        class_path = metadata_path + class_folder
        images = os.listdir(class_path)
        images = [image for image in images if image.endswith('.png')]
        random.shuffle(images) 

        train_size = int(train_ratio * len(images))
        train_images = images[:train_size]
        test_images = images[train_size:]

        for image in train_images:
            src_path = class_path + '/' + image
            dst_path = data_path + 'train/' + class_folder + '/' + image
            os.makedirs(data_path + 'train/' + class_folder, exist_ok=True)
            shutil.copy(src_path, dst_path)

        for image in test_images:
            src_path = class_path + '/' + image
            dst_path = data_path + 'test/' + class_folder + '/' + image
            os.makedirs(data_path + 'test/' + class_folder, exist_ok=True)
            shutil.copy(src_path, dst_path)

print("数据集划分完成！")
