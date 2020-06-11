# Author: Acer Zhang
# Datetime:2020/6/3 10:22
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import PIL.Image as Image
import numpy as np


def reader(data_path, image_size, is_val: bool = False):
    def switch_reader():
        # 读取标签数据
        with open(data_path + "/OCR_100P.txt", 'r') as f:
            labels = f.read()
        # 判断是否是验证集
        if is_val:
            index_range = range(600, 1000)
        else:
            index_range = range(1, 600)
        # 抽取数据使用迭代器返回
        for index in index_range:
            im = Image.open(data_path + "/" + str(index) + ".jpg")  # 使用Pillow读取图片
            im = np.array(im).reshape(1, image_size).astype(np.float32)  # NCHW格式
            im /= 255  # 归一化以提升训练效果
            lab = labels[index - 1]  # 因为循环中i是从1开始迭代的，所有这里需要减去1
            yield im, int(lab)

    return switch_reader  # 注意！此处不需要带括号
