# Author: Acer Zhang
# Datetime:2020/2/8 17:09
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
import numpy as np
import PIL.Image as Image

data_path = "./data"
save_model_path = "./model"

# 读取标签数据
with open(data_path + "/OCR_100P.txt", 'rt') as f:
    labels = f.read()


# 构建Reader
def reader():
    for i in range(1, 2000):
        im = Image.open(data_path + "/" + str(i) + ".jpg").convert('L')  # 使用Pillow读取图片
        im = np.array(im).reshape(1, 1, 30, 15).astype(np.float32)  # NCHW格式
        label = labels[i - 1]  # 因为循环中i是从1开始迭代的，所有这里需要减去1
        yield im, label


def normalized(sample):
    im, label = sample
    im /= 255
    return im, label


reader = fluid.io.xmap_readers(normalized, reader, process_num=8, buffer_size=10)
