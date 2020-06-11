# Author: Acer Zhang
# Datetime:2020/6/3 10:23
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle
import paddle.nn as nn
from paddle.incubate import hapi as h_api

from Paddle_CV_Example.Paddle_classify.simple.make_reader import reader

# 运行设备选择
USE_GPU = False
# Data文件夹所在路径 - 推荐在自己电脑上使用时选择绝对路径写法 例如:"C:\DLExample\Paddle_classify\data" 可以有效避免因路径问题读不到文件
DATA_PATH = "../data"
# 分类数量，0~9个数字，所以是10分类任务
CLASSIFY_NUM = 10
# CHW格式 - 通道数、高度、宽度
IMAGE_SHAPE_C = 3
IMAGE_SHAPE_H = 30
IMAGE_SHAPE_W = 15

# 计算图片像素点个数
IMAGE_SIZE = IMAGE_SHAPE_C * IMAGE_SHAPE_H * IMAGE_SHAPE_W


# 定义网络结构
class SampleCNN(h_api.model.Model):
    def __init__(self, name_scope):
        super(SampleCNN, self).__init__()
        self.linear1 = nn.Linear(input_dim=IMAGE_SIZE, output_dim=200, act="relu")
        self.linear2 = nn.Linear(input_dim=200, output_dim=100, act="relu")
        self.linear3 = nn.Linear(input_dim=100, output_dim=10, act="softmax")

    # 定义网络结构的前向计算过程
    def forward(self, inputs):
        hidden1 = self.linear1(inputs)
        hidden2 = self.linear2(hidden1)
        hidden3 = self.linear3(hidden2)
        return hidden3


# 定义输出层
img_define = [h_api.model.Input(shape=[-1, IMAGE_SIZE], dtype="float32", name="img")]
label_define = [h_api.model.Input(shape=[-1, 1], dtype="int64", name="label")]

# 实例化网络对象并定义优化器逻辑
model = SampleCNN("test_model")
optimizer = paddle.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=model.parameters())
model.prepare(optimizer=optimizer,
              loss_function=h_api.loss.CrossEntropy(),
              metrics=h_api.metrics.Accuracy(),
              inputs=img_define,
              labels=label_define,
              device='gpu' if USE_GPU else "cpu")


# 这里的reader是刚刚已经定义好的，代表训练和测试的数据
model.fit(train_data=reader(DATA_PATH, IMAGE_SIZE),
          eval_data=reader(DATA_PATH, IMAGE_SIZE, is_val=True),
          batch_size=128,
          epochs=10,
          save_dir="output/")
