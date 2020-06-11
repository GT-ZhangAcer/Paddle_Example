# Author: Acer Zhang
# Datetime:2020/6/3 14:53
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle
from paddle.nn import Linear
from paddle.incubate.hapi.model import Model, Input
from paddle.incubate.hapi.loss import CrossEntropy
from paddle.incubate.hapi.metrics import Accuracy
from paddle.incubate.hapi.datasets.mnist import MNIST as MnistDataset


class Mnist(Model):
    def __init__(self, name_scope):
        super(Mnist, self).__init__()
        self.fc = Linear(input_dim=784, output_dim=10, act="softmax")

    # 定义网络结构的前向计算过程
    def forward(self, inputs):
        inputs = paddle.tensor.reshape(inputs, shape=[-1, 784])
        outputs = self.fc(inputs)
        return outputs


# 定义输入数据格式
inputs = [Input([None, 1,28,28], 'float32', name='image')]
labels = [Input([None, 1], 'int64', name='label')]

# 声明网络结构
model = Mnist("mnist")
optimizer = paddle.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=model.parameters())
# 使用高层API，prepare() 完成训练的配置
model.prepare(optimizer, CrossEntropy(), Accuracy(), inputs, labels, device='cpu')

# 定义数据读取器
train_dataset = MnistDataset(mode='train')
val_dataset = MnistDataset(mode='test')
# 启动训练
model.fit(train_dataset, val_dataset, batch_size=100, epochs=10, log_freq=100, save_dir="./output/")
