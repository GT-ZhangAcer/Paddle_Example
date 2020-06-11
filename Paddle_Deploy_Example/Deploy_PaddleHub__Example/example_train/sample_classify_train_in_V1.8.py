# Author: Acer Zhang
# Datetime:2020/6/11 14:52
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来显示中文

path = "./"
params_dirname = path + "test.inference.model"
print("训练后文件夹路径" + params_dirname)
# 参数初始化
# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace()
exe = fluid.Executor(place)

# 加载数据 - 数据请在https://github.com/GT-ZhangAcer/Paddle_Example下找到mini_classify_data.zip并解压
datatype = 'float32'
with open(path + "data/ocrData.txt", 'rt') as f:
    a = f.read()


def data_reader():
    def reader():
        for i in range(1, 800):
            im = Image.open(path + "data/" + str(i) + ".jpg").convert('L')
            im = np.array(im).reshape(1, 1, 30, 15).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            '''
            img = paddle.dataset.image.load_image(path + "data/" + str(i+1) + ".jpg")'''
            label_info = a[i - 1]
            yield im, label_info

    return reader


# 定义网络
x = fluid.layers.data(name="x", shape=[1, 30, 15], dtype=datatype)
label = fluid.layers.data(name='label', shape=[1], dtype='int64')


def sample_net(ipt):
    fc1 = fluid.layers.fc(input=ipt, size=100, act='relu', name='fc1')
    fc2 = fluid.layers.fc(input=fc1, size=10, act='softmax', name='fc2')
    return fc2


net = sample_net(x)  # CNN模型

cost = fluid.layers.cross_entropy(input=net, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=net, label=label, k=1)
# 定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)
# 数据传入设置
batch_reader = paddle.batch(reader=data_reader(), batch_size=2048)
feeder = fluid.DataFeeder(place=place, feed_list=[x, label])
prog = fluid.default_startup_program()
exe.run(prog)

trainNum = 50
accL = []
for i in range(trainNum):
    for batch_id, data in enumerate(batch_reader()):
        outs = exe.run(
            feed=feeder.feed(data),
            fetch_list=[label, avg_cost, acc])  # feed为数据表 输入数据和标签数据
        accL.append(outs[2])

    pross = float(i) / trainNum
    print("当前训练进度百分比为：" + str(pross * 100)[:3].strip(".") + "%")

path = params_dirname
plt.figure(1)
plt.title('正确率指标')
plt.xlabel('迭代次数')
plt.plot(range(50), accL)
plt.show()

fluid.io.save_inference_model(params_dirname, ['x'], [net], exe)
