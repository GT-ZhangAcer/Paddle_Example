# Author:  Acer Zhang
# Datetime:2019/10/27
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
import numpy as np
import PIL.Image as Image

data_path = "./data"
save_model_path = "./model"


# 构建Reader

def switch_reader(is_val: bool = False):
    def reader():
        # 读取标签数据
        with open(data_path + "/OCR_100P.txt", 'r') as f:
            labels = f.read()
        # 判断是否是验证集
        if is_val:
            index_range = range(1501, 2000)
        else:
            index_range = range(1, 1500)
        # 抽取数据使用迭代器返回
        for index in index_range:
            im = Image.open(data_path + "/" + str(index) + ".jpg").convert('L')  # 使用Pillow读取图片
            im = np.array(im).reshape(1, 1, 30, 15).astype(np.float32)  # NCHW格式
            im /= 255  # 归一化以提升训练效果
            lab = labels[index - 1]  # 因为循环中i是从1开始迭代的，所有这里需要减去1
            yield im, int(lab)

    return reader  # 注意！此处不需要带括号


# 划分mini_batch
batch_size = 128
train_reader = fluid.io.batch(reader=switch_reader(), batch_size=batch_size)
val_reader = fluid.io.batch(reader=switch_reader(is_val=True), batch_size=batch_size)

# 定义网络输入格式
img = fluid.data(name="img", shape=[-1, 1, 30, 15], dtype="float32")
label = fluid.data(name='label', shape=[-1, 1], dtype='int64')

# 定义网络
hidden = fluid.layers.fc(input=img, size=200, act='relu')
# 第二个全连接层，激活函数为ReLU
hidden = fluid.layers.fc(input=hidden, size=200, act='relu')
# 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
net_out = fluid.layers.fc(input=hidden, size=10, act='softmax')
# 使用API来计算正确率
acc = fluid.layers.accuracy(input=net_out, label=label)
# 将上方的结构克隆出来给测试程序使用
eval_prog = fluid.default_main_program().clone(for_test=True)
# 定义损失函数
loss = fluid.layers.cross_entropy(input=net_out, label=label)
avg_loss = fluid.layers.mean(loss)

# 定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_loss)

# 定义执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)

# 数据传入设置
feeder = fluid.DataFeeder(place=place, feed_list=[img, label])
prog = fluid.default_startup_program()
exe.run(prog)

# 开始训练
Epoch = 10
for i in range(Epoch):
    batch_loss = None
    batch_acc = None
    # 训练集 只看loss来判断模型收敛情况
    for batch_id, data in enumerate(train_reader()):
        outs = exe.run(
            feed=feeder.feed(data),
            fetch_list=[loss])
        batch_loss = np.average(outs[0])
    # 验证集 只看准确率来判断收敛情况
    for batch_id, data in enumerate(val_reader()):
        outs = exe.run(program=eval_prog,
                       feed=feeder.feed(data),
                       fetch_list=[acc])
        batch_acc = np.average(outs[0])
    print("Epoch:", i, "\tLoss:{:3f}".format(batch_loss), "\tAcc:{:2f} %".format(batch_acc * 100))

# 保存模型
fluid.io.save_inference_model(save_model_path, ['img'], [net_out], exe)
