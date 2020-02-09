# Author:  Acer Zhang
# Datetime:2019/10/27
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
import numpy as np
import PIL.Image as Image

data_path = "./data"
save_model_path = "./model"

# 读取标签数据
with open(data_path + "/OCR_100P.txt", 'rt') as f:
    a = f.read()


# 构建Reader
def reader(mode: str = "Train"):
    """
    数据读取器
    :param mode: Train or Eval
    :return:
    """

    def req_one_data():
        if mode == "Train":
            sta = 1
            end = 1501
        else:
            sta = 1501
            end = 2001
        for i in range(sta, end):
            im = Image.open(data_path + "/" + str(i) + ".jpg").convert('L')  # 使用Pillow读取图片
            im = np.array(im).reshape(1, 1, 30, 15).astype(np.float32)  # 转为Numpy格式并补上Batch_size的1
            im = im / 255.0 * 2.0 - 1.0
            label_info = a[i - 1]
            yield im, label_info

    return req_one_data


# 定义网络输入格式
img = fluid.layers.data(name="img", shape=[1, 30, 15], dtype="float32")
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 定义网络
hidden = fluid.layers.fc(input=img, size=200, act='relu')
# 第二个全连接层，激活函数为ReLU
hidden = fluid.layers.fc(input=hidden, size=200, act='relu')
# 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
net_out = fluid.layers.fc(input=hidden, size=10, act='softmax')

# 定义损失函数
loss = fluid.layers.cross_entropy(input=net_out, label=label)
avg_loss = fluid.layers.mean(loss)
acc = fluid.layers.accuracy(input=net_out, label=label, k=1)
# 克隆一个用做测试集的程序
test_prog = fluid.default_main_program().clone(for_test=True)
# 定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_loss)

# 定义执行器
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)

# 数据传入设置
train_reader = fluid.io.batch(reader=reader("Train"), batch_size=512)
test_reader = fluid.io.batch(reader=reader("Eval"), batch_size=512)
feeder = fluid.DataFeeder(place=place, feed_list=[img, label])
prog = fluid.default_startup_program()
exe.run(prog)

# 开始训练
Epoch = 50
for i in range(Epoch):
    now_loss = None
    now_acc = None
    # 训练集 只看loss来判断模型收敛情况
    for batch_id, data in enumerate(train_reader()):
        outs = exe.run(
            feed=feeder.feed(data),
            fetch_list=[loss])
        now_loss = outs[0]
    # 测试集 只看准确率来判断收敛情况
    for batch_id, data in enumerate(test_reader()):
        outs = exe.run(program=test_prog,
                       feed=feeder.feed(data),
                       fetch_list=[acc])
        now_acc = outs[0]
    print("loss:", now_loss, "acc:", now_acc)

# 保存模型
fluid.io.save_inference_model(save_model_path, ['x'], [net_out], exe)
