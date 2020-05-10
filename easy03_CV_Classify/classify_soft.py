# Author: Acer Zhang
# Datetime:2020/5/10 13:30
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
import numpy as np
import PIL.Image as Image

data_path = "../easy07_visual_feature_map/data"
save_model_path = "./model2"

BATCH_SIZE = 32


# 构建Reader

def switch_reader(is_val: bool = False):
    def reader():
        # 读取标签数据
        with open(data_path + "/OCR_100P.txt", 'r') as f:
            labels = f.read()
        # 判断是否是验证集
        if is_val:
            index_range = range(500, 800)
        else:
            index_range = range(1, 500)
        # 抽取数据使用迭代器返回
        for index in index_range:
            im = Image.open(data_path + "/" + str(index) + ".jpg").convert('L')  # 使用Pillow读取图片
            im = np.array(im).reshape(1, 1, 30, 15).astype(np.float32)  # NCHW格式
            im /= 255  # 归一化以提升训练效果
            img_label = int(labels[index - 1])  # 图片对应的标签值
            # 创建soft_label 前1-10个位置为0~9的概率，11、12代表奇偶的概率，12为奇数，11为偶数
            lab = np.zeros([1, 12], dtype=np.float32)
            if img_label % 2 == 0:
                # 若label为偶数(此时包括0在内了)
                lab[0][-2] = 1
            else:
                lab[0][-1] = 1
            lab[0][img_label] = 1
            yield im, lab

    return reader  # 注意！此处不需要带括号


# 划分mini_batch

train_reader = fluid.io.batch(reader=switch_reader(), batch_size=BATCH_SIZE)
val_reader = fluid.io.batch(reader=switch_reader(is_val=True), batch_size=BATCH_SIZE)

# 定义网络输入格式
img = fluid.data(name="img", shape=[-1, 1, 30, 15], dtype="float32")
label = fluid.data(name='label', shape=[-1, 12], dtype='float32')

# 定义网络
conv1 = fluid.layers.conv2d(input=img,
                            num_filters=16,
                            filter_size=3,
                            act="relu")
conv2 = fluid.layers.conv2d(input=conv1,
                            num_filters=32,
                            filter_size=3,
                            act="relu")
conv3 = fluid.layers.conv2d(input=conv2,
                            num_filters=12,
                            filter_size=1,
                            act="relu")
# 以softmax为激活函数的全连接输出层，输出层的大小为数字的个数类别A数字0-9[10] + 类别B奇偶数[2]
net_out = fluid.layers.fc(input=conv3, size=12)


# 将上方的结构克隆出来给测试程序使用
eval_prog = fluid.default_main_program().clone(for_test=True)

# 定义多标签损失函数
loss = fluid.layers.sigmoid_cross_entropy_with_logits(net_out, label)
avg_loss = fluid.layers.mean(loss)

# 定义优化方法
sgd_optimizer = fluid.optimizer.Adam(learning_rate=0.01)
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
    # 训练集 只看loss来判断模型收敛情况
    for batch_id, data in enumerate(train_reader()):
        outs = exe.run(
            feed=feeder.feed(data),
            fetch_list=[loss])
        batch_loss = np.average(outs[0])
    print("Epoch:", i, "\tLoss:{:3f}".format(batch_loss))

# 保存模型
fluid.io.save_inference_model(save_model_path, ['img'], [net_out], exe)
