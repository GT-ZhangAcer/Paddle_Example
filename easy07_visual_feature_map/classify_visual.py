# Author: Acer Zhang
# Datetime:2020/5/1 11:03
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
import numpy as np
import PIL.Image as Image

data_path = "data"
save_model_path = "../easy03_CV_Classify/model"


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


def draw(results, save_path, img_num: int = 10):
    """
    可视化对应层结果
    :param save_path: 图像保存时的文件路径以及文件名
    :param results: 经卷积/池化层后的运算结果，格式为NCHW - 例： 128(batch size), 32(num_filters=16), 30(H), 15(W)
    :param img_num: 采样数
    """

    def fix_value(ipt):
        pix_max = np.max(ipt)
        pix_min = np.min(ipt)
        base_value = np.abs(pix_min) + np.abs(pix_max)
        base_rate = 255 / base_value
        pix_left = base_rate * pix_min
        ipt = ipt * base_rate - pix_left
        ipt[ipt < 0] = 0.
        ipt[ipt > 255] = 1.
        return ipt

    im_list = None
    for result_i, result in enumerate(results):
        if result_i == img_num - 1:
            break
        result = np.sum(result, axis=0)
        im = fix_value(result)
        if im_list is None:
            im_list = im
        else:
            im_list = np.append(im_list, im, axis=1)
    im = Image.fromarray(np.array(im_list).astype('uint8')).convert("RGB")
    im.save(save_path + ".jpg")


# 划分mini_batch
batch_size = 128
train_reader = fluid.io.batch(reader=switch_reader(), batch_size=batch_size)
val_reader = fluid.io.batch(reader=switch_reader(is_val=True), batch_size=batch_size)

# 定义网络输入格式
img = fluid.data(name="img", shape=[-1, 1, 30, 15], dtype="float32")
label = fluid.data(name='label', shape=[-1, 1], dtype='int64')

# 定义网络 - 两组卷积层
conv1 = fluid.layers.conv2d(input=img,
                            num_filters=2,
                            filter_size=3,
                            act='relu')
conv2 = fluid.layers.conv2d(input=conv1,
                            num_filters=20,
                            filter_size=3,
                            act='relu')
# 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
net_out = fluid.layers.fc(input=conv2, size=10, act='softmax')

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
            fetch_list=[loss, conv1, conv2, img])
        batch_loss = np.average(outs[0])
        draw(outs[1], "./visual_img/EPOCH_" + str(i) + "Conv1")
        draw(outs[2], "./visual_img/EPOCH_" + str(i) + "Conv2")
    # 验证集 只看准确率来判断收敛情况
    for batch_id, data in enumerate(val_reader()):
        outs = exe.run(program=eval_prog,
                       feed=feeder.feed(data),
                       fetch_list=[acc])
        batch_acc = np.average(outs[0])
    print("Epoch:", i, "\tLoss:{:3f}".format(batch_loss), "\tAcc:{:2f} %".format(batch_acc * 100))
