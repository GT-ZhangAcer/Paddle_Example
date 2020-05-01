# Author: Acer Zhang
# Datetime:2020/4/27 9:09
# Copyright belongs to the author.
# Please indicate the source for reprinting.


import paddle.fluid as fluid
import numpy as np


main_prog = fluid.default_main_program()
main_prog.seed = 1

# 定义数据输入格式
x = fluid.data(name="x", shape=[-1, 1], dtype="float32")
y = fluid.data(name="y", shape=[-1, 1], dtype="float32")

# 指定初始化权重，并设置Name方便后期跟踪
def init_fc(ipt, name):
    initializer_w = fluid.ParamAttr(learning_rate=1.,
                                    initializer=fluid.initializer.Constant(value=1.0),
                                    name="initializer_w" + str(name),
                                    trainable=True)
    initializer_b = fluid.ParamAttr(learning_rate=1.,
                                    initializer=fluid.initializer.Constant(value=0.),
                                    name="initializer_b" + str(name),
                                    trainable=True)
    fc = fluid.layers.fc(input=ipt, size=1, param_attr=initializer_w, bias_attr=initializer_b)
    return fc

# 定义三层全连接网络
layer_1 = init_fc(x, 1)
layer_2 = init_fc(layer_1, 2)
layer_3 = init_fc(layer_2, 3)

# 定义损失函数
loss = fluid.layers.square_error_cost(input=layer_3, label=y)
avg_loss = fluid.layers.mean(loss)

# 定义优化器
opt = fluid.optimizer.SGD(learning_rate=0.1)
# 在program中进行反向传播
opt.minimize(avg_loss)

# 初始化环境
place = fluid.CPUPlace()  # 初始化运算环境
start = fluid.default_startup_program()  # 初始化训练框架环境
exe = fluid.Executor(place)  # 初始化执行器
exe.run(start)

for i in range(3):
    # 定义数据
    x_ = np.array([[1]]).astype("float32")
    y_ = np.array([[2]]).astype("float32")
    # 传入数据
    exe.run(feed={"x": x_, "y": y_}, fetch_list=[loss])
    # 打印网络层的参数值
    for name in range(1, 4):
        parameter = fluid.global_scope().find_var("initializer_w" + str(name)).get_tensor()
        value = float(str(parameter).split("\n")[-2].replace("\tdata:", "")[2:-1])
        bool_text = "已更新" if value != 1.0 else "\033[1;31m已冻结\033[0m"
        print("Epoch", i, "第{}组w参数:".format(name), str(value)[: 3], "\t原始值为：1.0\t", bool_text)
    print("-" * 10)
