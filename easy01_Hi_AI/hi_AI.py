# Author:  Acer Zhang
# Datetime:2019/10/7
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
import numpy as np

# 定义数据 y = 10x + 3
data_X = [[1], [2], [3], [5], [10]]
data_Y = [[13], [23], [33], [53], [103]]

# 定义张量格式
x = fluid.data(name="x", shape=[-1, 1], dtype="float32")
y = fluid.data(name="y", shape=[-1, 1], dtype="float32")

# 定义神经网络
out = fluid.layers.fc(input=x, size=1)

# 定义损失函数
loss = fluid.layers.square_error_cost(input=out, label=y)
avg_loss = fluid.layers.mean(loss)

# 定义优化器
opt = fluid.optimizer.SGD(learning_rate=0.01)
opt.minimize(avg_loss)

# 初始化环境
place = fluid.CPUPlace()  # 初始化运算环境
start = fluid.default_startup_program()  # 初始化训练框架环境
exe = fluid.Executor(place)  # 初始化执行器
exe.run(start)

# 开始训练
for i in range(100):
    for x_, y_ in zip(data_X, data_Y):
        x_ = np.array(x_).reshape(1, 1).astype("float32")
        y_ = np.array(y_).reshape(1, 1).astype("float32")
        info = exe.run(feed={"x": x_, "y": y_}, fetch_list=[loss])
        if i % 10 == 0:
            print("EPOCH:", i, "损失值：", info[0])

fluid.io.save_inference_model(dirname="infer.model", feeded_var_names=["x"], target_vars=[out], executor=exe)
