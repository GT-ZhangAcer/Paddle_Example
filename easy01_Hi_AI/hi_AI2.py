# Author:  Acer Zhang
# Datetime:2019/10/7
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
import paddle
import numpy as np


# 重写Reader
def reader():
    def req_one_data():
        for i in range(10):
            data_X = [i]
            data_Y = [i * 10 + 3]
            data_X = np.array(data_X).reshape(1, 1).astype("float32")
            data_Y = np.array(data_Y).reshape(1, 1).astype("float32")
            yield data_X, data_Y

    return req_one_data


# 初始化项目环境
hi_ai_program = fluid.Program()  # 空白程序
start_program = fluid.Program()  # 用于初始化框架的空白程序

with fluid.program_guard(main_program=hi_ai_program, startup_program=start_program):
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
exe = fluid.Executor(place)  # 初始化执行器
exe.run(start_program)

# 定义数据传输格式
train_reader = paddle.batch(reader=reader(), batch_size=10)
train_feeder = fluid.DataFeeder(feed_list=[x, y], place=place, program=hi_ai_program)

# 开始训练
for epoch in range(100):
    for data in train_reader():
        info = exe.run(program=hi_ai_program,
                       feed=train_feeder.feed(data),
                       fetch_list=[loss])
        if epoch % 10 == 0:
            print("EPOCH:", epoch, "损失值：", info[0])

fluid.io.save_inference_model(dirname="infer.model",
                              feeded_var_names=["x"],
                              target_vars=[out],
                              executor=exe,
                              main_program=hi_ai_program)
