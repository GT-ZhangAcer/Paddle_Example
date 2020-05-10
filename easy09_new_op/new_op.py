# Author: Acer Zhang
# Datetime:2020/3/10 21:43
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
import numpy

gpu = fluid.CPUPlace()
exe = fluid.Executor(gpu)

train_data = [[1], [2], [3], [4], [5]]
y_true = [[3], [6], [9], [12], [15]]

# 定义网络
x = fluid.data(name="x", shape=[-1, 1], dtype="float32")
print(x.)
y = fluid.data(name="y", shape=[-1, 1], dtype="float32")

y_predict = fluid.layers.fc(input=x, size=1)
loss = fluid.default_main_program().current_block().create_var(name="loss_tmp", shape=[1], dtype="float32")


def square_error_cost(ipt_a, ipt_b):
    a = numpy.array(ipt_a)
    b = numpy.array(ipt_b)
    ab_cost = numpy.square(a - b)
    ab_cost = numpy.mean(ab_cost)
    return ab_cost


# 正常版本
def backward_square_error_cost(out, target, ab_mean, d_ab_mean):
    a = numpy.array(out)
    b = numpy.array(target)
    d_ab_mean = numpy.array(d_ab_mean)
    d = numpy.array(2 * (a - b)) * d_ab_mean
    return d, 0


# 正常版本
fluid.layers.py_func(func=square_error_cost, x=[y_predict, y], backward_func=backward_square_error_cost,
                     out=loss)

# 定义优化方法
# loss = fluid.layers.mean(loss)
sgd_optimizer = fluid.optimizer.Adam(learning_rate=0.1)
sgd_optimizer.minimize(loss)

# 开始训练，迭代100次

exe.run(fluid.default_startup_program())

for i in range(100):

    for data_id in range(len(y_true)):
        data_x = numpy.array(train_data[data_id]).astype("float32").reshape((1, 1))
        data_y = numpy.array(y_true[data_id]).astype("float32").reshape((1, 1))
        outs = exe.run(
            feed={'x': data_x, 'y': data_y},
            fetch_list=[loss])
        print("loss:", outs[0])
