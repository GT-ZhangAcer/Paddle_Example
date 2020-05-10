# Author: Acer Zhang
# Datetime:2020/5/3 11:41
# Copyright belongs to the author.
# Please indicate the source for reprinting.

init_w = 0.5
lr = 0.1


def forward(x, param):
    w = param
    out = x * w
    return out


def loss(out, target):
    cost = (target - out) * (target - out)
    print("cost:{:.4f}".format(cost), "out:{:.4f}".format(out))


def backward(out, param, target):
    w = param
    tmp_w = -2 * (target - out)
    w = w - tmp_w * lr
    return w


train_param = init_w
for i in range(1, 10):
    train_target = i * 3
    train_out = forward(i, train_param)
    loss(train_out, train_target)
    train_param = backward(train_out, train_param, train_target)

for i in range(10):
    print("测试集：{:.4f}".format(forward(i, train_param)), "期望值：", i * 3)
print("最终参数:", train_param)
