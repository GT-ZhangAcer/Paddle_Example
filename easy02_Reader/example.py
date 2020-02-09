# Author: Acer Zhang
# Datetime:2020/2/8 15:59
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid

# 初始化环境
place = fluid.CPUPlace()
exe = fluid.Executor(place)


# 写一个简单的Reader

def reader():
    data = [i for i in range(10)]
    for sample in data:
        yield sample


batch_size = 10

reader = fluid.io.shuffle(reader, buf_size=5)
train_reader = fluid.io.batch(reader, batch_size=batch_size)

# train_feeder = fluid.DataFeeder(feed_list=['sentence'], place=place)

for i, data in enumerate(train_reader()):
    print("第", i, "轮\t", len(data), "\t", data)

