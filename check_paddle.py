# Author: Acer Zhang
# Datetime:2020/1/16 17:50
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
import numpy as np

a = np.array([1])
b = np.array([2])
a_data = fluid.data(name="a_data", shape=[1], dtype="int64")
b_data = fluid.data(name="a_data", shape=[1], dtype="int64")
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
out = exe.run(feed={"a_data": a}, fetch_list=[a_data, b_data])

print(out)
