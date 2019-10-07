# Author:  Acer Zhang
# Datetime:2019/10/7
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
import numpy as np

# 定义测试数据
test_data = [[100], [567], [20]]

# 初始化预测环境
exe = fluid.Executor(place=fluid.CPUPlace())

# 读取模型
Program, feed_target_names, fetch_targets = fluid.io.load_inference_model(dirname="infer.model",
                                                                          executor=exe)
# 开始预测
for x_ in test_data:
    x_ = np.array(x_).reshape(1, 1).astype("float32")
    out = exe.run(program=Program, feed={feed_target_names[0]: x_}, fetch_list=fetch_targets)
    print(out[0])