# Author: Acer Zhang
# Datetime:2020/5/10 14:29
# Copyright belongs to the author.
# Please indicate the source for reprinting.

# Author: Acer Zhang
# Datetime:2020/5/10 14:19
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
import numpy as np
import PIL.Image as Image

img_path = r"D:\DLExample\easy07_visual_feature_map\data\1.jpg"
save_model_path = "./model2"

img1 = Image.open(img_path).convert('L')
img1 = np.array(img1).reshape(1, 1, 30, 15).astype(np.float32)  # NCHW格式
img1 /= 255  # 归一化以提升训练效果

# 防止Notebook中出现冲突等问题，使用新的ScopeS、来保证程序的健壮性。
new_scope = fluid.Scope()

place = fluid.CPUPlace()
exe = fluid.Executor(place)

with fluid.scope_guard(new_scope):
    # 读取预测模型
    infer_program, feed_name, fetch_list = fluid.io.load_inference_model(save_model_path, exe)
    outs = exe.run(program=infer_program, feed={feed_name[0]: img1}, fetch_list=fetch_list)
    out = outs[0][0]  # 第1个batch和第1个fetch结果
    print("概率分布:", np.round(out, 2))  # 保留2位整数
    print("推理结果:\t分类A[数字]", np.argmax(out[:10]), "\t分类B[奇偶性]", np.argmax(out[10:]))  # 获取概率最高的标签索引
