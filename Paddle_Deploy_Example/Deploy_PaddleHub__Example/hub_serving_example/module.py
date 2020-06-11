# Author: Acer Zhang
# Datetime:2020/6/11 15:06
# Copyright belongs to the author.
# Please indicate the source for reprinting.

# 更新该Demo时使用的版本 PaddlePaddle1.8.1 + Hub1.7.1
import os

import numpy as np
from PIL import Image
import paddlehub as hub

import paddle.fluid as fluid

from paddlehub.module.module import moduleinfo, serving


@moduleinfo(
    name="Sample_Module",  # 模型名
    version="1.0.0",  # 模型版本
    summary="Simple Example Deployment server PaddleHub",  # 模型说明 - 在Hub list命令下可以展示
    author="Acer Zhang",
    author_email="None",
    type="nlp/classify")  # 举例 nlp/xxx 在Serving部署后接口为:地址+端口+/nlp/xxx
class SampleModule(hub.Module):
    def _initialize(self, use_gpu: bool = False):
        """
        模型初始化
        :param use_gpu: 是否使用GPU
        """

        model_path = os.path.join(self.directory, "predict_model")
        # 此处要想得到更高性能，建议参考最新官方文档使用Paddle Inference，下方写法推理速度较慢。
        self.place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)

        # 返回的分别是推理Program，输入字段，返回字段
        self.model = fluid.io.load_inference_model(model_path, self.exe)

    @serving
    def predict(self, array):
        """
        进行推理
        :param array: Numpy对象
        :return:
        """
        array = np.array(array).astype("float32")

        # model[0] - 推理Program ，self.model[1][0] - 输入字段的第一个字段，此处在保存预测模型时已设置
        outputs = self.exe.run(self.model[0], feed={self.model[1][0]: array}, fetch_list=self.model[2])
        ret = []
        # 增加Batch Size不为1的情况支持
        for sample in outputs:
            label = np.array(sample).argmax()
            ret.append(label)
        # 避免Json不支持，此处返回str数据类型
        return str(ret)


# Debug - 若直接在IDE中运行该py文件则进行调试
if __name__ == "__main__":
    # 此处需要自己修改路径
    img_path = r"D:\DLExample\Paddle_Deploy_Example\Deploy_PaddleHub__Example\sample_img\1009.jpg"
    # 图片预处理
    im = Image.open(img_path).convert('L')
    im = np.array(im).reshape(1, 1, 30, 15).astype("float32")
    im = im / 255.0 * 2.0 - 1.0
    debug_module = SampleModule()
    # 因为JSON无法识别Numpy对象，所以在在Serving逻辑中加入了转numpy和转字符串
    out = debug_module.predict(im.tolist())
    print("推理结果为:", out)
