# Author: Acer Zhang
# Datetime:2020/6/11 15:53
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import requests
import json

import numpy as np
from PIL import Image

# 启动命令 hub serving start --modules Sample_Module
img_path = r"D:\DLExample\Paddle_Deploy_Example\Deploy_PaddleHub__Example\sample_img\1008.jpg"
# 图片预处理
im = Image.open(img_path).convert('L')
im = np.array(im).reshape(1, 1, 30, 15).astype(np.float32)
im = im / 255.0 * 2.0 - 1.0
input_dict = {"array": im.tolist()}

url = "http://127.0.0.1:8866/predict/Sample_Module"
headers = {"Content-Type": "application/json"}

r = requests.post(url=url, headers=headers, data=json.dumps(input_dict))

out = eval(json.dumps(r.json(), indent=4, ensure_ascii=False))
print("推理结果为:", out["results"])
