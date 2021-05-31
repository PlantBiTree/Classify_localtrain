import cv2
from PIL import Image
import requests as req
from io import BytesIO
import numpy as np
import os
import math
import codecs
import random
from models.resnet50 import preprocess_input
# 本地路径获取图片信息
def preprocess_img(img_path,img_size):
    try:
        img = Image.open(img_path)
        # if img.format:
        # resize_scale = img_size / max(img.size[:2])
        # img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
        img = img.resize((256, 256))
        img = img.convert('RGB')
        # img.show()
        img = np.array(img)
        imgs = []
        for _ in range(10):
            i = random.randint(0, 32)
            j = random.randint(0, 32)
            imgg = img[i:i + 224, j:j + 224]
            imgg = preprocess_input(imgg)
            imgs.append(imgg)
        return imgs
    except Exception as e:
        print('发生了异常data_process：', e)
        return 0




# url获取图片数组信息
def preprocess_img_from_Url(img_path,img_size):
    try:
        response = req.get(img_path)
        img = Image.open(BytesIO(response.content))
        img = img.resize((256, 256))
        img = img.convert('RGB')
        # img.show()
        img = np.array(img)
        imgs = []
        for _ in range(10):
            i = random.randint(0, 32)
            j = random.randint(0, 32)
            imgg = img[i:i + 224, j:j + 224]
            imgg = preprocess_input(imgg)
            imgs.append(imgg)
        return imgs
    except Exception as e:
        print('发生了异常data_process：', e)
        return 0