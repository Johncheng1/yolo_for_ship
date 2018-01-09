import tensorflow as tf 
import numpy as np
import cv2
weights_small='weights/YOLO_small.ckpt'
# 定义基本的网络类
class Net:
    def __init__(self, mode, weight_file):
        print('创建网络类')
        self.mode = mode
        # 从ckpt文件里面把权重啥的都给读出来
        weights = load(weight_file)
        # 把网络结构搭起来
        build()
    def build(self):
        # 定义网络的基本结构
        pass
    def load(self, weight_file):
        # 加载权重文件
        pass
    def init_weights(self):
        # 初始化权重
        pass
    def read_weights(self):
        # 读取ckpt文件
        pass
    def conv_layer(self, layer_name, inputs, filter, stride, trainable = False):
        pass
    def max_pool(self, layer_name, inputs):
        pass
    def fc_layer(self, layer_name, inputs, shape, is_load, is_output):
        pass
    def loss(self):
        pass