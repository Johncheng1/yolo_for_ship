import tensorflow as tf
import numpy as np
class net   
    def __init__(self, layer_names, weights)    
        self.layer_names = layer_names
        self.weights = weights
    def build(self) 
        pass
    """
    Parameters
        idx       层数对应编号
        inputs    输入数据
        filter    [width,height,input_channels,output_channels]
        stride    步长
    Returns
        卷积计算的结果
    """
    def conv_layer(self, idx, inputs, filter, stride)   
        channels = inputs.get_shape()[3]