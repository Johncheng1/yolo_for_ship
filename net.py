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
    def conv_layer(self, ids, inputs, filter, stride, trainable = False):   
        weight = tf.get_variable(name=self.layer_names[ids[0]], trainable = trainable, shape = filter, initializer = tf.contrib.layers.xavier_initializer() )
        bias = tf.get_variable(name=self.layer_names[ids[1]], trainable = trainable, shape = filter[-1], initializer = tf.constant_initializer(0.0) )