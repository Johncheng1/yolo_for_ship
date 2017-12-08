import tensorflow as tf
import numpy as np
import read_ckpt
import cv2
class Net:   
    def __init__(self, layer_names, weights):    
        self.layer_names = layer_names
        self.weights = weights
        self.alpha = 0.1
        self.build()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.load(self.sess)
    """
        神经网络的结构
    """
    def build(self): 
        self.input = tf.placeholder('float32',[None,448,448,3])
        self.output = tf.placeholder('float32',[None,7*7*6]) 
        
        self.pre = self.conv_layer(0, self.input, [7,7,3,64], 2)
        self.pre = self.max_pool(0, self.pre)

        self.result = self.pre

    def load(self, session):
        session.run(tf.get_collection('weights')[0].assign(self.weights[0]))
        session.run(tf.get_collection('weights')[1].assign(self.weights[1]))

    """
    Parameters
        layer_name  层名
        inputs      输入数据
        filter      [width,height,input_channels,output_channels]
        stride      步长
    Returns
        卷积计算的结果
    """
    def conv_layer(self, layer_name, inputs, filter, stride, trainable = False):   
        weight = tf.get_variable(name='w_'+layer_name, trainable = trainable, shape = filter, initializer = tf.contrib.layers.xavier_initializer() )
        bias = tf.get_variable(name='b_'+layer_name, trainable = trainable, shape = [ filter[-1] ], initializer = tf.constant_initializer(0.0) )

        tf.add_to_collection('weights', weight)
        tf.add_to_collection('weights',bias)    


        inputs = tf.nn.conv2d(inputs, weight, [1,stride,stride,1], padding='SAME', name=layer_name+'_conv')
        inputs = tf.nn.bias_add(inputs, bias, name=layer_name+'_bias')
        inputs = tf.maximum(self.alpha*inputs, inputs, name=layer_name+'_leaky_relu')

        return inputs
    """
    Parameters
        layer_name  层对应编号
        inputs      输入数据
    Returns
        maxpool结果
    """
    def max_pool(self, layer_name, inputs):
        inputs = tf.nn.max_pool(inputs, [1,2,2,1], strides = [1,2,2,1], padding = 'SAME', name = layer_name)
        return inputs
    def run(self, input):
        return self.sess.run(self.result,feed_dict={self.input:input})
"""
img = cv2.imread('juanji.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
inputs = np.zeros((1,448,448,3),dtype='float32')
inputs[0] = (img/255.0)*2.0-1.0
net = Net(read_ckpt.layer_names, read_ckpt.weights)
a = net.run(inputs)
np.save('juanji.npy',a)
"""