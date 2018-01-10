import tensorflow as tf 
import numpy as np
import cv2
weights_small='weights/YOLO_small.ckpt'
layer_names = ['conv1s', 'max_pool1', 'conv2', 'max_pool2',
            'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'max_pool3',
            'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv4_5', 'conv4_6', 'conv4_7', 'conv4_8', 'conv4_9', 'conv4_10', 'max_pool4',
            'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'conv5_5', 'conv5_6s',
            'conv6_1', 'conv6_2',
            'transform',
            'full_7', 'full_8', 'full_9']
# 定义基本的网络类
class Net:
    # mode
    # 0 => 完整模式；1 => 只有全连接层的模式; 2=> 物体检测的模式; 3 => 只有卷积层的模式
    def __init__ ( self, mode, weight_file, layer_names):
        print('创建网络类')
        self.alpha = 0.1
        self.mode = mode
        self.layer_names = layer_names
        # 从ckpt文件里面把权重啥的都给读出来
        weights = self.read_weights(weight_file,mode)
        # 先定义输入输出
        if mode == 0:   # 舰船检测的完整模式
            self.input = tf.placeholder('float32', [None, 448, 448, 3])
            self.output = tf.placeholder('float32', [None, 7*7*5])
        elif mode == 1: # 只有全连接层
            self.input = tf.placeholder('float32', [None, 7*7*1024])
            self.output = tf.placeholder('float32', [None, 7*7*5])
        elif mode == 2: # 完整的物体检测模式
            self.input = tf.placeholder('float32', [None, 448, 448, 3])
            self.output = tf.placeholder('float32', [None, 7*7*30])
        elif mode == 3: # 只要卷积层
            self.input = tf.placeholder('float32', [None, 448, 448, 3])
            self.output = tf.placeholder('float32', [None, 7*7*1024])
        # 把网络结构搭起来
        self.build(layer_names, weights)

    def build(self, layer_names, weights):
        self.temp = self.input
        i = 0
        # 定义网络的基本结构
        for layer_name in layer_names:
            ''' print(weights[i].shape[0])
            i = i + 1 '''
            if layer_name[0] == 'c':
                if layer_name[-1] == 's':
                    self.temp = self.conv_layer(layer_name[:-1], self.temp, weights[i].shape, 2)
                    print('这是一个卷积层 => '+str(layer_name) + ' => 尺寸为' + str(weights[i].shape))
                else:
                    self.temp = self.conv_layer(layer_name, self.temp, weights[i].shape, 1)
                    print('这是一个卷积层 => '+str(layer_name) + ' => 尺寸为' + str(weights[i].shape))
                i = i + 2
            elif layer_name[0] == 'm':
                # self.pre = self.max_pool("max_pool_3", self.pre)
                self.temp = self.max_pool(layer_name, self.temp)
                print('这是一个池化层 => '+str(layer_name))
            elif layer_name[0] == 'f':
                print('这是一个全连接层 => '+str(layer_name))
            elif layer_name[0] == 't':
                self.temp= tf.transpose(self.temp,(0,3,1,2))
                self.temp = tf.reshape(self.temp, [-1, 7*7*1024 ]) 



    def read_weights(self, weights_file, mode):
        weights = []
        # 加载权重文件
        if mode != 1:
            reader = tf.train.NewCheckpointReader(weights_file)
            name = 'Variable'
            weights.append(reader.get_tensor(name))
            for i in range(1,54):
                name = 'Variable_' + str(i)
                weights.append(reader.get_tensor(name))
        
        return weights

    def load_weights(self):
        # 初始化权重
        pass

    def conv_layer(self, layer_name, inputs, filter, stride, trainable = False):   
        weight = tf.get_variable(name='w_'+layer_name, trainable = trainable, shape = filter, initializer = tf.contrib.layers.xavier_initializer() )
        bias = tf.get_variable(name='b_'+layer_name, trainable = trainable, shape = [ filter[-1] ], initializer = tf.constant_initializer(0.0) )

        tf.add_to_collection('weights', weight)
        tf.add_to_collection('weights',bias)    

        inputs = tf.nn.conv2d(inputs, weight, [1,stride,stride,1], padding='SAME', name=layer_name+'_conv')
        inputs = tf.nn.bias_add(inputs, bias, name=layer_name+'_bias')
        inputs = tf.maximum(self.alpha*inputs, inputs, name=layer_name+'_leaky_relu')

        return inputs

    def max_pool(self, layer_name, inputs):
        inputs = tf.nn.max_pool(inputs, [1,2,2,1], strides = [1,2,2,1], padding = 'SAME', name = layer_name)
        return inputs

    def fc_layer(self, layer_name, inputs, shape, is_read_weights, is_output):
        weight = tf.get_variable(name='w_'+layer_name, trainable = True, shape = shape, initializer = tf.contrib.layers.xavier_initializer() )
        bias = tf.get_variable(name='b_'+layer_name, trainable = True, shape = [ shape[-1] ], initializer = tf.constant_initializer(0.0) )

        tf.add_to_collection(layer_name, weight)
        tf.add_to_collection(layer_name,bias)   

        inputs = tf.add(tf.matmul(inputs, weight), bias)

        if is_output:
            inputs = tf.maximum(0.1*inputs,inputs,name=layer_name+'_leaky_relu') 
        
        return inputs

    def loss(self):
        pass
if __name__ == '__main__':
    net = Net(0, weights_small, layer_names)