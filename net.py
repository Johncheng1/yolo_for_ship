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
        
        self.pre = self.input
        index = 0

        for l in self.layer_names:
            if l[:3] == 'con':
                if l[-1] == 's':
                    self.pre = self.conv_layer(l[:-1], self.pre, self.weights[index].shape, 2)
                    print('build the '+l[:-1]+' the kernel size is '+ str(self.weights[index].shape))
                else:
                    self.pre = self.conv_layer(l, self.pre, self.weights[index].shape, 1)
                    print('build the '+l[:-1]+' the kernel size is '+ str(self.weights[index].shape))
                index += 2
            elif l[:3] == 'max':
                self.pre = self.max_pool("max_pool_3", self.pre)
                print('build the ' + l +'!')    
        # 转换一下好搞那个全连接层 
        self.pre= tf.transpose(self.pre,(0,3,1,2))
        self.pre = tf.reshape(self.pre, [-1, 7*7*1024 ])

        self.pre = self.fc_layer('fc_7', self.pre, [7*7*1024,512], True, False)
        self.pre = self.fc_layer('fc_8', self.pre, [512,4096], True, False)
        self.pre = self.fc_layer('fc_9', self.pre, [4096,7*7*6], False, True)
                
        self.result = self.pre
    """
        载入权重参数
    """
    def load(self, session):
        i = 0
        for w in tf.get_collection('weights'):
            session.run(w.assign(self.weights[i]))
            i = i + 1
            print('loadded the %sth params' % (i))
        #session.run(tf.get_collection('weights')[0].assign(self.weights[0]))
        #session.run(tf.get_collection('weights')[1].assign(self.weights[1]))
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
        layer_name  层名称
        inputs      输入数据
    Returns
        maxpool结果
    """
    def max_pool(self, layer_name, inputs):
        inputs = tf.nn.max_pool(inputs, [1,2,2,1], strides = [1,2,2,1], padding = 'SAME', name = layer_name)
        return inputs
    """
    Parameters
        layer_name  层名称
        inputs      输入数据
        shape       神经元数目
        is_load     是否载入预训练数据
    Returns
        全连接层
    """
    def fc_layer(self, layer_name, inputs, shape, is_load, is_output):
        weight = tf.get_variable(name='w_'+layer_name, trainable = True, shape = shape, initializer = tf.contrib.layers.xavier_initializer() )
        bias = tf.get_variable(name='b_'+layer_name, trainable = True, shape = [ shape[-1] ], initializer = tf.constant_initializer(0.0) )
        if is_load:
            tf.add_to_collection('weights', weight)
            tf.add_to_collection('weights',bias)   

        inputs = tf.add(tf.matmul(inputs, weight), bias)
        if is_output:
            inputs = tf.maximum(0.1*inputs,inputs,name=layer_name+'_leaky_relu') 
        return inputs
    """
        损失函数的计算
    """
    def loss(self, session, result, label, size):
        # 预测类别
        class_pre = tf.slice(self.result, [0, 0], [size, 7*7], name='class_pre')
        class_label =tf.size(label, [0,0], [size, 7*7], name='class_label')
        # 预测是否存在于方块
        prob_pre =  tf.slice(self.result, [0, 7*7], [size, 7*7], name='prob_pre')
        prob_label =  tf.slice(label, [0, 7*7], [size, 7*7], name='prob_label')
        # 预测横纵坐标
        pos_pre = tf.slice(self.result, [0, 7*7*2], [size, 7*7*4], name='pos_pre')
        pos_label = tf.slice(label, [0, 7*7*2], [size, 7*7*4], name='pos_label')
        # 预测长宽尺寸
        size_pre = tf.slice(self.result, [0, 7*7*4], [size, 7*7*6], name='size_pre')
        size_label = tf.slice(label, [0, 7*7*4], [size, 7*7*6], name='size_label')
    """
        训练
    """
    def train(self,session):
        pass
    def run(self, input):
        return self.sess.run(self.result,feed_dict={self.input:input})

img = cv2.imread('juanji.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
inputs = np.zeros((1,448,448,3),dtype='float32')
inputs[0] = (img/255.0)*2.0-1.0
net = Net(read_ckpt.layer_names, read_ckpt.weights)
a = net.run(inputs)
print(a.shape)
np.save('juanji.npy',a)