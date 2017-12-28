import tensorflow as tf
import numpy as np
import read_ckpt
import cv2
import load_fc
class Net:   
    """
    模式
    0：正常模式，完整版本
    1：只要卷积层的版本
    2：只要全连接层，用于训练的版本
    3: 只是没有全连接层
    """
    def __init__(self, layer_names, weights, mode=0):    
        self.layer_names = layer_names
        self.weights = weights
        self.alpha = 0.1
        self.batch_size = 128
        self.step = 100000
        self.mode = mode
        self.build(mode)
        if mode == 2:
            self.set_training()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.load(self.sess,mode)
    """
        神经网络的结构
    """
    def build(self,mode): 
        if mode == 2:
            self.input = tf.placeholder('float32',[None,7*7*1024])
        else:
            self.input = tf.placeholder('float32',[None,448,448,3])
        self.output = tf.placeholder('float32',[None,7*7*6]) 

        self.pre = self.input

        if mode == 0:
            # 正常模式
            self.pre = self.build_conv()
            # 转换一下好搞那个全连接层 
            self.pre= tf.transpose(self.pre,(0,3,1,2))
            self.pre = tf.reshape(self.pre, [-1, 7*7*1024 ]) 
            
            self.pre = self.fc_layer('fc_7', self.pre, [7*7*1024,512], True, False)
            self.pre = self.fc_layer('fc_8', self.pre, [512,4096], True, False)
            self.pre = self.fc_layer('fc_9', self.pre, [4096,7*7*6], False, True) 
        elif mode == 1:
            # 只要卷积层
            self.pre = self.build_conv()
        elif mode == 2:
            # 只要全连接层
            self.pre = self.fc_layer('fc_7', self.pre, [7*7*1024,512], True, False)
            #self.fc1 = self.pre
            self.pre = self.fc_layer('fc_8', self.pre, [512,4096], True, False)
            #self.fc2 = self.pre
            self.pre = self.fc_layer('fc_9', self.pre, [4096,7*7*6], False, True)
            #self.fc3 = self.pre
        elif mode == 3:
            self.pre = self.build_conv()
            # 转换一下好搞那个全连接层 
            self.pre= tf.transpose(self.pre,(0,3,1,2))
            self.pre = tf.reshape(self.pre, [-1, 7*7*1024 ]) 
        
        
        self.result = self.pre
    def build_conv(self):
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
        return self.pre 
          
    """
        载入权重参数
    """
    def load(self, session, mode):
        i = 0
        if mode == 2:
            i = i + 48
        for w in tf.get_collection('weights'):
            session.run(w.assign(self.weights[i]))
            i = i + 1
            print('loadded the %sth params' % (i))
        if mode ==0 or mode == 2:
            fc = np.load('fc.npy')
            fc_7_w = tf.get_collection('fc_7')[0]
            fc_7_b = tf.get_collection('fc_7')[1]
            fc_8_w = tf.get_collection('fc_8')[0]
            fc_8_b = tf.get_collection('fc_8')[1]
            fc_9_w = tf.get_collection('fc_9')[0]
            fc_9_b = tf.get_collection('fc_9')[1]
            cul = [fc_7_w, fc_7_b, fc_8_w, fc_8_b, fc_9_w, fc_9_b]
            for i in range(6):
                session.run(cul[i].assign(fc[i]))
                print('全连接层第%s个权重加载完毕'%(i+1))
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
        #if is_load:
        tf.add_to_collection(layer_name, weight)
        tf.add_to_collection(layer_name,bias)   

        inputs = tf.add(tf.matmul(inputs, weight), bias)
        if is_output:
            inputs = tf.maximum(0.1*inputs,inputs,name=layer_name+'_leaky_relu') 
        return inputs
    """
        损失函数的计算
    """
    def loss(self, result, label, size):
        # 预测类别
        class_pre = tf.slice(self.result, [0, 0], [size, 7*7], name='class_pre')
        class_label =tf.slice(label, [0,0], [size, 7*7], name='class_label')
        # 预测是否存在于方块
        prob_pre =  tf.slice(self.result, [0, 7*7], [size, 7*7], name='prob_pre')
        prob_label =  tf.slice(label, [0, 7*7], [size, 7*7], name='prob_label')
        # 预测横纵坐标
        pos_pre = tf.slice(self.result, [0, 7*7*2], [size, 7*7*2], name='pos_pre')
        pos_label = tf.slice(label, [0, 7*7*2], [size, 7*7*2], name='pos_label')
        # 预测长宽尺寸
        size_pre = tf.slice(self.result, [0, 7*7*4], [size, 7*7*2], name='size_pre')
        size_label = tf.slice(label, [0, 7*7*4], [size, 7*7*2], name='size_label')
        # 损失
        class_loss = tf.reduce_sum(tf.square( class_pre - class_label ))
        prob_loss = tf.reduce_sum(tf.square( prob_pre - prob_label ))
        pos_loss = tf.reduce_sum(tf.square( pos_pre - pos_label ))
        size_loss = tf.reduce_sum(tf.square( size_pre - size_label ))
        loss = 0.5*(class_loss + prob_loss) + 5*(pos_loss + size_loss)

        return loss
    """
        训练
    """
    def set_training(self):
        self.loss = self.loss(self.result, self.output, self.batch_size)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(self.loss)
    def run(self, input):
        return self.sess.run(self.result,feed_dict={self.input:input})
    """
        将数据集转换为仅需要训练全连接层的输入和标签
    """
    def get_fc_dataset(self):
        ''' dataset = np.load("train.npy")
        dataset1 = []
        for data in dataset:
            a = self.sess.run(self.result,feed_dict={self.input:[data[0]]})
            dataset1.append([a[0], data[1]])
            print("this is the %s epoch and the shape is %s" % (a[:5],a.shape))
        #np.save('conv_result.npy', dataset1) '''
        dataset = np.load('dataset.npy')
        size = len(dataset)
        fc_dataset = []
        for data in dataset:
            img = cv2.cvtColor(data[0],cv2.COLOR_BGR2RGB)
            img = ( img / 255 ) * 2 - 1
            a = self.sess.run(self.result,feed_dict={self.input:[ img ]})
            fc_data = [ a, data[1] ]
            fc_dataset.append(fc_data)
            print("this is the %s/%s picture" % (len(fc_dataset), size))
        np.save('fc_dataset.npy',fc_dataset)
    def train_fc(self):
        print('=======================开始训练了哈==========================')
        for i in range(self.step):
            # 读点数据进来
            data, label = load_fc.get_train_data(load_fc.dataset, self.batch_size)
            self.sess.run(self.optimizer, feed_dict={self.input: data, self.output: label})
            if i%20 ==0:
                l = self.sess.run(self.loss, feed_dict={self.input: data, self.output: label})
                print("the %s epoch loss is %s" % (i,l))
                
                if i > 1000 and self.mode == 2:
                    #fc1,fc2,fc3 = self.sess.run([self.fc1,self.fc2,self.fc3], feed_dict={self.input: data, self.output: label})
                    # 跑了一下午程序，存错变量了
                    fc_7_w = tf.get_collection('fc_7')[0]
                    fc_7_b = tf.get_collection('fc_7')[1]
                    fc_8_w = tf.get_collection('fc_8')[0]
                    fc_8_b = tf.get_collection('fc_8')[1]
                    fc_9_w = tf.get_collection('fc_9')[0]
                    fc_9_b = tf.get_collection('fc_9')[1]
                    cul = [fc_7_w, fc_7_b, fc_8_w, fc_8_b, fc_9_w, fc_9_b]
                    c = self.sess.run(cul, feed_dict={self.input: data, self.output: label}) #session.run(tf.get_collection('weights')[0].assign(self.weights[0]))
                    c = np.array(c)
                    np.save('fc.npy',c)

    def trans_to_npy(self):
        pass

''' img = cv2.imread('juanji.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
inputs = np.zeros((1,448,448,3),dtype='float32')
inputs[0] = (img/255.0)*2.0-1.0
net = Net(read_ckpt.layer_names, read_ckpt.weights)
a = net.run(inputs)
print(a.shape)
np.save('juanji.npy',a) '''
# 这个是用来对全连接层进行训练的
net = Net(read_ckpt.layer_names, read_ckpt.weights,2)
net.train_fc()
# 这个是完整的网络
''' img = cv2.imread('dataset/img0.png')
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
inputs = np.zeros((1,448,448,3),dtype='float32')
inputs[0] = (img/255.0)*2.0-1.0
net = Net(read_ckpt.layer_names, read_ckpt.weights, 0)
a = net.run(inputs)
print(a.shape)
np.save('result.npy',a) '''