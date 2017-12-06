import tensorflow as tf
import numpy as np
import cv2

weights_file='YOLO_small.ckpt'

class TinyNet:
    def __init__(self, weights_file='YOLO_small.ckpt'):
        print('创建网络')
        self.weights_file = weights_file
        self.reader = tf.train.NewCheckpointReader(weights_file)
        self.get_weights()
        self.init_weights()
        self.build()
    def cul(self,input):
        return self.sess.run(self.result,feed_dict={self.input:input})

    def build(self):
        self.input = tf.placeholder('float32',[None,448,448,3])
        self.output = tf.placeholder('float32',[None,7*7*6])
        # 第一层
        self.x = tf.nn.conv2d(self.input, self.w_1, [1,2,2,1], padding='SAME', name='w_conv_1')
        self.x = tf.nn.bias_add(self.x, self.b_1, name = 'bias_conv_1' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_1')
        self.x = tf.nn.max_pool(self.x, [1,2,2,1], strides = [1,2,2,1], padding = 'SAME', name = 'maxpool_1')
        # 第二层 输出 56*56*192
        self.x = tf.nn.conv2d(self.x, self.w_2, [1,1,1,1], padding='SAME', name='w_conv_2')
        self.x = tf.nn.bias_add(self.x, self.b_2, name = 'bias_conv_2' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_2')
        self.x = tf.nn.max_pool(self.x, [1,2,2,1], strides = [1,2,2,1], padding = 'SAME', name = 'maxpool_2')
        
        # 第三层 输出28*28*512
        self.x = tf.nn.conv2d(self.x, self.w_3, [1,1,1,1], padding='SAME', name='w_conv_3')
        self.x = tf.nn.bias_add(self.x, self.b_3, name = 'bias_conv_3' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_3')
        self.x = tf.nn.conv2d(self.x, self.w_4, [1,1,1,1], padding='SAME', name='w_conv_4')
        self.x = tf.nn.bias_add(self.x, self.b_4, name = 'bias_conv_4' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_4')
        self.x = tf.nn.conv2d(self.x, self.w_5, [1,1,1,1], padding='SAME', name='w_conv_5')
        self.x = tf.nn.bias_add(self.x, self.b_5, name = 'bias_conv_5' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_5')
        self.x = tf.nn.conv2d(self.x, self.w_6, [1,1,1,1], padding='SAME', name='w_conv_6')
        self.x = tf.nn.bias_add(self.x, self.b_6, name = 'bias_conv_6' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_6')
        self.x = tf.nn.max_pool(self.x, [1,2,2,1], strides = [1,2,2,1], padding = 'SAME', name = 'maxpool_3')
        
        # 第四层
        self.x = tf.nn.conv2d(self.x, self.w_7, [1,1,1,1], padding='SAME', name='w_conv_7')
        self.x = tf.nn.bias_add(self.x, self.b_7, name = 'bias_conv_7' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_7')
        self.x = tf.nn.conv2d(self.x, self.w_8, [1,1,1,1], padding='SAME', name='w_conv_8')
        self.x = tf.nn.bias_add(self.x, self.b_8, name = 'bias_conv_8' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_8')
        self.x = tf.nn.conv2d(self.x, self.w_9, [1,1,1,1], padding='SAME', name='w_conv_9')
        self.x = tf.nn.bias_add(self.x, self.b_9, name = 'bias_conv_9' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_9')
        self.x = tf.nn.conv2d(self.x, self.w_10, [1,1,1,1], padding='SAME', name='w_conv_10')
        self.x = tf.nn.bias_add(self.x, self.b_10, name = 'bias_conv_10' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_10')
        self.x = tf.nn.conv2d(self.x, self.w_11, [1,1,1,1], padding='SAME', name='w_conv_11')
        self.x = tf.nn.bias_add(self.x, self.b_11, name = 'bias_conv_11' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_11')
        self.x = tf.nn.conv2d(self.x, self.w_12, [1,1,1,1], padding='SAME', name='w_conv_12')
        self.x = tf.nn.bias_add(self.x, self.b_12, name = 'bias_conv_12' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_12')
        self.x = tf.nn.conv2d(self.x, self.w_13, [1,1,1,1], padding='SAME', name='w_conv_13')
        self.x = tf.nn.bias_add(self.x, self.b_13, name = 'bias_conv_13' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_13')
        self.x = tf.nn.conv2d(self.x, self.w_14, [1,1,1,1], padding='SAME', name='w_conv_14')
        self.x = tf.nn.bias_add(self.x, self.b_14, name = 'bias_conv_14' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_14')
        self.x = tf.nn.conv2d(self.x, self.w_15, [1,1,1,1], padding='SAME', name='w_conv_15')
        self.x = tf.nn.bias_add(self.x, self.b_15, name = 'bias_conv_15' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_15')
        self.x = tf.nn.conv2d(self.x, self.w_16, [1,1,1,1], padding='SAME', name='w_conv_16')
        self.x = tf.nn.bias_add(self.x, self.b_16, name = 'bias_conv_16' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_16')
        self.x = tf.nn.max_pool(self.x, [1,2,2,1], strides = [1,2,2,1], padding = 'SAME', name = 'maxpool_4')
        
        # 第五层
        self.x = tf.nn.conv2d(self.x, self.w_17, [1,1,1,1], padding='SAME', name='w_conv_17')
        self.x = tf.nn.bias_add(self.x, self.b_17, name = 'bias_conv_17' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_17')
        self.x = tf.nn.conv2d(self.x, self.w_18, [1,1,1,1], padding='SAME', name='w_conv_18')
        self.x = tf.nn.bias_add(self.x, self.b_18, name = 'bias_conv_18' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_18')
        self.x = tf.nn.conv2d(self.x, self.w_19, [1,1,1,1], padding='SAME', name='w_conv_19')
        self.x = tf.nn.bias_add(self.x, self.b_19, name = 'bias_conv_19' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_19')
        self.x = tf.nn.conv2d(self.x, self.w_20, [1,1,1,1], padding='SAME', name='w_conv_20')
        self.x = tf.nn.bias_add(self.x, self.b_20, name = 'bias_conv_20' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_20')
        self.x = tf.nn.conv2d(self.x, self.w_21, [1,1,1,1], padding='SAME', name='w_conv_21')
        self.x = tf.nn.bias_add(self.x, self.b_21, name = 'bias_conv_21' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_21')
        self.x = tf.nn.conv2d(self.x, self.w_22, [1,2,2,1], padding='SAME', name='w_conv_22')
        self.x = tf.nn.bias_add(self.x, self.b_22, name = 'bias_conv_22' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_22')
        # 第六层
        self.x = tf.nn.conv2d(self.x, self.w_23, [1,1,1,1], padding='SAME', name='w_conv_23')
        self.x = tf.nn.bias_add(self.x, self.b_23, name = 'bias_conv_23' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_23')
        self.x = tf.nn.conv2d(self.x, self.w_24, [1,1,1,1], padding='SAME', name='w_conv_24')
        self.x = tf.nn.bias_add(self.x, self.b_24, name = 'bias_conv_24' )
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_24')
        # 全连接层1
        self.x= tf.transpose(self.x,(0,3,1,2))
        self.x = tf.reshape(self.x, [-1, 7*7*1024 ])
        self.x = tf.add(tf.matmul(self.x, self.tf_w25), self.tf_b25)
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_full_1')
        # 全连接层2
        self.x = tf.add(tf.matmul(self.x, self.tf_w26), self.tf_b26)
        self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_full_2')
        # 全连接层3
        self.x = tf.add(tf.matmul(self.x, self.tf_w27), self.tf_b27)
        #self.x = tf.nn.conv2d(self.x, self.w_2, [1,1,1,1], padding='SAME', name='w_conv_2')
        #self.x = tf.nn.bias_add(self.x, self.b_2, name = 'bias_conv_2' )
        #self.x = tf.maximum(0.1*self.x,self.x,name='leaky_relu_conv_2')
        
        self.result = self.x


        self.sess = tf.Session()
        
        self.sess.run(tf.global_variables_initializer())
        

        self.load_weights(self.sess)

        print('计算图创建完毕')
        
    def load_weights(self,session):
        session.run(self.tf_w1.assign(self.w_1))
        session.run(self.tf_b1.assign(self.b_1))
        session.run(self.tf_w2.assign(self.w_2))
        session.run(self.tf_b2.assign(self.b_2))
        session.run(self.tf_w3.assign(self.w_3))
        session.run(self.tf_b3.assign(self.b_3))
        session.run(self.tf_w4.assign(self.w_4))
        session.run(self.tf_b4.assign(self.b_4))
        session.run(self.tf_w5.assign(self.w_5))
        session.run(self.tf_b5.assign(self.b_5))
        session.run(self.tf_w6.assign(self.w_6))
        session.run(self.tf_b6.assign(self.b_6))
        session.run(self.tf_w7.assign(self.w_7))
        session.run(self.tf_b7.assign(self.b_7))
        session.run(self.tf_w8.assign(self.w_8))
        session.run(self.tf_b8.assign(self.b_8))
        session.run(self.tf_w9.assign(self.w_9))
        session.run(self.tf_b9.assign(self.b_9))
        session.run(self.tf_w10.assign(self.w_10))
        session.run(self.tf_b10.assign(self.b_10))
        session.run(self.tf_w11.assign(self.w_11))
        session.run(self.tf_b11.assign(self.b_11))
        session.run(self.tf_w12.assign(self.w_12))
        session.run(self.tf_b12.assign(self.b_12))
        session.run(self.tf_w13.assign(self.w_13))
        session.run(self.tf_b13.assign(self.b_13))
        session.run(self.tf_w14.assign(self.w_14))
        session.run(self.tf_b14.assign(self.b_14))
        session.run(self.tf_w15.assign(self.w_15))
        session.run(self.tf_b15.assign(self.b_15))
        session.run(self.tf_w16.assign(self.w_16))
        session.run(self.tf_b16.assign(self.b_16))

        session.run(self.tf_w17.assign(self.w_17))
        session.run(self.tf_b17.assign(self.b_17))
        session.run(self.tf_w18.assign(self.w_18))
        session.run(self.tf_b18.assign(self.b_18))
        session.run(self.tf_w19.assign(self.w_19))
        session.run(self.tf_b19.assign(self.b_19))
        session.run(self.tf_w20.assign(self.w_20))
        session.run(self.tf_b20.assign(self.b_20))
        session.run(self.tf_w21.assign(self.w_21))
        session.run(self.tf_b21.assign(self.b_21))
        session.run(self.tf_w22.assign(self.w_22))
        session.run(self.tf_b22.assign(self.b_22))
        session.run(self.tf_w23.assign(self.w_23))
        session.run(self.tf_b23.assign(self.b_23))
        session.run(self.tf_w24.assign(self.w_24))
        session.run(self.tf_b24.assign(self.b_24))
        session.run(self.tf_w25.assign(self.w_25))
        session.run(self.tf_b25.assign(self.b_25))
        session.run(self.tf_w26.assign(self.w_26))
        session.run(self.tf_b26.assign(self.b_26))
        #session.run(self.tf_w27.assign(np.zeros([4096,7*7*6])))
        #session.run(self.tf_b27.assign(np.zeros(7*7*6)))
        #session.run(self.tf_w27.assign(self.w_27))
        #session.run(self.tf_b27.assign(self.b_27))
        print('权重加载完毕')

    def init_weights(self,trainable= True):
        # 先这么恶心的写着吧。。。。看能不能算数
        self.tf_w1 = tf.get_variable(name='tf_w1', trainable = trainable, shape = [7, 7, 3, 64], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b1 = tf.get_variable(name='tf_b1',trainable = trainable,shape = [ 64 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w2 = tf.get_variable(name='tf_w2', trainable = trainable, shape = [3, 3, 64, 192], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b2 = tf.get_variable(name='tf_b2',trainable = trainable,shape = [ 192 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w3 = tf.get_variable(name='tf_w3', trainable = trainable, shape = [1, 1, 192, 128], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b3 = tf.get_variable(name='tf_b3',trainable = trainable,shape = [ 128 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w4 = tf.get_variable(name='tf_w4', trainable = trainable, shape = [3, 3, 128, 256], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b4 = tf.get_variable(name='tf_b4',trainable = trainable,shape = [ 256 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w5 = tf.get_variable(name='tf_w5', trainable = trainable, shape = [1, 1, 256, 256], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b5 = tf.get_variable(name='tf_b5',trainable = trainable,shape = [ 256 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w6 = tf.get_variable(name='tf_w6', trainable = trainable, shape = [3, 3, 256, 512], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b6 = tf.get_variable(name='tf_b6',trainable = trainable,shape = [ 512 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w7 = tf.get_variable(name='tf_w7', trainable = trainable, shape = [1, 1, 512, 256], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b7 = tf.get_variable(name='tf_b7',trainable = trainable,shape = [ 256 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w8 = tf.get_variable(name='tf_w8', trainable = trainable, shape = [3, 3, 256, 512], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b8 = tf.get_variable(name='tf_b8',trainable = trainable,shape = [ 512 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w9 = tf.get_variable(name='tf_w9', trainable = trainable, shape = [1, 1, 512, 256], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b9 = tf.get_variable(name='tf_b9',trainable = trainable,shape = [ 256 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w10 = tf.get_variable(name='tf_w10', trainable = trainable, shape = [3, 3, 256, 512], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b10 = tf.get_variable(name='tf_b10',trainable = trainable,shape = [ 512 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w11 = tf.get_variable(name='tf_w11', trainable = trainable, shape = [1, 1, 512, 256], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b11 = tf.get_variable(name='tf_b11',trainable = trainable,shape = [ 256 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w12 = tf.get_variable(name='tf_w12', trainable = trainable, shape = [3, 3, 256, 512], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b12 = tf.get_variable(name='tf_b12',trainable = trainable,shape = [ 512 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w13 = tf.get_variable(name='tf_w13', trainable = trainable, shape = [1, 1, 512, 256], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b13 = tf.get_variable(name='tf_b13',trainable = trainable,shape = [ 256 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w14 = tf.get_variable(name='tf_w14', trainable = trainable, shape = [3, 3, 256, 512], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b14 = tf.get_variable(name='tf_b14',trainable = trainable,shape = [ 512 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w15 = tf.get_variable(name='tf_w15', trainable = trainable, shape = [1, 1, 512, 512], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b15 = tf.get_variable(name='tf_b15',trainable = trainable,shape = [ 512 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w16 = tf.get_variable(name='tf_w16', trainable = trainable, shape = [3, 3, 512, 1024], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b16 = tf.get_variable(name='tf_b16',trainable = trainable,shape = [ 1024 ], initializer = tf.constant_initializer(0.0) )
        
        self.tf_w17 = tf.get_variable(name='tf_w17', trainable = trainable, shape = [1, 1, 1024, 512], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b17 = tf.get_variable(name='tf_b17',trainable = trainable,shape = [ 512 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w18 = tf.get_variable(name='tf_w18', trainable = trainable, shape = [3, 3, 512, 1024], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b18 = tf.get_variable(name='tf_b18',trainable = trainable,shape = [ 1024 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w19 = tf.get_variable(name='tf_w19', trainable = trainable, shape = [1, 1, 1024, 512], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b19 = tf.get_variable(name='tf_b19',trainable = trainable,shape = [ 512 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w20 = tf.get_variable(name='tf_w20', trainable = trainable, shape = [3, 3, 512, 1024], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b20 = tf.get_variable(name='tf_b20',trainable = trainable,shape = [ 1024 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w21 = tf.get_variable(name='tf_w21', trainable = trainable, shape = [3, 3, 1024, 1024], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b21 = tf.get_variable(name='tf_b21',trainable = trainable,shape = [ 1024 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w22 = tf.get_variable(name='tf_w22', trainable = trainable, shape = [3, 3, 1024, 1024], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b22 = tf.get_variable(name='tf_b22',trainable = trainable,shape = [ 1024 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w23 = tf.get_variable(name='tf_w23', trainable = trainable, shape = [3, 3, 1024, 1024], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b23 = tf.get_variable(name='tf_b23',trainable = trainable,shape = [ 1024 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w24 = tf.get_variable(name='tf_w24', trainable = trainable, shape = [3, 3, 1024, 1024], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b24 = tf.get_variable(name='tf_b24',trainable = trainable,shape = [ 1024 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w25 = tf.get_variable(name='tf_w25', trainable = True, shape = [50176, 512 ], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b25 = tf.get_variable(name='tf_b25',trainable = True,shape = [ 512 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w26 = tf.get_variable(name='tf_w26', trainable = True, shape = [512 , 4096], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b26 = tf.get_variable(name='tf_b26',trainable = True,shape = [ 4096 ], initializer = tf.constant_initializer(0.0) )

        self.tf_w27 = tf.get_variable(name='tf_w27', trainable = True, shape = [4096 , 7*7*6], initializer = tf.contrib.layers.xavier_initializer() )
        self.tf_b27 = tf.get_variable(name='tf_b27',trainable = True,shape = [ 7*7*6 ], initializer = tf.constant_initializer(0.0) )
        #self.tf_w27 = tf.get_variable(name='tf_w27', trainable = True, shape = [4096 , 1470], initializer = tf.contrib.layers.xavier_initializer() )
        #self.tf_b27 = tf.get_variable(name='tf_b27',trainable = True,shape = [ 1470 ], initializer = tf.constant_initializer(0.0) )
        print('权重获取完毕')

    def get_weights(self):
        # 挨个把里面的值读出来，暂时先这样，以后再想办法搞个npy文件
        self.w_1 = self.reader.get_tensor('Variable')
        self.b_1 = self.reader.get_tensor('Variable_1')
        self.w_2 = self.reader.get_tensor('Variable_2')
        self.b_2 = self.reader.get_tensor('Variable_3')
        self.w_3 = self.reader.get_tensor('Variable_4')
        self.b_3 = self.reader.get_tensor('Variable_5')
        self.w_4 = self.reader.get_tensor('Variable_6')
        self.b_4 = self.reader.get_tensor('Variable_7')
        self.w_5 = self.reader.get_tensor('Variable_8')
        self.b_5 = self.reader.get_tensor('Variable_9')
        self.w_6 = self.reader.get_tensor('Variable_10')
        self.b_6 = self.reader.get_tensor('Variable_11')
        self.w_7 = self.reader.get_tensor('Variable_12')
        self.b_7 = self.reader.get_tensor('Variable_13')
        self.w_8 = self.reader.get_tensor('Variable_14')
        self.b_8 = self.reader.get_tensor('Variable_15')
        self.w_9 = self.reader.get_tensor('Variable_16')
        self.b_9 = self.reader.get_tensor('Variable_17')
        self.w_10 = self.reader.get_tensor('Variable_18')
        self.b_10 = self.reader.get_tensor('Variable_19')
        self.w_11 = self.reader.get_tensor('Variable_20')
        self.b_11 = self.reader.get_tensor('Variable_21')
        self.w_12 = self.reader.get_tensor('Variable_22')
        self.b_12 = self.reader.get_tensor('Variable_23')
        self.w_13 = self.reader.get_tensor('Variable_24')
        self.b_13 = self.reader.get_tensor('Variable_25')
        self.w_14 = self.reader.get_tensor('Variable_26')
        self.b_14 = self.reader.get_tensor('Variable_27')
        self.w_15 = self.reader.get_tensor('Variable_28')
        self.b_15 = self.reader.get_tensor('Variable_29')
        self.w_16 = self.reader.get_tensor('Variable_30')
        self.b_16 = self.reader.get_tensor('Variable_31')
        self.w_17 = self.reader.get_tensor('Variable_32')
        self.b_17 = self.reader.get_tensor('Variable_33')
        self.w_18 = self.reader.get_tensor('Variable_34')
        self.b_18 = self.reader.get_tensor('Variable_35')
        self.w_19 = self.reader.get_tensor('Variable_36')
        self.b_19 = self.reader.get_tensor('Variable_37')
        self.w_20 = self.reader.get_tensor('Variable_38')
        self.b_20 = self.reader.get_tensor('Variable_39')
        self.w_21 = self.reader.get_tensor('Variable_40')
        self.b_21 = self.reader.get_tensor('Variable_41')
        self.w_22 = self.reader.get_tensor('Variable_42')
        self.b_22 = self.reader.get_tensor('Variable_43')
        self.w_23 = self.reader.get_tensor('Variable_44')
        self.b_23 = self.reader.get_tensor('Variable_45')
        self.w_24 = self.reader.get_tensor('Variable_46')
        self.b_24 = self.reader.get_tensor('Variable_47')
        self.w_25 = self.reader.get_tensor('Variable_48')
        self.b_25 = self.reader.get_tensor('Variable_49')
        self.w_26 = self.reader.get_tensor('Variable_50')
        self.b_26 = self.reader.get_tensor('Variable_51')
        #self.w_27 = self.reader.get_tensor('Variable_52')
        #self.b_27 = self.reader.get_tensor('Variable_53')

net = TinyNet()
img = cv2.imread('juanji.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
inputs = np.zeros((1,448,448,3),dtype='float32')
inputs[0] = (img/255.0)*2.0-1.0
output = net.cul(inputs)
print(output.shape)
#np.save('conv.npy',output)

"""
probs = np.zeros((7,7,2,20))
print(output.shape)
class_probs = np.reshape(output[0:980],(7,7,20))
scales = np.reshape(output[980:1078],(7,7,2))
boxes = np.reshape(output[1078:],(7,7,2,4))
offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))
#for j in class_probs:
#    for i in j:
#        print(np.where(i==np.max(i)))
"""