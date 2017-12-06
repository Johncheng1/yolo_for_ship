import tensorflow as tf
import numpy as np
weights_file='YOLO_small.ckpt'
#reader = tf.train.NewCheckpointReader(weights_file)
#a = reader.get_tensor('Variable')
#print(a.shape)
weights = []
names = [ 'Variable_' + str(x) for x in range(54)]
names[0] = 'Variable'
reader = tf.train.NewCheckpointReader(weights_file)
weights.append(reader.get_tensor('Variable'))

layers = ['conv1','max_pool1','conv2','max_pool2',
            'conv3_1','conv3_2','conv3_3','conv3_4','max_pool3',
            'conv4_1','conv4_2','conv4_3','conv4_4','conv4_5','conv4_6','conv4_7','conv4_8','conv4_9','conv4_10','max_pool4']