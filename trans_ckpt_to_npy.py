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
#print(weights.shape)