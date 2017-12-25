import tensorflow as tf
import numpy as np
import cv2
#import load
def get_matrix(data):
    matrix = np.zeros([6,7,7])
    #cv2.imshow('image',data[0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    x = int (data[0] // 64)
    y = int (data[1] // 64)
    px = (data[0] - 64 * x) / 64
    py = (data[1] - 64 * y) / 64
    width = data[2] / 64
    height = data[3] / 64
    matrix[0][y][x] = 1
    matrix[1][y][x] = 1
    matrix[2][y][x] = px
    matrix[3][y][x] = py
    matrix[4][y][x] = width
    matrix[5][y][x] = height
    
    matrix = matrix.reshape(6*7*7)
    #matrix = matrix.reshape([6,7,7])
    return matrix
''' input = tf.Variable([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]], name="counter")
result = tf.slice(input, [0,2], [2,4])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
a = sess.run(result)
print(load.train_data[0]) '''
''' dataset = np.load('fc_dataset.npy')
label = dataset[0][1]
m = get_matrix(label)
m = m.reshape([6,7,7])
np.save('result.npy',m) '''
m = np.load('result.npy')
#print(m)
s = m[0]*m[1]
s_y = np.where(s>0.2)[0][0]
s_x = np.where(s>0.2)[1][0]
pos_y = (m[3][s_y][s_x] + s_y) * 64
pos_x = (m[2][s_y][s_x] + s_x) * 64
pos_x = int(pos_x)
pos_y = int(pos_y)
w = int(m[4][s_y][s_x] * 32)
h = int(m[5][s_y][s_x] * 32)
img = cv2.imread('dataset/img0.png')
cv2.rectangle(img, (pos_x-w,pos_y-h), (pos_x+w,pos_y+h), (255,255,255), 1)
cv2.imshow("shape", img)
cv2.waitKey(0)
cv2.destroyAllWindows()