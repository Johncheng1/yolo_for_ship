import tensorflow as tf
import numpy as np
import cv2
import net
import read_ckpt
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
def sigmoid(x):
    return 1/(1+np.exp(-x))
''' m = np.load('result.npy')
m0 = m.reshape([6,7,7])[0]
m1 = m.reshape([6,7,7])[1]
print(np.where(m0>0.7))
print(np.where(m1>0.7))
m = m.reshape([6,7,7])
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
cv2.destroyAllWindows()  '''
img = cv2.imread('dataset/img.png')
#img = cv2.imread('dataset_for_orign/2017-12-02_15-33-51.png')
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#img=cv2.resize(img,(448,448),interpolation=cv2.INTER_CUBIC)
inputs = np.zeros((1,448,448,3),dtype='float32')
inputs[0] = (img/255.0)*2.0-1.0
yolo = net.Net(read_ckpt.layer_names, read_ckpt.weights, 0)
output = yolo.run(inputs)
m = output.reshape([6,7,7])
print(m[0])
th = 0.5

s = m[0] 
s_y_l = np.where(s>th)[0]
s_x_l = np.where(s>th)[1]
for i in range(len(list(s_y_l))): 
    
    s_y = s_y_l[i]
    s_x = s_x_l[i]
    
    print(m[0][s_y][s_x])
    
    pos_y = (m[3][s_y][s_x] + s_y) * 64
    pos_x = (m[2][s_y][s_x] + s_x) * 64
    pos_x = int(pos_x)
    pos_y = int(pos_y)
    w = int(m[4][s_y][s_x] * 32)
    h = int(m[5][s_y][s_x] * 32)
    cv2.rectangle(img, (pos_x-w,pos_y-h), (pos_x+w,pos_y+h), (255,255,255), 1)

img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

cv2.imshow("shape", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
''' a = np.arange(10)
s = np.where(a>5)
print(s)
for i in s[0]:
    print(i) '''