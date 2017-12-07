import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn import preprocessing
def zuhe(width,height,size,input):
    mat = np.zeros([height*size, width*size ])
    for j in range(width):
        for i in range(height):
            index = i * width + j
            img = input[:,:,index]
            #print(img)
            mat[i*size:(i+1)*size,j*size:(j+1)*size] = img
            #print(a.shape)
    return mat
input = np.load('juanji.npy')[0]
print(input[:,:,0].shape)
mat = zuhe(8,8,112,input)
#scaler = preprocessing.MinMaxScaler()
#mat = scaler.fit_transform(mat) * 255
plt.imshow(mat, cmap='gray')
plt.xticks([],plt.yticks([]))  # to hide tick values on X and Y axis
plt.show()
