import numpy as np
import cv2
def read_dataset(filename):
    dataset = np.load('dataset.npy')
    return dataset
def get_matrix(data):
    matrix = np.zeros([6,7,7])
    #cv2.imshow('image',data[0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    data = data[1]
    
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
def get_train_data(dataset):
    train_data = []
    train_label = []
    for data in dataset:
        label = get_matrix(data)
        train_label.append(label)
        train_data.append(data[0])
    return train_data, train_label

filename = 'dataset.npy'
dataset = read_dataset(filename)
train_data, train_label = get_train_data(dataset)