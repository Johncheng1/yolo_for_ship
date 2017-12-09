import numpy as np
def get_train_data(dataset, batch_size):
    index = np.random.uniform(0,4542,size=128).astype(int)
    train_data = []
    train_label = []
    for i in index:
        train_data.append(dataset[i][0][0])
        label = get_matrix(dataset[i][1])
        train_label.append(label)
    return train_data, train_label
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

dataset = np.load('fc_dataset.npy')
''' a,b = get_train_data(dataset, 10)
print(a[0]) '''