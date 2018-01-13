# 检测船的位置用的
import net
import cv2
import numpy as np
if __name__ == "__main__":
    img = cv2.imread('test.jpg')
    img = cv2.resize(img,(448,448),interpolation=cv2.INTER_CUBIC)
    inputs = np.zeros((1,448,448,3),dtype='float32')
    inputs[0] = (img/255.0)*2.0-1.0
    net = net.Net(2, net.weights_small, net.layer_names)
    out = net.run(inputs)
    print(out)