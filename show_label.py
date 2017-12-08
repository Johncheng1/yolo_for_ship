import numpy as np
import cv2
dataset = np.load('dataset.npy')
#cv2.imwrite("juanji.png",dataset[0][0])
for data in dataset:
    img = data[0]
    rec = data[1]
    x1 = int(rec[0] - rec[2] / 2)
    x2 = int(rec[0] + rec[2] / 2)
    y1 = int(rec[1] - rec[3] / 2)
    y2 = int(rec[1] + rec[3] / 2)
    cv2.rectangle(img,(x1,y1),(x2,y2),(55,255,155),1)
    cv2.imshow('image',img)
    cv2.waitKey(0)
cv2.destroyAllWindows()