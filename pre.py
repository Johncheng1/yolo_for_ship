import os
import cv2
import numpy as np
path='dataset_for_orign'
def get_rec():
    rec = []
    rec.append([47,0,130,50])
    rec.append([961,109,1047,167])
    rec.append([280,515,420,592])
    rec.append([1420,619,1497,671])

    for i in range(4):
        data = rec[i]
        temp = [(data[0]+data[2])/2, (data[1]+data[3])/2, data[2]-data[0], data[3]-data[1]]
        rec[i] = temp
    return rec
def is_in_rec(x, y, rec):
    x1 = rec[0] - rec[2] / 2
    x2 = rec[0] + rec[2] / 2
    y1 = rec[1] - rec[3] / 2
    y2 = rec[1] + rec[3] / 2
    if x < x1 and x+448 > x2 and y < y1 and y+448 > y2:
        return True
    else:
        return False
def read_img(filename,rec):
    img = cv2.imread(path + '/' + filename)
    # shape[0]是高 shape[1]是宽
    width = img.shape[1]
    height = img.shape[0]
    dataset = []
    if 448 < width and 448 < height:
        for y in range(0,height - 448, 5):
            for x in range(0,width - 448, 5):
                for data in rec:
                    if is_in_rec(x, y, data) == True:
                        mat = img[y:y+448,x:x+448,:]
                        t_rec = [data[0] - x, data[1] - y, data[2], data[3]]
                        dataset.append([mat, t_rec])
                        #name = filename[:-4] + '_' + str(x) + '_' + str(y) + '.png'
                        #cv2.imwrite('dataset/'+name,mat)
    dataset = np.array(dataset)
    np.save('dataset.npy',dataset)
"""
for dirpath,dirnames,filenames in os.walk(path):
    for filename in filenames:
            print(filename)
"""
rec = get_rec()
read_img('2017-12-02_15-28-38.png', rec)