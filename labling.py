import tensorflow as tf
import cv2
img = cv2.imread('dataset_for_orign/2017-12-02_15-28-38.png')

#cv2.rectangle(img,(47,0),(130,50),(55,255,155),1)

#cv2.rectangle(img,(961,109),(1047,167),(55,255,155),1)

#cv2.rectangle(img,(280,515),(420,592),(55,255,155),1)

#cv2.rectangle(img,(1420,619),(1497,671),(55,255,155),1)
rec = []
rec.append([47,0,130,50])
rec.append([961,109,1047,167])
rec.append([280,515,420,592])
rec.append([1420,619,1497,671])

for i in range(4):
    data = rec[i]
    temp = [(data[0]+data[2])/2, (data[1]+data[3])/2, data[2]-data[0], data[3]-data[1]]
    rec[i] = temp

print(rec)

"""
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""