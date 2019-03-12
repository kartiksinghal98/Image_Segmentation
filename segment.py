import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img=cv2.imread("./sample-img.jpg")
#img = cv2.resize(img, (0,0), fx=0.2, fy=0.2) 
#print(img.shape)
#img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def segmentation(img,dominant_colors):

    all_pixels=img.reshape((-1,img.shape[2]))
    km=KMeans(n_clusters=dominant_colors)

    km.fit(all_pixels)

    centers=np.array(km.cluster_centers_,dtype="uint8")
    color=[]

    for each_row in centers:
        color.append(each_row)

    new_img=np.zeros((img.shape[0]*img.shape[1],img.shape[2]),dtype="uint8")

    #print(new_img.shape[0])

    for ix in range(new_img.shape[0]):
        new_img[ix]=color[km.labels_[ix]]

    new_img=new_img.reshape((img.shape[0],img.shape[1],img.shape[2]))

    return new_img


new_img=segmentation(img,3)
 
cv2.imshow("original",img)
cv2.imshow("segmented",new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
