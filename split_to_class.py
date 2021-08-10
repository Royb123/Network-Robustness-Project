# Setup
import os
import numpy as np
import sklearn
import torch
import keras
import os, sys
import keras
import numpy as np
import tensorflow as tf
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import csv


img_h = 28
img_w = 28
img_layers = 1
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, img_h , img_w,img_layers) / 255.0
clip_up = 1
clip_down = 0
num_of_pic= 784

def get_class_indexes(dataset,labels,digit_to_analyze):

    cnt= 0
    indexes = []
    for i in range( len(dataset) ):
      if labels[i] == digit_to_analyze:
        indexes.append(i)
        cnt+=1
        if cnt==num_of_pic:
            break

    return indexes

def create_class_array(dataset,indexes): #returns the images in one stack. array of the images arrays
  digits = []
  cnt = 0
  for i in indexes:
    im = dataset[i]
    b = im.reshape(-1,img_layers*img_h*img_w)
    b = np.insert(b,0,digit_to_analyze)
    if cnt == 0:
        digits = b
    else:
        digits = np.vstack((digits,b))
    cnt += 1
  return digits


for digit_to_analyze in range(10):
    class_indexes= get_class_indexes(train_images,train_labels,digit_to_analyze)
    images_in_class= create_class_array(train_images,class_indexes)

    #plot to check if correct difits
    #img_1 = images_in_class[10][1:].reshape(1,img_h*img_w*img_layers)
    #img_2 =  images_in_class[99][1:].reshape(1,img_h*img_w*img_layers)
    #plt.imshow(img_1.reshape(img_h , img_w), cmap='gray', vmin=0, vmax=1)
    #plt.show()
    #plt.imshow(img_2.reshape(img_h , img_w), cmap='gray', vmin=0, vmax=1)
    #plt.show()


    new_file_name = "mnist_class_digit_"+str(digit_to_analyze)+".csv"
    with open(new_file_name, 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(images_in_class)
