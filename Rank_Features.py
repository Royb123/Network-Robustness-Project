import subprocess
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#%matplotlib inline
import math
from time import time
import datetime
import os
import sys
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import csv
import keras

"""
cmd arguments:
    1. network file name
    2 & 3. amount of features to scan (from x to y)
    4. limit on the delta (from 1 to ..)
"""

###########SPLIT TO CLASSES
img_h = 28
img_w = 28
img_layers = 1
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, img_h , img_w,img_layers) 
clip_up = 1
clip_down = 0


def get_class_indexes(dataset,labels,digit_to_analyze):
    cnt= 0
    indexes = []
    for i in range( len(dataset) ):
      if labels[i] == digit_to_analyze:
        indexes.append(i)
        cnt+=1
        if cnt==784:
            break
    return indexes


def create_class_array(dataset,indexes,digit_to_analyze): # returns the images in one stack. array of the images arrays
  digits = []
  cnt = 0
  for i in indexes:
    im = dataset[i]
    b = im.reshape(-1,img_layers*img_h*img_w)
    b = np.insert(b,0,int(digit_to_analyze))
    if cnt == 0:
        digits = b
    else:
        digits = np.vstack((digits,b))
    cnt += 1
  return digits

images_in_class_list=[]
for digit_to_analyze in range(10):
    class_indexes= get_class_indexes(train_images,train_labels,digit_to_analyze)
    images_in_class= create_class_array(train_images,class_indexes,digit_to_analyze)

    #plot to check if correct difits
    #img_1 = images_in_class[10][1:].reshape(1,img_h*img_w*img_layers)
    #img_2 =  images_in_class[99][1:].reshape(1,img_h*img_w*img_layers)
    #plt.imshow(img_1.reshape(img_h , img_w), cmap='gray', vmin=0, vmax=1)
    #plt.show()
    #plt.imshow(img_2.reshape(img_h , img_w), cmap='gray', vmin=0, vmax=1)
    #plt.show()
    lst=[]
    new_file_name = "mnist_class_digit_"+str(digit_to_analyze)+".csv"
    for image in images_in_class:
        lst=list(image)
        lst[0]=int(lst[0])
        images_in_class_list.append(lst)

    with open(new_file_name, 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(images_in_class_list)
    images_in_class_list.clear()
    os.system("cp "+new_file_name+" ../data/")

###########PUT IMAGES INTO PCA
cnt = 0
img_h = 28
img_w = 28
img_layers = 1
pca_dim = img_h*img_w*img_layers
num_of_images = 784
new_file = ''
network_name = sys.argv[1]
# put MNIST images into list
for digit_to_analyze in range(10):
    cmd1 = ''
    cmd2 = ''
    if digit_to_analyze == 0:
        cmd1 = 'cp ../data/mnist_test.csv ../data/mnist_test.csv_ORIGINAL'
    else:
        cmd1 = 'cp ../data/mnist_test.csv '+new_file
    new_file = '../data/mnist_class_digit_' + str(digit_to_analyze) + '.csv'
    os.system(cmd1)
    cmd2 = 'mv '+new_file+' ../data/mnist_test.csv'
    os.system(cmd2)
    # load data for the class digit
    file_name = "mnist_class_digit_" + str(digit_to_analyze)+".csv"
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    # arrange data
    pca_train_data = []
    cnt = 0
    for image in data:
        b = image[1:]
        b = np.array(b)
        b = b.astype(np.float64)
        pca_train_data.append(b)

    # calculate pca
    if len(pca_train_data) < pca_dim:
        pca_dim = len(pca_train_data)

    pca = PCA(n_components=pca_dim)
    pca.fit(pca_train_data)
    date = datetime.datetime.now()
    day = str(date.day)
    month =str(date.month)
    hour = str(date.hour)
    minute = str(date.minute)
    timestamp_str = month+'_'+day+'_'+hour+'_'+minute
    results_file_name = '/root/results/'+network_name+'/features_rank_'+network_name+'_'+timestamp_str+'.txt'
    new_results_file = open(results_file_name, 'a')
    # change images for different features
    for pca_feature_to_change in range(int(sys.argv[2]),int(sys.argv[3])):
        for change_value in range(1,int(sys.argv[4])):
            changed_images = []
            for i in range(num_of_images):
                img_org = pca_train_data[i].reshape(1, img_w * img_h)/255
                img_pca = pca.transform(img_org)[0]
                img_pca[pca_feature_to_change] += change_value
                img = pca.inverse_transform(img_pca)
                img = np.insert(img, 0, digit_to_analyze)
                changed_images.append(img)

            # save changed images
            new_file_name = "digit_"+str(digit_to_analyze)+"_feature_"+str(pca_feature_to_change)+"_delta_"+str(change_value)+".csv"
            with open(new_file_name, 'w+', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(changed_images)

            # with is like your try .. finally block in this case
            with open('__main__.py', 'r') as file:
                # read a list of lines into data
                data = file.readlines()
            # now change the 2nd line, note that you have to add a newline
            data[29] = 'changed_images=\"digit_'+str(digit_to_analyze)+'_feature_'+str(pca_feature_to_change)+'_delta_' +str(change_value)+'.csv\"\n'
            # and write everything back
            with open('__main__.py', 'w') as file:
                file.writelines(data)

            output = subprocess.check_output(
                    ['python3', '.', '--netname', '/root/models/'+network_name,
                     '--domain', 'deepzono', '--dataset', 'mnist'])
            output_str = str(output)
            ind1 = output_str.find("analysis precision")
            len_of_analysis = len("analysis precision")
            new_result_str = output_str[ind1 + len_of_analysis:]
            new_results_file.write("Network: " + network_name+" test of digit "+str(digit_to_analyze)+" delta" + str(change_value) + " , feature: " + str(pca_feature_to_change))
            new_results_file.write(new_result_str+'\n')
            # delete file
            os.system('rm '+new_file_name)


#########################################

for digit_to_analyze in range(10):
    new_file_name = "mnist_class_digit_"+str(digit_to_analyze)+".csv"
    os.system("rm ../data/"+new_file_name)
    os.system("rm "+new_file_name)




