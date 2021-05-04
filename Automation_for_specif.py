import subprocess
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
from time import time
import datetime
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import csv
import keras

IMG_HIGHT = 28
IMG_WIDTH = 28
IMG_LAYERS = 1
MAX_PCA_DIM = IMG_HIGHT * IMG_WIDTH * IMG_LAYERS
NUM_OF_IMAGES = 784


(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
print("how many??"+str(len(train_images)))
#train_images = train_images.reshape(-1, IMG_HIGHT , IMG_WIDTH, IMG_LAYERS)
clip_up = 1
clip_down = 0


def get_class_indexes(dataset,labels,digit_to_analyze, start_ind):
    cnt= 0
    indexes = []
    for i in range(start_ind, len(dataset) ):
      if labels[i] == digit_to_analyze:
        indexes.append(i)
        cnt+=1
        if cnt==NUM_OF_IMAGES:
            break
    return indexes


def create_class_array(dataset,indexes,digit_to_analyze): # returns the images in one stack. array of the images arrays
  digits = []
  cnt = 0
  for i in indexes:
    im = dataset[i]
    b = im.reshape(-1,IMG_LAYERS*IMG_HIGHT*IMG_WIDTH)
    b = np.insert(b,0,int(digit_to_analyze))
    if cnt == 0:
        digits = b
    else:
        digits = np.vstack((digits,b))
    cnt += 1
  return digits

def save_data_to_csv(new_file_name, data_to_save):
    with open(new_file_name, 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_to_save)

def save_class_to_csv(new_file_name, data_to_save):
    images_in_class_list=[]
    for image in data_to_save:
        lst=list(image)
        lst[0]=int(lst[0])
        images_in_class_list.append(lst)
    save_data_to_csv(new_file_name, images_in_class_list)
    images_in_class_list.clear()
    os.system("cp "+new_file_name+" ../data/")

def load_data(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def calc_pca(pca_train_data, pca_dim):
    pca = PCA(n_components=pca_dim)
    pca.fit(pca_train_data)
    return pca

def create_img_by_delta(delta, feature, img_org, pca):
    reshaped_img = img_org.reshape(1, IMG_WIDTH*IMG_HIGHT)/255
    img_pca = pca.transform(reshaped_img)[0]
    img_pca[feature] += delta
    img = pca.inverse_transform(img_pca)
    img = np.insert(img, 0, digit_to_analyze)
    return img

digit_to_analyze= 1
delta= 1
features= [57,15,31,10,2,6]
network_name='mnist_relu_6_100.tf'
#for PCA train
class_indexes= get_class_indexes(train_images,train_labels,digit_to_analyze,0)
images_in_class= create_class_array(train_images,class_indexes,digit_to_analyze)
save_class_to_csv("pca_mnist_class_digit_"+str(digit_to_analyze)+".csv", images_in_class)

#for testing predictions
tst_class_indexes = get_class_indexes(train_images, train_labels, digit_to_analyze, 2*NUM_OF_IMAGES)
tst_images_in_class = create_class_array(train_images, tst_class_indexes, digit_to_analyze)
save_class_to_csv("tst_mnist_class_digit_" + str(digit_to_analyze) + ".csv", tst_images_in_class)


pca_data = load_data("pca_mnist_class_digit_" + str(digit_to_analyze)+".csv")
pred_data = load_data("tst_mnist_class_digit_" + str(digit_to_analyze)+".csv")

# arrange data & PCA
pca_train_data = [np.array(image[1:]).astype(np.float64) for image in pca_data]
tst_pred_data = [np.array(image[1:]).astype(np.float64) for image in pred_data]
pca_dim = min(MAX_PCA_DIM, len(pca_train_data))
pca = calc_pca(pca_train_data, pca_dim)

results_file_name = 'results_pred_spec.txt'
new_results_file = open(results_file_name, 'a')


# change images for different features
for feature in features:
    changed_images = []
    for i in range(NUM_OF_IMAGES):
        print(i)
        changed_images.append(create_img_by_delta(delta, feature, tst_pred_data[i], pca))

    # save changed images
    new_file_name= "digit_"+str(digit_to_analyze)+"_feature_"+str(feature)+"_delta_"+str(delta)+".csv"
    save_data_to_csv(new_file_name, changed_images)

    # with is like your try .. finally block in this case
    with open('__main__.py', 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    # now change the 2nd line, note that you have to add a newline
    data[29] = 'changed_images=\"digit_'+str(digit_to_analyze)+'_feature_'+str(feature)+'_delta_' +str(delta)+'.csv\"\n'
    # and write everything back
    with open('__main__.py', 'w') as file:
        file.writelines(data)

    output = subprocess.check_output(
            ['python3', '.', '--netname', '/root/ERAN_9/tf_verify/models/'+network_name,
             '--domain', 'deepzono', '--dataset', 'mnist'])
    output_str = str(output)
    ind1 = output_str.find("analysis precision")
    len_of_analysis = len("analysis precision")
    new_result_str = output_str[ind1 + len_of_analysis:]
    new_results_file.write("Network: " + network_name+" test of digit "+str(digit_to_analyze)+" delta" + str(delta) + " , feature: " + str(feature))
    new_results_file.write(new_result_str+'\n')
    # delete file
    os.system('rm '+new_file_name)


#########################################

new_file_name = "pca_mnist_class_digit_"+str(digit_to_analyze)+".csv"
os.system("rm ../data/"+new_file_name)
os.system("rm "+new_file_name)





