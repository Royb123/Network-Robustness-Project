import subprocess
import numpy as np
from sklearn.decomposition import PCA
import csv
import os
import time
import  keras
import datetime
import sys

"""
cmd arguments:
    1. network file name
    2. start index of images to search from
"""
IMG_HIGHT = 28
IMG_WIDTH = 28
IMG_LAYERS = 1
MAX_PCA_DIM = IMG_HIGHT * IMG_WIDTH * IMG_LAYERS
NUM_OF_IMAGES = 784
start_indx = sys.argv[2]

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, IMG_HIGHT , IMG_WIDTH, IMG_LAYERS)
#IMGS_PATH = r"C:\Users\zinki\Desktop\Technion\integrativy\14.12.2020"

def get_class_indexes(dataset,labels,digit_to_analyze, start_ind):
    cnt= 0
    indexes = []
    for i in range(0, len(dataset)):
        if labels[i] == digit_to_analyze:
            if cnt >= start_ind:
               indexes.append(i)
            cnt += 1
        if cnt == NUM_OF_IMAGES+start_ind:
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

def save_single_img_csv(new_file_name, data_to_save):
    with open(new_file_name, 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data_to_save)

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
def run_in_ERAN(epsilon):
    output = subprocess.check_output(
        ['python3', '.', '--netname', '/root/models/' + network_name,
         '--epsilon', str(epsilon), '--domain', 'deepzono', '--dataset', 'mnist'])
    output_str = str(output)
    ind1 = output_str.find("analysis precision")
    len_of_analysis = len("analysis precision")
    new_result_str = output_str[ind1 + len_of_analysis+2]
    print('\n\n\n\n\n\n\n\n\n'+ str(new_result_str)+'\n\n\n\n\n')
    return new_result_str

'''
def binary_search():
    new_result_str = run_in_ERAN(0.5)
    epsilon = 0
    mid = 0.5
    min_ep = 0.0
    max_ep = 1.0
    cnt =0
    while cnt<=10:
        if new_result_str=='0':
            max_ep= mid
            mid= (max_ep+min_ep)/2.0
            new_result_str = run_in_ERAN(mid)
            cnt+=1
        else:
            cnt+=1
            min_ep = mid
            mid = (min_ep+max_ep)/2.0
            new_result_str = run_in_ERAN(mid)

    return mid
'''
def binary_search(lower_bound, upper_bound):
    mid = (lower_bound+upper_bound)/2
    new_result_str = run_in_ERAN(mid)
    epsilon = 0
    last_epsilon = -1
    mid = (lower_bound+upper_bound)/2
    min_ep = lower_bound
    max_ep = upper_bound
    cnt = 0
    while "{:.4f}".format(last_epsilon) != "{:.4f}".format(epsilon):
        if new_result_str == '0':
            max_ep = mid
            mid = (max_ep+min_ep)/2
            new_result_str = run_in_ERAN(mid)
            cnt += 1
        else:
            last_epsilon = epsilon
            epsilon = mid
            cnt += 1
            min_ep = mid
            mid = (min_ep+max_ep)/2
            new_result_str = run_in_ERAN(mid)

    return epsilon,cnt

def main():
    # put MNIST images into list
    digit_to_analyze = 0
    EPSILON= 0.2
    data = load_data(digit_to_analyze)
    img= data[10][1:]
    img=[float(i) for i in img]
    plot_changed_images(EPSILON, img)


for digit_to_analyze in range(10):
    #for PCA train
    class_indexes= get_class_indexes(train_images,train_labels,digit_to_analyze,int(start_indx))
    images_in_class= create_class_array(train_images,class_indexes,digit_to_analyze)
    save_class_to_csv("pca_mnist_class_digit_"+str(digit_to_analyze)+".csv", images_in_class)
    #for testing predictions
    '''tst_class_indexes = get_class_indexes(train_images, train_labels, digit_to_analyze, NUM_OF_IMAGES+1)
    tst_images_in_class = create_class_array(train_images, tst_class_indexes, digit_to_analyze)
    save_class_to_csv("tst_mnist_class_digit_" + str(digit_to_analyze) + ".csv", images_in_class)'''

#use PCA

new_file = ''
network_name = sys.argv[1]
# put MNIST images into list
for digit_to_analyze in range(1):
    cmd1 = ''
    cmd2 = ''
    if digit_to_analyze == 0:
        cmd1 = 'cp ../data/mnist_test.csv ../data/mnist_test.csv_ORIGINAL'
    else:
        cmd1 = 'cp ../data/mnist_test.csv '+new_file
    new_file = '../data/pca_mnist_class_digit_' + str(digit_to_analyze) + '.csv'
    os.system(cmd1)
    cmd2 = 'mv '+new_file+' ../data/mnist_test.csv'
    os.system(cmd2)


    pca_data = load_data("pca_mnist_class_digit_" + str(digit_to_analyze)+".csv")

    # arrange data & PCA
    pca_train_data = [np.array(image).astype(np.float64) for image in pca_data]
    
    date = datetime.datetime.now()
    day = str(date.day)
    month =str(date.month)
    hour = str(date.hour)
    minute = str(date.minute)
    timestamp_str = month+'_'+day+'_'+hour+'_'+minute

    results_file_name = '/root/results/'+network_name+'/binary_srch_epsilon_accurate_'+network_name+'_'+str(start_indx)+'_to_'+str(int(start_indx)+NUM_OF_IMAGES)+'.txt'
    new_results_file = open(results_file_name, 'a')

    # change images for different features
    for feature in range(83,84): #temporary
        for delta in range(0,1):

            for i in range(NUM_OF_IMAGES):
                changed_image = pca_train_data[i] 

                # clear file
                changed_image=changed_image.astype(int)
                open('../data/mnist_test.csv', 'w').close()

                # save changed image
                # save_data_to_csv(new_file_name, changed_image)
                save_single_img_csv('../data/mnist_test.csv', changed_image)


                #check for epsilon zero
                rounds= -4
                max_epsilon = -1
                start = time.time()
                new_result_str = run_in_ERAN(0.0)
                end = time.time()
                if new_result_str == '1':
                    start = time.time()
                    max_epsilon,rounds = binary_search(0,0.2)
                    end = time.time()
                new_results_file.write("img "+str(i)+" Network: " + network_name+" test of digit "+str(digit_to_analyze)+" delta" + str(delta) + " , feature: " + str(feature)+", max epsilon ")
                new_results_file.write(str(max_epsilon)+' , num of rounds: '+str(rounds)+' time: ' +str(end-start)+'\n')

                # delete file
                open('../data/mnist_test.csv', 'w').close()
            

#########################################

'''for digit_to_analyze in range(10):
    new_file_name = "pca_mnist_class_digit_"+str(digit_to_analyze)+".csv"
    os.system("rm ../data/"+new_file_name)
    os.system("rm "+new_file_name)'''




