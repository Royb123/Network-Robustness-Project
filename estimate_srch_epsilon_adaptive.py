"""
for each image:
- get the PCA matrix
- find the closest image from the "training set" according to strongest features

- cmd arguments:
    1. network name (including suffix like ".tf")
    2. start index of images to look for (for example: 785)
    3. start index for the accurate epsilon file to compare to
    4. k, num of images closest to given image

"""
import subprocess
import numpy as np
from sklearn.decomposition import PCA
import csv
import os
import keras
import time
import sys
import datetime
IMG_HIGHT = 28
IMG_WIDTH = 28
IMG_LAYERS = 1
MAX_PCA_DIM = IMG_HIGHT * IMG_WIDTH * IMG_LAYERS
NUM_OF_IMAGES = 784
network_name = sys.argv[1]
start_indx_ep = sys.argv[3]
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, IMG_HIGHT , IMG_WIDTH, IMG_LAYERS) 
# IMGS_PATH = r"C:\Users\zinki\Desktop\Technion\integrativy\14.12.2020"


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
    reshaped_img = img_org.reshape(1, IMG_WIDTH*IMG_HIGHT)
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
    return new_result_str


def score_func(pca_image, k, features_lst, pca_images_set,epsilon_lst):
    k_closest_images = []
    ind = 0
    max_sum = 0
    #norm_pca_image=[(float(i)-min(pca_image))/(max(pca_image)-min(pca_image)) for i in pca_image]
    for image in pca_images_set:
        if epsilon_lst[ind]==-1.0:
            ind+=1
            continue
        sum_i = 0
        #norm_pca_image_set=[(float(i)-min(image))/(max(image)-min(image)) for i in image]
        for feature in features_lst:
            sum_i += abs(pca_image[feature]-image[feature])
        if len(k_closest_images) < k:
            k_closest_images.append((ind, sum_i))
            if sum_i>max_sum:
                max_sum=sum_i
        else:
            if sum_i < max_sum:
                curr_max = max(k_closest_images, key=lambda t: t[1])
                k_closest_images.remove(curr_max)
                k_closest_images.append((ind, sum_i))
                new_max = max(k_closest_images, key=lambda t: t[1])
                max_sum = new_max[1]

        ind += 1
    res=[]
    adaptive_res=[]
    for ind in k_closest_images:
        print(ind[0])
        res.append(epsilon_lst[ind[0]])
    print(res)
    print("##########################3")
    """
    res.sort()
    i=4
    adaptive_res.append(res[i])
    for i in range(5,8):
        if res[i] - res[i-1] > 0.0008:
            break
        adaptive_res.append(res[i])
    for i in range(3,1,-1):
        if res[i+1] - res[i] > 0.0008:
            break
        adaptive_res.append(res[i])
    """
    print("ressssss "+str(res))
    return res


def estimate_epsilon(k, k_images_lst, epsilon_lst):
    epsilon = 0
    for image in k_images_lst:
        epsilon += epsilon_lst[image[0]]

    epsilon = epsilon/k
    return epsilon


def create_epsilon_lst():
    epsilons = []
    eps_file = results_file_name = '/root/results/'+network_name+'/binary_srch_epsilon_accurate_'+network_name+'_0'+'_to_'+str(0+NUM_OF_IMAGES)+'.txt'
    with open(eps_file, 'r') as file:
        data = file.readlines()

    for line in data:
        new_limit = line.split(' ')
        new_limit = new_limit[14]
        new_limit = new_limit[:6]
        new_ep = float(new_limit)
        epsilons.append(new_ep)
    
    print(epsilons)
    return epsilons


def binary_search(lower_bound, upper_bound):
    mid = (lower_bound+upper_bound)/2
    new_result_str = run_in_ERAN(mid)
    epsilon = 0
    last_epsilon = -1
    #mid = (lower_bound+upper_bound)/2
    min_ep = lower_bound
    max_ep = upper_bound
    cnt = 0
    initial_check = run_in_ERAN(min_ep)
    if initial_check == '0':
        epsilon = -3
        return epsilon,cnt
    
    initial_check = run_in_ERAN(max_ep+0.0001)
    if initial_check == '1':
        epsilon = -2
        return epsilon,cnt

    while min_ep != max_ep:
        if round(max_ep-min_ep,4)==0.0001:
            new_result_str = run_in_ERAN(max_ep)
            cnt+=1
            if new_result_str == '1':
                return max_ep,cnt
            else:
                return min_ep,cnt
        if new_result_str == '0':
            max_ep = mid
            mid = round((max_ep+min_ep)/2,4)
            new_result_str = run_in_ERAN(mid)
            cnt += 1
        else:
            last_epsilon = epsilon
            epsilon = mid
            cnt += 1
            min_ep = mid
            mid = round((min_ep+max_ep)/2,4)
            new_result_str = run_in_ERAN(mid)

    return min_ep,cnt


pca_matrix = 0
images_in_class = 0
start_indx = int(sys.argv[2])
pca_img_lst=[]
for digit_to_analyze in range(1):  # just for zero
    # for PCA train
    class_indexes = get_class_indexes(train_images,train_labels,digit_to_analyze,start_indx)
    images_in_class = create_class_array(train_images,class_indexes,digit_to_analyze)
    #images_in_class =  load_data("pca_mnist_class_digit_" + str(digit_to_analyze) + ".csv")
    pca_data = load_data("pca_mnist_class_digit_" + str(digit_to_analyze) + ".csv")
    pca_train_data = [np.array(image[1:]).astype(np.float64) for image in pca_data]
    images_for_mnist_test = [np.array(image).astype(np.float64) for image in pca_data]
    pca_matrix = calc_pca(pca_train_data,MAX_PCA_DIM)


    for img in pca_train_data:
        reshaped_img = img.reshape(1, IMG_WIDTH * IMG_HIGHT)/255.0
        new_img_pca = pca_matrix.transform(reshaped_img)[0]
        pca_img_lst.append(new_img_pca)
    # ##### pca_train_data = [np.array(image).astype(np.float64) for image in pca_data]
    # for testing predictions


# use PCA
eps_lst = create_epsilon_lst()
closest_images_ep = []
features = [76,9,8,69,5,13,77,25,19,42]  # fill this
date = datetime.datetime.now()
day = str(date.day)
month =str(date.month)
hour = str(date.hour)
minute = str(date.minute)
timestamp_str = month+'_'+day+'_'+hour+'_'+minute
results_file_name = '/root/results/'+network_name+'/estimate_binary_srch_epsilon_'+network_name+'_imgs_'+str(start_indx)+'_to_'+str(int(start_indx)+NUM_OF_IMAGES)+'_'+timestamp_str+'.txt'
new_results_file = open(results_file_name, 'a')
closest_k = 0
if len(sys.argv)==3:
    closest_k = 10
else:
    closest_k = sys.argv[4]

new_results_file.write('using k = '+str(closest_k)+', features list: '+str(features)+', '+str(len(features))+' features \n')
cnt= NUM_OF_IMAGES+1
img_cnt = 0
for image in images_in_class:
    reshaped_img = image[1:].reshape(1, IMG_WIDTH * IMG_HIGHT)/255.0
    img_pca = pca_matrix.transform(reshaped_img)[0]
    closest_images_ep = score_func(img_pca,int(closest_k),features,pca_img_lst,eps_lst)
    lower = min(closest_images_ep)
    upper = max(closest_images_ep)
    changed_image = image #images_for_mnist_test[img_cnt]
    # for i in range(NUM_OF_IMAGES):
    changed_image = changed_image.astype(int)  # create_img_by_delta(delta, feature, pca_train_data[i], pca)

    # clear file
    open('../data/mnist_test.csv', 'w').close()

    # save changed image
    # save_data_to_csv(new_file_name, changed_image)
    save_single_img_csv('../data/mnist_test.csv', changed_image)
    # check for epsilon zero
    max_epsilon = -1
    start = time.time()
    new_result_str = run_in_ERAN(0.0)
    end = time.time()
    if new_result_str == '1':
        start = time.time()
        max_epsilon = binary_search(lower,upper)
        end = time.time()
    new_results_file.write(
        "img " + str(cnt) + " Network: " + network_name + " test of digit " + str(digit_to_analyze) +
                ", max epsilon ")
    new_results_file.write(str(max_epsilon) +' time:  '+str(end-start)+' lower bound: '+str(lower)+' upper bound: '+str(upper)+' size of cluster: '+str(len(closest_images_ep))+' \n')

    # delete file
    open('../data/mnist_test.csv', 'w').close()
    cnt+=1
    img_cnt+=1

new_results_file.close()
    






