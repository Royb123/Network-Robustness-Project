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
from sklearn.cluster import KMeans
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
start_indx = int(sys.argv[2])
#start_indx_ep = sys.argv[3]
cmd_k = sys.argv[3]
num_of_kmean_runs = sys.argv[4]
kmean_tolerance = sys.argv[5]
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
        cnt += 1
    else:
        digits = np.vstack((digits,b))

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
    for image in pca_images_set:
        sum_i = 0
        for feature in features_lst:
            sum_i += abs(image[feature]-pca_image[feature])
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
        res.append(epsilon_lst[ind[0]])
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

    return epsilons


def binary_search(lower_bound, upper_bound):
    mid = (lower_bound+upper_bound)/2
    new_result_str = run_in_ERAN(mid)
    epsilon = 0
    last_epsilon = -1
    mid = (lower_bound+upper_bound)/2
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


def create_single_img_features_array(image, features_list):
    feature_cnt = 0
    single_img_features_array = []

    for feature in features_list:
        if feature_cnt == 0:
            single_img_features_array = image[feature]
            feature_cnt += 1
        else:
            single_img_features_array = np.append(single_img_features_array, image[feature])

    return single_img_features_array


def create_features_array(images_in_features_plane, features_list):
    images_features_array = []
    images_cnt = 0
    for image in images_in_features_plane:
        single_img_features = create_single_img_features_array(image, features_list)

        if images_cnt == 0:
            images_features_array = single_img_features
            images_cnt += 1
        else:
            images_features_array = np.vstack((images_features_array, single_img_features))

    return images_features_array


def compute_clusters_epsilons(images_features_array, num_clusters, epsilon_lst, num_of_kmean_runs, kmean_tolerance):
    clusters_epsilons = []
    k_means = KMeans(n_cluster= num_clusters, n_init = num_of_kmean_runs, tol = kmean_tolerance,
                     precompute_distances = True, algorithm = "full").fit(images_features_array)  #note that maximum number of iterations of the k-means algorithm for a single run is 300.
    clusters_labels = k_means.labels_
    for cluster in range(num_clusters):   #extract array of images in cluster then extract max and min eps for each cluster
        cluster_cnt = 0
        tmp_max_eps = 0
        tmp_min_eps = np.inf
        for img_indx in range(len(images_features_array)):
            if clusters_labels == cluster:
                tmp_max_eps = max(tmp_max_eps, epsilon_lst[img_indx])
                tmp_min_eps = min(tmp_min_eps, eps_lst[img_indx])
        if cluster_cnt == 0:
            clusters_epsilons = ([tmp_min_eps,tmp_max_eps])
            cluster_cnt += 1
        else:
            clusters_epsilons = np.vstack(clusters_epsilons, ([tmp_min_eps,tmp_max_eps]))

    return clusters_epsilons, k_means #first element is min then max


pca_matrix = 0
epsilon_tst_images = 0
pca_img_lst = []  # images in features plane

date = datetime.datetime.now()
day = str(date.day)
month =str(date.month)
hour = str(date.hour)
minute = str(date.minute)
timestamp_str = month+'_'+day+'_'+hour+'_'+minute


for digit_to_analyze in range(1):  # just for zero
    # new images to binary search according to estimation
    tst_indexes = get_class_indexes(train_images, train_labels, digit_to_analyze, start_indx)
    epsilon_tst_images = create_class_array(train_images, tst_indexes, digit_to_analyze)
    # images with known epsilon, used for estimating the new images epsilon and calculating the pca matrix.
    images_epsilon_estimate = load_data("pca_mnist_class_digit_" + str(digit_to_analyze) + ".csv")
    estimate_epsilon_data = [np.array(image[1:]).astype(np.float64) for image in images_epsilon_estimate]

    pca_matrix = calc_pca(estimate_epsilon_data, MAX_PCA_DIM)

    for img in estimate_epsilon_data:
        reshaped_img = img.reshape(1, IMG_WIDTH * IMG_HIGHT)/255.0
        img_pca = pca_matrix.transform(reshaped_img)[0]
        pca_img_lst.append(img_pca)

    # using PCA to test binary search
    features = [28, 34, 65, 3, 49, 53, 37, 47, 44, 64]  # fill this
    cnt = NUM_OF_IMAGES + 1
    eps_lst = create_epsilon_lst()
    clusters_epsilons = []
    images_features_array = create_features_array(pca_img_lst, features)
    results_file_name = '/root/results/'+network_name+'/estimate_binary_srch_epsilon_'+network_name+'_imgs_'+str(start_indx)+'_to_'+str(int(start_indx)+NUM_OF_IMAGES)+'_'+timestamp_str+'.txt'
    new_results_file = open(results_file_name, 'a')
    if len(sys.argv) == 3:
        closest_k = 10
    else:
        closest_k = cmd_k
    num_of_clusters = round(NUM_OF_IMAGES/closest_k)
    new_results_file.write('using k = '+str(closest_k)+', features list: '+str(features)+', '+str(len(features))+' features \n')

    for image in epsilon_tst_images:
        reshaped_img = image[1:].reshape(1, IMG_WIDTH * IMG_HIGHT)/255.0
        img_pca = pca_matrix.transform(reshaped_img)[0]
        img_features_array = create_single_img_features_array(image, features)
        clusters_epsilons, k_means = compute_clusters_epsilons(images_features_array, num_of_clusters, eps_lst, num_of_kmean_runs, kmean_tolerance)
        img_cluster = k_means.predict(img_features_array)
        img_epsilons = clusters_epsilons[img_cluster[0]]
        lower = img_epsilons[0]
        upper = img_epsilons[1]

        changed_image = image
        changed_image = changed_image.astype(int)

        # clear file
        open('../data/mnist_test.csv', 'w').close()

        save_single_img_csv('../data/mnist_test.csv', changed_image)
        # check for epsilon zero
        max_epsilon = -1
        start = time.time()
        new_result_str = run_in_ERAN(0.0)
        end = time.time()
        if new_result_str == '1':
            start = time.time()
            max_epsilon = binary_search(lower, upper)
            end = time.time()
        new_results_file.write(
            "img " + str(cnt) + " Network: " + network_name + " test of digit " + str(digit_to_analyze) +
            ", max epsilon ")
        new_results_file.write(str(max_epsilon) + ' time:  ' + str(end-start) + ' lower bound: ' + str(lower) +
                               ' upper bound: ' + str(upper) + ' size of cluster: ' + str(closest_k) + ' \n')

        # delete file
        open('../data/mnist_test.csv', 'w').close()
        cnt += 1


    new_results_file.close()







