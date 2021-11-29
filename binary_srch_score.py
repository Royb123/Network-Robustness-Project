# This is a sample Python script.
import keras
import numpy as np
import matplotlib.pyplot as plt
import random
from operator import itemgetter

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


dataset_labels_setup = {
        'mnist': ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
        'cifar': ('airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks')
    }

dataset_test_labels_setup_func = {
        'mnist': lambda tl, i: tl[i],
        'cifar': lambda tl, i: tl[i][0]
    }


class Dataset(object):
    def __init__(self, name, width, height, train_images, train_labels, test_images, test_labels):
        self.name = name
        self.width = width
        self.height = height
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.labels = []
        self.organized_images = {}
        self.set_up_labels()

    def set_up_labels(self):
        if self.name in dataset_labels_setup:
            self.labels = dataset_labels_setup[self.name]
        else:
            raise Exception('only {} are supported'.format(dataset_labels_setup.keys()))

    def organize_to_labels(self):
        """
        input is from class dataset, output is a dictionary, keys are according to labels
        """
        organized_images = dict.fromkeys(self.labels)
        for i in range(len(self.test_labels)):
            key = self.labels[dataset_test_labels_setup_func[self.name](self.test_labels,i)]
            if organized_images[key] is None:
                organized_images[key] = [self.test_images[i]]
            else:
                organized_images[key] = np.append(organized_images[key], [self.test_images[i]], axis=0)
        self.organized_images = organized_images


def plot(image, label, name):  # input- image (without label). no output. plots image

    print(label)
    cmap_type = 'viridis'
    if name == 'mnist':
        cmap_type = 'gray'
    plt.imshow(image, cmap=cmap_type)
    plt.show()


def load_dataset(dataset_name, debug=False):

    name = dataset_name.lower().strip()

    if name == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        img_width = img_height = 28
    elif name == "cifar":
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
        img_width = img_height = 32
    else:
        raise Exception('only mnist and cifar are supported')

    images_dataset = Dataset(name, img_width, img_height, train_images, train_labels, test_images, test_labels)
    images_dataset.organize_to_labels()

    if debug:
        plot(images_dataset.test_images[2], images_dataset.test_labels[2], images_dataset.name)

    return images_dataset


def binary_search(lower_bound, upper_bound, is_in_range, floating_point=4):

    if is_in_range(lower_bound) == 0:
        print("epsilon is out of range, too small")
        return -1, 0
    if is_in_range(upper_bound) == 1:
        print('epsilon is out of range, too high')
        return -2, 0

    cnt = 0

    while ("{:.%sf}" % floating_point).format(upper_bound) != ("{:.%sf}" % floating_point).format(lower_bound):
        mid = (lower_bound + upper_bound)/2
        if is_in_range(mid) == 1:  # if epsilon >= mid
            lower_bound = mid
        else:
            upper_bound = mid
        cnt += 1
    return mid, cnt


def ERAN_demmy_func(mid, i):
    if mid <= 0.0004+i*0.0001:
        return 1
    else:
        return 0


def score_func(dataset, epsilon=0):
    # TODO
    pass


def choose_index(range_list):
    image_index = random.choice(range(len(range_list)))
    return image_index


def restart_images_range(dataset, lower_bound, upper_bound):
    """
            input is a list of images, it returns 2D list containing
            initialized range
    """
    range_list = []
    for i in range(len(dataset)):
        range_list.append([i, dataset[i], lower_bound, upper_bound])
    return range_list


def find_all_epsilons(images_bounderies, is_in_range, floating_point=4):

    epsilon_list = []
    while images_bounderies:
        i = choose_index(images_bounderies)
        upper_bound = images_bounderies[i][2]
        lower_bound = images_bounderies[i][3]
        mid_epsilon = (upper_bound+lower_bound)/2
        # mid_epsilon = (images_bounderies[i]["upper_bound"] + images_bounderies[i]["lower_bound"]) / 2
        is_robust = is_in_range(mid_epsilon, images_bounderies[i][0])
        if is_robust == 1:
            for j in range(i, len(images_bounderies)):
                images_bounderies[j][2] = mid_epsilon
        else:
            for k in range(0, i+1):
                images_bounderies[k][3] = mid_epsilon

        if ("{:.%sf}" % floating_point).format(images_bounderies[i][2])\
                == ("{:.%sf}" % floating_point).format(images_bounderies[i][3]):
            epsilon = ("{:.%sf}" % floating_point).format(images_bounderies[i][2])
            image_index = images_bounderies[i][0]
            images_bounderies.pop(i)
            epsilon_list.append([image_index, epsilon])

    return epsilon_list

def is_in_range_using_ERAN_by_CMD():
    pass


def is_in_range_using_ERAN_by_import():
    pass

if __name__ == "__main__":
    # epsilon, num_of_runs = binary_search(-132, 9879595, ERAN_demmy_func)
    # print('epsilon is '+str(epsilon)+' num of runs is '+str(num_of_runs))

    images = load_dataset('mnist')

    # plot(images.organized_images['horses'][1236], '3', 'cifar')
    # datasets = [] # TODO
    # datasets = sorted(datasets, key=score_func(datasets))

    # TODO change restart_images_range use with class
    NUM_OF_IMAGES = 10
    images_bounderies_list = restart_images_range(images.organized_images['0'][:NUM_OF_IMAGES], 0, 0.6)
    eps = find_all_epsilons(images_bounderies_list, is_in_range=ERAN_demmy_func)
    print(sorted(eps,key=itemgetter(0)))