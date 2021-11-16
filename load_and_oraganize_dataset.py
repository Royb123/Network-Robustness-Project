# This is a sample Python script.
import keras
import numpy as np
import matplotlib.pyplot as plt
import random

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


