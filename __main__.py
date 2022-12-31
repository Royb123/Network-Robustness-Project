"""
  Copyright 2020 ETH Zurich, Secure, Reliable, and Intelligent Systems Lab

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""
import random
import sys
import os

import argparse
from config import config

import logging

import keras
import numpy as np
import matplotlib.pyplot as plt
import csv
import subprocess
import heapq
import re
import time
import json
import traceback
from multiprocessing import Process, Queue, Pool
from itertools import product


EPS_IS_LOWER = -1
EPS_IS_HIGHER = -2
EPS_UNDEFINED = -3
IMG_UNRECOGNIZABLE = -4

MAX_EPS = 0.05
MIN_EPS = 0

VERSION = "3.1"
PRECISION = 4

USE_SUBPROCESS_AND_WAIT = True
TEST = False
LOGGER_PATH = r"/root/logging/user_logger"
OUTCOMES_PATH = '/root/ERAN/tf_verify/outcomes_{version}.json'.format(version=VERSION)


dataset_labels_setup = {
        'mnist': ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
        'cifar': ('airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks')
    }

dataset_labels_setup_func = {
        'mnist': lambda tl, i: tl[i],
        'cifar': lambda tl, i: tl[i][0]
    }

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def isnetworkfile(fname):
    _, ext = os.path.splitext(fname)
    if ext not in ['.pyt', '.meta', '.tf','.onnx', '.pb']:
        raise argparse.ArgumentTypeError('only .pyt, .tf, .onnx, .pb, and .meta formats supported')
    return fname

def parse_args():
    parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .pb, .pyt, .tf, .meta, and .onnx')
    parser.add_argument('--epsilon', type=float, default=config.epsilon, help='the epsilon for L_infinity perturbation')
    parser.add_argument('--zonotope', type=str, default=config.zonotope, help='file to specify the zonotope matrix')
    parser.add_argument('--subset', type=str, default=config.subset, help='suffix of the file to specify the subset of the test dataset to use')
    parser.add_argument('--target', type=str, default=config.target, help='file specify the targets for the attack')
    parser.add_argument('--epsfile', type=str, default=config.epsfile, help='file specify the epsilons for the L_oo attack')
    parser.add_argument('--specnumber', type=int, default=config.specnumber, help='the property number for the acasxu networks')
    parser.add_argument('--domain', type=str, default=config.domain, help='the domain name can be either deepzono, refinezono, deeppoly or refinepoly')
    parser.add_argument('--dataset', type=str, default=config.dataset, help='the dataset, can be either mnist, cifar10, acasxu, or fashion')
    parser.add_argument('--complete', type=str2bool, default=config.complete,  help='flag specifying where to use complete verification or not')
    parser.add_argument('--timeout_lp', type=float, default=config.timeout_lp,  help='timeout for the LP solver')
    parser.add_argument('--timeout_milp', type=float, default=config.timeout_milp,  help='timeout for the MILP solver')
    parser.add_argument('--timeout_complete', type=float, default=config.timeout_milp,  help='timeout for the complete verifier')
    parser.add_argument('--numproc', type=int, default=config.numproc,  help='number of processes for MILP / LP / k-ReLU')
    parser.add_argument('--sparse_n', type=int, default=config.sparse_n,  help='Number of variables to group by k-ReLU')
    parser.add_argument('--use_default_heuristic', type=str2bool, default=config.use_default_heuristic,  help='whether to use the area heuristic for the DeepPoly ReLU approximation or to always create new noise symbols per relu for the DeepZono ReLU approximation')
    parser.add_argument('--use_milp', type=str2bool, default=config.use_milp,  help='whether to use milp or not')
    parser.add_argument('--refine_neurons', action='store_true', default=config.refine_neurons, help='whether to refine intermediate neurons')
    parser.add_argument('--mean', nargs='+', type=float, default=config.mean, help='the mean used to normalize the data with')
    parser.add_argument('--std', nargs='+', type=float, default=config.std, help='the standard deviation used to normalize the data with')
    parser.add_argument('--data_dir', type=str, default=config.data_dir, help='data location')
    parser.add_argument('--geometric_config', type=str, default=config.geometric_config, help='config location')
    parser.add_argument('--num_params', type=int, default=config.num_params, help='Number of transformation parameters')
    parser.add_argument('--num_tests', type=int, default=config.num_tests, help='Number of images to test')
    parser.add_argument('--from_test', type=int, default=config.from_test, help='Number of images to test')
    parser.add_argument('--debug', action='store_true', default=config.debug, help='Whether to display debug info')
    parser.add_argument('--attack', action='store_true', default=config.attack, help='Whether to attack')
    parser.add_argument('--geometric', '-g', dest='geometric', default=config.geometric, action='store_true', help='Whether to do geometric analysis')
    parser.add_argument('--input_box', default=config.input_box,  help='input box to use')
    parser.add_argument('--output_constraints', default=config.output_constraints, help='custom output constraints to check')
    parser.add_argument('--normalized_region', type=str2bool, default=config.normalized_region, help='Whether to normalize the adversarial region')
    parser.add_argument('--spatial', action='store_true', default=config.spatial, help='whether to do vector field analysis')
    parser.add_argument('--t-norm', type=str, default=config.t_norm, help='vector field norm (1, 2, or inf)')
    parser.add_argument('--delta', type=float, default=config.delta, help='vector field displacement magnitude')
    parser.add_argument('--gamma', type=float, default=config.gamma, help='vector field smoothness constraint')

    # Logging options
    parser.add_argument('--logdir', type=str, default=None, help='Location to save logs to. If not specified, logs are not saved and emitted to stdout')
    parser.add_argument('--logn ame', type=str, default=None, help='Directory of log files in `logdir`, if not specified timestamp is used')


    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(config, k, v)
    config.json = vars(args)




def block_print():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

def main_run_eran_wrapper(img_input, input_epsilon, queue=None):
    from eran_runner import main_run_eran

    ret = main_run_eran(img_input, input_epsilon)
    if queue:
        queue.put(ret)

    return ret

def run_eran(img_input, input_epsilon, supress_print=False):
    if supress_print:
        block_print()

    if USE_SUBPROCESS_AND_WAIT:
        q = Queue()
        p = Process(target=main_run_eran_wrapper, args=(img_input, input_epsilon, q))
        p.start()
        p.join()
        ret = q.get()

    else:
        ret = main_run_eran_wrapper(img_input, input_epsilon)

    if supress_print:
        enable_print()

    return ret


formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

user_logger = setup_logger("user_logger", LOGGER_PATH)

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
        self.dict_to_eran = {}
        self.create_dict_to_eran()
        self.set_up_labels()

    def set_up_labels(self):
        if self.name in dataset_labels_setup:
            self.labels = dataset_labels_setup[self.name]
        else:
            raise Exception('only {} are supported'.format(dataset_labels_setup.keys()))

    def organize_to_labels(self, test_imags=False):
        """
        input is from class dataset, output is a dictionary, keys are according to labels
        """
        labels = self.train_labels
        images = self.train_images

        if test_imags:
            labels = self.test_labels
            images = self.test_images

        organized_images = dict.fromkeys(self.labels)
        for i in range(len(labels)):
            key = self.labels[dataset_labels_setup_func[self.name](labels, i)]
            if organized_images[key] is None:
                organized_images[key] = [images[i]]
            else:
                organized_images[key] = np.append(organized_images[key], [images[i]], axis=0)
        self.organized_images = organized_images

    def create_dict_to_eran(self):
        """
        input is organized dictionary, output is the image in a single array,
         with the label as the first object in array
        """
        dict_to_eran = dict.fromkeys(self.labels)
        for label in self.labels:
            dict_to_eran[label] = [np.insert(self.organized_images[label][0], 0, label)]
            for k in range(1, len(self.organized_images[label])):
                dict_to_eran[label].append(np.insert(self.organized_images[label][k], 0, label))
        self.dict_to_eran = dict_to_eran


class Image(object):
    def __init__(self, image, index):
        self.image = image #label + 28x28 image if MNIST
        self.index = index

    def __eq__(self, other):
        return self.index == other.index

class Epsilon(float):
    def __eq__(self, other):
        # because of the problem of floating point in python
        return abs(self.real - other.real ) <= (10 ** (-1 * PRECISION) + 10 ** (-1 * (PRECISION + 1)))


def plot(image, label, name):  # input- image (without label). no output. plots image

    print(label)
    cmap_type = 'viridis'
    if name == 'mnist':
        cmap_type = 'gray'
    plt.imshow(image, cmap=cmap_type)
    plt.show()


def ready_image_for_eran(image, label):
    image_with_label = np.insert(image, 0, label)
    return image_with_label


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


def binary_search(img, lower_bound, upper_bound, is_in_range):
    user_logger.info(" low {}. up {}.".format(lower_bound, upper_bound))

    lower_bound = max(lower_bound - (10 ** (-1 * PRECISION)) / 2, MIN_EPS)
    upper_bound = min(upper_bound + (10 ** (-1 * PRECISION)) / 2, MAX_EPS)

    if is_in_range([img], lower_bound)[2] == 0:
        user_logger.info("epsilon is out of range, too small")
        user_logger.info(" runs {}.".format( 2))
        return EPS_IS_LOWER, 2

    if is_in_range([img], upper_bound)[2] == 1:
        user_logger.info('epsilon is out of range, too high')
        user_logger.info(" runs {}.".format(2))
        return EPS_IS_HIGHER, 2

    cnt = 2

    point = 10**(-(PRECISION + 1))
    while (upper_bound - lower_bound) > point:
        mid = round(((lower_bound + upper_bound)/2), PRECISION+2)
        if is_in_range([img], mid)[2] == 1:  # if epsilon >= mid
            lower_bound = mid
        else:
            upper_bound = mid
        cnt += 1

    user_logger.info("eps {}. runs {}.".format(lower_bound, cnt))
    return round(lower_bound, PRECISION), cnt


def eran_dummy_func(mid, i):
    if mid <= 0.0004+i*0.0001:
        return 1
    else:
        return 0


def confidence_score_func(img):
    labels_confidence = run_eran([img.image], MIN_EPS)[0][0]
    two_highest_conf_lbl = heapq.nlargest(2, labels_confidence)
    return abs(two_highest_conf_lbl[0] - two_highest_conf_lbl[1])

def random_score_func(img):
    return random.random()


def choose_index(range_list):
    #image_index = random.choice(range(len(range_list)))
    image_index = max(0, (round(len(range_list)/2)-1))
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


def find_all_epsilons(images_boundaries, is_in_range, floating_point=2):

    cnt = 0
    epsilon_list = []
    while images_boundaries:
        cnt = cnt + 1
        i = choose_index(images_boundaries)
        upper_bound = images_boundaries[i][3]
        lower_bound = images_boundaries[i][2]
        mid_epsilon = round(((upper_bound+lower_bound)/2), PRECISION+1)
        # mid_epsilon = (images_boundaries[i]["upper_bound"] + images_boundaries[i]["lower_bound"]) / 2
        is_robust = is_in_range([images_boundaries[i][1]], mid_epsilon)[2]
        if is_robust == 1:
            for j in range(i, len(images_boundaries)):
                images_boundaries[j][2] = mid_epsilon
        else:
            for k in range(0, i+1):
                images_boundaries[k][3] = mid_epsilon

        if images_boundaries[i][3]-images_boundaries[i][2] <= (10**(-PRECISION)):
            epsilon = round(images_boundaries[i][2], PRECISION)
            image_index = images_boundaries[i][0]
            images_boundaries.pop(i)
            epsilon_list.append([image_index, epsilon])

    return epsilon_list, cnt


def load_data_from_csv(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


def load_cheat_eps_from_txt(file_name, num_of_imgs):
    eps_file = open(file_name)
    eps_array = []
    for position, line in enumerate(eps_file):
        new_eps = float(re.findall('max epsilon (.*?) ,', line)[0])
        eps_array.append(new_eps)
        if position == num_of_imgs:
            break
    return eps_array


def load_cheat_eps_from_csv(file_name, num_imgs):
    eps_array = []
    num_of_runs = 0
    line_count = 0
    with open(file_name) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            if line_count == num_imgs:
                break
            eps_array.append((Epsilon(row[1]), int(row[0])))
            num_of_runs += int(row[2])
    return eps_array, num_of_runs


def create_cheat_sheet_csv(images, images_index, is_in_range, file_name):
    if not os.path.exists(file_name):
        header = ['index', 'max_epsilon', 'num_of_runs']
        if len(images) != len(images_index):
            raise Exception('indexes list and images list must be the same length')
        cheat_sheet = []
        for i in range(len(images)):
            max_eps, cnt = binary_search(images[i], MIN_EPS, MAX_EPS, is_in_range)
            user_logger.info("bin srch: index {}. eps {}. runs {} ".format(i, max_eps, cnt))
            cheat_sheet.append([images_index[i], max_eps, cnt])
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(cheat_sheet)


def save_single_img_csv(new_file_name, data_to_save):
    with open(new_file_name, 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data_to_save)


def run_eran_in_cmd(epsilon, image, LABEL=0):
    """
        this runs ERAN using the cmd. in order to do so, need to open a new folder for this specific run,
        the function saves specific image with label at the start, in the folder data, name mnist.test.csv, from which eran then takes
         the image.
        """
    image_with_label = np.insert(image, 0, LABEL)
    open('../data/mnist_test.csv', 'w').close()
    save_single_img_csv('../data/mnist_test.csv', image_with_label)

    output = subprocess.check_output(
        ['python3', '.', '--netname', '/root/' + config.netname,
         '--epsilon', str(epsilon), '--domain', 'deepzono', '--dataset', 'mnist'])

    output_str = str(output)
    return output_str


def is_in_range_using_eran_by_cmd(epsilon, image):
    output_str = run_eran_in_cmd(epsilon, image)
    new_result = re.findall('analysis precision  ([0,1])', output_str)
    return new_result[0]


# TODO run eran only once
def labels_confidence_using_eran_by_cmd(epsilon, image):
    output_str = run_eran_in_cmd(epsilon, image)
    labels_confidence_str = re.findall('nub  \[(.*?)\]', output_str)
    confidence_array = np.array(labels_confidence_str[0].split(',')).astype(float)

    return confidence_array

def check_bad_image(eps):
    if eps < MIN_EPS:
        # Epsilon is out of global boundaries
        if eps == EPS_IS_LOWER:
            # Bad image
            user_logger.info("epsilon = 0 - the network does not classify this img")
        elif eps == EPS_IS_HIGHER:
            user_logger.error("Epsilon > MAX !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            user_logger.error("Something weird with this img")
            raise Exception("Something weird with this img")

def epsilon_out_of_iter_range(img_eps, img, lower, upper, is_in_range):
    if img_eps == EPS_IS_LOWER:
        # Epsilon is smaller than lower bound
        img_eps, num_of_runs_after_mistake = binary_search(img.image, MIN_EPS, lower, is_in_range)
        user_logger.warning("out of scope lower rng binary_search: ")
    elif img_eps == EPS_IS_HIGHER:
        # Epsilon is bigger than upper bound
        img_eps, num_of_runs_after_mistake = binary_search(img.image, upper, MAX_EPS, is_in_range)
        user_logger.warning("out of scope rng upper binary_search: ")
    else:
        raise Exception("Error: binary_search")

    check_bad_image(img_eps)
    return num_of_runs_after_mistake, lower, upper, img_eps

def epsilon_out_of_iter_restart(img, is_in_range):
    img_eps, num_of_runs_after_mistake = binary_search(img.image, MIN_EPS, MAX_EPS, is_in_range)

    return num_of_runs_after_mistake, MIN_EPS, MAX_EPS, MIN_EPS, MAX_EPS, img_eps


def get_all_eps_with_mistakes_control(imgs, lower=MIN_EPS, upper=MAX_EPS, is_in_range=run_eran):
    user_logger.info("rng binary_search: ")

    if imgs:
        mid_indx = round(len(imgs)/2)
        mid_img = imgs[mid_indx]
        mid_img_eps, num_of_runs = binary_search(mid_img.image, lower, upper, is_in_range)

        if mid_img_eps < MIN_EPS:
            if not (mid_img_eps == EPS_IS_LOWER and lower == MIN_EPS):
                # otherwise the image is bad image

                # num_of_runs_after_mistake, new_lower ,new_upper, mid_img_eps = epsilon_out_of_iter_range(mid_img_eps, mid_img, lower, upper, is_in_range)
                num_of_runs_after_mistake, new_lower ,new_upper, lower, upper, mid_img_eps = epsilon_out_of_iter_restart(mid_img, is_in_range)
                num_of_runs += num_of_runs_after_mistake
            else:
                new_upper = upper
                new_lower = lower
        else:
            # Epsilon is in bounderies
            new_upper = new_lower = mid_img_eps

        lower_list = imgs[:mid_indx]
        lower_eps, lower_eps_runs = get_all_eps_with_mistakes_control(lower_list, lower, new_upper, is_in_range)

        upper_list = imgs[mid_indx+1:]
        upper_eps, upper_eps_runs = get_all_eps_with_mistakes_control(upper_list, new_lower, upper, is_in_range)

        epsilon_list = lower_eps + [(Epsilon(mid_img_eps), int(mid_img.index))] + upper_eps
        total_runs = num_of_runs + lower_eps_runs + upper_eps_runs
        return epsilon_list, total_runs

    else:
        return [], 0

def get_all_eps_ignore_method(imgs, lower=MIN_EPS, upper=MAX_EPS, is_in_range=run_eran):
    user_logger.info("rng binary_search_ignore: ")

    if imgs:
        mid_indx = round(len(imgs)/2)
        mid_img = imgs[mid_indx]
        mid_img_eps, num_of_runs = binary_search(mid_img.image, lower, upper, is_in_range)

        user_logger.info("img_ind - {}. eps found first search - {}".format(mid_img.index, mid_img_eps))

        if mid_img_eps < MIN_EPS:
            if not (mid_img_eps == EPS_IS_LOWER and lower == MIN_EPS):
                # otherwise the image is bad image
                mid_img_eps, num_of_runs_after_mistake = binary_search(mid_img.image, MIN_EPS, MAX_EPS, is_in_range)

                user_logger.info("img_ind - {}. eps found second search - {}".format(mid_img.index, mid_img_eps))

                if mid_img_eps != EPS_IS_LOWER:
                    user_logger.info("ignore - the order is wrong. add second running")
                    # otherwise the image is bad image
                    num_of_runs += num_of_runs_after_mistake

            imgs.pop(mid_indx)
            reduced_eps_list, reduced_eps_runs = get_all_eps_ignore_method(imgs, lower, upper, is_in_range)

            epsilon_list = reduced_eps_list[:mid_indx] + [(Epsilon(mid_img_eps), int(mid_img.index))] + reduced_eps_list[mid_indx:]
            total_runs = num_of_runs + reduced_eps_runs

        else:
            # Epsilon is in bounderies
            new_upper = new_lower = mid_img_eps

            lower_list = imgs[:mid_indx]
            lower_eps, lower_eps_runs = get_all_eps_ignore_method(lower_list, lower, new_upper, is_in_range)

            upper_list = imgs[mid_indx+1:]
            upper_eps, upper_eps_runs = get_all_eps_ignore_method(upper_list, new_lower, upper, is_in_range)

            epsilon_list = lower_eps + [(Epsilon(mid_img_eps), int(mid_img.index))] + upper_eps
            total_runs = num_of_runs + lower_eps_runs + upper_eps_runs

        return epsilon_list, total_runs

    else:
        return [], 0

def get_all_eps_with_mistakes_control_ignore_method(imgs, lower=MIN_EPS, upper=MAX_EPS, before_lower=MIN_EPS,
                                                    before_upper=MAX_EPS, is_in_range=run_eran):
    user_logger.info("rng binary_search_mistakes_control_ignore: ")


    if not imgs:
        return [], 0

    else:
        mid_indx = round(len(imgs)/2)
        mid_img = imgs[mid_indx]
        mid_img_eps, num_of_runs = binary_search(mid_img.image, lower, upper, is_in_range)

        if mid_img_eps >= MIN_EPS:
            # Epsilon in bounderies
            lower_list = imgs[:mid_indx]
            lower_eps, lower_eps_runs = get_all_eps_with_mistakes_control_ignore_method(lower_list, lower,
                                            mid_img_eps, before_lower, upper, is_in_range)

            upper_list = imgs[mid_indx + 1:]
            upper_eps, upper_eps_runs = get_all_eps_with_mistakes_control_ignore_method(upper_list, mid_img_eps,
                                            upper, lower, before_upper, is_in_range)

        else:

            # check which boundary is wrong
            if mid_img_eps == EPS_IS_LOWER:
                if lower == MIN_EPS:
                    # image is bad image, ignore and restart
                    user_logger.info("ignore mistakes_control - bad_image")
                    imgs.pop(mid_indx)
                    reduced_eps_list, reduced_eps_runs = get_all_eps_with_mistakes_control_ignore_method(imgs,
                                                             lower, upper, before_lower, before_upper, is_in_range)
                    return reduced_eps_list[:mid_indx] + [(Epsilon(EPS_IS_LOWER), int(mid_img.index))] + reduced_eps_list[
                                                                        mid_indx:], num_of_runs + reduced_eps_runs

                mid_img_eps, num_of_runs_after_mistake = binary_search(mid_img.image, MIN_EPS, upper, is_in_range)
                check_bad_image(mid_img_eps)

                if mid_img_eps == EPS_IS_LOWER:
                    # image is bad image, ignore and restart
                    user_logger.info("ignore mistakes_control - bad_image")
                    imgs.pop(mid_indx)
                    reduced_eps_list, reduced_eps_runs = get_all_eps_with_mistakes_control_ignore_method(imgs,
                                                            lower, upper, before_lower, before_upper, is_in_range)
                    return reduced_eps_list[:mid_indx] + [(Epsilon(EPS_IS_LOWER), int(mid_img.index))] + reduced_eps_list[
                                                                       mid_indx:], num_of_runs + reduced_eps_runs

                user_logger.info("ignore mistakes_control - the order is wrong. add second running")
                num_of_runs += num_of_runs_after_mistake

                if mid_img_eps < before_lower:
                    # img order is wrong, ignore and restart
                    imgs.pop(mid_indx)
                    reduced_eps_list, reduced_eps_runs = get_all_eps_with_mistakes_control_ignore_method(imgs,
                                                             lower, upper, before_lower, before_upper, is_in_range)
                    return reduced_eps_list[:mid_indx] + [(Epsilon(mid_img_eps), int(mid_img.index))] + reduced_eps_list[
                                                                        mid_indx:], num_of_runs + reduced_eps_runs

                # lower or img_eps is wrong
                lower_list = imgs[:mid_indx]
                lower_eps, lower_eps_runs = get_all_eps_with_mistakes_control_ignore_method(lower_list, before_lower,
                                                upper, MIN_EPS, before_upper, is_in_range)

                upper_list = imgs[mid_indx + 1:]
                upper_eps, upper_eps_runs = get_all_eps_with_mistakes_control_ignore_method(upper_list, mid_img_eps,
                                                upper, before_lower, before_upper, is_in_range)

            elif mid_img_eps == EPS_IS_HIGHER:
                mid_img_eps, num_of_runs_after_mistake = binary_search(mid_img.image, lower, MAX_EPS, is_in_range)
                check_bad_image(mid_img_eps)

                num_of_runs += num_of_runs_after_mistake

                if mid_img_eps > before_upper:
                    # img order is wrong, ignore and restart
                    imgs.pop(mid_indx)
                    reduced_eps_list, reduced_eps_runs = get_all_eps_with_mistakes_control_ignore_method(imgs,
                                                             lower, upper, before_lower, before_upper, is_in_range)
                    return reduced_eps_list[:mid_indx] + [(Epsilon(mid_img_eps), int(mid_img.index))] + reduced_eps_list[mid_indx:], num_of_runs + reduced_eps_runs

                # upper or img_eps is wrong
                lower_list = imgs[:mid_indx]
                lower_eps, lower_eps_runs = get_all_eps_with_mistakes_control_ignore_method(lower_list,
                                                lower, mid_img_eps, before_lower, before_upper, is_in_range)

                upper_list = imgs[mid_indx + 1:]
                upper_eps, upper_eps_runs = get_all_eps_with_mistakes_control_ignore_method(upper_list,
                                                   lower, before_upper, before_lower, MAX_EPS, is_in_range)

            else:
                raise Exception("false return value of binary_search")

        epsilon_list = lower_eps + [(Epsilon(mid_img_eps), int(mid_img.index))] + upper_eps
        total_runs = num_of_runs + lower_eps_runs + upper_eps_runs
        return epsilon_list, total_runs


def save_epsilons_to_csv(eps_list, num_of_iter, path):
    header = ['max_epsilon', 'index', 'num_of_runs']
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(eps_list)
        writer.writerow([num_of_iter, num_of_iter, num_of_iter])

def create_default_json_file(path):
    if not os.path.exists(path):
        with open(path, "w+") as f:
            json.dump({}, f)

def save_runs_num(runs_num_file, runs_num, options, network, precision=PRECISION):
    user_logger.debug("pasten - {}".format(runs_num_file))
    create_default_json_file(runs_num_file)
    with open(runs_num_file, "r") as f:
        runs_num_dict = json.load(f)

    # using json.dumps(key) only for using jsom.dump on dictionery with key as key
    key = json.dumps((os.path.basename(network), options, precision, VERSION))

    if key in runs_num_dict:
        if runs_num not in runs_num_dict[key]:
            user_logger.warning("new runs_num for same key. save new num {}".format(runs_num))
            runs_num_dict[key].append(runs_num)
    else:
        runs_num_dict[key] = [runs_num]

    with open(runs_num_file, "w") as f:
        json.dump(runs_num_dict, f)

def sort_img_correctly(indexed_imgs_list, num_imgs, eps_file_path):
    eps_arr, _ = load_cheat_eps_from_csv(eps_file_path, num_imgs)
    user_logger.info("sort_img_correctly: loaded {}".format(eps_arr))
    sorted_by_ind_eps_arr = sorted(eps_arr, key=lambda eps: eps[1])
    user_logger.info("sort_img_correctly: sorted epsilons {}".format(sorted_by_ind_eps_arr))
    sorted_imgs = sorted(indexed_imgs_list, key=lambda img: sorted_by_ind_eps_arr[img.index][0])
    user_logger.info("sort_img_correctly: sorted imgs {}".format([img.index for img in sorted_imgs]))

    return sorted_imgs

def get_score_func_sort_correctly(imgs, size, eps_file_path):
    sorted_imgs = sort_img_correctly(create_indexed_img_list_from_dataset(imgs), size, eps_file_path)

    def test_score_func(img):
        return sorted_imgs.index(img)

    return test_score_func

def create_indexed_img_list_from_dataset(imgs_list):
    return [Image(imgs_list[i], i) for i in range(len(imgs_list))]


def rng_search_all_epsilons_sorted_by_score_func(imgs_list, method, score_func=confidence_score_func):
    imgs = create_indexed_img_list_from_dataset(imgs_list)
    sorted_imgs = sorted(imgs, key=lambda img: score_func(img))
    epsilons, runs_num = method(sorted_imgs)
    sorted_epsilons = sorted(epsilons, key=lambda eps: eps[1])
    return sorted_epsilons, runs_num

def check_all_outputs_equal(eps_list):
    if all([i == eps_list[0] for i in eps_list]):
        user_logger.info("epsilon lists are identical")
    else:
        for epsilons in eps_list:
            user_logger.error("epsilons - {}".format(epsilons))
        user_logger.error('epsilon lists not identical')

        raise Exception('epsilon lists not identical')


def run_and_check_range_sizes_X_labels(sizes, labels, methods, score_funcs):
    user_logger.info("######################## start logging ########################")

    str_labels = [str(label) for label in labels]
    for size, label in product(sizes, str_labels):
        user_logger.info("############# start logging size {}, label {} #############".format(size, label))

        imgs_list = load_imgs(label, size)
        epsilons_list = []
        basename_for_log = "netname_{}_label_{}_size_{}".format(os.path.basename(config.netname), str(label),
                                                                str(size))

        q = Queue(len(methods) * len(score_funcs) * 2)
        processes = [Process(target=check_epsilons_by_method_with_time, args=(imgs_list, size, basename_for_log, met, s_func, q))
                     for met, s_func in product(methods, score_funcs)]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        while not q.empty():
            epsilons_list.append(q.get())

        check_all_outputs_equal(epsilons_list)
        user_logger.info("############# end logging size {}, label {} #############".format(size, label))


    user_logger.info("######################## end of logging ########################")

def check_epsilons_rng_binary_sorted_by_score_func(imgs_list, method, score_func, name_for_log):
    user_logger.info("start rng_binary {}".format(name_for_log))

    rng_bin_srch_epsilons, rng_bin_srch_runs_num = rng_search_all_epsilons_sorted_by_score_func(imgs_list, method=method, score_func=score_func)

    user_logger.info('rng_binary {} # num of runs: {}'.format(name_for_log, rng_bin_srch_runs_num))
    user_logger.info('rng_binary {} # epsilons: {}'.format(name_for_log, rng_bin_srch_epsilons))

    rng_path = '/root/ERAN/tf_verify/rng_binary_srch_score_{}.csv'.format(name_for_log)
    save_epsilons_to_csv(rng_bin_srch_epsilons, rng_bin_srch_runs_num, rng_path)
    save_runs_num(OUTCOMES_PATH, rng_bin_srch_runs_num, options=name_for_log, network=config.netname)

    return rng_bin_srch_epsilons

def check_epsilons_naive_img_one_by_one(imgs_list, size, name_for_log):
    user_logger.info("start naive {}".format(name_for_log))

    eps_file_path = './cheat_sheet_round_{}.csv'.format(name_for_log)

    create_cheat_sheet_csv(imgs_list, range(size), run_eran, eps_file_path)
    naive_epsilons, naive_runs_num = load_cheat_eps_from_csv(eps_file_path, size)

    user_logger.info('Naive {} # num of runs: {}'.format(name_for_log, naive_runs_num))
    user_logger.info('naive {} # epsilons: {}'.format(name_for_log, naive_epsilons))

    save_runs_num(OUTCOMES_PATH, naive_runs_num, options="naive_{}".format(name_for_log), network=config.netname)

    return naive_epsilons, eps_file_path

def check_epsilons_by_method_with_time(imgs_list, size, basename_for_log, method_string, score_func_string, queue):
    start_time = time.time()

    # Check method
    if method_string == "ignore":
        method_func = get_all_eps_ignore_method
    elif method_string == "ignore_mistake_control":
        method_func = get_all_eps_with_mistakes_control_ignore_method
    else:
        raise Exception("unknown method")

    # Check naive score_func
    if "naive" in score_func_string:
        naive_ret, eps_file_path = check_epsilons_naive_img_one_by_one(imgs_list, size, basename_for_log)
        queue.put(naive_ret)

        if score_func_string == "naive_and_sorted_correctly":

            sorted_correctly_score_func = get_score_func_sort_correctly(imgs_list, size, eps_file_path)
            queue.put(check_epsilons_rng_binary_sorted_by_score_func(imgs_list, method_func, sorted_correctly_score_func,
                                                             "{}_sorted_correctly_method_{}".format(basename_for_log, method_string)))

    else:
        # Check score_func
        if score_func_string == "confidence":
            score_func = confidence_score_func
        elif score_func_string == "random":
            score_func = random_score_func
        else:
            raise Exception("unknown score_func")

        queue.put(check_epsilons_rng_binary_sorted_by_score_func(imgs_list, method_func, score_func,
                                                                "{}_method_{}_scored_{}".format(basename_for_log,
                                                                method_string, score_func_string)))

    end_time = time.time()
    elapsed_time = (start_time - end_time) / 60  # convert to minutes

    user_logger.info('Execution time: {} minutes'. format(elapsed_time))

def load_imgs(label, size):
    images = load_dataset('mnist') #TODO change using config.netname
    images.create_dict_to_eran()
    dataset = images.dict_to_eran
    return dataset[label][:size]

def main():
    """
    confidence_array -> array of scores per class for the images
    epsilon -> the epsilon used in ERAN
    num_of_verified -> number of pictures which were verifeid
    num_of_classified -> number of pictures which were classified correctly
    :return:
    """
    parse_args()

    sizes = [1024,]
    labels = [0, 1, 2, 4, 5, 7, 8,]
    methods = ["ignore_mistake_control",]
    score_funcs = ["naive", "random", "confidence"]
    run_and_check_range_sizes_X_labels(sizes, labels, methods, score_funcs)


    sizes = [16 * 2 ** i for i in range(6)]
    labels = [3, 6]
    methods = ["ignore_mistake_control",]
    score_funcs = ["naive", "confidence"]
    run_and_check_range_sizes_X_labels(sizes, labels, methods, score_funcs)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        exc_line = traceback.format_exc()
        user_logger.error(exc_line)