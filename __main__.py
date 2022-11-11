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


import sys
import os
cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../deepg/code/')
import numpy as np
from eran import ERAN
from read_net_file import *
from read_zonotope_file import read_zonotope
import tensorflow as tf
import csv
import time
from tqdm import tqdm
from ai_milp import *
import argparse
from config import config
from constraint_utils import *
import re
import itertools
from multiprocessing import Pool, Value
import onnxruntime.backend as rt
import logging
import torch
import spatial
from copy import deepcopy
import matplotlib.pyplot as plt
sys.path.insert(0, '/root/load_dataset')
import keras
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import subprocess
import heapq
from operator import itemgetter
import re
import time
import json
import traceback
from multiprocessing import Process, Queue

########################### This is Dana's functions ###########################
if True:
    #ZONOTOPE_EXTENSION = '.zt'
    EPS = 10**(-9)

    is_tf_version_2=tf.__version__[0]=='2'

    if is_tf_version_2:
        tf = tf.compat.v1


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


    def parse_input_box(text):
        intervals_list = []
        for line in text.split('\n'):
            if line!="":
                interval_strings = re.findall("\[-?\d*\.?\d+, *-?\d*\.?\d+\]", line)
                intervals = []
                for interval in interval_strings:
                    interval = interval.replace('[', '')
                    interval = interval.replace(']', '')
                    [lb,ub] = interval.split(",")
                    intervals.append((np.double(lb), np.double(ub)))
                intervals_list.append(intervals)

        # return every combination
        boxes = itertools.product(*intervals_list)
        return list(boxes)


    def show_ascii_spec(lb, ub, n_rows, n_cols, n_channels):
        print('==================================================================')
        for i in range(n_rows):
            print('  ', end='')
            for j in range(n_cols):
                print('#' if lb[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
            print('  |  ', end='')
            for j in range(n_cols):
                print('#' if ub[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
            print('  |  ')
        print('==================================================================')


    def normalize(image, means, stds, dataset):
        # normalization taken out of the network
        if len(means) == len(image):
            for i in range(len(image)):
                image[i] -= means[i]
                if stds!=None:
                    image[i] /= stds[i]
        elif dataset == 'mnist'  or dataset == 'fashion':
            for i in range(len(image)):
                image[i] = (image[i] - means[0])/stds[0]
        elif(dataset=='cifar10'):
            count = 0
            tmp = np.zeros(3072)
            for i in range(1024):
                tmp[count] = (image[count] - means[0])/stds[0]
                count = count + 1
                tmp[count] = (image[count] - means[1])/stds[1]
                count = count + 1
                tmp[count] = (image[count] - means[2])/stds[2]
                count = count + 1

            if(is_conv):
                for i in range(3072):
                    image[i] = tmp[i]
            else:
                count = 0
                for i in range(1024):
                    image[i] = tmp[count]
                    count = count+1
                    image[i+1024] = tmp[count]
                    count = count+1
                    image[i+2048] = tmp[count]
                    count = count+1


    def normalize_plane(plane, mean, std, channel, is_constant):
        plane_ = plane.clone()

        if is_constant:
            plane_ -= mean[channel]

        plane_ /= std[channel]

        return plane_


    def normalize_poly(num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights, uexpr_dim, means, stds, dataset):
        # normalization taken out of the network
        if dataset == 'mnist' or dataset == 'fashion':
            for i in range(len(lexpr_cst)):
                lexpr_cst[i] = (lexpr_cst[i] - means[0]) / stds[0]
                uexpr_cst[i] = (uexpr_cst[i] - means[0]) / stds[0]
            for i in range(len(lexpr_weights)):
                lexpr_weights[i] /= stds[0]
                uexpr_weights[i] /= stds[0]
        else:
            for i in range(len(lexpr_cst)):
                lexpr_cst[i] = (lexpr_cst[i] - means[i % 3]) / stds[i % 3]
                uexpr_cst[i] = (uexpr_cst[i] - means[i % 3]) / stds[i % 3]
            for i in range(len(lexpr_weights)):
                lexpr_weights[i] /= stds[(i // num_params) % 3]
                uexpr_weights[i] /= stds[(i // num_params) % 3]


    def denormalize(image, means, stds, dataset):
        if dataset == 'mnist'  or dataset == 'fashion':
            for i in range(len(image)):
                image[i] = image[i]*stds[0] + means[0]
        elif(dataset=='cifar10'):
            count = 0
            tmp = np.zeros(3072)
            for i in range(1024):
                tmp[count] = image[count]*stds[0] + means[0]
                count = count + 1
                tmp[count] = image[count]*stds[1] + means[1]
                count = count + 1
                tmp[count] = image[count]*stds[2] + means[2]
                count = count + 1

            for i in range(3072):
                image[i] = tmp[i]


    def model_predict(base, input):
        if is_onnx:
            pred = base.run(input)
        else:
            pred = base.run(base.graph.get_operation_by_name(model.op.name), {base.graph.get_operations()[0].name + ':0': input})
        return pred


    def estimate_grads(specLB, specUB, dim_samples=3):
        specLB = np.array(specLB, dtype=np.float32)
        specUB = np.array(specUB, dtype=np.float32)
        inputs = [((dim_samples - i) * specLB + i * specUB) / dim_samples for i in range(dim_samples + 1)]
        diffs = np.zeros(len(specLB))

        # refactor this out of this method
        if is_onnx:
            runnable = rt.prepare(model, 'CPU')
        elif sess is None:
            runnable = tf.Session()
        else:
            runnable = sess

        for sample in range(dim_samples + 1):
            pred = model_predict(runnable, inputs[sample])

            for index in range(len(specLB)):
                if sample < dim_samples:
                    l_input = [m if i != index else u for i, m, u in zip(range(len(specLB)), inputs[sample], inputs[sample+1])]
                    l_input = np.array(l_input, dtype=np.float32)
                    l_i_pred = model_predict(runnable, l_input)
                else:
                    l_i_pred = pred
                if sample > 0:
                    u_input = [m if i != index else l for i, m, l in zip(range(len(specLB)), inputs[sample], inputs[sample-1])]
                    u_input = np.array(u_input, dtype=np.float32)
                    u_i_pred = model_predict(runnable, u_input)
                else:
                    u_i_pred = pred
                diff = np.sum([abs(li - m) + abs(ui - m) for li, m, ui in zip(l_i_pred, pred, u_i_pred)])
                diffs[index] += diff
        return diffs / dim_samples


    progress = 0.0
    def print_progress(depth):
        if config.debug:
            global progress, rec_start
            progress += np.power(2.,-depth)
            sys.stdout.write('\r%.10f percent, %.02f s' % (100 * progress, time.time()-rec_start))


    def acasxu_recursive(specLB, specUB, max_depth=10, depth=0):
        hold,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
        global failed_already
        if hold:
            print_progress(depth)
            return hold
        elif depth >= max_depth:
            if failed_already.value and config.complete:
                verified_flag, adv_examples = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                print_progress(depth)
                if verified_flag == False:
                    if adv_examples!=None:
                        #print("adv image ", adv_image)
                        for adv_image in adv_examples:
                            hold,_,nlb,nub,_,_ = eran.analyze_box(adv_image, adv_image, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
                            #print("hold ", hold, "domain", domain)
                            if hold == False:
                                print("property violated at ", adv_image, "output_score", nlb[-1])
                                failed_already.value = 0
                                break
                return verified_flag
            else:
                return False
        else:
            grads = estimate_grads(specLB, specUB)
            # grads + small epsilon so if gradient estimation becomes 0 it will divide the biggest interval.
            smears = np.multiply(grads + 0.00001, [u-l for u, l in zip(specUB, specLB)])

            #start = time.time()
            #nn.set_last_weights(constraints)
            #grads_lower, grads_upper = nn.back_propagate_gradiant(nlb, nub)
            #smears = [max(-grad_l, grad_u) * (u-l) for grad_l, grad_u, l, u in zip(grads_lower, grads_upper, specLB, specUB)]
            index = np.argmax(smears)
            m = (specLB[index]+specUB[index])/2

            result =  failed_already.value and acasxu_recursive(specLB, [ub if i != index else m for i, ub in enumerate(specUB)], max_depth, depth + 1)
            result = failed_already.value and result and acasxu_recursive([lb if i != index else m for i, lb in enumerate(specLB)], specUB, max_depth, depth + 1)
            return result


    def get_tests(dataset, geometric):
        if geometric:
            csvfile = open('../deepg/code/datasets/{}_test.csv'.format(dataset), 'r')
        else:
            if config.subset == None:
                csvfile = open('../data/{}_test.csv'.format(dataset), 'r')
            else:
                filename = '../data/'+ dataset+ '_test_' + config.subset + '.csv'
                csvfile = open(filename, 'r')
        tests = csv.reader(csvfile, delimiter=',')

        return tests


    def init_domain(d):
        if d == 'refinezono':
            return 'deepzono'
        elif d == 'refinepoly':
            return 'deeppoly'
        else:
            return d

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
        parser.add_argument('--logname', type=str, default=None, help='Directory of log files in `logdir`, if not specified timestamp is used')


        args = parser.parse_args()
        for k, v in vars(args).items():
            setattr(config, k, v)
        config.json = vars(args)


    def main_run_eran(img_input, input_epsilon, queue=None):
        confidence_arrays=[]

        if config.specnumber and not config.input_box and not config.output_constraints:
            config.input_box = '../data/acasxu/specs/acasxu_prop_' + str(config.specnumber) + '_input_prenormalized.txt'
            config.output_constraints = '../data/acasxu/specs/acasxu_prop_' + str(config.specnumber) + '_constraints.txt'

        assert config.netname, 'a network has to be provided for analysis.'

        #if len(sys.argv) < 4 or len(sys.argv) > 5:
        #    print('usage: python3.6 netname epsilon domain dataset')
        #    exit(1)

        netname = config.netname
        filename, file_extension = os.path.splitext(netname)

        is_trained_with_pytorch = file_extension==".pyt"
        is_saved_tf_model = file_extension==".meta"
        is_pb_file = file_extension==".pb"
        is_tensorflow = file_extension== ".tf"
        is_onnx = file_extension == ".onnx"
        assert is_trained_with_pytorch or is_saved_tf_model or is_pb_file or is_tensorflow or is_onnx, "file extension not supported"

        #epsilon = config.epsilon
        epsilon = input_epsilon
        #assert (epsilon >= 0) and (epsilon <= 1), "epsilon can only be between 0 and 1"

        zonotope_file = config.zonotope
        zonotope = None
        zonotope_bool = (zonotope_file!=None)
        if zonotope_bool:
            zonotope = read_zonotope(zonotope_file)

        domain = config.domain

        if zonotope_bool:
            assert domain in ['deepzono', 'refinezono'], "domain name can be either deepzono or refinezono"
        elif not config.geometric:
            assert domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly'], "domain name can be either deepzono, refinezono, deeppoly or refinepoly"

        dataset = config.dataset

        if zonotope_bool==False:
           assert dataset in ['mnist', 'cifar10', 'acasxu', 'fashion'], "only mnist, cifar10, acasxu, and fashion datasets are supported"

        constraints = None
        if config.output_constraints:
            constraints = get_constraints_from_file(config.output_constraints)

        mean = 0
        std = 0

        complete = (config.complete==True)

        if(dataset=='acasxu'):
            print("netname ", netname, " specnumber ", config.specnumber, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)
        else:
            print("netname ", netname, " epsilon ", epsilon, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)

        non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault', 'IsVariableInitialized', 'Placeholder', 'Identity']

        sess = None
        if is_saved_tf_model or is_pb_file:
            netfolder = os.path.dirname(netname)

            tf.logging.set_verbosity(tf.logging.ERROR)

            sess = tf.Session()
            if is_saved_tf_model:
                saver = tf.train.import_meta_graph(netname)
                saver.restore(sess, tf.train.latest_checkpoint(netfolder+'/'))
            else:
                with tf.gfile.GFile(netname, "rb") as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    sess.graph.as_default()
                    tf.graph_util.import_graph_def(graph_def, name='')
            ops = sess.graph.get_operations()
            last_layer_index = -1
            while ops[last_layer_index].type in non_layer_operation_types:
                last_layer_index -= 1
            model = sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0')
            eran = ERAN(model, sess)

        else:
            if(zonotope_bool==True):
                num_pixels = len(zonotope)
            elif(dataset=='mnist'):
                num_pixels = 784
            elif (dataset=='cifar10'):
                num_pixels = 3072
            elif(dataset=='acasxu'):
                num_pixels = 5
            if is_onnx:
                model, is_conv = read_onnx_net(netname)
            else:
                model, is_conv, means, stds = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch)
            eran = ERAN(model, is_onnx=is_onnx)

        if not is_trained_with_pytorch:
            if dataset == 'mnist' and not config.geometric:
                means = [0]
                stds = [1]
            elif dataset == 'acasxu':
                means = [1.9791091e+04,0.0,0.0,650.0,600.0]
                stds = [60261.0,6.28318530718,6.28318530718,1100.0,1200.0]
            else:
                means = [0.5, 0.5, 0.5]
                stds = [1, 1, 1]

        is_trained_with_pytorch = is_trained_with_pytorch or is_onnx

        if config.mean is not None:
            means = config.mean
            stds = config.std

        os.sched_setaffinity(0,cpu_affinity)

        correctly_classified_images = 0
        verified_images = 0


        if dataset:
            if config.input_box is None:
                tests = get_tests(dataset, config.geometric)
            else:
                tests = open(config.input_box, 'r').read()

        def init(args):
            global failed_already
            failed_already = args

        if dataset=='acasxu':
            if config.debug:
                print('Constraints: ', constraints)
            boxes = parse_input_box(tests)
            total_start = time.time()
            for box_index, box in enumerate(boxes):
                specLB = [interval[0] for interval in box]
                specUB = [interval[1] for interval in box]
                normalize(specLB, means, stds, dataset)
                normalize(specUB, means, stds, dataset)


                rec_start = time.time()

                _,nn,nlb,nub,_ ,_= eran.analyze_box(specLB, specUB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
                # expensive min/max gradient calculation
                nn.set_last_weights(constraints)
                grads_lower, grads_upper = nn.back_propagate_gradiant(nlb, nub)


                smears = [max(-grad_l, grad_u) * (u-l) for grad_l, grad_u, l, u in zip(grads_lower, grads_upper, specLB, specUB)]
                split_multiple = 20 / np.sum(smears)

                num_splits = [int(np.ceil(smear * split_multiple)) for smear in smears]
                step_size = []
                for i in range(5):
                    if num_splits[i]==0:
                        num_splits[i] = 1
                    step_size.append((specUB[i]-specLB[i])/num_splits[i])
                #sorted_indices = np.argsort(widths)
                #input_to_split = sorted_indices[0]
                #print("input to split ", input_to_split)

                #step_size = widths/num_splits
                #print("step size", step_size,num_splits)
                start_val = np.copy(specLB)
                end_val = np.copy(specUB)
                flag = True
                _,nn,_,_,_,_ = eran.analyze_box(specLB, specUB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
                start = time.time()
                #complete_list = []
                multi_bounds = []

                for i in range(num_splits[0]):
                    specLB[0] = start_val[0] + i*step_size[0]
                    specUB[0] = np.fmin(end_val[0],start_val[0]+ (i+1)*step_size[0])

                    for j in range(num_splits[1]):
                        specLB[1] = start_val[1] + j*step_size[1]
                        specUB[1] = np.fmin(end_val[1],start_val[1]+ (j+1)*step_size[1])

                        for k in range(num_splits[2]):
                            specLB[2] = start_val[2] + k*step_size[2]
                            specUB[2] = np.fmin(end_val[2],start_val[2]+ (k+1)*step_size[2])
                            for l in range(num_splits[3]):
                                specLB[3] = start_val[3] + l*step_size[3]
                                specUB[3] = np.fmin(end_val[3],start_val[3]+ (l+1)*step_size[3])
                                for m in range(num_splits[4]):

                                    specLB[4] = start_val[4] + m*step_size[4]
                                    specUB[4] = np.fmin(end_val[4],start_val[4]+ (m+1)*step_size[4])

                                    # add bounds to input for multiprocessing map
                                    multi_bounds.append((specLB.copy(), specUB.copy()))


                                    # --- VERSION WITHOUT MULTIPROCESSING ---
                                    #hold,_,nlb,nub = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
                                    #if not hold:
                                    #    if complete==True:
                                    #       verified_flag,adv_image = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                                    #       #complete_list.append((i,j,k,l,m))
                                    #       if verified_flag==False:
                                    #          flag = False
                                    #          assert 0
                                    #    else:
                                    #       flag = False
                                    #       break
                                    #if config.debug:
                                    #    sys.stdout.write('\rsplit %i, %i, %i, %i, %i %.02f sec' % (i, j, k, l, m, time.time()-start))

                #print(time.time() - rec_start, "seconds")
                #print("LENGTH ", len(multi_bounds))
                failed_already = Value('i',1)
                try:
                    with Pool(processes=10, initializer=init, initargs=(failed_already,)) as pool:
                        res = pool.starmap(acasxu_recursive, multi_bounds)

                    if all(res):
                        print("AcasXu property", config.specnumber, "Verified for Box", box_index, "out of",len(boxes))
                    else:
                        print("AcasXu property", config.specnumber, "Failed for Box", box_index, "out of",len(boxes))
                except Exception as e:
                    print("AcasXu property", config.specnumber, "Failed for Box", box_index, "out of",len(boxes),"because of an exception ",e)

                print(time.time() - rec_start, "seconds")
            print("Total time needed:", time.time() - total_start, "seconds")

        elif zonotope_bool:
            perturbed_label, nn, nlb, nub,_ = eran.analyze_zonotope(zonotope, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
            print("nlb ",nlb[-1])
            print("nub ",nub[-1])
            if(perturbed_label!=-1):
                print("Verified")
            elif(complete==True):
                constraints = get_constraints_for_dominant_label(perturbed_label, 10)
                verified_flag,adv_image = verify_network_with_milp(nn, zonotope, [], nlb, nub, constraints)
                if(verified_flag==True):
                    print("Verified")
                else:
                    print("Failed")
            else:
                 print("Failed")


        elif config.geometric:

            total, attacked, standard_correct, tot_time = 0, 0, 0, 0
            correct_box, correct_poly = 0, 0
            cver_box, cver_poly = [], []
            if config.geometric_config:
                transform_attack_container = get_transform_attack_container(config.geometric_config)
                for i, test in enumerate(tests):
                    if config.from_test and i < config.from_test:
                        continue

                    if config.num_tests is not None and i >= config.num_tests:
                        break
                    set_transform_attack_for(transform_attack_container, i, config.attack, config.debug)
                    attack_params = get_attack_params(transform_attack_container)
                    attack_images = get_attack_images(transform_attack_container)
                    print('Test {}:'.format(i))

                    image = np.float64(test[1:])
                    if config.dataset == 'mnist' or config.dataset == 'fashion':
                        n_rows, n_cols, n_channels = 28, 28, 1
                    else:
                        n_rows, n_cols, n_channels = 32, 32, 3

                    spec_lb = np.copy(image)
                    spec_ub = np.copy(image)

                    normalize(spec_lb, means, stds, config.dataset)
                    normalize(spec_ub, means, stds, config.dataset)

                    label, nn, nlb, nub,_,_ = eran.analyze_box(spec_lb, spec_ub, 'deeppoly', config.timeout_lp, config.timeout_milp,
                                                           config.use_default_heuristic)
                    print('Label: ', label)

                    begtime = time.time()
                    if label != int(test[0]):
                        print('Label {}, but true label is {}, skipping...'.format(label, int(test[0])))
                        print('Standard accuracy: {} percent'.format(standard_correct / float(i + 1) * 100))
                        continue
                    else:
                        standard_correct += 1
                        print('Standard accuracy: {} percent'.format(standard_correct / float(i + 1) * 100))

                    dim = n_rows * n_cols * n_channels

                    ok_box, ok_poly = True, True
                    k = config.num_params + 1 + 1 + dim

                    attack_imgs, checked, attack_pass = [], [], 0
                    cex_found = False
                    if config.attack:
                        for j in tqdm(range(0, len(attack_params))):
                            params = attack_params[j]
                            values = np.array(attack_images[j])

                            attack_lb = values[::2]
                            attack_ub = values[1::2]

                            normalize(attack_lb, means, stds, config.dataset)
                            normalize(attack_ub, means, stds, config.dataset)
                            attack_imgs.append((params, attack_lb, attack_ub))
                            checked.append(False)

                            predict_label, _, _, _, _, _ = eran.analyze_box(
                                attack_lb[:dim], attack_ub[:dim], 'deeppoly',
                                config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                            if predict_label != int(test[0]):
                                print('counter-example, params: ', params, ', predicted label: ', predict_label)
                                cex_found = True
                                break
                            else:
                                attack_pass += 1
                    print('tot attacks: ', len(attack_imgs))

                    lines = get_transformations(transform_attack_container)
                    print('Number of lines: ', len(lines))
                    assert len(lines) % k == 0

                    spec_lb = np.zeros(config.num_params + dim)
                    spec_ub = np.zeros(config.num_params + dim)

                    expr_size = config.num_params
                    lexpr_cst, uexpr_cst = [], []
                    lexpr_weights, uexpr_weights = [], []
                    lexpr_dim, uexpr_dim = [], []

                    ver_chunks_box, ver_chunks_poly, tot_chunks = 0, 0, 0

                    for i, line in enumerate(lines):
                        if i % k < config.num_params:
                            # read specs for the parameters
                            values = line
                            assert len(values) == 2
                            param_idx = i % k
                            spec_lb[dim + param_idx] = values[0]
                            spec_ub[dim + param_idx] = values[1]
                            if config.debug:
                                print('parameter %d: [%.4f, %.4f]' % (param_idx, values[0], values[1]))
                        elif i % k == config.num_params:
                            # read interval bounds for image pixels
                            values = line
                            spec_lb[:dim] = values[::2]
                            spec_ub[:dim] = values[1::2]
                            # if config.debug:
                            #     show_ascii_spec(spec_lb, spec_ub)
                        elif i % k < k - 1:
                            # read polyhedra constraints for image pixels
                            tokens = line
                            assert len(tokens) == 2 + 2 * config.num_params

                            bias_lower, weights_lower = tokens[0], tokens[1:1 + config.num_params]
                            bias_upper, weights_upper = tokens[config.num_params + 1], tokens[2 + config.num_params:]

                            assert len(weights_lower) == config.num_params
                            assert len(weights_upper) == config.num_params

                            lexpr_cst.append(bias_lower)
                            uexpr_cst.append(bias_upper)
                            for j in range(config.num_params):
                                lexpr_dim.append(dim + j)
                                uexpr_dim.append(dim + j)
                                lexpr_weights.append(weights_lower[j])
                                uexpr_weights.append(weights_upper[j])
                        else:
                            assert (len(line) == 0)
                            for p_idx in range(config.num_params):
                                lexpr_cst.append(spec_lb[dim + p_idx])
                                for l in range(config.num_params):
                                    lexpr_weights.append(0)
                                    lexpr_dim.append(dim + l)
                                uexpr_cst.append(spec_ub[dim + p_idx])
                                for l in range(config.num_params):
                                    uexpr_weights.append(0)
                                    uexpr_dim.append(dim + l)
                            normalize(spec_lb[:dim], means, stds, config.dataset)
                            normalize(spec_ub[:dim], means, stds, config.dataset)
                            normalize_poly(config.num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights,
                                           uexpr_dim, means, stds, config.dataset)

                            for attack_idx, (attack_params, attack_lb, attack_ub) in enumerate(attack_imgs):
                                ok_attack = True
                                for j in range(num_pixels):
                                    low, up = lexpr_cst[j], uexpr_cst[j]
                                    for idx in range(config.num_params):
                                        low += lexpr_weights[j * config.num_params + idx] * attack_params[idx]
                                        up += uexpr_weights[j * config.num_params + idx] * attack_params[idx]
                                    if low > attack_lb[j] + EPS or attack_ub[j] > up + EPS:
                                        ok_attack = False
                                if ok_attack:
                                    checked[attack_idx] = True
                                    # print('checked ', attack_idx)
                            if config.debug:
                                print('Running the analysis...')

                            t_begin = time.time()
                            perturbed_label_poly, _, _, _, _, _ = eran.analyze_box(
                                spec_lb, spec_ub, 'deeppoly',
                                config.timeout_lp, config.timeout_milp, config.use_default_heuristic, None,
                                lexpr_weights, lexpr_cst, lexpr_dim,
                                uexpr_weights, uexpr_cst, uexpr_dim,
                                expr_size)
                            perturbed_label_box, _, _, _, _, _ = eran.analyze_box(
                                spec_lb[:dim], spec_ub[:dim], 'deeppoly',
                                config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                            t_end = time.time()

                            print('DeepG: ', perturbed_label_poly, '\tInterval: ', perturbed_label_box, '\tlabel: ', label,
                                  '[Time: %.4f]' % (t_end - t_begin))

                            tot_chunks += 1
                            if perturbed_label_box != label:
                                ok_box = False
                            else:
                                ver_chunks_box += 1

                            if perturbed_label_poly != label:
                                ok_poly = False
                            else:
                                ver_chunks_poly += 1

                            lexpr_cst, uexpr_cst = [], []
                            lexpr_weights, uexpr_weights = [], []
                            lexpr_dim, uexpr_dim = [], []

                    total += 1
                    if ok_box:
                        correct_box += 1
                    if ok_poly:
                        correct_poly += 1
                    if cex_found:
                        assert (not ok_box) and (not ok_poly)
                        attacked += 1
                    cver_poly.append(ver_chunks_poly / float(tot_chunks))
                    cver_box.append(ver_chunks_box / float(tot_chunks))
                    tot_time += time.time() - begtime

                    print('Verified[box]: {}, Verified[poly]: {}, CEX found: {}'.format(ok_box, ok_poly, cex_found))
                    assert not cex_found or not ok_box, 'ERROR! Found counter-example, but image was verified with box!'
                    assert not cex_found or not ok_poly, 'ERROR! Found counter-example, but image was verified with poly!'


            else:
                for i, test in enumerate(tests):
                    if config.from_test and i < config.from_test:
                        continue

                    if config.num_tests is not None and i >= config.num_tests:
                        break

                    attacks_file = os.path.join(config.data_dir, 'attack_{}.csv'.format(i))
                    print('Test {}:'.format(i))

                    image = np.float64(test[1:])
                    if config.dataset == 'mnist' or config.dataset == 'fashion':
                        n_rows, n_cols, n_channels = 28, 28, 1
                    else:
                        n_rows, n_cols, n_channels = 32, 32, 3

                    spec_lb = np.copy(image)
                    spec_ub = np.copy(image)

                    normalize(spec_lb, means, stds, config.dataset)
                    normalize(spec_ub, means, stds, config.dataset)

                    label, nn, nlb, nub, _, _ = eran.analyze_box(spec_lb, spec_ub, 'deeppoly', config.timeout_lp, config.timeout_milp,
                                                           config.use_default_heuristic)
                    print('Label: ', label)

                    begtime = time.time()
                    if label != int(test[0]):
                        print('Label {}, but true label is {}, skipping...'.format(label, int(test[0])))
                        print('Standard accuracy: {} percent'.format(standard_correct / float(i + 1) * 100))
                        continue
                    else:
                        standard_correct += 1
                        print('Standard accuracy: {} percent'.format(standard_correct / float(i + 1) * 100))

                    dim = n_rows * n_cols * n_channels

                    ok_box, ok_poly = True, True
                    k = config.num_params + 1 + 1 + dim

                    attack_imgs, checked, attack_pass = [], [], 0
                    cex_found = False
                    if config.attack:
                        with open(attacks_file, 'r') as fin:
                            lines = fin.readlines()
                            for j in tqdm(range(0, len(lines), config.num_params + 1)):
                                params = [float(line[:-1]) for line in lines[j:j + config.num_params]]
                                tokens = lines[j + config.num_params].split(',')
                                values = np.array(list(map(float, tokens)))

                                attack_lb = values[::2]
                                attack_ub = values[1::2]

                                normalize(attack_lb, means, stds, config.dataset)
                                normalize(attack_ub, means, stds, config.dataset)
                                attack_imgs.append((params, attack_lb, attack_ub))
                                checked.append(False)

                                predict_label, _, _, _, _, _ = eran.analyze_box(
                                    attack_lb[:dim], attack_ub[:dim], 'deeppoly',
                                    config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                                if predict_label != int(test[0]):
                                    print('counter-example, params: ', params, ', predicted label: ', predict_label)
                                    cex_found = True
                                    break
                                else:
                                    attack_pass += 1
                    print('tot attacks: ', len(attack_imgs))
                    specs_file = os.path.join(config.data_dir, '{}.csv'.format(i))
                    with open(specs_file, 'r') as fin:
                        lines = fin.readlines()
                        print('Number of lines: ', len(lines))
                        assert len(lines) % k == 0

                        spec_lb = np.zeros(config.num_params + dim)
                        spec_ub = np.zeros(config.num_params + dim)

                        expr_size = config.num_params
                        lexpr_cst, uexpr_cst = [], []
                        lexpr_weights, uexpr_weights = [], []
                        lexpr_dim, uexpr_dim = [], []

                        ver_chunks_box, ver_chunks_poly, tot_chunks = 0, 0, 0

                        for i, line in enumerate(lines):
                            if i % k < config.num_params:
                                # read specs for the parameters
                                values = np.array(list(map(float, line[:-1].split(' '))))
                                assert values.shape[0] == 2
                                param_idx = i % k
                                spec_lb[dim + param_idx] = values[0]
                                spec_ub[dim + param_idx] = values[1]
                                if config.debug:
                                    print('parameter %d: [%.4f, %.4f]' % (param_idx, values[0], values[1]))
                            elif i % k == config.num_params:
                                # read interval bounds for image pixels
                                values = np.array(list(map(float, line[:-1].split(','))))
                                spec_lb[:dim] = values[::2]
                                spec_ub[:dim] = values[1::2]
                                # if config.debug:
                                #     show_ascii_spec(spec_lb, spec_ub)
                            elif i % k < k - 1:
                                # read polyhedra constraints for image pixels
                                tokens = line[:-1].split(' ')
                                assert len(tokens) == 2 + 2 * config.num_params + 1

                                bias_lower, weights_lower = float(tokens[0]), list(map(float, tokens[1:1 + config.num_params]))
                                assert tokens[config.num_params + 1] == '|'
                                bias_upper, weights_upper = float(tokens[config.num_params + 2]), list(
                                    map(float, tokens[3 + config.num_params:]))

                                assert len(weights_lower) == config.num_params
                                assert len(weights_upper) == config.num_params

                                lexpr_cst.append(bias_lower)
                                uexpr_cst.append(bias_upper)
                                for j in range(config.num_params):
                                    lexpr_dim.append(dim + j)
                                    uexpr_dim.append(dim + j)
                                    lexpr_weights.append(weights_lower[j])
                                    uexpr_weights.append(weights_upper[j])
                            else:
                                assert (line == 'SPEC_FINISHED\n')
                                for p_idx in range(config.num_params):
                                    lexpr_cst.append(spec_lb[dim + p_idx])
                                    for l in range(config.num_params):
                                        lexpr_weights.append(0)
                                        lexpr_dim.append(dim + l)
                                    uexpr_cst.append(spec_ub[dim + p_idx])
                                    for l in range(config.num_params):
                                        uexpr_weights.append(0)
                                        uexpr_dim.append(dim + l)
                                normalize(spec_lb[:dim], means, stds, config.dataset)
                                normalize(spec_ub[:dim], means, stds, config.dataset)
                                normalize_poly(config.num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights,
                                               uexpr_dim, means, stds, config.dataset)

                                for attack_idx, (attack_params, attack_lb, attack_ub) in enumerate(attack_imgs):
                                    ok_attack = True
                                    for j in range(num_pixels):
                                        low, up = lexpr_cst[j], uexpr_cst[j]
                                        for idx in range(config.num_params):
                                            low += lexpr_weights[j * config.num_params + idx] * attack_params[idx]
                                            up += uexpr_weights[j * config.num_params + idx] * attack_params[idx]
                                        if low > attack_lb[j] + EPS or attack_ub[j] > up + EPS:
                                            ok_attack = False
                                    if ok_attack:
                                        checked[attack_idx] = True
                                        # print('checked ', attack_idx)
                                if config.debug:
                                    print('Running the analysis...')

                                t_begin = time.time()
                                perturbed_label_poly, _, _, _ , _, _ = eran.analyze_box(
                                    spec_lb, spec_ub, 'deeppoly',
                                    config.timeout_lp, config.timeout_milp, config.use_default_heuristic, None,
                                    lexpr_weights, lexpr_cst, lexpr_dim,
                                    uexpr_weights, uexpr_cst, uexpr_dim,
                                    expr_size)
                                perturbed_label_box, _, _, _, _, _ = eran.analyze_box(
                                    spec_lb[:dim], spec_ub[:dim], 'deeppoly',
                                    config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                                t_end = time.time()

                                print('DeepG: ', perturbed_label_poly, '\tInterval: ', perturbed_label_box, '\tlabel: ', label,
                                      '[Time: %.4f]' % (t_end - t_begin))

                                tot_chunks += 1
                                if perturbed_label_box != label:
                                    ok_box = False
                                else:
                                    ver_chunks_box += 1

                                if perturbed_label_poly != label:
                                    ok_poly = False
                                else:
                                    ver_chunks_poly += 1

                                lexpr_cst, uexpr_cst = [], []
                                lexpr_weights, uexpr_weights = [], []
                                lexpr_dim, uexpr_dim = [], []

                    total += 1
                    if ok_box:
                        correct_box += 1
                    if ok_poly:
                        correct_poly += 1
                    if cex_found:
                        assert (not ok_box) and (not ok_poly)
                        attacked += 1
                    cver_poly.append(ver_chunks_poly / float(tot_chunks))
                    cver_box.append(ver_chunks_box / float(tot_chunks))
                    tot_time += time.time() - begtime

                    print('Verified[box]: {}, Verified[poly]: {}, CEX found: {}'.format(ok_box, ok_poly, cex_found))
                    assert not cex_found or not ok_box, 'ERROR! Found counter-example, but image was verified with box!'
                    assert not cex_found or not ok_poly, 'ERROR! Found counter-example, but image was verified with poly!'

            print('Attacks found: %.2f percent, %d/%d' % (100.0 * attacked / total, attacked, total))
            print('[Box]  Provably robust: %.2f percent, %d/%d' % (100.0 * correct_box / total, correct_box, total))
            print('[Poly] Provably robust: %.2f percent, %d/%d' % (100.0 * correct_poly / total, correct_poly, total))
            print('Empirically robust: %.2f percent, %d/%d' % (100.0 * (total - attacked) / total, total - attacked, total))
            print('[Box]  Average chunks verified: %.2f percent' % (100.0 * np.mean(cver_box)))
            print('[Poly]  Average chunks verified: %.2f percent' % (100.0 * np.mean(cver_poly)))
            print('Average time: ', tot_time / total)

        elif config.input_box is not None:
            boxes = parse_input_box(tests)
            index = 1
            correct = 0
            for box in boxes:
                specLB = [interval[0] for interval in box]
                specUB = [interval[1] for interval in box]
                normalize(specLB, means, stds, dataset)
                normalize(specUB, means, stds, dataset)
                hold, nn, nlb, nub,_ = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
                if hold:
                    print('constraints hold for box ' + str(index) + ' out of ' + str(sum([1 for b in boxes])))
                    correct += 1
                else:
                    print('constraints do NOT hold for box ' + str(index) + ' out of ' + str(sum([1 for b in boxes])))

                index += 1

            print('constraints hold for ' + str(correct) + ' out of ' + str(sum([1 for b in boxes])) + ' boxes')

        elif config.spatial:

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if config.dataset in ['mnist', 'fashion']:
                height, width, channels = 28, 28, 1
            else:
                height, width, channels = 32, 32, 3

            for idx, test in enumerate(tests):

                if idx < config.from_test:
                    continue

                if (config.num_tests is not None) and (config.from_test + config.num_tests == idx):
                    break

                image = torch.from_numpy(
                    np.float64(test[1:len(test)]) / np.float64(255)
                ).reshape(1, height, width, channels).permute(0, 3, 1, 2).to(device)
                label = np.int(test[0])

                specLB = image.clone().permute(0, 2, 3, 1).flatten().cpu()
                specUB = image.clone().permute(0, 2, 3, 1).flatten().cpu()

                normalize(specLB, means, stds, dataset)
                normalize(specUB, means, stds, dataset)

                predicted_label, nn, nlb, nub, _, _ = eran.analyze_box(
                    specLB=specLB, specUB=specUB, domain=init_domain(domain),
                    timeout_lp=config.timeout_lp, timeout_milp=config.timeout_milp,
                    use_default_heuristic=config.use_default_heuristic
                )

                print(f'concrete {nlb[-1]}')

                if label != predicted_label:
                    print(f'img {idx} not considered, correct_label {label}, classified label {predicted_label}')
                    continue

                correctly_classified_images += 1
                start = time.time()

                transformer = getattr(
                    spatial, f'T{config.t_norm.capitalize()}NormTransformer'
                )(image, config.delta)
                box_lb, box_ub = transformer.box_constraints()

                lower_bounds = box_lb.permute(0, 2, 3, 1).flatten()
                upper_bounds = box_ub.permute(0, 2, 3, 1).flatten()

                normalize(lower_bounds, means, stds, dataset)
                normalize(upper_bounds, means, stds, dataset)

                specLB, specUB = lower_bounds.clone(), upper_bounds.clone()
                LB_N0, UB_N0 = lower_bounds.clone(), upper_bounds.clone()

                expr_size = 0
                lexpr_weights = lexpr_cst = lexpr_dim = None
                uexpr_weights = uexpr_cst = uexpr_dim = None
                lower_planes = upper_planes = None
                deeppoly_spatial_constraints = milp_spatial_constraints = None

                if config.gamma < float('inf'):

                    expr_size = 2
                    lower_planes, upper_planes = list(), list()
                    lexpr_weights, lexpr_cst, lexpr_dim = list(), list(), list()
                    uexpr_weights, uexpr_cst, uexpr_dim = list(), list(), list()

                    linear_lb, linear_ub = transformer.linear_constraints()

                    for channel in range(image.shape[1]):
                        lb_a, lb_b, lb_c = linear_lb[channel]
                        ub_a, ub_b, ub_c = linear_ub[channel]

                        linear_lb[channel][0] = normalize_plane(
                            lb_a, means, stds, channel, is_constant=True
                        )
                        linear_lb[channel][1] = normalize_plane(
                            lb_b, means, stds, channel, is_constant=False
                        )
                        linear_lb[channel][2] = normalize_plane(
                            lb_c, means, stds, channel, is_constant=False
                        )

                        linear_ub[channel][0] = normalize_plane(
                            ub_a, means, stds, channel, is_constant=True
                        )
                        linear_ub[channel][1] = normalize_plane(
                            ub_b, means, stds, channel, is_constant=False
                        )
                        linear_ub[channel][2] = normalize_plane(
                            ub_c, means, stds, channel, is_constant=False
                        )

                    for i in range(3):
                        lower_planes.append(
                            torch.cat(
                                [
                                    linear_lb[channel][i].unsqueeze(-1)
                                    for channel in range(image.shape[1])
                                ], dim=-1
                            ).flatten().tolist()
                        )
                        upper_planes.append(
                            torch.cat(
                                [
                                    linear_ub[channel][i].unsqueeze(-1)
                                    for channel in range(image.shape[1])
                                ], dim=-1
                            ).flatten().tolist()
                        )

                    deeppoly_spatial_constraints = {'gamma': config.gamma}

                    for key, val in transformer.flow_constraint_pairs.items():
                        deeppoly_spatial_constraints[key] = val.cpu()

                    milp_spatial_constraints = {
                        'delta': config.delta, 'gamma': config.gamma,
                        'channels': image.shape[1], 'lower_planes': lower_planes,
                        'upper_planes': upper_planes,
                        'add_norm_constraints': transformer.add_norm_constraints,
                        'neighboring_indices': transformer.flow_constraint_pairs
                    }

                    num_pixels = image.flatten().shape[0]
                    num_flows = 2 * num_pixels

                    flows_LB = torch.full((num_flows,), -config.delta).to(device)
                    flows_UB = torch.full((num_flows,), config.delta).to(device)

                    specLB = torch.cat((specLB, flows_LB))
                    specUB = torch.cat((specUB, flows_UB))

                    lexpr_cst = deepcopy(lower_planes[0]) + flows_LB.tolist()
                    uexpr_cst = deepcopy(upper_planes[0]) + flows_UB.tolist()

                    lexpr_weights = [
                        v for p in zip(lower_planes[1], lower_planes[2]) for v in p
                    ] + torch.zeros(2 * num_flows).tolist()
                    uexpr_weights = [
                        v for p in zip(upper_planes[1], upper_planes[2]) for v in p
                    ] + torch.zeros(2 * num_flows).tolist()

                    lexpr_dim = torch.cat([
                        num_pixels + torch.arange(num_flows),
                        torch.zeros(2 * num_flows).long()
                    ]).tolist()
                    uexpr_dim = torch.cat([
                        num_pixels + torch.arange(num_flows),
                        torch.zeros(2 * num_flows).long()
                    ]).tolist()

                perturbed_label, _, nlb, nub, failed_labels, _ = eran.analyze_box(
                    specLB=specLB.cpu(), specUB=specUB.cpu(), domain=domain,
                    timeout_lp=config.timeout_lp, timeout_milp=config.timeout_milp,
                    use_default_heuristic=config.use_default_heuristic,
                    label=label, lexpr_weights=lexpr_weights, lexpr_cst=lexpr_cst,
                    lexpr_dim=lexpr_dim, uexpr_weights=uexpr_weights,
                    uexpr_cst=uexpr_cst, uexpr_dim=uexpr_dim, expr_size=expr_size,
                    spatial_constraints=deeppoly_spatial_constraints
                )
                end = time.time()

                print(f'nlb {nlb[-1]} nub {nub[-1]} adv labels {failed_labels}')

                if perturbed_label == label:
                    print(f'img {idx} verified {label}')
                    verified_images += 1
                    print(end - start, "seconds")
                    continue

                if (not complete) or (domain not in ['deeppoly', 'deepzono']):
                    print(f'img {idx} Failed')
                    print(end - start, "seconds")
                    continue

                verified_flag, adv_image = verify_network_with_milp(
                    nn=nn, LB_N0=LB_N0, UB_N0=UB_N0, nlb=nlb, nub=nub,
                    constraints=get_constraints_for_dominant_label(
                        predicted_label, failed_labels=failed_labels
                    ), spatial_constraints=milp_spatial_constraints
                )

                if verified_flag:
                    print(f'img {idx} Verified as Safe {label}')
                    verified_images += 1
                else:
                    print(f'img {idx} Failed')

                end = time.time()
                print(end - start, "seconds")

            print(f'analysis precision {verified_images} / {correctly_classified_images}')

        else:
            target = []
            if config.target != None:
                targetfile = open(config.target, 'r')
                targets = csv.reader(targetfile, delimiter=',')
                for i, val in enumerate(targets):
                    target = val


            if config.epsfile != None:
                epsfile = open(config.epsfile, 'r')
                epsilons = csv.reader(epsfile, delimiter=',')
                for i, val in enumerate(epsilons):
                    eps_array = val

            tests = img_input
            for i, test in enumerate(tests):
                if config.from_test and i < config.from_test:
                    continue

                if config.num_tests is not None and i >= config.from_test + config.num_tests:
                    break
                image = np.float64(test)/np.float64(255)
                specLB = np.copy(image)
                specUB = np.copy(image)

                normalize(specLB, means, stds, dataset)
                normalize(specUB, means, stds, dataset)

                label,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                confidence_arrays.append(nlb[-1])
                #for number in range(len(nub)):
                #    for element in range(len(nub[number])):
                #        if(nub[number][element]<=0):
                #            print('False')
                #        else:
                #            print('True')
                if config.epsfile!= None:
                    epsilon = np.float64(eps_array[i])
                print("concrete ", nlb[-1])
                #if(label == int(test[0])):
                if(label == int(test[0])):
                    perturbed_label = None
                    if config.normalized_region==True:
                        specLB = np.clip(image - epsilon,0,1)
                        specUB = np.clip(image + epsilon,0,1)
                        normalize(specLB, means, stds, dataset)
                        normalize(specUB, means, stds, dataset)
                    else:
                        specLB = specLB - epsilon
                        specUB = specUB + epsilon
                    start = time.time()
                    if config.target == None:
                        prop = -1
                    else:
                        prop = int(target[i])
                    perturbed_label, _, nlb, nub,failed_labels, x = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic,label=label, prop=prop)
                    print("nlb ", nlb[-1], " nub ", nub[-1],"adv labels ", failed_labels)
                    if(perturbed_label==label):
                        print("img", i, "Verified", label)
                        verified_images += 1
                    else:
                        if complete==True:
                            constraints = get_constraints_for_dominant_label(label, failed_labels)
                            verified_flag,adv_image = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                            if(verified_flag==True):
                                print("img", i, "Verified as Safe", label)
                                verified_images += 1
                            else:

                                if adv_image != None:
                                    cex_label,_,_,_,_,_ = eran.analyze_box(adv_image[0], adv_image[0], 'deepzono', config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                                    if(cex_label!=label):
                                        denormalize(adv_image[0], means, stds, dataset)
                                        print("img", i, "Verified unsafe with adversarial image ", adv_image, "cex label", cex_label, "correct label ", label)
                                print("img", i, "Failed")
                        else:

                            if x != None:
                                cex_label,_,_,_,_,_ = eran.analyze_box(x,x,'deepzono',config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                                print("cex label ", cex_label, "label ", label)
                                if(cex_label!=label):
                                    denormalize(x,means, stds, dataset)
                                    print("img", i, "Verified unsafe with adversarial image ", x, "cex label ", cex_label, "correct label ", label)
                                else:
                                    print("img", i, "Failed")
                            else:
                                print("img", i, "Failed")

                    correctly_classified_images +=1
                    end = time.time()
                    print(end - start, "seconds")
                else:
                    print("img",i,"not considered, correct_label", int(test[0]), "classified label ", label)

            print('analysis precision ',verified_images,'/ ', correctly_classified_images)

            if queue:
                queue.put((confidence_arrays,epsilon,verified_images,correctly_classified_images))
            return confidence_arrays,epsilon,verified_images,correctly_classified_images
########################### End of Dana's functions ############################



#from geometric_constraints import *

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


EPS_IS_LOWER = -1
EPS_IS_HIGHER = -2
EPS_UNDEFINED = -3
IMG_UNRECOGNIZABLE = -4

MAX_EPS = 0.05
MIN_EPS = 0

PRECISION = 4
USE_SUBPROCESS_AND_WAIT = True
TEST = False
LOGGER_PATH = r"/root/logging/user_logger"


dataset_labels_setup = {
        'mnist': ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
        'cifar': ('airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks')
    }

dataset_test_labels_setup_func = {
        'mnist': lambda tl, i: tl[i],
        'cifar': lambda tl, i: tl[i][0]
    }


def block_print():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

def run_eran(img_input, input_epsilon, supress_print=False):
    if supress_print:
        block_print()

    if USE_SUBPROCESS_AND_WAIT:
        q = Queue()
        p = Process(target=main_run_eran, args=(img_input, input_epsilon, q))
        p.start()
        p.join()
        ret = q.get()

    else:
        ret = main_run_eran(img_input, input_epsilon)

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

    def organize_to_labels(self):
        """
        input is from class dataset, output is a dictionary, keys are according to labels
        """
        organized_images = dict.fromkeys(self.labels)
        for i in range(len(self.test_labels)):
            key = self.labels[dataset_test_labels_setup_func[self.name](self.test_labels, i)]
            if organized_images[key] is None:
                organized_images[key] = [self.test_images[i]]
            else:
                organized_images[key] = np.append(organized_images[key], [self.test_images[i]], axis=0)
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


class Epsilon(float):
    def __eq__(self, other):
        return abs(self.real - other.real ) < 10 ** PRECISION


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

    if is_in_range([img], lower_bound)[2] == 0:
        user_logger.info("epsilon is out of range, too small")
        user_logger.info(" runs {}.".format( 2))
        return EPS_IS_LOWER, 2

    if is_in_range([img], upper_bound)[2] == 1:
        user_logger.info('epsilon is out of range, too high')
        user_logger.info(" runs {}.".format(2))
        return EPS_IS_HIGHER, 2

    cnt = 2

    point = 10**(-PRECISION)
    while (upper_bound - lower_bound) > point:
        mid = round(((lower_bound + upper_bound)/2), PRECISION+1)
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


def score_func(img, epsilon=MIN_EPS):
    labels_confidence = run_eran([img], epsilon)[0][0]
    two_highest_conf_lbl = heapq.nlargest(2, labels_confidence)
    return abs(two_highest_conf_lbl[0] - two_highest_conf_lbl[1])


def test_score_func(a, cheat_sheet):

    return cheat_sheet[a[0]]


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
        if position == num_of_imgs-1:
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


def get_all_eps_with_mistakes_control(imgs, lower=MIN_EPS, upper=MAX_EPS, is_in_range=run_eran):
    user_logger.info("rng binary_search: ")

    if imgs:
        mid_indx = round(len(imgs)/2)

        mid_img = imgs[mid_indx]
        mid_img_eps, num_of_runs = binary_search(mid_img.image, lower, upper, is_in_range)
        if mid_img_eps < MIN_EPS:
            if mid_img_eps == EPS_IS_LOWER:
                mid_img_eps, num_of_runs_after_mistake = binary_search(mid_img.image, MIN_EPS, lower, is_in_range)
                user_logger.warning("out of scope lower rng binary_search: ")
            elif mid_img_eps == EPS_IS_HIGHER:
                mid_img_eps, num_of_runs_after_mistake = binary_search(mid_img.image, upper, MAX_EPS, is_in_range)
                user_logger.warning("out of scope rng upper binary_search: ")
            else:
                raise Exception("Error: binary_search")

            num_of_runs += num_of_runs_after_mistake

            if mid_img_eps < MIN_EPS:
                # epsilon is out of boundaries
                new_upper = upper
                new_lower = lower
                if mid_img_eps == EPS_IS_HIGHER:
                    user_logger.error("GREAT EPSILON!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            else:
                new_upper = max(upper, mid_img_eps)
                new_lower = min(lower, mid_img_eps)

        else:
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
def save_runs_num(runs_num_file, runs_num, method, label, num_of_images, network, precision=PRECISION):
    user_logger.info("pasten - {}".format(runs_num_file))
    create_default_json_file(runs_num_file)
    with open(runs_num_file, "r") as f:
        runs_num_dict = json.load(f)

    # using json.dumps(key) only for using jsom.dump on dictionery with key as key
    key = json.dumps((method, os.path.basename(network), label, num_of_images, precision))

    if key in runs_num_dict:
        if runs_num not in runs_num_dict[key]:
            user_logger.error("new runs_num for same key. save new num {}".format(runs_num))
            runs_num_dict[key].append(runs_num)
    else:
        runs_num_dict[key] = [runs_num]

    with open(runs_num_file, "w") as f:
        json.dump(runs_num_dict, f)

def sort_img_correctly(indexed_imgs_list, num_imgs, eps_file_path):
    eps_arr, _ = load_cheat_eps_from_csv(eps_file_path, num_imgs)
    user_logger.info("sort_img_correctly: loaded {}".format(eps_arr))
    sorted_eps_arr = sorted(eps_arr, key=lambda eps: eps[1])
    user_logger.info("sort_img_correctly: sorted epsilons {}".format(sorted_eps_arr))
    sorted_imgs = sorted(indexed_imgs_list, key=lambda img: sorted_eps_arr[img.index][0])
    user_logger.info("sort_img_correctly: sorted imgs {}".format([img.index for img in sorted_imgs]))
    return sorted_imgs

def sort_img_by_confidence(indexed_imgs_list):
    return sorted(indexed_imgs_list, key=lambda img: score_func(img.image))

def sort_img_by_score(indexed_imgs_list, num_imgs, eps_file_path):
    if TEST:
        return sort_img_correctly(indexed_imgs_list, num_imgs, eps_file_path)
    return sort_img_by_confidence(indexed_imgs_list)


def create_indexed_img_list_from_dataset(imgs_list):
    return [Image(imgs_list[i], i) for i in range(len(imgs_list))]


def rng_search_all_epsilons(imgs_list, num_imgs, eps_file_path):
    imgs = create_indexed_img_list_from_dataset(imgs_list)
    sorted_imgs = sort_img_by_score(imgs, num_imgs, eps_file_path)
    epsilons, runs_num = get_all_eps_with_mistakes_control(sorted_imgs)
    sorted_epsilons = sorted(epsilons, key=lambda eps: eps[1])
    return sorted_epsilons, runs_num

def run_and_check_range_sizes_X_labels(labels, sizes):
    for label in labels:
        run_and_check_range_sizes(label, sizes)
def run_and_check_range_sizes(label, sizes):
    for num_imgs in sizes:
        p = Process(target=run_and_check_one_iteration, args=(num_imgs, str(label)))
        p.start()

def run_and_check_one_iteration(num_imgs, label):
    eps_file_path = './cheat_sheet_round_label_{}_indx_0_to_{}_precision_{}.csv'.format(str(label), str(num_imgs), str(PRECISION))

    start_time = time.time()
    user_logger.info("######################## start logging ########################")
    images = load_dataset('mnist')
    images.create_dict_to_eran()
    dataset = images.dict_to_eran
    imgs_list = dataset[label][:num_imgs-1]

    # create cheat sheet
    create_cheat_sheet_csv(imgs_list, range(num_imgs-1), run_eran, eps_file_path)

    naive_epsilons, naive_runs_num = load_cheat_eps_from_csv(eps_file_path, num_imgs)

    # new and pretty binary search
    rng_bin_srch_epsilons, rng_bin_srch_runs_num = rng_search_all_epsilons(imgs_list, num_imgs, eps_file_path)
    end_time = time.time()
    elapsed_time = (start_time-end_time)/60 #convert to minutes

    user_logger.info('Execution time: {} minutes'. format(elapsed_time))
    user_logger.info('Network: {network}, number of images: {img_num}, digit: {digit}'.format(network=config.netname, img_num=num_imgs, digit=label, ))
    user_logger.info('Naive approach num of runs: {}'.format(naive_runs_num))
    user_logger.info('Ranged binary search approach num of runs: {}'.format(rng_bin_srch_runs_num))
    user_logger.info('rng_bin_srch_epsilons: {}'.format(rng_bin_srch_epsilons))
    user_logger.info('naive_epsilons: {}'.format(naive_epsilons))
    user_logger.info('List are identical: {}'.format(rng_bin_srch_epsilons == naive_epsilons))

    rng_path = '/root/ERAN/tf_verify/rng_binary_srch_score' + str(label) + '_indx_0_to_' + str(num_imgs) \
               + '_precision_' + str(PRECISION) + '.csv'
    save_epsilons_to_csv(rng_bin_srch_epsilons, rng_bin_srch_runs_num, rng_path)

    runs_num_path = '/root/ERAN/tf_verify/outcomes.json'
    save_runs_num(runs_num_path, naive_runs_num, method="naive", label=label, num_of_images=num_imgs, network=config.netname)
    save_runs_num(runs_num_path, rng_bin_srch_runs_num, method="rng_bin_srch_by_confidence", label=label, num_of_images=num_imgs, network=config.netname)

    user_logger.info("######################## end of logging ########################")

def main():
    """
    confidence_array -> array of scores per class for the images
    epsilon -> the epsilon used in ERAN
    num_of_verified -> number of pictures which were verifeid
    num_of_classified -> number of pictures which were classified correctly
    :return:
    """
    parse_args()

    # run_and_check_one_iteration(256,'0')
    # run_and_check_one_iteration(512,'0')
    #
    # run_and_check_one_iteration(512, '2')
    # run_and_check_one_iteration(1024, '2')

    # sizes = [8 * (2 ** i) for i in range(8)]
    # run_and_check_range_sizes('2', sizes)
    # sizes = [8 * (2 ** i) for i in range(2)]
    # labels = [2,3]
    # run_and_check_range_sizes_X_labels(labels, sizes)

    sizes = [8 * (2 ** i) for i in range(7)]
    labels = range(10)
    run_and_check_range_sizes_X_labels(labels, sizes)
    run_and_check_one_iteration(1024, '2')

if __name__ == "__main__":
    try:
        main()
    except Exception:
        exc_line = traceback.format_exc()
        user_logger.error(exc_line)