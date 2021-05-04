import csv
import sys
import numpy as np


def save_single_img_csv(new_file_name, data_to_save):
    with open(new_file_name, 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data_to_save)


def load_data(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


params = sys.argv #  1st arg - digit to analyze
                    #  2nd  arg - num of immg to analyze
digit_to_analyze = int(params[1])
img_num = int(params[2])
pca_data = load_data("pca_mnist_class_digit_" + str(digit_to_analyze) + ".csv")

# arrange data & PCA
pca_train_data = [np.array(image).astype(np.float64) for image in pca_data]

# change images for different features
changed_image = pca_train_data[img_num]  #

changed_image = changed_image.astype(int)
open('../data/mnist_test.csv', 'w').close()
#  save changed image
#  save_data_to_csv(new_file_name, changed_image)
save_single_img_csv('../data/mnist_test.csv', changed_image)
