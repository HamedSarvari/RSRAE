import numpy as np
from Autoencoder_utils_torch import Read_data, find_outlier_inlier
import torch
import torch.utils.data as data_utils

# Takes a dataset name and if inliers= True, returns a numpy array of only inlier data with labels
def get_data_label(ds_name, data_add, inliers=False, mnist=False):

        if mnist == True:
            data, labels = Read_data(ds_name, data_add, type='MNIST')
        else:
            data, labels = Read_data(ds_name, data_add)

        if inliers == True:
            outlier_d, inlier_d = find_outlier_inlier(data, labels)
            data_np = np.array(inlier_d)
            labels = ['no'] * data_np.shape[0]
            return inlier_d, labels
        else:
            data_np = np.array(data)
            return data_np, labels

def load_od_datasets(ds_name, ds_add, mode= 'unsupervised', is_mnist= False):

    # keep all data to create test data loader

    x_all_data, labels_all_data = get_data_label(ds_name, ds_add, inliers=False, mnist= is_mnist)
    y_train_all_data = np.where(labels_all_data == 'yes', 1, 0).reshape(-1, 1)
    indices_all_data = np.arange(len(x_all_data)).reshape(-1, 1)

    if mode == 'unsupervised':
        x_train, labels = get_data_label(ds_name, ds_add, inliers=False, mnist= is_mnist)
        y_train = np.where(labels == 'yes', 1, 0).reshape(-1, 1)
        indices = np.arange(len(x_train)).reshape(-1, 1)

    if mode == 'semisupervised':
        x_train, labels = get_data_label(ds_name, ds_add, inliers=True, mnist= is_mnist)
        y_train = np.array([0] * x_train.shape[0])
        indices = np.arange(len(x_train)).reshape(-1, 1)

    return x_train, y_train

