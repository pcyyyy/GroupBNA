
import torch

from sklearn.decomposition import PCA
import os
import numpy as np


folder_path = '/run/user/0/stage_feature/new_data/all_timecourse'

def SVD_count(tuple):
    node_feature_matrix = []
    for i in range(90):
        columns_data = []
        for matrix in tuple:
            matrix = matrix[:120, :]
            column = matrix[:, i]
            columns_data.append(column)
        new_matrix = np.vstack(columns_data).T
        node_feature_matrix.append(new_matrix)

    left_singular_matrices = []

    for matrix in node_feature_matrix:
        matrix = torch.Tensor(matrix)

        u, s, v = torch.linalg.svd(matrix)
        u[u< 0] = 0

        left_singular_matrices.append(u)

    columns = []
    for mat in left_singular_matrices:
        column = mat[:, 1]
        columns.append(column)
    feature_weight = np.vstack(columns).T

    return feature_weight


def count_all_group():

    matrix_tuple = tuple(
        np.loadtxt(os.path.join(folder_path, filename))
        for filename in os.listdir(folder_path)
        if filename.endswith(".txt")
    )

    feature_weight=SVD_count(matrix_tuple)

    row_mean = np.mean(feature_weight, axis=1)

    sorted_indices = np.argsort(row_mean)

    top_40_indices = sorted_indices[-40:][::-1]
    bottom_40_indices = sorted_indices[:40]

    print("排名前40的行索引：", top_40_indices)
    print("排名最后40的行索引：", bottom_40_indices)

    return top_40_indices,bottom_40_indices

def count_single_group(label):
    AD_filename=os.listdir('/run/user/0/stage_feature/new_data/AD/timecourse')
    emci_filename=os.listdir('/run/user/0/stage_feature/new_data/emci/timecourse')
    mci_filename=os.listdir('/run/user/0/stage_feature/new_data/mci/timecourse')
    lmci_filename=os.listdir('/run/user/0/stage_feature/new_data/lmci/timecourse')
    nc_filename=os.listdir('/run/user/0/stage_feature/new_data/NC/timecourse')
    delete_path='/run/user/0/stage_feature/new_data/delete_feature'

    AD_matrix_tuple = tuple(np.loadtxt(os.path.join(delete_path,item)) for item in AD_filename)
    emci_matrix_tuple = tuple(np.loadtxt(os.path.join(delete_path,item))for item in emci_filename)
    mci_matrix_tuple = tuple(np.loadtxt(os.path.join(delete_path,item)) for item in mci_filename)
    lmci_matrix_tuple = tuple(np.loadtxt(os.path.join(delete_path,item)) for item in lmci_filename)
    nc_matrix_tuple = tuple(np.loadtxt(os.path.join(delete_path,item)) for item in nc_filename)

    if label==0:
        return SVD_count(AD_matrix_tuple)
    elif label==1:
        return SVD_count(lmci_matrix_tuple)
    elif label==2:
        return SVD_count(mci_matrix_tuple)
    elif label==3:
        return SVD_count(emci_matrix_tuple)
    else:
        return SVD_count(nc_matrix_tuple)


if __name__ == '__main__':
    top40,bottom40=count_all_group()
    top40=top40.T
    delete_row=np.concatenate((top40, bottom40))
    input_folder = '/run/user/0/stage_feature/new_data/all_timecourse'
    output_folder = '/run/user/0/stage_feature/new_data/delete_feature'

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        matrix = np.loadtxt(file_path)
        matrix = matrix[:120, :]

        matrix1 = np.delete(matrix, delete_row, axis=0)

        output1 = os.path.join(output_folder, filename)

        np.savetxt(output1, matrix1)

    AD_feature=count_single_group(0)
    print(AD_feature)