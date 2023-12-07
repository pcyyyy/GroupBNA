import os
import csv
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import scipy.sparse as sp
from torch_geometric.data import DataLoader


data_folder='/run/user/0/stage_feature/new_data'


def get_ids(num_subjects=None):
    """
    return:
        subject_IDs    : list of all subject IDs
    """
    subject_IDs = []
    file_AD=[os.path.splitext(file)[0] for file in os.listdir(os.path.join(data_folder,"AD"+"/timecourse"))]
    file_lmci=[os.path.splitext(file)[0] for file in os.listdir(os.path.join(data_folder,"lmci"+"/timecourse"))]
    file_mci = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(data_folder, "mci" + "/timecourse"))]
    file_emci = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(data_folder, "emci" + "/timecourse"))]
    file_nc = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(data_folder, "NC" + "/timecourse"))]

    subject_IDs.append(file_AD)
    subject_IDs.append(file_lmci)
    subject_IDs.append(file_mci)
    subject_IDs.append(file_emci)
    subject_IDs.append(file_nc)


    return subject_IDs

def get_labels():
    # labels = ["AD"] * 45 + ["MCI"] * 45  + ["NC"] * 45
    labels = ["AD"] * 25 + ["lmci"] * 25 + ["MCI"] * 25 + ["emci"] * 25+ ["nc"] * 25
    return labels

def my_custom_transform(data):

    return data
def my_custom_pretransform(data):

    return data

class MyCustomDataset(Dataset):
    def __init__(self, data_list, transform, pre_transform):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def transform_dataset_original(subject_list,label):
    dataset = []
    i=0
    for list in subject_list:
        for subject in list:
            path_adj = "/run/user/0/GCA/data/new/all_FCN"
            path_node = "/run/user/0/GCA/data/new/all_timecourse"
            adj_fl = os.path.join(path_adj, subject + ".txt")
            node_fl = os.path.join(path_node, subject + ".txt")
            x1_temp = np.loadtxt(node_fl, dtype=np.float32).T
            if min(x1_temp)<-100:
                x1_temp=x1_temp/100
            x1_temp=x1_temp+100

            x1 = x1_temp[:, :100]
            pca=PCA(n_components=20)
            x1=pca.fit_transform(x1)
            x1 = torch.from_numpy(x1)
            x1 = x1.to(torch.float32)

            adj_matrix = np.loadtxt(adj_fl, dtype=np.float64)
            adj_matrix[adj_matrix < 0]=0
            edge_index_temp = sp.coo_matrix(adj_matrix)
            values = edge_index_temp.data

            indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
            edge_index_A = torch.tensor(indices, dtype=torch.long)
            v = torch.tensor(values, dtype=torch.float64)

            y1_label=label[i]
            if y1_label in "AD":
                y1=0
            elif y1_label in "MCI":
                y1 = 1
            else :
                y1=2

            i+=1

            dataset.append(
                Data(x=x1, edge_index=edge_index_A, y=torch.tensor(y1, dtype=torch.long),
                     edge_attr=v))

    Dataset = MyCustomDataset(dataset, transform=my_custom_transform, pre_transform=my_custom_pretransform)

    return Dataset



def transform_dataset_feature(subject_list,label):
    dataset = []
    i = 0
    for list in subject_list:
        for subject in list:
            path_adj = "/run/user/0/stage_feature/new_data/all_fcn"
            path_node = "/run/user/0/stage_feature/new_data/delete_feature"
            adj_fl = os.path.join(path_adj, subject + ".txt")
            node_fl = os.path.join(path_node, subject + ".txt")
            x1_temp = np.loadtxt(node_fl, dtype=np.float32).T
            x1_temp[x1_temp < 0] = 0

            x1 = torch.from_numpy(x1_temp)
            x1 = x1.to(torch.float32)


            adj_matrix = np.loadtxt(adj_fl, dtype=np.float64)
            adj_matrix[adj_matrix < 0] = 0
            edge_index_temp = sp.coo_matrix(adj_matrix)
            values = edge_index_temp.data

            indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
            edge_index_A = torch.tensor(indices, dtype=torch.long)
            v = torch.tensor(values, dtype=torch.float64)

            # 标签0为AD，标签1为lmci，标签2为mci,标签3为emci,标签4为nc
            y1_label = label[i]
            if y1_label in "AD":
                y1 = 0
            elif y1_label in "lmci":
                y1 = 1
            elif y1_label in "MCI":
                y1 = 2
            elif y1_label in "emci":
                y1 = 3
            else:
                y1 = 4

            i += 1

            dataset.append(
                Data(x=x1, edge_index=edge_index_A, y=torch.tensor(y1, dtype=torch.long),
                     edge_attr=v))

    Dataset = MyCustomDataset(dataset, transform=my_custom_transform, pre_transform=my_custom_pretransform)

    return Dataset


if __name__ == '__main__':
    subject_IDs = get_ids()
    labels = get_labels()
    # FCN_dataset = transform_dataset_original(subject_IDs, label=labels)
    FCN_dataset = transform_dataset_feature(subject_IDs, label=labels)





