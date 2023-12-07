import numpy as np
import argparse
import os.path as osp
import random
import nni
from torch_geometric.data import Data, DataLoader
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected
from Stage_BGL.SVD import count_single_group
from sklearn.manifold import TSNE

from Stage_BGL.Dataset import get_ids,get_labels,transform_dataset_original,transform_dataset_feature
from get_param.get import get_param
from Stage_BGL.model import Encoder, GRACE
from Stage_BGL.augmentation import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from Stage_BGL.evaluate import log_regression, MulticlassEvaluator
from Stage_BGL.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def x1_x2(data,AD_feature,LMCI_feature,MCI_feature,EMCI_feature,NC_feature,edge_id,drop_rate):

    def drop_edge(idx: int,data):

        if param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'],
                                      threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')


    for i in range((data.batch_size)):
        if param['drop_scheme'] == 'degree':
            drop_weights = degree_drop_weights(data[i].edge_index).to(device)
        else:
            drop_weights = None

        if data.y[i]==0:
            feature_martix=AD_feature
        elif data.y[i] == 1:
            feature_martix = LMCI_feature
        elif data.y[i] == 2:
            feature_martix = MCI_feature
        elif data.y[i]==3:
            feature_martix = EMCI_feature
        else:
            feature_martix = NC_feature
        feature_martix=feature_martix.T

        feature_martix=torch.from_numpy(feature_martix).to(device)

        if param['drop_scheme'] == 'degree':
            edge_index_ = to_undirected(data.edge_index)
            node_deg = degree(edge_index_[1])
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data[i].x, node_c=feature_martix).to(device)
            else:
                feature_weights = feature_drop_weights(data[i].x, node_c=feature_martix).to(device)
        else:
            feature_weights = torch.ones((data[i].x.size(1),)).to(device)

        data[i].edge_index = drop_edge(edge_id,data[i])
        data[i].x = drop_feature(data[i].x, drop_rate)

        if param['drop_scheme'] in ['pr', 'degree', 'evc']:
            data[i].x = drop_feature_weighted_2(data[i].x, feature_weights, drop_rate)


    return data



def train(dataset,device,AD_feature,LMCI_feature,MCI_feature,EMCI_feature,NC_feature):
    model.train().to(device)
    optimizer.zero_grad()


    dataset=dataset.shuffle()
    loader1 = DataLoader(dataset, batch_size=10, shuffle=False)
    total_loss = 0



    for data in loader1:

        data=data.to(device)


        data1=x1_x2(data,AD_feature, LMCI_feature, MCI_feature, EMCI_feature, NC_feature,edge_id=1,drop_rate=param['drop_feature_rate_1'])
        data2 = x1_x2(data, AD_feature, LMCI_feature, MCI_feature, EMCI_feature, NC_feature, edge_id=2,
                      drop_rate=param['drop_feature_rate_2'])

        z1 = model(data1.x, data1.edge_index)
        z2 = model(data2.x, data2.edge_index)

        loss = model.loss(z1, z2)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()


    return total_loss


def eval_acc(dataset,final=False):
    model.eval().to(device)

    evaluator = MulticlassEvaluator()

    for data in dataset:
        data=data.to(device)

    if final:
        score,plot=log_regression(model, dataset, evaluator, Plot=True,test_device= device,split='rand:0.2', num_epochs=1000, preload_split=split)
    else:
        score,plot = log_regression(model, dataset, evaluator, test_device= device,split='rand:0.2', num_epochs=1000, preload_split=split)

    train_acc,test_acc,f1=score.values()

    if final and use_nni:
        nni.report_final_result(train_acc)
        # nni.report_final_result(auc)
        nni.report_final_result(f1)
    elif use_nni:
        nni.report_intermediate_result(train_acc)
        nni.report_intermediate_result(f1)
        # nni.report_intermediate_result(auc)

    return train_acc,test_acc,f1,plot




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--param', type=str, default='local:default.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 500,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()

    # parse param
    sp = get_param(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)

    subject_IDs = get_ids()
    labels = get_labels()
    FCN_dataset = transform_dataset_feature(subject_IDs, label=labels)

    AD_feature=count_single_group(0)
    LMCI_feature = count_single_group(1)
    MCI_feature=count_single_group(2)
    EMCI_feature = count_single_group(3)
    NC_feature=count_single_group(4)


    # generate split
    split = generate_split(len(FCN_dataset), train_ratio=0.1, val_ratio=0.1)

    if args.save_split:
        torch.save(split, args.save_split)
    elif args.load_split:
        split = torch.load(args.load_split)

    total_train_loss=0
    num_features=FCN_dataset[0].num_features

    encoder = Encoder(num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    for epoch in range(1, param['num_epochs'] + 1):


        loss = train(FCN_dataset,device,AD_feature,LMCI_feature,MCI_feature,EMCI_feature,NC_feature)

        log = args.verbose.split(',')

        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

        if epoch % 100 == 0:
            train_acc,test_acc,f1,plot = eval_acc(dataset=FCN_dataset)

            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, train_acc={train_acc},test_acc={test_acc},F1_score={f1} ')

    train_acc,test_acc,f1,plot = eval_acc(dataset=FCN_dataset,final=True)


    if 'final' in log:
        print(f'train_acc={train_acc},test_acc={test_acc},F1_score={f1}')
        plot.show()
