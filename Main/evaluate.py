from typing import Optional
import numpy as np

import torch
from torch.optim import Adam
import torch.nn as nn
from torch_geometric.data import Data,DataLoader

from sklearn.metrics import f1_score,roc_auc_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from Stage_BGL.model import LogReg


def get_idx_split(dataset, split):
    if split[:4] == 'rand':
        train_ratio = float(split.split(':')[1])
        num_Graphs = len(dataset)
        train_size = int(num_Graphs * train_ratio)
        indices = torch.randperm(num_Graphs)
        return {
            'train': indices[:train_size],
            'val': indices[train_size:2 * train_size],
            'test': indices[2 * train_size:]
        }


def log_regression(model,
                   dataset,
                   evaluator,
                   num_epochs: int = 5000,
                   test_device: Optional[str] = None,
                   split: str = 'rand:0.1',
                   verbose: bool = False,
                   preload_split=None,
                   Plot=False):
    data_example=dataset[0].to(test_device)
    z1=model(data_example.x,data_example.edge_index)
    z1=z1.detach()
    num_classes = 5
    num_hidden = z1.size(1)
    classifier = LogReg(num_hidden, num_classes).to(test_device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
    split = get_idx_split(dataset, split)
    split = {k: v.to(test_device) for k, v in split.items()}
    f = nn.LogSoftmax(dim=-1)
    nll_loss = nn.NLLLoss()

    best_test_acc = 0
    best_test_f1=0
    best_test_auc = 0
    best_val_acc = 0
    best_val_f1=0
    best_val_auc = 0
    best_epoch = 0


    loader1=DataLoader(dataset[split['train']],batch_size=8,shuffle=False)
    loader2=DataLoader(dataset[split['val']],batch_size=8,shuffle=False)
    loader3=DataLoader(dataset[split['test']], batch_size=8, shuffle=False)
    loader4= DataLoader(dataset,batch_size=1,shuffle=False)


    plt.figure(figsize=(10, 6))
    for epoch in range(num_epochs):
        classifier.train()

        best_train_acc=0
        y_train = torch.tensor([]).to(test_device)
        cla_train = torch.tensor([]).to(test_device)
        for data1 in loader1:
            data1=data1.to(test_device)
            z = model(data1.x, data1.edge_index)
            z = z.detach().to(test_device)
            y = data1.y
            y_train = torch.cat([y_train, y])
            cla_train_temp = classifier(z, data1.batch)
            cla_train = torch.cat([cla_train, cla_train_temp])
            cla = classifier(z, data1.batch)
            loss = nll_loss(f(cla), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_acc, train_f1 = evaluator.eval({
            'y_true': y_train.view(-1, 1),
            'y_pred': cla_train.argmax(1).view(-1, 1)
        }).values()
        if best_train_acc < train_acc:
            best_train_acc = train_acc
            best_epoch = epoch



        if epoch==(num_epochs-1) and Plot==True:
            cla_plot=torch.tensor([]).to(test_device)
            for data4 in loader4:
                data4 = data4.to(test_device)
                z = model(data4.x, data4.edge_index)
                z = z.detach().to(test_device)
                cla = classifier(z, data4.batch)
                cla_plot= torch.cat([cla, cla_plot])

            label = cla_plot.argmax(1).view(-1, 1)
            cla_plot=cla_plot.detach().cpu().numpy()
            label=label.detach().cpu().numpy()
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(cla_plot)
            label_mapping = {
                0: "AD",
                1: "LMCI",
                2: "MCI",
                3: "EMCI",
                4: "NC"
            }
            for i in range(5):
                indices = np.where(label == i)
                class_label = label_mapping[i]  # 获取对应的字符串标签
                plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=f' {class_label}')

            plt.legend(loc='upper left',bbox_to_anchor=(1, 1))
            plt.title('Brain Network Clustering')




        if (epoch + 1) % 20 == 0:
            y_val = torch.tensor([]).to(test_device)
            y_test = torch.tensor([]).to(test_device)
            cla_test=torch.tensor([]).to(test_device)
            cla_val=torch.tensor([]).to(test_device)
            if 'val' in split:
                # val split is available
                for data3 in loader3:
                    data3 = data3.to(test_device)
                    z3 = model(data3.x, data3.edge_index)
                    z3 = z3.detach().to(test_device)
                    y = data3.y
                    y_test=torch.cat([y_test,y])
                    cla_test_temp = classifier(z3, data3.batch)
                    cla_test=torch.cat([cla_test,cla_test_temp])
                test_acc, test_f1 = evaluator.eval({
                    'y_true': y_test.view(-1, 1),
                    'y_pred': cla_test.argmax(1).view(-1, 1)
                }).values()

                for data2 in loader2:
                    data2 = data2.to(test_device)
                    z2 = model(data2.x, data2.edge_index)
                    z2 = z2.detach().to(test_device)
                    y = data2.y
                    y_val = torch.cat([y_val, y])
                    cla_val_temp = classifier(z2, data2.batch)
                    cla_val = torch.cat([cla_val, cla_val_temp])
                val_acc ,val_f1= evaluator.eval({
                    'y_true': y_val.view(-1, 1),
                    'y_pred': cla_val.argmax(1).view(-1, 1)
                }).values()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch
                if val_f1 > best_val_f1:
                    best_val_f1=val_f1
                    best_test_f1 = test_f1
                    best_epoch = epoch
                # if val_auc > best_val_auc:
                #     best_val_auc=val_auc
                #     best_test_auc = test_auc
                #     best_epoch = epoch
            else:
                for data3 in loader3:
                    data3 = data3.to(test_device)
                    z3 = model(data3.x, data3.edge_index)
                    z3 = z3.detach().to(test_device)
                    y = data3.y
                    y_test=torch.cat([y_test,y])
                    cla_test_temp = classifier(z3, data3.batch)
                    cla_test = torch.cat([cla_test, cla_test_temp])
                acc, f1,auc = evaluator.eval({
                    'y_true': y.test.view(-1, 1),
                    'y_pred': cla_test.argmax(1).view(-1, 1)
                }).values()
                if best_test_acc < acc:
                    best_test_acc = acc
                    best_epoch = epoch
                if best_test_f1 < f1:
                    best_test_f1 = f1
                    best_epoch = epoch
                # if best_test_auc < auc:
                #     best_test_auc = auc
                #     best_epoch = epoch
            if verbose:
                print(f'logreg epoch {epoch}: best test acc {best_test_acc}')


    return {'train_acc':best_train_acc,'test_acc': best_test_acc,'f1':best_test_f1},plt



class MulticlassEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        total = y_true.size(0)
        correct = (y_true == y_pred).to(torch.float32).sum()
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        f1 = f1_score(y_true_np, y_pred_np, average='micro')

        return  (correct / total).item(),f1

    def eval(self, res):
        acc, f1  = self._eval(**res)
        return {'acc': acc, 'f1': f1}
