import os
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected
import pickle
import gzip
import random
import sys
import math
sys.path.append("/home/rohan/nn-recoil-electron-tracking/")
sys.path.append("/global/home/users/rbohra/RecoilElectronTracking")
sys.path.append("/Users/rohan/cosi/nn-recoil-electron-tracking")
import EventData

def euclidean_dist(event: EventData, i, j):
    """euclidean dist between hitpoint i and j"""
    return math.sqrt((event.X[i] - event.X[j])**2 + (event.Y[i] - event.Y[j])**2 + (event.Z[i] - event.Z[j])**2)

class GraphDataset(Dataset):
    def __init__(self, filename, directed=True, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(None, transform, pre_transform)

        self.directed = directed

        self.event_list = []
        with gzip.open(filename, "rb") as file_handle:
            self.event_list = pickle.load(file_handle)   
        
    @property
    def raw_file_names(self):
        return self.graph_files

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.event_list)
        
    def get_undirected(self, idx):
        event = self.event_list[idx]
        N = len(event.E)
        E = N*(N-1)//2

        permutation = [i for i in range(N)]
        random.shuffle(permutation)

        X = torch.zeros((N, 4))
        y = torch.zeros((E, 1))
        edge_index = torch.zeros((2, E), dtype=torch.int64)
        edge_attr = torch.zeros((E, 2)) # just use euclidean dist, energy diff for now

        c = 0
        for i in range(N):
            for j in range(i+1, N):
                edge_index[0][c] = i
                edge_index[1][c] = j
                edge_attr[c][0] = euclidean_dist(event, permutation[i], permutation[j])
                edge_attr[c][1] = abs(event.E[permutation[i]]-event.E[permutation[j]])
                c += 1

        c = 0
        for i in range(N):
            X[i][0] = event.X[permutation[i]]
            X[i][1] = event.Y[permutation[i]]
            X[i][2] = event.Z[permutation[i]]
            X[i][3] = event.E[permutation[i]]

            for j in range(i+1, N):
                if -1 <= permutation[j] - permutation[i] <= 1:
                    y[c] = 1
                else:
                    y[c] = 0
                c += 1
        
        return Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, permutation=permutation, eventdata=event)

    def get_original(self, idx, num_node_features=4, num_edge_features=5):
        event = self.event_list[idx]
        N = len(event.E)
        E = N*(N-1)

        permutation = [i for i in range(N)]
        random.shuffle(permutation)

        X = torch.zeros((N, 4))
        y = torch.zeros((E, 1))
        edge_index = torch.zeros((2, E), dtype=torch.int64)
        edge_attr = torch.zeros((E, 5)) # just use euclidean dist, energy diff for now, maybe add dx, dy, dz?

        c = 0
        for i in range(N):
            X[i][0] = event.X[permutation[i]]
            X[i][1] = event.Y[permutation[i]]
            X[i][2] = event.Z[permutation[i]]
            X[i][3] = event.E[permutation[i]]

            for j in range(N):
                if i == j:
                    continue
                if permutation[j] == permutation[i] + 1:
                    y[c] = 1
                else:
                    y[c] = 0
                c += 1

        for i in range(num_node_features):
            mean, std = torch.mean(X[:, i]), torch.std(X[:, i])
            X[:, i] = (X[:, i]-mean)/std
        
        X = torch.nan_to_num(X)
        c = 0
        for i in range(N):
            for j in range(N):
                if i != j:
                    edge_index[0][c] = i
                    edge_index[1][c] = j
                    edge_attr[c][0] = euclidean_dist(event, permutation[i], permutation[j])
                    edge_attr[c][1] = event.E[permutation[i]]-event.E[permutation[j]]
                    edge_attr[c][2] = event.X[permutation[i]]-event.X[permutation[j]]
                    edge_attr[c][3] = event.Y[permutation[i]]-event.Y[permutation[j]]
                    edge_attr[c][4] = event.Z[permutation[i]]-event.Z[permutation[j]]
                    c += 1

        for i in range(num_edge_features):
            mean, std = torch.mean(edge_attr[:, i]), torch.std(edge_attr[:, i])
            edge_attr[:, i] = (edge_attr[:, i]-mean)/std
        edge_attr = torch.nan_to_num(edge_attr)

        return Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, permutation=permutation, eventdata=event)

    def get_unpermuted(self, idx, num_node_features, num_edge_features):
        event = self.event_list[idx]
        N = len(event.E)
        E = N*(N-1)

        X = torch.zeros((N, 4))
        y = torch.zeros((E, 1))
        edge_index = torch.zeros((2, E), dtype=torch.int64)
        edge_attr = torch.zeros((E, 5)) # just use euclidean dist, energy diff for now, maybe add dx, dy, dz?

        c = 0
        for i in range(N):
            X[i][0] = event.X[i]
            X[i][1] = event.Y[i]
            X[i][2] = event.Z[i]
            X[i][3] = event.E[i]

            for j in range(N):
                if i == j:
                    continue
                if j == i + 1:
                    y[c] = 1
                else:
                    y[c] = 0
                c += 1

        for i in range(num_node_features):
            mean, std = torch.mean(X[:, i]), torch.std(X[:, i])
            X[:, i] = (X[:, i]-mean)/std
        
        X = torch.nan_to_num(X)
        c = 0
        for i in range(N):
            for j in range(N):
                if i != j:
                    edge_index[0][c] = i
                    edge_index[1][c] = j
                    edge_attr[c][0] = euclidean_dist(event, i, j)
                    edge_attr[c][1] = event.E[i]-event.E[j]
                    edge_attr[c][2] = event.X[i]-event.X[j]
                    edge_attr[c][3] = event.Y[i]-event.Y[j]
                    edge_attr[c][4] = event.Z[i]-event.Z[j]
                    c += 1

        for i in range(num_edge_features):
            mean, std = torch.mean(edge_attr[:, i]), torch.std(edge_attr[:, i])
            edge_attr[:, i] = (edge_attr[:, i]-mean)/std
        edge_attr = torch.nan_to_num(edge_attr)

        return Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, eventdata=event)

    def get(self, idx, num_node_features=4, num_edge_features=5):
        if not self.directed:
            return self.get_undirected(idx)
        return self.get_unpermuted(idx, num_node_features=4, num_edge_features=5)
             
    def get_event(self, idx):
        return self.event_list[idx]

if __name__ == '__main__':
    n = 10
    ds = GraphDataset('../data/RecoilElectrons.10k.data', directed=True)
    a = [random.randint(0, len(ds)) for i in range(n)]
    for i in a:
        data = ds.get(i)
        