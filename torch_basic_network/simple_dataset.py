import torch
from torch.utils.data import Dataset, DataLoader

import random
import pickle
import sys

sys.path.append("/Users/rohan/cosi/nn-recoil-electron-tracking")
from EventData import EventData

MAX_N = 15
NODE_FEATURES = 4 # x, y, z, e

class SimpleDataset(Dataset):
    def __init__(self, filename):
        # we randomly generate an array of ints that will act as data
        self.event_list = []
        with open(filename, "rb") as file_handle:
            self.event_list = pickle.load(file_handle)   

    def __len__(self):
        return len(self.event_list)

    def __getitem__(self, idx):
        event = self.event_list[idx]
        N = len(event.E)
        E = N*(N-1)

        permutation = [i for i in range(N)]
        random.shuffle(permutation)

        X = torch.zeros((MAX_N, 4))
        y_size = MAX_N * (MAX_N-1)
        y = torch.zeros(y_size)

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
                if c == y_size:
                    print(N, y_size)
                    break

        for i in range(NODE_FEATURES):
            mean, std = torch.mean(X[:, i]), torch.std(X[:, i])
            X[:, i] = (X[:, i]-mean)/std
        X = torch.nan_to_num(X)
        return X, y

if __name__ == '__main__':
    ds = SimpleDataset('../data/RecoilElectrons.1k.data')
    for i in range(10):
        X, y = ds[i]
        print(X.size(), y.size())
        print(X)
        print(y)