import os 
import argparse
from time import time
import random
import sys

import numpy as np
import torch
import torch_geometric
import torch.nn.functional as F
from torch import optim
from torch_geometric.data import Data, DataLoader
from torch.optim.lr_scheduler import StepLR

from interaction_network import InteractionNetwork
from dataset import GraphDataset

sys.path.append("/Users/rohan/cosi/nn-recoil-electron-tracking")
from Visualizer import save_pred_projections

def most_probable_edges(y_hat, num_edges):
    """select num_edges most probable edges, regardless of whether they form a path"""
    threshold = sorted(y_hat)[-num_edges-1]
    pred_path = torch.tensor([1.0 if edge_probability > threshold else 0.0 for edge_probability in y_hat], requires_grad=False)
    return pred_path

def reverse_permutation(pred_path, edge_index, permutation):
    n = len(permutation)
    pd = {}
    for i, ord in enumerate(permutation):
        pd[ord] = i
    path_edges = []
    for i in range(len(pred_path)):
        if pred_path[i] == 1:
            from_node = edge_index[0][i]
            to_node = edge_index[1][i]
            path_edges.append((from_node, to_node))

    unpermuted_path_edges = []
    for i, j in path_edges:
        unpermuted_path_edges.append((permutation[i], permutation[j]))

    return unpermuted_path_edges
    
def get_inference(model: InteractionNetwork, data: Data):
    output = model.forward(data.x, data.edge_index, data.edge_attr)
    return output

def main():
    parser = argparse.ArgumentParser(description='PyG Interaction Network Implementation')
    parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                        help='how many batches to wait before logging inference status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--hidden-size', type=int, default=40,
                        help='Number of hidden units per layer')
    parser.add_argument('--model_save', type=str, default='./trained_models/model_epoch_100.pt',
                        help='model save')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--data_set', type=str, default='../data/RecoilElectrons.10k.data',
                        help='data set')
    args = parser.parse_args()

    test_set = GraphDataset(args.data_set)
    
    model_state_dict = torch.load(args.model_save)
    model = InteractionNetwork(args.hidden_size)
    model.load_state_dict(model_state_dict)

    num_samples = len(test_set)

    correct_paths = 0
    avg_path_accuracy = 0

    for i in range(len(test_set)):
        data = test_set[i]
        y_hat = get_inference(model, data).squeeze()
        data.y = data.y.squeeze()
        num_nodes = data.x.size()[0]
        num_edges = num_nodes*(num_nodes-1)
        pred_path = most_probable_edges(y_hat, num_nodes-1)
        path_accuracy = torch.dot(data.y, pred_path) / (num_nodes-1)

        avg_path_accuracy += path_accuracy
        if path_accuracy >= 0.99: # floating pt issues
            correct_paths += 1

        if False and i % args.log_interval == 0:
            print(f"Results for {i} / {num_samples}")
            print(f"Correctly predicted paths: {correct_paths} / {i}")
            print(f"Average dot product: {avg_path_accuracy / i}")

        unpermuted_pred_path = reverse_permutation(pred_path, data.edge_index, data.permutation)
        save_pred_projections(test_set.get_event(i), 
                              save_file='temp',
                              show_vectors=False,
                              show_track=True,
                              pred_edge_list=unpermuted_pred_path)

        if args.verbose:
            print(">>>>")
            print(data.y)
            print(pred_path)
            print(path_accuracy)
        
    print(f"Final Results:")
    print(f"proportion correctly predicted: {correct_paths / num_samples}")
    print(f"average dot product: {avg_path_accuracy / num_samples}")
    """
    Last Run gave:
    Final Results:
    Correctly predicted paths: 0.6491
    Average dot product: 0.7969236373901367    
    """

if __name__ == '__main__':
    main()
