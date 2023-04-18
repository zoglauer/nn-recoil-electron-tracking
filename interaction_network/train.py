import os 
import argparse
from time import time
import sys

import numpy as np
import torch
print(torch.__version__)
import torch_geometric
import torch.nn.functional as F
from torch import optim
from torch_geometric.data import Data, DataLoader
from torch.optim.lr_scheduler import StepLR

from interaction_network import InteractionNetwork
from dataset import GraphDataset
sys.path.append("/Users/rohan/cosi/nn-recoil-electron-tracking")
sys.path.append("/home/rohan/nn-recoil-electron-tracking/")
MODEL_ITERATION = 'modelv6_directed_batchsize16_hidden60'

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_t0 = time()
    losses = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr)
        y, output = data.y, output
        loss = F.binary_cross_entropy(output, y, reduction='mean')
        loss.backward()
        optimizer.step()
        if False and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            #print(data.x.size(), data.edge_attr.size(), data.edge_index.size(), output.size())
        losses.append(loss.item())
    print("...epoch time: {0}s".format(time()-epoch_t0))
    print("...epoch {}: train loss={}".format(epoch, np.mean(losses)))
    return np.mean(losses)

def validate(model, device, val_loader):
    model.eval()
    opt_thlds, accs = [], []
    for batch_idx, data in enumerate(val_loader):
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr)
        y, output = data.y, output
        loss = F.binary_cross_entropy(output, y, reduction='mean').item()
        
        # define optimal threshold (thld) where TPR = TNR 
        diff, opt_thld, opt_acc = 100, 0, 0
        best_tpr, best_tnr = 0, 0
        for thld in np.arange(0.001, 0.5, 0.001):
            TP = torch.sum((y==1) & (output>thld)).item()
            TN = torch.sum((y==0) & (output<thld)).item()
            FP = torch.sum((y==0) & (output>thld)).item()
            FN = torch.sum((y==1) & (output<thld)).item()
            acc = (TP+TN)/(TP+TN+FP+FN)
            if TP + FN == 0:
                TPR = 0
            else:
                TPR = TP/(TP+FN)
            if TN + FP == 0:
                TNR = 0
            else:
                TNR = TN/(TN+FP)
            delta = abs(TPR-TNR)
            if (delta < diff): 
                diff, opt_thld, opt_acc = delta, thld, acc

        opt_thlds.append(opt_thld)
        accs.append(opt_acc)

    print("...val accuracy=", np.mean(accs))
    return np.mean(opt_thlds) 

def test(model, device, test_loader, thld=0.5):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr)
            TP = torch.sum((data.y==1).squeeze() & 
                           (output>thld).squeeze()).item()
            TN = torch.sum((data.y==0).squeeze() & 
                           (output<thld).squeeze()).item()
            FP = torch.sum((data.y==0).squeeze() & 
                           (output>thld).squeeze()).item()
            FN = torch.sum((data.y==1).squeeze() & 
                           (output<thld).squeeze()).item()            
            acc = (TP+TN)/(TP+TN+FP+FN)
            loss = F.binary_cross_entropy(output, data.y, 
                                          reduction='mean').item()
            accs.append(acc)
            losses.append(loss)
            #print(f"acc={TP+TN}/{TP+TN+FP+FN}={acc}")

    print('...test loss: {:.4f}\n...test accuracy: {:.4f}'
          .format(np.mean(losses), np.mean(accs)))
    return np.mean(losses), np.mean(accs)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyG Interaction Network Implementation')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--step-size', type=int, default=8,
                        help='Learning rate step size')
    parser.add_argument('--pt', type=str, default='2',
                        help='Cutoff pt value in GeV (default: 2)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--hidden-size', type=int, default=40,
                        help='Number of hidden units per layer')
    parser.add_argument('--model-iteration', type=str, default=MODEL_ITERATION,
                        help='Name of model iteration')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume from provided save model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("use_cuda={0}".format(use_cuda))    
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 4}
    
    train_set = GraphDataset('../data/RecoilElectrons.100k.data', directed=True)
    train_loader = DataLoader(train_set, **params)
    test_set = GraphDataset('../data/RecoilElectrons.10k.data', directed=True)
    test_loader = DataLoader(test_set, **params)
    val_set = GraphDataset('../data/RecoilElectrons.1k.data', directed=True)
    val_loader = DataLoader(val_set, **params)
    
    model = InteractionNetwork(args.hidden_size).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    print('total trainable params:', total_trainable_params)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(optimizer)
    scheduler = StepLR(optimizer, step_size=args.step_size,
                       gamma=args.gamma)
    print(f"scheduler ( \n\tgamma: {args.gamma}\n\tstep size: {args.step_size}\n)")

    output = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    start_epoch = 1

    if args.resume:
        output = torch.load(f"{args.model_iteration}/loss_dict.bin")
        start_epoch = len(output['train_loss']) + 1
        print(f"Resuming at epoch {start_epoch}")
        model_state_dict = torch.load(f"{args.model_iteration}/model_epoch_{start_epoch-1}.pt")
        model = InteractionNetwork(args.hidden_size)
        model.load_state_dict(model_state_dict)


    for epoch in range(start_epoch, args.epochs + 1):
        print("---- Epoch {} ----".format(epoch))
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        thld = validate(model, device, val_loader)
        print('...optimal threshold', thld)
        test_loss, test_acc = test(model, device, test_loader, thld=thld)
        scheduler.step()

        output['train_loss'].append(train_loss)
        output['test_loss'].append(test_loss)
        output['test_acc'].append(test_acc)

        if args.save_model:
            torch.save(model.state_dict(),
                       f"{args.model_iteration}/model_epoch_{epoch}.pt")
            torch.save(output, f"{args.model_iteration}/loss_dict.bin")

if __name__ == '__main__':
    main()
