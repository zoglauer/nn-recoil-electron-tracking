import os 
import argparse
from time import time
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

sys.path.append("/global/home/users/rbohra/RecoilElectronTracking/interaction_network/")
from simple_dataset import SimpleDataset
sys.path.append("/global/home/users/rbohra/RecoilElectronTracking")

from simple_model import SimpleModel

MODEL_ITERATION = 'modelv1'

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_t0 = time()
    losses = []
    for batch_idx, data in enumerate(train_loader):
        X, y = data
        X = torch.flatten(X, start_dim=1)
        X.to(device)
        y.to(device)
        optimizer.zero_grad()
        output = model(X).softmax(1)
        loss = F.binary_cross_entropy(output, y, reduction='mean')
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        losses.append(loss.item())
        
    print("...epoch time: {0}s".format(time()-epoch_t0))
    print("...epoch {}: train loss={}".format(epoch, np.mean(losses)))
    return np.mean(losses)

def test(model, device, test_loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            X, y = data
            X = torch.flatten(X, start_dim=1)
            X.to(device)
            y.to(device)
            output = model(X).softmax(1)
            loss = F.binary_cross_entropy(output, y, 
                                          reduction='mean').item()
            losses.append(loss)
            #print(f"acc={TP+TN}/{TP+TN+FP+FN}={acc}")

    print('...test loss: {:.4f}'.format(np.mean(losses)))
    return np.mean(losses)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyG Interaction Network Implementation')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--step-size', type=int, default=8,
                        help='Learning rate step size')
    parser.add_argument('--log-interval', type=int, default=100000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--hidden-size', type=int, default=40,
                        help='Number of hidden units per layer')
    parser.add_argument('--model-iteration', type=str, default=MODEL_ITERATION,
                        help='Name of model iteration')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume from provided save model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='do not use cuda')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(f"torch v{torch.__version__}")
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
    
    train_set = SimpleDataset('../data/RecoilElectrons.100k.data')
    train_loader = DataLoader(train_set, **params)
    test_set = SimpleDataset('../data/RecoilElectrons.10k.data')
    test_loader = DataLoader(test_set, **params)
    val_set = SimpleDataset('../data/RecoilElectrons.1k.data')
    val_loader = DataLoader(val_set, **params)
    
    model = SimpleModel(args.hidden_size).to(device)
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
        model = SimpleModel(args.hidden_size)
        model.load_state_dict(model_state_dict)


    for epoch in range(start_epoch, args.epochs + 1):
        print("---- Epoch {} ----".format(epoch))
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        scheduler.step()

        output['train_loss'].append(train_loss)
        output['test_loss'].append(test_loss)

        torch.save(model.state_dict(),
                    f"{args.model_iteration}/model_epoch_{epoch}.pt")
        torch.save(output, f"{args.model_iteration}/loss_dict.bin")

if __name__ == '__main__':
    main()
