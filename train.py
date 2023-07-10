from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics as metrics
import numpy as np
from tqdm import tqdm
from dataset import load_dataset, GeoDatset, make_x_y
from torch.utils.data import DataLoader
from model import CNN_model
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    conf_matrix = np.zeros((21,21)) # initialize confusion matrix
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # determine index with maximal log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # update confusion matrix
            conf_matrix = conf_matrix + metrics.confusion_matrix(
                          target.cpu(),pred.cpu(),labels=list(range(21)))
            
        # print confusion matrix
        np.set_printoptions(precision=4, suppress=True)
        print(type(conf_matrix))
        print(conf_matrix)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default="Dataset",help='directory of dataset')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--mom',type=float,default=0.8,help='momentum')
    parser.add_argument('--epochs',type=int,default=10,help='number of training epochs')
    parser.add_argument('--no_cuda',action='store_true',default=False,help='disables CUDA')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # fetch and load training data
    train_X, train_y, val_X, val_y = load_dataset(args.dataset)

    train_dataset = GeoDatset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,collate_fn=make_x_y)

    # fetch and load test data
    val_dataset = GeoDatset(val_X, val_y)
    test_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=make_x_y)

    net = CNN_model().to(device)

    if list(net.parameters()):
        # use SGD optimizer
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom)

        # training and testing loop
        for epoch in range(1, args.epochs + 1):
            train(args, net, device, train_loader, optimizer, epoch)
            test(args, net, device, test_loader)
        
if __name__ == '__main__':
    main()
