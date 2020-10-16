
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as udata
import torch.optim as optim
from torch.distributions import multivariate_normal as mvn

import torchvision
import torchvision.datasets as tdatasets
from torchvision import transforms

import matplotlib.pyplot as plt

import os
import os.path as osp
import argparse

from model import MNIST_CNN
from utils import train, test

here = osp.abspath(osp.dirname(__file__))
parent = osp.abspath(osp.dirname(here))


Z = lambda n, beta: mvn.MultivariateNormal(torch.zeros(n), torch.diag(torch.ones(n) * beta**2))


def train_MNIST_CNN(device, beta=0.05):
    # Zero mean AWGN
    Z_1 = Z(16, beta)
    Z_2 = Z(16, beta)
    Z_3 = Z(128, beta)

    # Create data folder
    data_path = osp.join(parent, "data")
    if not osp.exists(data_path):
        os.mkdir(data_path)

    # Download training and test sets
    trainset = tdatasets.MNIST(root=data_path, train=True, download=True, 
                                transform=transforms.ToTensor())
    train_loader = udata.DataLoader(trainset, batch_size=32, shuffle=True)

    testset = tdatasets.MNIST(root=data_path, train=False, download=True,
                                transform=transforms.ToTensor())
    test_loader = udata.DataLoader(testset, batch_size=32, shuffle=False)
    
    # Instantiate the model and train it for 128 epochs
    model = MNIST_CNN(Z_1, Z_2, Z_3, device).to(device)
    model.initialize_parameters()

    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    optimizer = optim.SGD(model.parameters(), lr = 5*10**(-3))
    for epoch in range(1, 129):
        train_loss = 0.
        test_loss = 0.

        train_acc = 0.
        test_acc = 0.

        for train_x, train_y in train_loader:
            train_x, train_y = train_x.to(device), train_y.to(device)
            batch_train_loss = train(model, train_x, train_y, optimizer)
            _, batch_train_acc = test(model, train_x, train_y)

            train_loss += batch_train_loss
            train_acc += batch_train_acc

        for test_x, test_y in test_loader:
            test_x, test_y = test_x.to(device), test_y.to(device)
            batch_test_loss, batch_test_acc = test(model, test_x, test_y)

            test_loss += batch_test_loss
            test_acc += batch_test_acc

        train_losses.append(train_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))
        
        train_accs.append(train_acc/len(train_loader))
        test_accs.append(test_acc/len(test_loader))

        print(f"Epoch: {epoch:02d},",
                f"Training loss: {train_losses[-1]:.4f},",
                f"Training accuracy: {100*train_accs[-1]:.2f}%,",
                f"Test loss: {test_losses[-1]:.4f},",
                f"Test accuracy: {100*test_accs[-1]:.2f}%")

    return model, train_losses, test_losses, train_accs, test_accs


def main(device, beta):
    if device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("Cuda device not available. Moving to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    model, train_losses, test_losses, train_accs, test_accs = train_MNIST_CNN(device, beta)

    # Create results folder
    results_path = osp.join(parent, "results")
    if not osp.exists(results_path):
        os.mkdir(results_path)

    # Save model
    torch.save(model.state_dict(), osp.join(results_path, "model.pt"))

    # Save losses
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(["Train", "Test"])
    plt.xlabel("Loss")
    plt.ylabel("Epoch")
    plt.savefig(osp.join(results_path, "loss.png"), format=png)
    plt.show()

    # Save accuracies
    plt.plot(train_accs)
    plt.plot(test_accs)
    plt.legend(["Train", "Test"])
    plt.xlabel("Accuracy")
    plt.ylabel("Epoch")
    plt.savefig(osp.join(results_path, "accuracy.png"), format=png)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Estimating MI in MNIST CNN")
    parser.add_argument("--device", type=str, default="cuda", help="Device", 
                        choices=["cuda", "cpu"])
    parser.add_argument("--beta", type=float, default=0.05, 
                        help="Standard deviation of AWGN")
    args = parser.parse_args()
    main(args.device, args.beta)