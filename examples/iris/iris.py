
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.distributions import multivariate_normal as mvn
from torch.autograd import Variable

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from bokeh.palettes import Category20_20, Category20b_20, Accent8

import sys
import os
import os.path as osp
import argparse

get_folder = lambda filename: osp.abspath(osp.dirname(filename))

here = get_folder(__file__) # iris
parent = get_folder(here) # examples
grandparent = get_folder(parent) # estimateMI

sys.path.insert(0, grandparent)
sys.path.insert(1, parent)
sys.path.insert(2, here)

from model import Net, train, test
from estimateMI.get_samples import get_samples
from estimateMI.KDE import KDE
from estimateMI.I_utils import compute_I


def create_folder(path):
    if not osp.exists(path):
        os.mkdir(path)


def get_iris():
    # load IRIS dataset
    dataset = pd.read_csv(osp.join(here, 'data', 'iris.csv'))

    # transform species to numerics
    dataset.loc[dataset.species=='Iris-setosa', 'species'] = 0
    dataset.loc[dataset.species=='Iris-versicolor', 'species'] = 1
    dataset.loc[dataset.species=='Iris-virginica', 'species'] = 2

    train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values,
                                                    dataset.species.values, test_size=0.2)

    # wrap up with Variable in pytorch
    train_X = Variable(torch.FloatTensor(train_X))
    test_X = Variable(torch.FloatTensor(test_X))

    train_y = Variable(torch.LongTensor(train_y.tolist()))
    test_y = Variable(torch.LongTensor(test_y.tolist()))

    print(f"{train_X.size(0)} training examples, {test_X.size(0)} test examples")

    return train_X, test_X, train_y, test_y


def iris(device, num_hid, lr, num_epochs, beta, n_MC, n_u, n_c, n_jobs):
    # Create "data" and "results" folder
    data_folder = osp.join(here, "data")
    results_folder = osp.join(here, "results")

    create_folder(data_folder)
    create_folder(results_folder)

    # Get iris dataset
    train_X, test_X, train_y, test_y = get_iris()
    train_X = train_X.to(device)
    test_X = test_X.to(device)
    train_y = train_y.to(device)
    test_y = test_y.to(device)

    # AWGN channel
    Z = lambda n, sigma: mvn.MultivariateNormal(torch.zeros(n), 
                                        torch.diag(torch.ones(n) * sigma**2))
    Z_ = Z(num_hid, beta)

    # Instantiate the model, optimizer and criterion
    noisy_model = Net(Z_, device).to(device)
    optimizer = Adam(noisy_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train model
    losses, train_accs, test_accs, I_1, I_2 = [], [], [], [], []
    for epoch in range(1, 1+num_epochs):
        loss = train(noisy_model, optimizer, criterion, train_X, train_y)
        train_acc = test(noisy_model, train_X, train_y.cpu())
        test_acc = test(noisy_model, test_X, test_y.cpu())
        u_S, c_S = get_samples(noisy_model, train_X)
        noisy_model.eval()
        z, _, _ = noisy_model(train_X)
            
        kde = KDE(n_jobs=n_jobs)
        I_T_1 = compute_I(u_S[0], c_S[0], Z_, n_MC, kde, device, n_u, n_c)
        I_T_2 = compute_I(u_S[1], c_S[1], Z_, n_MC, kde, device, n_u, n_c)

        print(f"Epoch: {epoch:02d}",
                f"Loss: {loss:.4f}",
                f"Train: {100 * train_acc:.2f}%",
                f"Test: {100 * test_acc:.2f}%",
                f"I(X,T_1]) = {I_T_1:.4f}",
                f"I(X,T_2) = {I_T_2:.4f}")

        losses.append(loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        I_1.append(I_T_1)
        I_2.append(I_T_2)

        # Create a folder for every epoch to store checkpoint and t-SNE embeddings
        chkpt = {
            "epoch" : epoch,
            "state_dict" : noisy_model.state_dict(),
            "loss" : loss,
            "train accuracy" : 100 * train_acc,
            "test accuracy" : 100 * test_acc,
            "I(X:T_1)" : I_T_1,
            "I(X:T_2)" : I_T_2
        }
        epoch_folder = osp.join(results_folder, f"{epoch}")

        create_folder(epoch_folder)
        torch.save(chkpt, osp.join(epoch_folder, f"{epoch}.pt"))

        x_coord, y_coord = zip(*TSNE().fit_transform(z.cpu().numpy()))
        colormap = np.array(Category20_20 + Category20b_20 + Accent8)
        labels = np.array([int(l) for l in train_y])
        plt.scatter(x_coord, y_coord, c=colormap[labels])
        plt.savefig(osp.join(epoch_folder, f"{epoch}.png"))
        plt.close()

    return losses, train_accs, val_accs, test_accs, I_1, I_2


def main(args):
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("CUDA not available. Switching to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # Get the training loss and the accuracies
    losses, train_accs, val_accs, test_accs, I_1, I_2 = iris(device, 
                                                            args.num_hid, 
                                                            args.lr, 
                                                            args.num_epochs, 
                                                            args.beta,
                                                            args.n_MC,
                                                            args.n_u,
                                                            args.n_c,
                                                            args.n_jobs)

    # Plot the training loss
    plt.plot(losses)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    loss_path = osp.join(here, "results", args.method, "loss.png")
    plt.savefig(loss_path, format="png")
    plt.close()
    print(f"{loss_path} created!")

    # Plot the accuracies
    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.plot(test_accs)
    plt.legend(["Train", "Val", "Test"])
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    accuracy_path = osp.join(here, "results", args.method, "accuracy.png")
    plt.savefig(accuracy_path, format="png")
    plt.close()
    print(f"{accuracy_path} created!")

    # Plot the MI estimates
    plt.plot(I_1)
    plt.plot(I_2)
    plt.legend(["I(X:T_1)", "I(X:T_2)"])
    plt.ylabel("Mutual information estimate")
    plt.xlabel("Epoch")
    MI_path = osp.join(here, "results", args.method, "MI.png")
    plt.savefig(MI_path, format="png")
    plt.close()
    print(f"{MI_path} created!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Iris MI estimation")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--num_hid", type=int, default=100,
                        help="Number of neurons in the hidden layer")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument("--beta", type=float, default=0.05,
                        help="Standard deviation for the AWGN channel")
    parser.add_argument("--n_MC", type=int, default=100,
                        help="Number of Monte Carlo simulations")
    parser.add_argument("--n_u", type=int, default=120,
                        help="Number of unconditional samples for fitting KDE")
    parser.add_argument("--n_c", type=int, default=120,
                        help="Number of conditional samples for fitting KDE")
    parser.add_argument("--n_jobs", type=int, default=50,
                        help="Number of jobs for parallel processing of KDE")
    args = parser.parse_args()

    main(args)