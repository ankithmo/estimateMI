
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import multivariate_normal as mvn

from sklearn.manifold import TSNE

from torch_geometric.datasets import WikiCS

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from bokeh.palettes import Category20_20, Category20b_20, Accent8

import sys
import os
import os.path as osp
import argparse

get_folder = lambda filename: osp.abspath(osp.dirname(filename))

here = get_folder(__file__) # WikiCS
parent = get_folder(here) # examples
grandparent = get_folder(parent) # estimateMI

sys.path.insert(0, grandparent)
sys.path.insert(1, parent)
sys.path.insert(2, here)

from MLP import MLP
from SAGE import SAGE
from utils import train, test
from estimateMI.get_samples import get_samples
from estimateMI.KDE import KDE
from estimateMI.I_utils import compute_I


def create_folder(path):
    if not osp.exists(path):
        os.mkdir(path)


def wikics(method, device, num_hid, lr, dropout, num_epochs, mask, beta, n_MC, 
            n_u, n_c, n_jobs):
    # Create "data" and "results" folder
    data_folder = osp.join(here, "data")
    results_folder = osp.join(here, "results")
    MLP_folder = osp.join(results_folder, "MLP")
    SAGE_folder = osp.join(results_folder, "SAGE")

    create_folder(data_folder)
    create_folder(results_folder)
    create_folder(MLP_folder)
    create_folder(SAGE_folder)

    # Get WikiCS dataset
    dataset = WikiCS(root=data_folder)
    data = dataset[0]
    X = data.x.to(device)
    y = data.y.to(device)

    # Get splits
    train_idx = data.train_mask[:, mask]
    val_idx = data.val_mask[:, mask]
    test_idx = data.test_mask

    # Get nodes
    nodes = torch.from_numpy(np.arange(data.num_nodes))
    train_nodes = nodes[train_idx]
    val_nodes = nodes[val_idx]
    test_nodes = nodes[test_idx]

    # AWGN channel
    Z = lambda n, sigma: mvn.MultivariateNormal(torch.zeros(n), 
                                        torch.diag(torch.ones(n) * sigma**2))
    Z_ = Z(num_hid, beta)

    # Instantiate the model, optimizer and criterion
    if method == "MLP":
        noisy_model = MLP(X.size(-1), num_hid, dataset.num_classes,
                            Z_, dropout, device).to(device)
    elif method == "SAGE":
        noisy_model = SAGE(X.size(-1), num_hid, dataset.num_classes,
                            Z_, dropout, device).to(device)
    else:
        raise ValueError("Invalid method")
    optimizer = Adam(noisy_model.parameters(), lr=lr)
    criterion = F.nll_loss
    
    # Train model
    losses, train_accs, val_accs, test_accs, I_1, I_2 = [], [], [], [], [], []
    for epoch in range(1, 1+num_epochs):
        if method == "MLP":
            loss = train(noisy_model, X, y, train_idx, optimizer, criterion)
            train_acc, val_acc, test_acc = test(noisy_model, X, y, train_idx, 
                                                val_idx, test_idx)
            u_S, c_S = get_samples(noisy_model, X[train_idx])
            noisy_model.eval()
            z, _, _ = noisy_model(X[train_idx])
        elif method == "SAGE":
            loss = train(noisy_model, X, y, train_idx, optimizer, criterion,
                            edge_index)
            train_acc, val_acc, test_acc = test(noisy_model, X, y, train_idx, 
                                                val_idx, test_idx, edge_index)
            u_S, c_S = get_samples(noisy_model, X, edge_index, train_idx)
            noisy_model.eval()
            z, _, _ = noisy_model(X, edge_index)
            z = z[train_idx]
            
        kde = KDE(n_jobs=n_jobs)
        I_T_1 = compute_I(u_S[0], c_S[0], Z_, n_MC, kde, device, n_u, n_c)
        I_T_2 = compute_I(u_S[1], c_S[1], Z_, n_MC, kde, device, n_u, n_c)

        print(f"Epoch: {epoch:02d}",
                f"Loss: {loss:.4f}",
                f"Train: {100 * train_acc:.2f}%",
                f"Valid: {100 * val_acc:.2f}%",
                f"Test: {100 * test_acc:.2f}%",
                f"I(X,T_1]) = {I_T_1:.4f}",
                f"I(X,T_2) = {I_T_2:.4f}")

        losses.append(loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        I_1.append(I_T_1)
        I_2.append(I_T_2)

        # Create a folder for every epoch to store checkpoint and t-SNE embeddings
        chkpt = {
            "epoch" : epoch,
            "state_dict" : noisy_model.state_dict(),
            "loss" : loss,
            "train accuracy" : 100 * train_acc,
            "valid accuracy" : 100 * val_acc,
            "test accuracy" : 100 * test_acc,
            "I(X:T_1)" : I_T_1,
            "I(X:T_2)" : I_T_2
        }
        if method == "MLP":
            epoch_folder = osp.join(MLP_folder, f"{epoch}")
        elif method == "SAGE":
            epoch_folder = osp.join(SAGE_folder, f"{epoch}")

        create_folder(epoch_folder)
        torch.save(chkpt, osp.join(epoch_folder, f"{epoch}.pt"))

        x_coord, y_coord = zip(*TSNE().fit_transform(z.cpu().numpy()))
        colormap = np.array(Category20_20 + Category20b_20 + Accent8)
        labels = np.array([int(l) for l in data.y[train_idx]])
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
    losses, train_accs, val_accs, test_accs, I_1, I_2 = wikics(args.method, 
                                                                device, 
                                                                args.num_hid, 
                                                                args.lr, 
                                                                args.dropout, 
                                                                args.num_epochs, 
                                                                args.mask, 
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
    parser = argparse.ArgumentParser("WikiCS MI estimation")
    parser.add_argument("--method", type=str, default="MLP", 
                        choices=["MLP", "SAGE"],
                        help="Method")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--num_hid", type=int, default=35,
                        help="Number of neurons in the hidden layer")
    parser.add_argument("--lr", type=float, default=0.03,
                        help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.35,
                        help="Dropout probability")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--mask", type=int, default=0,
                        choices=list(range(20)), 
                        help="Mask to choose the training and the validation sets")
    parser.add_argument("--beta", type=float, default=0.05,
                        help="Standard deviation for the AWGN channel")
    parser.add_argument("--n_MC", type=int, default=100,
                        help="Number of Monte Carlo simulations")
    parser.add_argument("--n_u", type=int, default=580,
                        help="Number of unconditional samples for fitting KDE")
    parser.add_argument("--n_c", type=int, default=580,
                        help="Number of conditional samples for fitting KDE")
    parser.add_argument("--n_jobs", type=int, default=50,
                        help="Number of jobs for parallel processing of KDE")
    args = parser.parse_args()

    main(args)