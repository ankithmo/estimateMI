
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import accuracy


def train(model, x, y, train_idx, optimizer, criterion, edge_index=None):
    """
        Training module for MLP

        Parameters:
            model : MLP
                MLP model
            x : torch.tensor of shape (num_examples, num_dims)
                Input tensor
            y : torch.tensor of shape (num_examples)
                Output tensor
            train_idx : torch.tensor of shape (num_examples)
                Boolean tensor which indicates which of the nodes are in the training set
            optimizer : torch.optim
                Optimization method
            criterion : torch.nn
                Cost function
            edge_index : torch.tensor with shape (2, num_edges), optional
                Edge index

        Returns:
            Cost : float
                Value computed by the cost function
    """
    model.train()

    optimizer.zero_grad()
    if edge_index is None:
        _, out, _ = model(x[train_idx])
        loss = criterion(out, y[train_idx])
    else:
        _, out, _ = model(x, edge_index)
        loss = criterion(out[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()
    

@torch.no_grad()
def test(model, x, y, train_idx, val_idx, test_idx, edge_index=None):
    """
        Test module for MLP

        Parameters:
            model : MLP
                MLP model
            x : torch.tensor of shape (num_examples, num_dims)
                Input tensor
            y : torch.tensor of shape (num_examples)
                Output tensor
            train_idx : torch.tensor of shape (num_examples)
                Boolean tensor which indicates which of the nodes are in the training set
            val_idx : torch.tensor of shape (num_examples)
                Boolean tensor which indicates which of the nodes are in the validation set
            test_idx : torch.tensor of shape (num_examples)
                Boolean tensor which indicates which of the nodes are in the test set
            edge_index : torch.tensor of shape (2, num_edges), optional
                Edge index 

        Returns:
            train_acc : float
                Accuracy of the training set
            val_acc : float
                Accuracy of the validation set
            test_acc : float
                Accuracy of the test set
    """
    model.eval()

    _, out, _ = model(x) if edge_index is None else model(x, edge_index)
    y_pred = out.argmax(dim=-1)

    train_acc = accuracy(y_pred[train_idx], y[train_idx])
    val_acc = accuracy(y_pred[val_idx], y[val_idx])
    test_acc = accuracy(y_pred[test_idx], y[test_idx])

    return train_acc, val_acc, test_acc