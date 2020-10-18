

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv

class SAGE(nn.Module):
    """
        SAGE model class

            Attributes:
                NN_1 : torch.nn.Linear
                    Input layer
                NN_2 : torch.nn.Linear
                    Hidden layer
                NN_3 : torch.nn.Linear
                    Output layer

                
    """
    def __init__(self, num_in, num_hid, num_out, mathcal_Z, dropout, device):
        """
            Initialization

                Parameters:
                    num_in : int
                        Number of neurons in the input layer
                    num_hid : int
                        Number of neurons in the hidden layer
                    num_out : int
                        Number of neurons in the output layer
                    mathcal_Z : torch.distributions.multivariate_normal
                        AWGN channel
                    dropout : float
                        Dropout probability
                    device : torch.device
                        Torch device

            Returns:

        """
        super(SAGE, self).__init__()

        # Linear layers
        self.NN_1 = SAGEConv(num_in, num_hid)
        self.NN_2 = SAGEConv(num_hid, num_hid)
        self.NN_3 = SAGEConv(num_hid, num_out)

        # Batch Normalization layers
        self.BN_1 = nn.BatchNorm1d(num_hid)
        self.BN_2 = nn.BatchNorm1d(num_hid)

        self.mathcal_Z = mathcal_Z
        self.dropout = dropout
        self.device = device

    def reset_parameters(self):
        """
            Reset the parameters
        """
        self.NN_1.reset_parameters()
        self.NN_2.reset_parameters()
        self.NN_3.reset_parameters()

        self.BN_1.reset_parameters()
        self.BN_2.reset_parameters()

    def forward(self, x, edge_index):
        """
            Forward module of MLP

                Parameters:
                    x : torch.tensor of shape (num_examples, num_dims)
                        Input tensor
                    edge_index : torch.tensor of shape (2, num_edges)
                        Input edge index
                    
                Returns:
                    y_pred : torch.tensor of shape (num_examples)
                        Predicted labels
                    [S_1, S_2]: list of length 2
                        Outputs of the hidden layers before passing through the AWGN channels

        """
        x = self.NN_1(x, edge_index)
        x = self.BN_1(x)
        S_1 = F.relu(x)

        if self.mathcal_Z is None:
            x = F.dropout(S_1, p=self.dropout, training=self.training)
        else:
            n_1 = torch.zeros(S_1.size(0))
            mathcal_z = self.mathcal_Z.sample(n_1.size()).to(self.device)
            T_1 = S_1 + mathcal_z if self.training else S_1
        
        x = self.NN_2(T_1, edge_index)
        x = self.BN_2(x)
        S_2 = F.relu(x)

        if self.mathcal_Z is None:
            x = F.dropout(S_2, p=self.dropout, training=self.training)
        else:
            n_2 = torch.zeros(S_2.size(0))
            mathcal_z = self.mathcal_Z.sample(n_2.size()).to(self.device)
            T_2 = S_2 + mathcal_z if self.training else S_2

        z = self.NN_3(T_2, edge_index)
        y_pred = torch.log_softmax(z, dim=-1)

        return z.detach(), y_pred, [S_1.detach(), S_2.detach()]