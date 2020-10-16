
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    """
        MNIST CNN as described in section B3 of https://arxiv.org/pdf/1810.05728.pdf
    """
    def __init__(self, mathcal_Z1, mathcal_Z2, mathcal_Z3, device):
        super(MNIST_CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm1d(128)

        # Pooling layers
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Zero mean AWGN
        self.mathcal_Z1 = mathcal_Z1
        self.mathcal_Z2 = mathcal_Z2
        self.mathcal_Z3 = mathcal_Z3

        self.dropout = 0.2
        self.device = device


    def initialize_parameters(self):
        # conv1
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1.bias.data.fill_(0)

        # conv2
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

        # fc1
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0)

        # fc2
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        self.bn3.reset_parameters()

        self.pool1.reset_parameters()
        self.pool2.reset_parameters()

        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


    def forward(self, img, noisy=False):
        #print(f"img: {img.size()}")
        conv_1 = self.conv1(img)
        #print(f"conv_1: {conv_1.size()}")
        bn_1 = self.bn1(conv_1)
        #print(f"bn_1: {bn_1.size()}")
        A_1 = torch.tanh(bn_1)
        #print(f"A_1: {A_1.size()}")

        if noisy:
            n_1 = torch.zeros(A_1.size(0))
            mathcal_Z_1 = self.mathcal_Z1.sample(n_1.size()).to(self.device)
            T_1 = A_1 + mathcal_Z_1 if self.training else A_1
        else:
            T_1 = F.dropout(A_1, p=0.2, training=self.training)
        #print(f"T_1: {T_1.size()}")

        LAYER_1 = self.pool1(T_1)
        #print(f"LAYER_1: {LAYER_1.size()}")
        
        conv_2 = self.conv2(LAYER_1)
        #print(f"conv_2: {conv_2.size()}")
        bn_2 = self.bn2(conv_2)
        #print(f"bn_2: {bn_2.size()}")
        A_2 = torch.tanh(bn_2)
        #print(f"A_2: {A_2.size()}")

        if noisy:
            n_2 = torch.zeros(A_2.size(0))
            mathcal_Z_2 = self.mathcal_Z2.sample(n_2.size()).to(self.device)
            T_2 = A_2 + mathcal_Z_2 if self.training else A_2
        else:
            T_2 = F.dropout(A_2, p=0.2, training=self.training)
        #print(f"T_2: {T_2.size()}")

        LAYER_2 = self.pool2(T_2)
        #print(f"LAYER_2: {LAYER_2.size()}")
        
        LAYER_2 = LAYER_2.view(img.size(0),-1)
        #print(f"LAYER_2: {LAYER_2.size()}")

        fc_1 = self.fc1(LAYER_2)
        #print(f"fc_1: {fc_1.size()}")
        bn_3 = self.bn3(fc_1)
        #print(f"bn_3: {bn_3.size()}")
        LAYER_3 = torch.tanh(bn_3)
        #print(f"LAYER_3: {LAYER_3.size()}")

        if noisy:
            n_3 = torch.zeros(LAYER_3.size(0))
            mathcal_Z_3 = self.mathcal_Z3.sample(n_3.size()).to(self.device)
            T_3 = LAYER_3 + mathcal_Z_3 if self.training else LAYER_3
        else:
            T_3 = F.dropout(LAYER_3, p=0.2, training=self.training)
        #print(f"T_3: {T_3.size()}")

        LAYER_4 = self.fc2(T_3)
        #print(f"LAYER_4: {LAYER_4.size()}")

        return LAYER_4, [LAYER_1, LAYER_2, LAYER_3, LAYER_4]