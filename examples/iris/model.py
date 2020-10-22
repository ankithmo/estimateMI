
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

class Net(nn.Module):
    # define nn
    def __init__(self, mathcal_Z, device):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)

        self.mathcal_Z = mathcal_Z
        self.device = device

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

    def forward(self, x):
        x = self.fc1(x)
        S_1 = F.relu(x)

        if self.mathcal_Z is None:
            x = F.dropout(S_1, p=self.dropout, training=self.training)
        else:
            n_1 = torch.zeros(S_1.size(0))
            mathcal_z = self.mathcal_Z.sample(n_1.size()).to(self.device)
            T_1 = S_1 + mathcal_z if self.training else S_1

        x = self.fc2(x)
        S_2 = F.relu(x)

        if self.mathcal_Z is None:
            x = F.dropout(S_2, p=self.dropout, training=self.training)
        else:
            n_2 = torch.zeros(S_2.size(0))
            mathcal_z = self.mathcal_Z.sample(n_2.size()).to(self.device)
            T_2 = S_2 + mathcal_z if self.training else S_2

        z = self.fc3(x)
        y_pred = self.softmax(z)

        return z.detach(), y_pred, [S_1.detach(), S_2.detach()]


def train(model, optimizer, criterion, X, y):
    model.train()
    optimizer.zero_grad()
    _, out, _ = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, X, y):
    _, out, _ = model(X)
    _, y_pred = torch.max(out, 1)
    y_pred = y_pred.cpu()
    acc = accuracy_score(y.data, y_pred.data)
    return acc