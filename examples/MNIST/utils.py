
import torch
import torch.nn as nn
import torch.nn.functional as F


loss_fn = nn.CrossEntropyLoss()


def train(model, x, y, optimizer):
    model.train()

    optimizer.zero_grad()
    out, _ = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, y):
    model.eval()

    out, _ = model(x)
    loss = loss_fn(out, y)
    y_pred = out.argmax(dim=-1)

    return loss.item(), sum(y.eq(y_pred)).item() / y.size(0)