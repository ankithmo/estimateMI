# estimateMI

This is a PyTorch implementation of the estimator proposed in the paper [*Estimating Differential Entropy under Gaussian Convolutions*](https://arxiv.org/abs/1810.11589). 
We provide an example to show the progressive geometric clustering proposed in the paper [*Estimating Information Flow in Deep Neural Networks*](https://arxiv.org/abs/1810.05728).

The same notations as used in the papers are followed in this package.

## How to run

#### 1. Dependencies
* [PyTorch](http://pytorch.org/)(1.6.0)
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)(1.6.1)

#### 2. Installation
`python setup.py install`

#### 3. Estimating MI
* `get_samples` is used to obtain the unconditional and the conditional i.i.d samples from the neural network.
* `estimate_h` estimates the differential entropy.
* `compute_I` estimates the mutual information.

The following code snippet shows how I(X:T_1) is estimated.
```
    u_S, c_S = get_samples(model, X)
    kde = KDE(n_jobs=n_jobs)
    I_T_1 = compute_I(u_S[0], c_S[0], Z_, n_MC, kde, device, n_u, n_c)
```

Please refer to `examples/WikiCS` for a detailed example of MI estimation for a neural network and a graph neural network.

#### 4. Contact
Please send me an email at ankithmo@usc.edu for any queries regarding this package.