
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

import numpy as np


class KDE(object):
    """
        Class of Kernel Density Estimation models. Modified version of
        https://github.com/lukasruff/Deep-SAD-PyTorch/blob/master/src/baselines/kde.py

            Attributes:
                kernel : str
                    Distibution of the kernel
                n_jobs : int
                    Number of jobs
                seed : int
                    Seed for reproducibility
                model : sklearn.neighbors.KernelDensity
                    Kernel density model
                bandwidth : float
                    Bandwidth for the kernel
    """

    def __init__(self, kernel='gaussian', n_jobs=-1, seed=None, **kwargs):
        """
            Initialize KDE instance

                Parameters:
                    kernel : str
                        Distribution of the kernel
                    n_jobs : int
                        Number of jobs
                    seed : int
                        Seed for reproducibility
        """
        self.kernel = kernel
        self.n_jobs = n_jobs
        self.seed = seed

        self.model = KernelDensity(kernel=kernel, **kwargs)
        self.bandwidth = None


    def train(self, dataset, device, batch_size=128): #n_jobs_dataloader,
        """
            Finds best bandwidth and fits KDE model on dataset

                Parameters:
                    dataset : torch.tensor of shape (num_examples, num_features)
                        Training dataset
                    device : torch.device
                        Torch device
                    batch_size : int
                        Batch size
        """

        # do not drop last batch for non-SGD optimization shallow_ssad
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                                    shuffle=True, drop_last=False)

        # Get data from loader
        X = ()
        for data in train_loader:
            inputs = data.to(device)
            X_batch = inputs.view(inputs.size(0), -1)
            # X_batch.shape = (batch_size, n_channels * height * width)
            X += (X_batch.cpu().data.numpy(),)
        X = np.concatenate(X)

        # Training
        if self.bandwidth is None:
            print("\nSearching over bandwidths...", end="")
            # use grid search cross-validation to select bandwidth
            params = {'bandwidth': np.logspace(0.5, 5, num=10, base=2)}
            hyper_kde = GridSearchCV(KernelDensity(kernel=self.kernel), params, 
                                     n_jobs=self.n_jobs, cv=5, verbose=0)
            print("Done!")

            print("Fitting data...", end="")
            hyper_kde.fit(X)
            self.bandwidth = hyper_kde.best_estimator_.bandwidth
            self.model = hyper_kde.best_estimator_
            print("Done!")
        else:
            self.model = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
            self.model.fit(X)


    def log_likelihood(self, samples, device):
        """
            Returns the log likelihood of samples

                Parameters:
                    samples : torch.tensor of shape (num_samples, num_features)
                        Samples
                    device : torch.device
                        Torch device

                Returns:
                    log likelihood : float
                        log likelihood of samples
        """
        log_g = self.model.score_samples(samples.cpu().data.numpy())
        log_g = log_g.reshape(log_g.shape[0],1)
        log_g = torch.from_numpy(log_g).to(device)
        return log_g