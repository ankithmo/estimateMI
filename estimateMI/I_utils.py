
import torch

from tqdm import tqdm

def fit_and_log_lik(S_n, kde, Z, device, n_g=None):
    """
        Fit KDE on n_g samples from S_n and compute the log likelihood on S_n+Z

            Parameters:
                S_n : torch.tensor
                    Samples
                Z : torch.distributions.multivariate_normal
                    Additive White Gaussian Noise $\mathcal{N}(0,\beta^{2}I)$
                device : torch.device
                    Torch device
                n_g : int, optional
                    Number of samples for fitting KDE

            Returns:
                log_likelihood : float
                    Log likelihood of S_n + Z
    """
    # Pick n_g samples from S_n (if required)
    S_n_g = S_n if n_g is None else S_n[torch.randperm(S_n.size(0)).tolist()[:n_g],:]
    assert S_n_g.size() == (n_g, S_n.size(1)), \
    ValueError(f"Expected tensor of shape ({n_g}, {S_n.size(1)}), got {S_n_g.size()} instead.")

    # Fit KDE on n_g samples
    kde.train(S_n_g, device)

    # Z for each sample for each MC
    n_zeros = torch.zeros(S_n.size(0))
    Z_j = Z.sample(n_zeros.size()).to(device)

    # Compute log-likelihood of T_n
    S_n_plus_Z_j = S_n + Z_j
    return kde.log_likelihood(S_n_plus_Z_j, device)


def estimate_h(S_n, Z, n_MC, kde, device, n_g, uncon):
    """
        Differential entropy estimator

            Parameters:
                S_n : torch.tensor
                    Samples
                Z : torch.distributions.multivariate_normal
                    Additive White Gaussian Noise $\mathcal{N}(0,\beta^{2}I)$
                n_MC : int
                    Number of Monte Carlo samples
                kde : KDE
                    Instance of KDE
                device : torch.device
                    Torch device
                n_g : int
                    Number of samples for fitting KDE
                uncon : bool
                    Whether we are performing unconditional sampling

            Returns:
                differential entropy estimate : float
                    Differential entropy estimate
    """
    log_g_list = []

    if uncon:
        with tqdm(total=n_MC) as pbar:
            for j in range(n_MC):
                log_g_list.append(fit_and_log_lik(S_n, kde, Z, device, n_g))
                pbar.update(1)
        pbar.close()
    else:
        for j in range(n_MC):
            log_g_list.append(fit_and_log_lik(S_n, kde, Z, device, n_g))

    log_g_list = torch.cat(log_g_list, axis=1)
    assert log_g_list.shape == (S_n.size(0), n_MC), \
    ValueError(f"Expected array of shape ({S_n.size(0)}, {n_MC}), got {log_g_list.shape} instead.")
  
    hat_Q = -torch.mean(log_g_list, axis=1)
    return torch.mean(hat_Q)


def compute_I(Us, Cs, Z, n_MC, kde, device, n_U, n_C):
    """
        Mutual information estimator

            Parameters:
                Us : torch.tensor of shape (num_examples, num_hidden)
                    Unconditional sampling
                Cs : torch.tensor of shape (num_examples, num_examples, num_hidden)
                    Conditional sampling
                Z : torch.normal.multivariate_normal
                    Additive White Gaussian Noise $\mathcal{N}(0,\beta^{2}I)$
                n_MC : int
                    Number of Monte Carlo samples
                kde : KDE
                    Instance of KDE
                device : torch.device
                    Torch device
                n_U : int
                    Number of unconditional samples to fit KDE on
                n_S : int
                    Number of conditional samples to fit KDE on

            Returns:
                hat_I : float
                    Mutual information estimate
    """
    # Estimate of unconditional entropy
    hat_h_MC_0 = estimate_h(Us, Z, n_MC, kde, device, n_U, uncon=True)
    print(f"hat_h_MC(S^n(0))={hat_h_MC_0}")

    # Estimate of conditional entropy
    hat_h_MC_x = []
    with tqdm(total=Cs.size(0)) as pbar:
        for m in range(Cs.size(0)):
            hat_h_MC_x.append(estimate_h(Cs[m,:,:], Z, n_MC, kde, device, n_C, uncon=False))
            pbar.set_description(f"hat_h_MC(S^n({m}))={hat_h_MC_x[-1]}")
            pbar.update(1)
        hat_h_MC_x = torch.tensor(hat_h_MC_x)
    pbar.close()

    # Estimate of mutual information
    hat_I = hat_h_MC_0.data - torch.mean(hat_h_MC_x)
    print(f"hat_I={hat_I}")

    return hat_I