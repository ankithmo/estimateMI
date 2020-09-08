
import torch
from tqdm import tqdm

def fit_and_log_lik(S_n, n_g, Z, device):
    """
        Fit KDE on n_g samples from S_n and compute the log likelihood on S_n+Z

        Args:
            - S_n: Samples
            - n_g: Number of samples for fitting KDE
            - Z: Additive White Gaussian Noise $\mathcal{N}(0,\sigma^{2}I)$
            - device: Torch device

        Return:
            - Log likelihood of S_n + Z
    """
    # Pick n_g samples from S_n
    S_n_g = S_n[torch.randperm(S_n.size(0)).tolist()[:n_g],:]
    assert S_n_g.size() == (n_g, S_n.size(1))

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

        Args:
            - S_n: Samples
            - Z: Additive White Gaussian Noise $\mathcal{N}(0,\sigma^{2}I)$
            - n_MC (int): Number of Monte Carlo samples
            - kde: Instance of KDE
            - device: Torch device
            - n_g (int): Number of samples for fitting KDE
            - uncon (bool): Whether or not we are performing unconditional sampling

        Returns:
            - Differential entropy estimate
    """
    log_g_list = []

    if uncon:
        with tqdm(total=n_MC) as pbar:
            for j in range(n_MC):
                log_g_list.append(fit_and_log_lik(S_n, n_g, Z, device))
                pbar.update(1)
        pbar.close()
    else:
        for j in range(n_MC):
            log_g_list.append(fit_and_log_lik(S_n, n_g, Z, device))

    log_g_list = torch.cat(log_g_list, axis=1)
    assert log_g_list.shape == (S_n.size(0), n_MC)
  
    hat_Q = -torch.mean(log_g_list, axis=1)
    return torch.mean(hat_Q)


def compute_I(Us, Cs, Z, n_MC, kde, device, n_U, n_C):
    """
        Mutual information estimator

        Args:
            - Us: Unconditional sampling
            - Cs: Conditional sampling
            - Z: Additive White Gaussian Noise $\mathcal{N}(0,\sigma^{2}I)$
            - n_MC (int): Number of Monte Carlo samples
            - kde: Instance of KDE
            - device: Torch device
            - n_U: Number of unconditional samples to fit KDE on
            - n_S: Number of conditional samples to fit KDE on

        Returns:
            - Mutual information estimate
    """
  # Estimate of unconditional entropy
  hat_h_MC_0 = estimate_h(u_S, Z, n_MC, kde, device, n_u, uncon=True)
  print(f"hat_h_MC(S^n(0))={hat_h_MC_0}")

  # Estimate of conditional entropy
  hat_h_MC_x = []
  with tqdm(total=c_S.size(0)) as pbar:
    for m in range(c_S.size(0)):
      hat_h_MC_x.append(estimate_h(c_S[m,:,:], Z, n_MC, kde, device, n_c, uncon=False))
      pbar.set_description(f"hat_h_MC(S^n({m}))={hat_h_MC_x[-1]}")
      pbar.update(1)
    hat_h_MC_x = torch.tensor(hat_h_MC_x)
  pbar.close()

  # Estimate of mutual information
  hat_I = hat_h_MC_0.data - torch.mean(hat_h_MC_x)
  print(f"hat_I={hat_I}")

  return hat_I