import torch
import os
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import time
import numpy as np
import random
from torch.distributions.constraints import positive
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.contrib.oed.eig import nmc_eig
from pyro.infer.predictive import Predictive
from pyro.util import warn_if_nan

from loss_functions import *
from gibbs_eig import gibbs_nmc_eig

import argparse
import pandas as pd

def iqr(tensor, dim=None, keepdim=False):
    q75 = torch.quantile(tensor, 0.75, dim=dim, keepdim=keepdim, interpolation="midpoint")
    q25 = torch.quantile(tensor, 0.25, dim=dim, keepdim=keepdim, interpolation="midpoint")
    return q75 - q25

# Get device for PyTorch (GPU or CPU for training)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_default_device(device)

# Set precision for PyTorch
torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=12)

# Model
def synthetic_data(xi, true_w=torch.tensor([2.0, 3.0]), noise_dist=dist.Normal(0, 0.5)):
    """Generate synthetic linear regression data."""
    y = sum(true_w[i] * xi**i for i in range(len(true_w)))
    y += noise_dist.sample(y.shape)
    return y

# Huber Model
def corrupted_synthetic_data(xi, true_w=torch.tensor([2.0, 3.0]), noise_dist=dist.Normal(0, 0.5), alpha=0.2):
    """Generate corrupted synthetic linear regression data."""
    # With alpha% chance, add a uniform outlier; otherwise, no outlier
    y = (synthetic_data(xi, true_w, noise_dist))
    outlier_mask = dist.Bernoulli(torch.tensor(alpha)).sample(y.shape)
    outlier = dist.Uniform(3 * noise_dist.stddev, 9 * noise_dist.stddev).sample(y.shape)
    y += - (outlier_mask * outlier)
    return y

def make_model_multivariate(mean, sd, noise_std=0.5):
    def model(xi):
        """Model function for Bayesian inference."""
        with pyro.plate_stack("plate", xi.shape[:-1]):
            beta = pyro.sample("beta", dist.MultivariateNormal(mean, sd))
            f_x = beta[..., 0].unsqueeze(-1) + beta[..., 1].unsqueeze(-1) * xi
            y = pyro.sample("y", dist.Normal(f_x, noise_std).to_event(1))
            return y
    return model

def get_c_exponential_decay(i, rate=0.3, q1=9, q2=1):
    return q1 * np.exp(-rate * i) + q2

def compute_predictive_distribution(model, xi, num_samples=1000):
    """
    Computes the predictive distribution for a given model and design points.

    Args:
        model: The Pyro model to sample from.
        xi: The design points (tensor).
        num_samples: Number of samples to draw from the predictive distribution.

    Returns:
        Samples from the predictive distribution (tensor).
    """
    predictive = Predictive(model, num_samples=num_samples)
    return predictive(xi)["y"]

def compute_rmse_predictive_vs_true(predictive_samples, true_values):
    """
    Computes the RMSE between the predictive distribution and the true model.

    Args:
        predictive_samples: Samples from the predictive distribution (tensor).
        true_values: True values from the model (tensor).

    Returns:
        RMSE value.
    """
    # Compute squared errors between predictive samples and true values
    squared_errors = torch.mean((predictive_samples - true_values) ** 2, dim=0)

    # Compute RMSE between predictive samples and true values
    rmse = torch.mean(torch.sqrt(squared_errors), dim=0)
    return rmse.item()

def median_heuristic_bandwidth_per_dim(x, y):
    """
    Compute median heuristic bandwidth per dimension (D,) from combined samples x and y.

    Args:
        x, y: Tensors of shape (N, D, 1)

    Returns:
        Tensor of shape (D,) with one bandwidth per dimension
    """
    x = x.squeeze(-1).to(torch.float32)  # (N, D)
    y = y.squeeze(-1).to(torch.float32)

    xy = torch.cat([x, y], dim=0)  # (2N, D)
    n = xy.shape[0]

    # Get all pairwise differences: (2N, 2N, D)
    diffs = xy.unsqueeze(0) - xy.unsqueeze(1)  # (2N, 2N, D)
    dists_sq = (diffs ** 2)  # (2N, 2N, D)

    # Extract upper triangle indices excluding diagonal
    i, j = torch.triu_indices(n, n, offset=1)
    upper_dists_sq = dists_sq[i, j, :]  # (num_pairs, D)

    # Median across pairs for each dimension → shape (D,)
    h_n = upper_dists_sq.median(dim=0).values
    sigma = torch.sqrt(h_n / 2)
    return sigma

def compute_mmd_vectorized_per_dim(x, y, bandwidths):
    """
    Compute MMD per dimension across 100 dimensions and average.

    Args:
        x, y: Tensors of shape (N, D, 1)
        bandwidths: Tensor of shape (D,) or float

    Returns:
        Scalar: average MMD over D dimensions
    """
    assert x.shape == y.shape and x.dim() == 3
    x = x.squeeze(-1).to(torch.float32)  # (N, D)
    y = y.squeeze(-1).to(torch.float32)

    N, D = x.shape

    x1 = x.unsqueeze(1)  # (N, 1, D)
    x2 = x.unsqueeze(0)  # (1, N, D)
    y1 = y.unsqueeze(1)
    y2 = y.unsqueeze(0)

    # Per-dimension bandwidths: shape (1, 1, D) for broadcasting
    bw = bandwidths.view(1, 1, D)

    k_xx = torch.exp(-((x1 - x2) ** 2) / (2 * bw ** 2))  # (N, N, D)
    k_yy = torch.exp(-((y1 - y2) ** 2) / (2 * bw ** 2))  # (N, N, D)
    k_xy = torch.exp(-((x1 - y2) ** 2) / (2 * bw ** 2))  # (N, N, D)

    # Unbiased MMD estimator (leave out diagonal terms for k_xx and k_yy)
    Nx = x.shape[0]
    Ny = y.shape[0]
    # k_xx: (Nx, Nx, D), k_yy: (Ny, Ny, D), k_xy: (Nx, Ny, D)

    # Remove diagonal for k_xx and k_yy
    if Nx > 1:
        mask_xx = ~torch.eye(Nx, dtype=torch.bool, device=x.device)
        k_xx_sum = k_xx[mask_xx].view(Nx, Nx - 1, D).sum(dim=(0, 1))
        mmd_xx = k_xx_sum / (Nx * (Nx - 1))
    else:
        mmd_xx = torch.zeros(D, device=x.device)

    if Ny > 1:
        mask_yy = ~torch.eye(Ny, dtype=torch.bool, device=y.device)
        k_yy_sum = k_yy[mask_yy].view(Ny, Ny - 1, D).sum(dim=(0, 1))
        mmd_yy = k_yy_sum / (Ny * (Ny - 1))
    else:
        mmd_yy = torch.zeros(D, device=y.device)

    k_xy_sum = k_xy.sum(dim=(0, 1))
    mmd_xy = 2 * k_xy_sum / (Nx * Ny) if Nx > 0 and Ny > 0 else torch.zeros(D, device=x.device)

    mmds = mmd_xx + mmd_yy - mmd_xy  # (D,)
    return torch.mean(mmds, dim=0).item()  # scalar

def compute_mean_log_likelihood(predictive_samples, true_values, model, xi, num_theta_samples=100, noise_std=2.0):
    """
    Vectorized computation of mean log-likelihood for the predictive distribution.

    Args:
        predictive_samples: Samples from the predictive distribution (tensor of shape [N, D, 1]).
        true_values: True values from the model (tensor of shape [N, D, 1]).
        model: The Pyro model to sample from.
        xi: The design points (tensor of shape [D, 1]).
        num_theta_samples: Number of samples to draw from the posterior distribution.
        noise_std: Standard deviation of the likelihood noise.

    Returns:
        Mean log-likelihood value (scalar).
    """
    D = xi.shape[0]  # Number of designs
    N = predictive_samples.shape[0]  # Number of y samples
    M = num_theta_samples  # Number of theta samples

    # Sample theta from the posterior for all designs at once
    posterior = Predictive(model, num_samples=M)
    theta_samples = posterior(xi)["beta"]  # shape: (M, D, 2)

    # Compute mean log-likelihood for each design, vectorized over samples
    # true_values: (N, D, 1) -> (N, D)
    y = true_values.squeeze(-1)  # (N, D)
    xi_flat = xi.squeeze(-1)     # (D,)

    # theta_samples: (M, D, 2)
    beta0 = theta_samples[..., 0]  # (M, D)
    beta1 = theta_samples[..., 1]  # (M, D)

    # Compute mean for each theta sample and design: (M, D)
    means = beta0 + beta1 * xi_flat  # (M, D)

    # For each design d, compute log-likelihood of all y[:, d] under all theta_samples[:, d]
    # Expand means to (M, D, N), y to (1, D, N)
    means_exp = means.unsqueeze(-1)  # (M, D, 1)
    y_exp = y.T.unsqueeze(0)         # (1, D, N)

    # Compute log_prob: (M, D, N)
    log_probs = dist.Normal(means_exp, noise_std).log_prob(y_exp)  # (M, D, N)

    # Average over theta samples (M) for each design and y: (D, N)
    log_likelihoods = torch.logsumexp(log_probs, dim=0) - np.log(M)  # (D, N)

    # Average over N samples and D designs
    mean_log_likelihood = torch.mean(torch.mean(log_likelihoods, dim=1), dim=0)  # (scalar)
    return mean_log_likelihood.item()

# Function to run the experiment for a given random seed
def run_experiment(random_s, noise_std=0.5, noise_dist=dist.Normal(0, 0.5), true_beta=torch.tensor([2.0, 3.0]), T=10, N=100, M=10, w=1.0, chosen_loss="score-matching-default", directory="./", file_for_table="./"):
    os.makedirs(directory, exist_ok=True)
    os.chdir(directory)

    prior_mean = torch.zeros(2)
    prior_var = torch.eye(2)
        
    def model(xi):
        """Model function for Bayesian inference."""
        with pyro.plate_stack("plate", xi.shape[:-1]):
            beta = pyro.sample("beta", dist.MultivariateNormal(prior_mean, prior_var))
            f_x = beta[..., 0].unsqueeze(-1) + beta[..., 1].unsqueeze(-1) * xi
            y = pyro.sample("y", dist.Normal(f_x, noise_std).to_event(1))
            return y

    def guide(xi):
        # The guide is initialised at the prior
        posterior_mean = pyro.param("posterior_mean", prior_mean.clone())
        posterior_var = pyro.param("posterior_var", prior_var.clone(), constraint=positive)
        pyro.sample("beta", dist.MultivariateNormal(posterior_mean, posterior_var))

    candidate_designs = torch.linspace(*(-4, 4), steps=100, dtype=torch.float).unsqueeze(-1)

    random.seed(random_s)
    np.random.seed(random_s)
    torch.manual_seed(random_s)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    ys = torch.tensor([])
    ls = torch.tensor([])
    history = [(prior_mean.detach().clone().cpu().numpy(), prior_var.detach().clone().cpu().numpy())]
    pyro.clear_param_store()
    current_model = make_model_multivariate(prior_mean, prior_var, noise_std=noise_std)

    #eigs = []
    #losses = []
    rmses = []
    mmds = []
    log_likelihoods = []

    print("Initial model statistics:")
    predictive_samples = compute_predictive_distribution(current_model, candidate_designs, num_samples=1000)
    true_values = synthetic_data(candidate_designs.expand(predictive_samples.shape[0], -1, -1), true_w=true_beta, noise_dist=noise_dist)
        
    rmse_predictive = compute_rmse_predictive_vs_true(predictive_samples, true_values)
    rmses.append(rmse_predictive)
    print(f"RMSE between predictive distribution and true model: {rmse_predictive}")

    mmd_predictive = compute_mmd_vectorized_per_dim(predictive_samples, true_values, bandwidths=median_heuristic_bandwidth_per_dim(predictive_samples, true_values))
    mmds.append(mmd_predictive)
    print(f"MMD between predictive distribution and true model: {mmd_predictive}")

    log_likelihood_predictive = compute_mean_log_likelihood(predictive_samples, true_values, current_model, candidate_designs, num_theta_samples=100, noise_std=noise_std)
    log_likelihoods.append(log_likelihood_predictive)
    print(f"Mean log-likelihood between predictive distribution and true model: {log_likelihood_predictive}")

    # Loss function
    dsm = score_matching_regression(w=w, mean_vec=prior_mean.unsqueeze(-1), covariance_mat=prior_var, std_y=noise_std)

    # Use the desired loss function
    if chosen_loss == "score-matching-weighted":
        gen_loss_fn = dsm.dm_weighted
    elif chosen_loss == "score-matching-default":
        gen_loss_fn = dsm.dm_normal
    elif chosen_loss == "neg-log":
        gen_loss_fn = dsm.negative_log_likelihood
    else:
        raise ValueError(f"Unknown loss function: {chosen_loss}. Choose from 'score-matching-weighted', 'score-matching-default', or 'neg-log'.")
    
    # Generalised ELBO function for SVI
    def generalised_elbo(model, guide, *args, **kwargs):
        """
        Computes the generalised ELBO as a single loss function.

        This function is used to compute the loss for the SVI step.
        It computes the ELBO by sampling from the guide and then computing the log probabilities of the model and guide.

        Args:
            model: The model function to compute the log probability of the data.
            guide: The guide function to compute the log probability of the latent variables.
            *args: Additional arguments to pass to the model and guide.
            **kwargs: Additional keyword arguments to pass to the model and guide.

        Returns:
            loss: The computed loss (negative ELBO) value.
        """

        # Run the guide to get q(theta)
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        guide_trace.compute_log_prob()

        # Sample theta from the guide
        theta = {name: site["value"] for name, site in guide_trace.nodes.items() if site["type"] == "sample"}["beta"]

        # Run the model with the same theta sample
        model_trace = pyro.poutine.trace(pyro.poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
        model_trace.compute_log_prob()

        # Extract data and experimental conditions
        x_data = args[0]
        y_data = {name: site["value"] for name, site in model_trace.nodes.items() if name == "y"}["y"]

        # Compute the loss term
        loss_sum = -w * torch.sum(torch.stack([gen_loss_fn(x, y, theta, predictive=predictive_samples_dict[float(x.cpu())], tau=tau, c_squared=c_squared) for x, y in zip(x_data, y_data)]), dim=0)
        
        # Compute the log probabilities of the model and guide, using loss_sum to replace the log likelihood
        log_p_t = loss_sum + model_trace.log_prob_sum(site_filter=lambda name, site: site["type"] == "sample" and site["is_observed"] is False)  # remove obs
        log_q_t = guide_trace.log_prob_sum()

        # Compute the ELBO
        elbo_particle = log_p_t - log_q_t

        # Loss is negative ELBO
        loss = -elbo_particle
        warn_if_nan(loss, "Generalised ELBO loss")
        return loss

    for experiment in range(T):
        print(f"Experiment {experiment + 1}")
        # Save timestamp
        start = time.time()

        tau = 1.0
        c_squared = None

        if chosen_loss == "neg-log":
            # Standard Bayesian EIG
            eig = nmc_eig(
                current_model,
                candidate_designs,
                ["y"],
                ["beta"],
                N=N, M=M
            )
        else:
            # Gibbs EIG
            eig = gibbs_nmc_eig(
                    current_model,
                    candidate_designs,       # design, or in this case, tensor of possible designs
                    lambda *args, **kwargs: gen_loss_fn(*args, predictive=predictive_samples, tau=tau, c_squared=c_squared, **kwargs), # loss function
                    ["y"],                     # site label of observations, could be a list
                    ["beta"],                 # site label of "targets" (latent variables), could also be list
                    N=N, M=M, w=w,
            )

        print(eig, eig.shape)

        best_l = torch.argmax(eig).float().detach()
        best_l = (int(best_l.item()))
        
        best_design = candidate_designs[best_l] # Not used since we already have a dataset

        print("Best design at xi =", round(best_design.detach().item(), 4))
        y = synthetic_data(best_design, true_w=true_beta, noise_dist=noise_dist) # Not used since we already have a dataset

        # Dataset from file provided before running run_experiment()
        ls = torch.tensor(x_dataaa)
        ys = torch.tensor(y_dataaa)
        
        print("Current history:")
        print(f"Designs: {ls} \nObservations: {ys}") # Not used since we already have a dataset

        # Create a dictionary mapping each design in ls to its predictive samples
        predictive_samples_dict = {}
        for l in ls:
            idx = (candidate_designs == l).all(dim=-1).nonzero(as_tuple=True)[0]
            predictive_samples_dict[l.item()] = predictive_samples[:, idx, :].squeeze(-1)  # (num_samples, 1)
            print(l.item(), torch.mean(predictive_samples_dict[l.item()], dim=0), torch.var(predictive_samples_dict[l.item()], dim=0), torch.quantile(predictive_samples_dict[l.item()], q=0.5, dim=0, interpolation="midpoint"), iqr(predictive_samples_dict[l.item()], dim=0) ** 2)

        conditioned_model = pyro.condition(model, {"y": ys})

        if chosen_loss == "neg-log":
            elbo_loss = Trace_ELBO()
        else:
            elbo_loss = generalised_elbo

        svi = SVI(conditioned_model,
                guide,
                Adam({"lr": 0.005}),
                loss=elbo_loss)
        num_iters = 10000
        for i in range(num_iters):
            svi.step(ls)

        posterior_mean = pyro.param("posterior_mean")
        posterior_var = pyro.param("posterior_var")

        print(f"Posterior mean: {posterior_mean.detach().clone().cpu().numpy()}")
        print(f"Posterior variance: {posterior_var.detach().clone().cpu().numpy()}")

        history.append((posterior_mean.detach().clone().cpu().numpy(), posterior_var.detach().clone().cpu().numpy()))
        current_model = make_model_multivariate(posterior_mean.detach().clone(), posterior_var.detach().clone(), noise_std=noise_std)

        predictive_samples = compute_predictive_distribution(current_model, candidate_designs, num_samples=1000)
        true_values = synthetic_data(candidate_designs.unsqueeze(0).expand(predictive_samples.shape[0], -1, -1), true_w=true_beta, noise_dist=noise_dist)
        
        rmse_predictive = compute_rmse_predictive_vs_true(predictive_samples, true_values)
        rmses.append(rmse_predictive)
        print(f"RMSE between predictive distribution and true model: {rmse_predictive}")

        mmd_predictive = compute_mmd_vectorized_per_dim(predictive_samples, true_values, bandwidths=median_heuristic_bandwidth_per_dim(predictive_samples, true_values))
        mmds.append(mmd_predictive)
        print(f"MMD between predictive distribution and true model: {mmd_predictive}")

        log_likelihood_predictive = compute_mean_log_likelihood(predictive_samples, true_values, current_model, candidate_designs, num_theta_samples=100, noise_std=noise_std)
        log_likelihoods.append(log_likelihood_predictive)
        print(f"Mean log-likelihood between predictive distribution and true model: {log_likelihood_predictive}")

        # Print time taken
        print("Time taken in seconds:", time.time() - start)

    print(f"Final history: {[(x, y) for x, y in history]}")

    # file_for_table contains the design history from whatever method was performed, which we use to compute the (Bayesian) EIG of all of these designs 
    # This allows us to compare between different methods in terms of the (Bayesian) EIG of the designs chosen, keeping everything in the same scale
    design_data_table, _ = print_history_for_seed(file_for_table, random_s)
    
    tabledesigns = torch.tensor(design_data_table).unsqueeze(-1)
    
    if chosen_loss == "neg-log":
        # Standard Bayesian EIG
        eigg = nmc_eig(
            current_model,
            tabledesigns,
            ["y"],
            ["beta"],
            N=N, M=M
        )
    else:
        # Gibbs EIG
        eigg = gibbs_nmc_eig(
                current_model,
                tabledesigns,       # design, or in this case, tensor of possible designs
                lambda *args, **kwargs: gen_loss_fn(*args, predictive=predictive_samples, tau=tau, c_squared=c_squared, **kwargs), # loss function
                ["y"],                     # site label of observations, could be a list
                ["beta"],                 # site label of "targets" (latent variables), could also be list
                N=N, M=M, w=w,
        )
    
    print(f"Final EIG at Seed {random_s}:", torch.sum(eigg, dim=0).cpu().numpy())

    # This is the important file with the (Bayesian) EIG results, in the last column
    with open(f"bayeseig_1.txt", "a") as file:
        file.write(f"{random_s};{rmses};{mmds};{log_likelihoods};{[eigg.tolist() if hasattr(eigg, 'tolist') else list(eigg)]}\n")

    with open(f"bayeseig_2.txt", "a") as file:
        eigs_list = [] #[e.tolist() if hasattr(e, "tolist") else list(e) for e in eigs]
        losses_list = [] #[l.tolist() if hasattr(l, "tolist") else list(l) for l in losses]
        history_list = [(h[0].tolist() if hasattr(h[0], "tolist") else list(h[0]),
                 h[1].tolist() if hasattr(h[1], "tolist") else list(h[1])) for h in history]
        file.write(f"{random_s};{eigs_list};{losses_list};{ls.tolist()};{ys.tolist()};{history_list}\n")

# Fixed dataset from performing BOED (one true model only, not collated over multiple models) - this is just to compute a posterior to then use in computing the EIG
file_for_bayesdata = "bayes_results_N10000_M100_std1.0_w1.0_2.txt"

# Function to read design and observation history from a CSV file for a given seed
def print_history_for_seed(file_path, seed):
    df = pd.read_csv(file_path, sep=";")
    row = df[df["Seed"] == seed]
    if row.empty:
        print(f"No entry found for seed {seed}")
        return
    design_history = eval(row.iloc[0]["Design History"])
    observation_history = eval(row.iloc[0]["Observation History"])
    return design_history, observation_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=50, help="Random seed for reproducibility")
    parser.add_argument("--N", type=int, default=10000, help="Number of samples for EIG estimation (N)")
    parser.add_argument("--M", type=int, default=100, help="Number of samples for EIG estimation (M)")
    parser.add_argument("--noise_std", type=float, default=1.2, help="Standard deviation of the noise")
    parser.add_argument("--w", type=float, default=1.0, help="Gibbs weighting parameter")
    # Keep the below as "neg-log" for standard Bayesian EIG, which is the point of this file
    parser.add_argument("--chosen_loss", type=str, default="neg-log", help="Chosen loss function, 'neg-log', 'score-matching-default', or 'score-matching-weighted'")
    parser.add_argument("--true_beta", type=float, nargs='+', default=[10.0, -7.0], help="True beta coefficients for the model")
    parser.add_argument("--T", type=int, default=1, help="Number of experiments to run")
    parser.add_argument("--directory", type=str, default="./", help="Directory to save results")
    parser.add_argument("--tableeig", type=str, default="./", help="Directory to save results")
    args = parser.parse_args()
    global x_dataaa, y_dataaa
    x_dataaa, y_dataaa = print_history_for_seed(file_for_bayesdata, args.seed)
    run_experiment(
            random_s=args.seed,
            noise_std=args.noise_std,
            noise_dist=dist.Normal(0, args.noise_std),
            true_beta=torch.tensor(args.true_beta),
            T=args.T,
            N=args.N,
            M=args.M,
            w=args.w,
            chosen_loss=args.chosen_loss,
            directory=args.directory,
            file_for_table=args.tableeig
        )
