import torch
from torch.distributions import transform_to
import argparse
import subprocess
import datetime
import pickle
import time
import os
from functools import partial
from contextlib import ExitStack
import logging
import numbers
import random
import copy

import pyro
import pyro.optim as optim
import pyro.distributions as dist
from pyro.contrib.util import iter_plates_to_shape, rexpand, rmv
from pyro.contrib.oed.eig import nmc_eig
import pyro.contrib.gp as gp

from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from pyro.optim import Adam, ClippedAdam
from pyro.util import warn_if_nan
import numpy as np
from pyro.infer.predictive import Predictive
from pyro.contrib.util import lexpand
from pyro import poutine
from pyro.infer.autoguide import AutoDiagonalNormal

from pyro.distributions.transforms import SplineAutoregressive, SplineCoupling, AffineAutoregressive, Permute
from pyro.nn import AutoRegressiveNN, DenseNN

from gibbs_eig import gibbs_nmc_eig
from loss_functions import score_matching_location

# Get device for PyTorch (GPU or CPU for training)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_device(device)
# print("Using device:", device)

# Set precision for PyTorch
torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=12)

def torch_isnan(x):
    """
    A convenient function to check if a Tensor contains any nan; also works with numbers
    """
    if isinstance(x, numbers.Number):
        return x != x
    return torch.isnan(x).any()

def torch_isinf(x):
    """
    A convenient function to check if a Tensor contains any +inf; also works with numbers
    """
    if isinstance(x, numbers.Number):
        return x == float('inf') or x == -float('inf')
    return (x == float('inf')).any() or (x == -float('inf')).any()

def is_bad(a):
    return torch_isnan(a) or torch_isinf(a)

def iqr(tensor, dim=None, keepdim=False):
    q75 = torch.quantile(tensor, 0.75, dim=dim, keepdim=keepdim, interpolation="midpoint")
    q25 = torch.quantile(tensor, 0.25, dim=dim, keepdim=keepdim, interpolation="midpoint")
    return q75 - q25

def get_c_exponential_decay(i, rate=0.3, q1=9, q2=1):
    return q1 * np.exp(-rate * i) + q2
    
def compute_predictive_distribution(model, xi, num_samples=1000):
    """
    Computes the predictive distribution for a given model and design points.

    Args:
        model: The Pyro model to sample from.
        xi: The design points (tensor).
        num_samples: Number of samples to draw from the predictive distribution.
        theta_samples: Samples from the posterior distribution of the model parameters.
    Returns:
        Samples from the predictive distribution (tensor).
    """
    expanded_design = lexpand(xi, num_samples)  # N copies of the model
    trace = poutine.trace(model).get_trace(expanded_design)
    return trace.nodes["y"]["value"]

    #predictive = Predictive(model, num_samples=num_samples)
    #return predictive(xi)["y"]

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

def compute_mean_log_likelihood(true_values, model, xi, num_theta_samples=100):
    """
    Vectorized computation of mean log-likelihood for the predictive distribution.

    Args:
        true_values: True values from the model (tensor of shape [N, D, 1]).
        model: The Pyro model to sample from.
        xi: The design points (tensor of shape [D, 1]).
        num_theta_samples: Number of samples to draw from the posterior distribution.
        noise_std: Standard deviation of the likelihood noise.

    Returns:
        Mean log-likelihood value (scalar).
    """
    M = num_theta_samples  # Number of theta samples

    y_dict = {"y": lexpand(true_values.unsqueeze(1), M)}
    #print(y_dict["y"].shape, "y_dict shape")
    # Resample M values of theta and compute conditional probabilities
    conditional_model = pyro.condition(model, data=y_dict)
    # Using (M, 1) instead of (M, N) - acceptable to re-use thetas between ys because
    # theta comes before y in graphical model
    reexpanded_design = lexpand(xi.unsqueeze(0), M, 1)  # sample M theta
    #print(xi.shape)
    #print(reexpanded_design.shape, "reexpanded_design shape")
    retrace = poutine.trace(conditional_model).get_trace(reexpanded_design)
    retrace.compute_log_prob()
    log_prob = (torch.logsumexp(retrace.nodes["y"]["log_prob"], dim=0) - np.log(M)).squeeze()
    mean_log_likelihood = torch.mean(torch.mean(log_prob, dim=0), dim=0)
    return mean_log_likelihood.item()

# TODO read from torch float spec
epsilon = torch.tensor(2**-22)

def make_location_model(empirical_prior, alpha, b, m, observation_sd, observation_label="y"):
    def location_model(design):
        if is_bad(design):
            raise ArithmeticError("bad design, contains nan or inf")
        batch_shape = design.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            #theta_shape = batch_shape + theta_mu.shape[-2:]
            #theta = pyro.sample("theta", dist.Normal(theta_mu.expand(theta_shape), theta_sig.expand(theta_shape)).to_event(2))
            theta = pyro.sample("theta", empirical_prior)
            #print("theta", theta.shape)
            #print("design", design.shape)
            distance = torch.square(theta - design).sum(dim=-1)
            #print("distance", distance.shape, distance)
            ratio = alpha / (m + distance)
            mu = b + ratio.sum(dim=-1, keepdims=True)
            #print("mu", mu.shape, mu)
            emission_dist = dist.Normal(torch.log(mu), observation_sd).to_event(1)
            y = pyro.sample(observation_label, emission_dist)
            return y

    return location_model

# Mean-Field Variational Inference (not good for location finding, use MCMC instead as already done)
def genelboguide(design, dim=1, k=2, d=2):
    theta_mu = pyro.param("theta_mu", torch.zeros(dim, 1, k, d))
    theta_sig = pyro.param("theta_sig", torch.ones(dim, 1, k, d), constraint=torch.distributions.constraints.positive)
    pyro.sample("theta", dist.Normal(theta_mu, theta_sig).to_event(2))

class EmpiricalPrior(dist.TorchDistribution):
    arg_constraints = {}
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(self, samples):
        self.samples = samples
        self.num_samples = samples.shape[0]
        batch_shape = torch.Size()
        event_shape = samples.shape[1:]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, sample_shape=torch.Size()):
        idx = torch.randint(0, self.num_samples, sample_shape, device=self.samples.device)
        return self.samples[idx]

    def log_prob(self, value):
        return torch.zeros(value.shape[:-len(self.event_shape)], device=value.device)

def main(num_steps, num_parallel, experiment_name, typs, inference, seed, lengthscale, variance, num_acquisition, obs_sd, w, N, M, chosen_loss, misspecification, actual_observation_sd, c_imq, b_use_expdecay, b_expdecay_imq, k, d, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: {}".format(loglevel))
    logging.basicConfig(level=numeric_level)

    output_dir = "./location/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not experiment_name:
        experiment_name = output_dir+f"seed{seed}:{datetime.datetime.now().isoformat()}"
    else:
        experiment_name = output_dir+experiment_name
    results_file = experiment_name + '.result_stream.pickle'
    try:
        os.remove(results_file)
    except OSError:
        logging.info("File {} does not exist yet".format(results_file))
    typs = typs.split(",")
    observation_sd = torch.tensor(obs_sd)

    for typ in typs:
        logging.info("Type {}".format(typ))
        pyro.clear_param_store()
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False

        #k, d = 2, 2
        num_bo_steps = 9
        design_dim = d

        alpha = 1
        b = 0.1
        m = 0.0001

        theta_mu  = torch.zeros(num_parallel, 1, k, d)
        theta_sig = torch.ones(num_parallel, 1, k, d)
        empirical_prior = dist.Normal(theta_mu, theta_sig).to_event(2)

        true_theta = empirical_prior.sample()
        print("true_theta", true_theta)

        # Define the true model based on the type of misspecification
        true_model = pyro.condition(make_location_model(empirical_prior, alpha, b, m, observation_sd), {"theta": true_theta})
        if misspecification == "none":
            data_real_world = true_model
        elif misspecification == "outlier":
            def data_real_world(design):
                normal_val = true_model(design)
                #print("normal_val", normal_val)
                outlier_mask = dist.Bernoulli(torch.tensor(0.3)).sample()
                #print("outlier_mask", outlier_mask)
                outlier = dist.Uniform(3 * observation_sd, 7 * observation_sd).sample()
                #print("outlier", outlier)
                normal_val += - (outlier_mask * outlier)
                #print("modified normal_val", normal_val)
                return normal_val
        elif misspecification == "error-dist":
            true_model = pyro.condition(make_location_model(empirical_prior, alpha, b, m, actual_observation_sd), {"theta": true_theta})
            data_real_world = true_model
        else:
            raise ValueError(f"Unknown misspecification type: {misspecification}. Choose from 'none', 'outlier', or 'error-dist'.")

        d_star_designs = torch.tensor([])
        ys = torch.tensor([])
        history = [(theta_mu.detach().clone().cpu().numpy(), theta_sig.detach().clone().cpu().numpy())]

        model = make_location_model(empirical_prior, alpha, b, m, observation_sd)

        rmses = []
        mmds = []
        log_likelihoods = []
        max_eigs = []

        # Loss function
        dsm = score_matching_location(w=w, obs_sd=observation_sd, b=b, m=m, a=alpha)

        # Use the desired loss function
        if chosen_loss == "score-matching-weighted":
            gen_loss_fn = dsm.dm_weighted
        elif chosen_loss == "score-matching-default":
            gen_loss_fn = dsm.dm_normal
        elif chosen_loss == "neg-log":
            gen_loss_fn = dsm.negative_log_likelihood
        else:
            raise ValueError(f"Unknown loss function: {chosen_loss}. Choose from 'score-matching-weighted', 'score-matching-default', or 'neg-log'.")

        # Generalised ELBO function for SVI, not used in the current implementation but kept for reference
        def generalised_elbo(model, guide, *args, num_particles=5, **kwargs):
            losses = []

            for _ in range(num_particles):
                guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
                guide_trace.compute_log_prob()

                theta = guide_trace.nodes["theta"]["value"]

                model_trace = pyro.poutine.trace(pyro.poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
                model_trace.compute_log_prob()

                x_data = args[0]
                y_data = model_trace.nodes["y"]["value"]

                predictive = pred_elbo_samples #predictive_samples_tensor[indices].squeeze(-1).permute(0, 2, 1, 3).squeeze(0)
                y_batch = y_data.unsqueeze(-1)

                loss_terms = gen_loss_fn(x_data, y_batch, theta, model, predictive=predictive, tau=tau, c_squared=c_squared)
                loss_sum = -w * loss_terms.sum()

                # Compute the log probabilities of the model and guide, using loss_sum to replace the log likelihood
                log_p_t = loss_sum + model_trace.log_prob_sum(site_filter=lambda name, site: site["type"] == "sample" and site["is_observed"] is False)  # remove obs
                log_q_t = guide_trace.log_prob_sum()

                # Compute the ELBO
                losses.append(-(log_p_t - log_q_t))
            
            loss = torch.stack(losses).mean()
            warn_if_nan(loss, "Generalised ELBO loss")
            return loss

        # Define the log potential function for MCMC sampling
        def make_log_potential(empirical_prior, x_data, y_data, predictive, tau, c_squared, model, gen_loss_fn, w):
            def log_potential(theta):
                loss_terms = gen_loss_fn(x_data, y_data.unsqueeze(-1), theta, model, predictive=predictive, tau=tau, c_squared=c_squared,)
                loss = loss_terms.sum()
                log_prior = dist.Normal(torch.zeros(num_parallel, 1, k, d), torch.ones(num_parallel, 1, k, d)).to_event(2).log_prob(theta)
                return -w * loss + log_prior

            return log_potential

        time_before_experiment = time.time()

        for experiment in range(num_steps):
            logging.info("Experiment {}".format(experiment + 1))
            results = {'typ': typ, 'step': experiment + 1, 'seed': seed, 'variance': variance,
                       'lengthscale': lengthscale, 'observation_sd': observation_sd, 'num_acquisition': num_acquisition}

            # Design phase
            t = time.time()

            tau = 1.0

            if c_imq == 0:
                c_squared = None
            else:
                c_squared = c_imq ** 2

            if b_use_expdecay == True:
                if b_expdecay_imq <= 0:
                    c_squared = None
                else:
                    c_squared = get_c_exponential_decay(experiment, rate=b_expdecay_imq, q1=1.8, q2=0.2) ** 2
            
            # Initialisation
            noise = torch.tensor(0.2).pow(2)
            # X = 100*rexpand(torch.rand((num_parallel, num_acq)), 4)
            X = -4 + 8 * torch.rand((num_parallel, num_acquisition, 1, design_dim))
            # Ensures designs are the same (assuming seed is the same) no matter the loss function used for computing metrics (fairer comparison of results)
            predictive_samples = compute_predictive_distribution(model, X, num_samples=1000).squeeze(-1).squeeze(-2).unsqueeze(-1)
            true_values = true_model(lexpand(X, predictive_samples.shape[0])).squeeze(-1).squeeze(-2).unsqueeze(-1)
            print(predictive_samples.shape, true_values.shape)

            rmse_predictive = compute_rmse_predictive_vs_true(predictive_samples, true_values)
            rmses.append(rmse_predictive)
            #print(f"RMSE between predictive distribution and true model: {rmse_predictive}")

            mmd_predictive = compute_mmd_vectorized_per_dim(predictive_samples, true_values, bandwidths=median_heuristic_bandwidth_per_dim(predictive_samples, true_values))
            mmds.append(mmd_predictive)
            #print(f"MMD between predictive distribution and true model: {mmd_predictive}")

            log_likelihood_predictive = compute_mean_log_likelihood(true_values, model, X.squeeze(0), num_theta_samples=100)
            log_likelihoods.append(log_likelihood_predictive)
            #print(f"Mean log-likelihood between predictive distribution and true model: {log_likelihood_predictive}")

            logging.info("RMSE {} \n MMD {} \n Log Likelihood {}".format(
                rmse_predictive, mmd_predictive, log_likelihood_predictive))
                
            results['rmse'], results['mmd'], results['log_likelihood'] = rmse_predictive, mmd_predictive, log_likelihood_predictive

            if typ in ['nmc', 'gibbs-nmc']:
                if typ == 'nmc':
                    def f(X):
                        return torch.cat([nmc_eig(model, X, ["y"],
                                                  ["theta"], N=N, M=M)
                                        ], dim=1)
                    
                elif typ == 'gibbs-nmc':
                    def f(X):
                        return torch.cat([gibbs_nmc_eig(model, X, 
                                                  lambda *args, **kwargs: gen_loss_fn(*args, model=model, predictive=predictive_samples, tau=tau, c_squared=c_squared, **kwargs), ["y"],
                                                  ["theta"], N=N, M=M, w=w).unsqueeze(0)
                                        ], dim=1)

                y = f(X)

                # Random search
                # # print(y.mean(1), y.max(1), y.min(1), y.std(1))
                # d_star_index = torch.argmax(y, dim=1)
                # # print(d_star_index.shape)
                # # print(d_star_index)
                # d_star_design = X[torch.arange(num_parallel), d_star_index, ...].unsqueeze(-2)

                # GPBO
                y = y.detach().clone()
                kernel = gp.kernels.Matern52(input_dim=1, lengthscale=torch.tensor(lengthscale),
                                             variance=torch.tensor(variance)) # y.var(unbiased=True) torch.tensor(variance)
                X = X.squeeze(-2).squeeze(0)
                constraint = torch.distributions.constraints.interval(-4., 4.)

                for i in range(num_bo_steps):
                    #print(f"bo step{i}")
                    Kff = kernel(X)
                    Kff += noise * torch.eye(Kff.shape[-1])
                    Lff = torch.linalg.cholesky(Kff) #Kff.cholesky(upper=False)
                    Xinit = -4 + 8 * torch.rand((num_parallel, num_acquisition, design_dim))
                    unconstrained_Xnew = transform_to(constraint).inv(Xinit).detach().clone().requires_grad_(True)
                    minimizer = torch.optim.LBFGS([unconstrained_Xnew], max_eval=20)

                    def gp_ucb1():
                        minimizer.zero_grad()
                        Xnew = transform_to(constraint)(unconstrained_Xnew).squeeze(0)
                        # Xnew.register_hook(lambda x: print('Xnew grad', x))
                        KXXnew = kernel(X, Xnew)
                        LiK = torch.triangular_solve(KXXnew, Lff, upper=False)[0]
                        Liy = torch.triangular_solve(y.unsqueeze(-1), Lff, upper=False)[0] #y.unsqueeze(-1).clamp(max=20.)
                        mean = rmv(LiK.transpose(-1, -2), Liy.squeeze(-1))
                        KXnewXnew = kernel(Xnew)
                        var = (KXnewXnew - LiK.transpose(-1, -2).matmul(LiK)).diagonal(dim1=-2, dim2=-1)
                        ucb = -(mean + 12*var.sqrt())
                        loss = ucb.sum()
                        torch.autograd.backward(unconstrained_Xnew,
                                                torch.autograd.grad(loss, unconstrained_Xnew, retain_graph=True))
                        return loss

                    minimizer.step(gp_ucb1)
                    X_acquire = transform_to(constraint)(unconstrained_Xnew).detach().clone()
                    predictive_samples = compute_predictive_distribution(model, X_acquire.unsqueeze(2), num_samples=1000).squeeze(-1).squeeze(-2).unsqueeze(-1)
                    # print('X_acquire', X_acquire)
                    y_acquire = f(X_acquire.unsqueeze(-2)).detach().clone()
                    # print('y_acquire', y_acquire)

                    X = torch.cat([X, X_acquire.squeeze(0)], dim=0)
                    #print('X after cat', X.shape)
                    y = torch.cat([y, y_acquire], dim=1)
                    #print('y after cat', y.shape)

                max_eig, d_star_index = torch.max(y, dim=1)
                logging.info('max EIG {}'.format(max_eig))
                results['max EIG'] = max_eig
                max_eigs.append(max_eig.item())
                #d_star_design = X[torch.arange(num_parallel), d_star_index, ...].unsqueeze(-2).unsqueeze(-2)
                d_star_design = X[d_star_index, ...].unsqueeze(-2).unsqueeze(-2)
                print('X_shape after BO', X.shape)

            elif typ == 'random':
                d_star_design = -4 + 8 * torch.rand((num_parallel, 1, 1, design_dim))

            #print(X, X.shape)
            #print(y, y.shape)
            elapsed = time.time() - t
            logging.info('elapsed design time {}'.format(elapsed))
            results['design_time'] = elapsed
            results['d_star_design'] = d_star_design
            logging.info('design {} {}'.format(d_star_design.squeeze(), d_star_design.shape))
            d_star_designs = torch.cat([d_star_designs, d_star_design], dim=-3)
            print('d_star_designs', d_star_designs.shape)
            y = data_real_world(d_star_design)
            ys = torch.cat([ys, y], dim=-1)
            logging.info('ys {} {}'.format(ys.squeeze(), ys.shape))
            results['y'] = y

            pred_elbo_samples = compute_predictive_distribution(model, d_star_designs, num_samples=1000).squeeze(-1).squeeze(-2).unsqueeze(-1)
            log_potential = make_log_potential(empirical_prior, d_star_designs, ys, pred_elbo_samples, tau, c_squared, model, gen_loss_fn, w)
            theta_init = dist.Normal(torch.zeros(num_parallel, 1, k, d), torch.ones(num_parallel, 1, k, d)).to_event(2).sample()

            nuts = NUTS(potential_fn=lambda params: -log_potential(params["theta"]))
            mcmc = MCMC(nuts, warmup_steps=10000, num_samples=100000, initial_params={"theta": theta_init})
            mcmc.run()

            # For SVI, not used in the current implementation but kept for reference
            # conditioned_model = pyro.condition(model, {"y": ys})
            # svi = SVI(conditioned_model,
            #         genelboguide,
            #         Adam({"lr": 0.005}),
            #         loss=generalised_elbo)
            # num_iters = 10000
            # for i in range(num_iters + 1):
            #     elbo = svi.step(d_star_designs)
            #     if i % 250 == 0:
            #         print(f'({i})', elbo)

            # theta_mu = pyro.param("theta_mu").detach().data.clone()
            # theta_sig = pyro.param("theta_sig").detach().data.clone()

            posterior_samples = mcmc.get_samples()["theta"]
            theta_mu = posterior_samples.mean(dim=0)
            theta_sig = posterior_samples.std(dim=0).clamp(min=1e-6)

            logging.info("theta_mu {} \n theta_sig {}".format(
                theta_mu.squeeze(), theta_sig.squeeze()))
            results['theta_mu'], results['theta_sig'] = theta_mu, theta_sig

            history.append((theta_mu.detach().clone().cpu().numpy(), theta_sig.detach().clone().cpu().numpy()))

            empirical_prior = EmpiricalPrior(posterior_samples.squeeze().squeeze())
            model = make_location_model(empirical_prior, alpha, b, m, observation_sd)
            #model = make_location_model(theta_mu, theta_sig, alpha, b, m, observation_sd)

            # Final loop should output final results
            if (experiment + 1) == num_steps:
                X = -4 + 8 * torch.rand((num_parallel, num_acquisition, 1, design_dim))

                # Ensures designs are the same (assuming seed is the same) no matter the loss function used for computing metrics (fairer comparison of results)
                predictive_samples = compute_predictive_distribution(model, X, num_samples=1000).squeeze(-1).squeeze(-2).unsqueeze(-1)
                true_values = true_model(lexpand(X, predictive_samples.shape[0])).squeeze(-1).squeeze(-2).unsqueeze(-1)

                rmse_predictive = compute_rmse_predictive_vs_true(predictive_samples, true_values)
                rmses.append(rmse_predictive)
                #print(f"RMSE between predictive distribution and true model: {rmse_predictive}")

                mmd_predictive = compute_mmd_vectorized_per_dim(predictive_samples, true_values, bandwidths=median_heuristic_bandwidth_per_dim(predictive_samples, true_values))
                mmds.append(mmd_predictive)
                #print(f"MMD between predictive distribution and true model: {mmd_predictive}")

                log_likelihood_predictive = compute_mean_log_likelihood(true_values, model, X.squeeze(0), num_theta_samples=100)
                log_likelihoods.append(log_likelihood_predictive)
                #print(f"Mean log-likelihood between predictive distribution and true model: {log_likelihood_predictive}")

                logging.info("RMSE {} \n MMD {} \n Log Likelihood {}".format(
                    rmse_predictive, mmd_predictive, log_likelihood_predictive))
                
                results['rmse'], results['mmd'], results['log_likelihood'] = rmse_predictive, mmd_predictive, log_likelihood_predictive

            final_elapsed = time.time() - time_before_experiment
            logging.info('elapsed total time {}'.format(final_elapsed))
            results['total_time'] = final_elapsed

            with open(results_file, 'ab') as f:
                pickle.dump(results, f)

        with open(f"_loc_find_results_{chosen_loss}_misspec_{misspecification}_N{N}_M{M}_assum_std{observation_sd}_w{round(w, 4)}_1.txt", "a") as file:
            file.write(f"{seed};{rmses};{mmds};{log_likelihoods};{max_eigs};{final_elapsed}\n")

        with open(f"_loc_find_results_{chosen_loss}_misspec_{misspecification}_N{N}_M{M}_assum_std{observation_sd}_w{round(w, 4)}_2.txt", "a") as file:
            #eigs_list = [e.tolist() if hasattr(e, "tolist") else list(e) for e in eigs]
            #losses_list = [l.tolist() if hasattr(l, "tolist") else list(l) for l in losses]
            history_list = [(h[0].tolist() if hasattr(h[0], "tolist") else list(h[0]),
                    h[1].tolist() if hasattr(h[1], "tolist") else list(h[1])) for h in history]
            file.write(f"{seed};{d_star_designs.tolist()};{ys.tolist()};{history_list}\n")

if __name__ == "__main__":
    # https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="Location Finding"
                                                 " iterated experiment design")
    parser.add_argument("--seed", type=int, default=50, help="Random seed for reproducibility")
    parser.add_argument("--T", type=int, default=10, help="Number of experiments to run")
    #parser.add_argument("--num-parallel", nargs="?", default=1, type=int)
    parser.add_argument("--name", nargs="?", default="nmc", type=str)
    parser.add_argument("--typs", nargs="?", default="gibbs-nmc", type=str)
    parser.add_argument("--inference", nargs="?", default="bayesian", type=str)
    parser.add_argument("--lengthscale", nargs="?", default=15.0, type=float)
    parser.add_argument("--variance", nargs="?", default=4.0, type=float)
    parser.add_argument("--loglevel", default="info", type=str)
    parser.add_argument("--num-acquisition", default=100, type=int)
    parser.add_argument("--observation-sd", default=0.5, type=float)
    parser.add_argument("--w", default=1.0, type=float, help="Learning rate for posterior")
    parser.add_argument("--N", type=int, default=10000, help="Number of samples for EIG estimation (N)")
    parser.add_argument("--M", type=int, default=100, help="Number of samples for EIG estimation (M)")
    parser.add_argument("--chosen-loss", default="neg-log", type=str,
                        help="Choose from 'score-matching-weighted' or 'score-matching-default'.")
    parser.add_argument("--misspecification", default="none", type=str,
                        help="Choose from 'none', 'outlier', or 'error-dist'.")
    parser.add_argument("--actual-observation-sd", default=0.5, type=float)
    parser.add_argument("--c-imq", default=0.0, type=float,
                        help="Input '0.0' for adaptive predictive variance (Laplante)." \
                        "Smaller values of c downweight observations faster.")
    parser.add_argument("--b-use-expdecay", type=str2bool, default=False, help="Whether to use exponential decay for c or not")
    parser.add_argument("--b-expdecay-imq", default=0.0, type=float,
                        help="Smaller values of b lead to slow decreases in c per experiment/timestep.")
    parser.add_argument("--k", default=2, type=int, help="Number of objects K in environment")
    parser.add_argument("--d", default=2, type=int, help="Number of dimensions D in environment")
    args = parser.parse_args()
    main(args.T, 1, args.name, args.typs, args.inference, args.seed, args.lengthscale, args.variance, args.num_acquisition,
         args.observation_sd, args.w, args.N, args.M, args.chosen_loss, args.misspecification, args.actual_observation_sd, args.c_imq, args.b_use_expdecay, args.b_expdecay_imq, args.k, args.d, args.loglevel)
