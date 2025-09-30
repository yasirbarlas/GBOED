import math
import warnings

import torch

import pyro
from pyro import poutine
from pyro.contrib.oed.search import Search
from pyro.contrib.util import lexpand
from pyro.infer import SVI, EmpiricalMarginal, Importance
from pyro.infer.autoguide.utils import mean_field_entropy
from pyro.util import warn_if_nan, warn_if_inf

def gibbs_nmc_eig(
    model,
    design,
    loss_fn,
    observation_labels,
    target_labels=None,
    N=100,
    M=10,
    w=1.0,
):
    """
    Gibbs NMC for computing the expected information gain (EIG) of a design. No current support for
    random effects in the model.
    Args:
        model: A Pyro model that takes a design as input and returns a sample of the data.
        design: A tensor representing the design points.
        loss_fn: A function that computes the loss given the design, data, and theta.
        observation_labels: Labels for the observed variables in the model.
        target_labels: Labels for the target variables in the model.
        N: Number of samples to draw from the model.
        M: Number of samples to draw from the model for the conditional probability.
        w: Weighting factor for the loss function.
    Returns:
        The expected information gain (EIG) of the design.
    """
    if isinstance(observation_labels, str):  # list of strings instead of strings
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    #if isinstance(w, float):
    #    w = [w]

    # Take N samples of the model
    expanded_design = lexpand(design, N)  # N copies of the model
    # Gather trace from the model, containing samples of theta and data y, for all designs separately as a tensor
    trace = poutine.trace(model).get_trace(expanded_design)

    # Sample y and theta from the model
    #y_value = sum(trace.nodes[l]["value"] for l in observation_labels)
    #theta_value = sum(trace.nodes[l]["value"] for l in target_labels)
    y_value = {l: trace.nodes[l]["value"] for l in observation_labels}
    theta_value = {l: trace.nodes[l]["value"] for l in target_labels}
    trace.compute_log_prob()
    # Compute log probability of the observed data
    log_prob = sum(trace.nodes[l]["log_prob"] for l in observation_labels)
    
    # Compute the loss for each sample
    #losses = torch.stack([-wi * loss_fn(expanded_design, y_value, theta_value) for wi in w]).mean(dim=0)
    losses = -w * loss_fn(expanded_design, y_value, theta_value)

    # Create importance weights, normalising numerator and denominator
    logsumexp_losses = torch.logsumexp(losses, dim=0)
    numer = torch.exp(losses - logsumexp_losses)

    sum_y_value = torch.sum(torch.exp(log_prob), dim=0)
    denom = (torch.exp(log_prob) / sum_y_value).squeeze()

    # Compute the importance weights and normalise overall fraction after normalising numerator and denominator
    frac = (numer / denom)
    fractionsum = torch.sum(frac, dim=0)
    fraction = frac / fractionsum

    #print(fraction.shape, fractionsum.shape, numer.shape, denom.shape)

    y_dict = {l: lexpand(trace.nodes[l]["value"], M) for l in observation_labels}
    # Resample M values of theta and compute conditional probabilities
    conditional_model = pyro.condition(model, data=y_dict)
    # Using (M, 1) instead of (M, N) - acceptable to re-use thetas between ys because
    # theta comes before y in graphical model
    reexpanded_design = lexpand(design, M, 1)  # sample M theta
    retrace = poutine.trace(conditional_model).get_trace(reexpanded_design)
    
    # Sample y and theta from the model
    #y_value_ret = sum(retrace.nodes[l]["value"] for l in observation_labels)
    #theta_value_ret = sum(retrace.nodes[l]["value"] for l in target_labels)

    y_value_ret = {l: retrace.nodes[l]["value"] for l in observation_labels}
    theta_value_ret = {l: retrace.nodes[l]["value"] for l in target_labels}

    # Compute the loss for each sample
    #losses_ret = torch.stack([-wi * loss_fn(reexpanded_design, y_value_ret, theta_value_ret) for wi in w]).mean(dim=0)
    losses_ret = -w * loss_fn(reexpanded_design, y_value_ret, theta_value_ret)

    # Compute second term of Gibbs EIG, approximating the Gibbs marginal likelihood of y
    marginal_lp = torch.logsumexp(losses_ret, dim=0) - math.log(M)

    # Compute the final Gibbs EIG
    final = (losses - marginal_lp) * fraction

    # Warning if any of the final values are NaN or Inf
    warn_if_nan(final, "Gibbs EIG contains NaN values")
    warn_if_inf(final, "Gibbs EIG contains Inf values")

    return final.sum(0)#, (losses * fraction).sum(0)

def nmc_eig(
    model,
    design,
    observation_labels,
    target_labels=None,
    N=100,
    M=10,
    M_prime=None,
    independent_priors=False,
):
    """Nested Monte Carlo estimate of the expected information
    gain (EIG). The estimate is, when there are not any random effects,

    .. math::

        \\frac{1}{N}\\sum_{n=1}^N \\log p(y_n | \\theta_n, d) -
        \\frac{1}{N}\\sum_{n=1}^N \\log \\left(\\frac{1}{M}\\sum_{m=1}^M p(y_n | \\theta_m, d)\\right)

    where :math:`\\theta_n, y_n \\sim p(\\theta, y | d)` and :math:`\\theta_m \\sim p(\\theta)`.
    The estimate in the presence of random effects is

    .. math::

        \\frac{1}{N}\\sum_{n=1}^N  \\log \\left(\\frac{1}{M'}\\sum_{m=1}^{M'}
        p(y_n | \\theta_n, \\widetilde{\\theta}_{nm}, d)\\right)-
        \\frac{1}{N}\\sum_{n=1}^N \\log \\left(\\frac{1}{M}\\sum_{m=1}^{M}
        p(y_n | \\theta_m, \\widetilde{\\theta}_{m}, d)\\right)

    where :math:`\\widetilde{\\theta}` are the random effects with
    :math:`\\widetilde{\\theta}_{nm} \\sim p(\\widetilde{\\theta}|\\theta=\\theta_n)` and
    :math:`\\theta_m,\\widetilde{\\theta}_m \\sim p(\\theta,\\widetilde{\\theta})`.
    The latter form is used when `M_prime != None`.

    :param function model: A pyro model accepting `design` as only argument.
    :param torch.Tensor design: Tensor representation of design
    :param list observation_labels: A subset of the sample sites
        present in `model`. These sites are regarded as future observations
        and other sites are regarded as latent variables over which a
        posterior is to be inferred.
    :param list target_labels: A subset of the sample sites over which the posterior
        entropy is to be measured.
    :param int N: Number of outer expectation samples.
    :param int M: Number of inner expectation samples for `p(y|d)`.
    :param int M_prime: Number of samples for `p(y | theta, d)` if required.
    :param bool independent_priors: Only used when `M_prime` is not `None`. Indicates whether the prior distributions
        for the target variables and the nuisance variables are independent. In this case, it is not necessary to
        sample the targets conditional on the nuisance variables.
    :return: EIG estimate, optionally includes full optimization history
    :rtype: torch.Tensor
    """

    if isinstance(observation_labels, str):  # list of strings instead of strings
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    # Take N samples of the model
    expanded_design = lexpand(design, N)  # N copies of the model
    # Gather trace from the model, containing samples of theta and data y, for all designs separately as a tensor
    trace = poutine.trace(model).get_trace(expanded_design)
    # Compute log-likelihoods
    trace.compute_log_prob()

    if M_prime is not None:
        y_dict = {
            l: lexpand(trace.nodes[l]["value"], M_prime) for l in observation_labels
        }
        theta_dict = {
            l: lexpand(trace.nodes[l]["value"], M_prime) for l in target_labels
        }
        theta_dict.update(y_dict)
        #print("M..y", lexpand(trace.nodes["y"]["value"], M_prime).shape)
        # Resample M values of u and compute conditional probabilities
        # WARNING: currently the use of condition does not actually sample
        # the conditional distribution!
        # We need to use some importance weighting
        conditional_model = pyro.condition(model, data=theta_dict)
        if independent_priors:
            reexpanded_design = lexpand(design, M_prime, 1)
        else:
            # Not acceptable to use (M_prime, 1) here - other variables may occur after
            # theta, so need to be sampled conditional upon it
            reexpanded_design = lexpand(design, M_prime, N)
            #print("reexpanded_design", reexpanded_design.shape)
        retrace = poutine.trace(conditional_model).get_trace(reexpanded_design)
        retrace.compute_log_prob()
        conditional_lp = sum(
            retrace.nodes[l]["log_prob"] for l in observation_labels
        ).logsumexp(0) - math.log(M_prime)
    else:
        # This assumes that y are independent conditional on theta
        # Furthermore assume that there are no other variables besides theta
        # Use computed log-probabilities from the model
        conditional_lp = sum(trace.nodes[l]["log_prob"] for l in observation_labels)

    y_dict = {l: lexpand(trace.nodes[l]["value"], M) for l in observation_labels}
    # Resample M values of theta and compute conditional probabilities
    conditional_model = pyro.condition(model, data=y_dict)
    # Using (M, 1) instead of (M, N) - acceptable to re-use thetas between ys because
    # theta comes before y in graphical model
    reexpanded_design = lexpand(design, M, 1)  # sample M theta
    retrace = poutine.trace(conditional_model).get_trace(reexpanded_design)
    retrace.compute_log_prob()
    marginal_lp = sum(
        retrace.nodes[l]["log_prob"] for l in observation_labels
    ).logsumexp(0) - math.log(M)

    terms = conditional_lp - marginal_lp
    nonnan = (~torch.isnan(terms)).sum(0).type_as(terms)
    terms[torch.isnan(terms)] = 0.0
    return terms.sum(0) / nonnan#, conditional_lp.sum(0) / nonnan