# Generalised Bayesian Optimal Experimental Design (GBOED)

Notes: This repository and its contents support the paper titled ['Robust Experimental Design via Generalised Bayesian Inference'](https://arxiv.org/abs/2511.07671). We recommend citing this paper if you plan on using the contents of this repository in any capacity for your own work/research. Where relevant, code was adapted and used from the following repositories [DAD](https://github.com/ae-foster/dad), [iDAD](https://github.com/desi-ivanova/idad), [RL-BOED](https://github.com/csiro-mlai/RL-BOED), [RL-BOED](https://github.com/yasirbarlas/RL-BOED), and [DFB](https://github.com/takuomatsubara/Discrete-Fisher-Bayes).

### Requirements
- Python 3.9+ - we use Python 3.9.13
- [PyTorch (with CUDA for GPU usage)](https://pytorch.org/get-started/locally/) - we use PyTorch 2.7.0 without GPU usage (CPU only version)
- All other requirements listed in [**requirements.txt**](requirements.txt) - specific versions are listed

## Background

Bayesian optimal experimental design (BOED) is a principled framework for conducting experiments that leverages Bayesian inference to quantify how much information one can expect to gain from selecting a certain design. However, accurate Bayesian inference relies on the assumption that one's statistical model of the data-generating process is correctly specified. If this assumption is violated, Bayesian methods can lead to poor inference and estimates of information gain. Generalised Bayesian (or Gibbs) inference is a more robust probabilistic inference framework that replaces the likelihood in the Bayesian update by a suitable loss function. In this work, we present *Generalised Bayesian Optimal Experimental Design (GBOED)*, an extension of Gibbs inference to the experimental design setting which achieves robustness in both design and inference. Using an extended information-theoretic framework, we derive a new acquisition function, the *Gibbs expected information gain (Gibbs EIG)*. Our empirical results demonstrate that GBOED enhances robustness to outliers and incorrect assumptions about the outcome noise distribution.

## Code

We explore three experimental design problems, namely Bayesian linear regression ([**regression.py**](regression.py)), pharmacokinetics ([**pharmacokinetic.py**](pharmacokinetic.py)), and location finding ([**location.py**](location.py)). The code files for each experimental design problem, aside from Pyro and the other requirements listed in [**requirements.txt**](requirements.txt), rely on the files in [**utils**](/utils). [**utils**](/utils) contains the loss functions used, and a Pyro nested Monte Carlo estimator of the Gibbs EIG. To run the code for a specific experimental design problem, one can either simply run the code file from their favourite IDE, or use software like SLURM for running the code over multiple random seeds. Examples of SLURM files are available in [**bash-scripts**](/bash-scripts). General files used for analysing the data after running (ideally) multiple seeds can be found in [**analyses**](/analyses).
