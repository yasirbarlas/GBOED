import torch
import pyro
from pyro.contrib.util import rexpand, rmv
from pyro import poutine

def iqr(tensor, dim=None, keepdim=False):
    q75 = torch.quantile(tensor, 0.75, dim=dim, keepdim=keepdim, interpolation="midpoint")
    q25 = torch.quantile(tensor, 0.25, dim=dim, keepdim=keepdim, interpolation="midpoint")
    return q75 - q25

def negative_log_likelihood(xi, y, theta, sigma=2.0, predictive=None):
    residuals = y - (theta[..., 0].unsqueeze(-1) + theta[..., 1].unsqueeze(-1) * xi)
    nll = (1 / 2) * torch.log(torch.tensor([2 * torch.pi * sigma**2])) + (1 / (2 * sigma**2)) * (residuals**2)
    return nll.squeeze()

def log_likelihood(model, x, y, theta):
    """
    Computes log-likelihood for observed y given a Pyro model.
    """
    y = y.clone().detach().requires_grad_(True)
    data_dict = theta
    #data_dict.update(theta)
    conditioned_model = pyro.condition(model, data=data_dict)
    model_trace = poutine.trace(conditioned_model).get_trace(x.clone().detach().requires_grad_(True))
    log_like = model_trace.nodes["y"]["fn"].log_prob(y).sum()
    return log_like, y

def derivatives_log_likelihood_wrt_y(model, x, y, theta):
    """
    Returns first and second derivatives of log-likelihood w.r.t. y.
    """
    log_like, y_var = log_likelihood(model, x, y, theta)
    grad1 = torch.autograd.grad(log_like, y_var, create_graph=True)[0]
    grad2 = torch.autograd.grad(grad1.sum(), y_var, retain_graph=True)[0]
    return grad1, grad2

def power_likelihood(xi, y, theta, sigma=0.5, c=1.0):
    residuals = y - (theta[..., 0].unsqueeze(-1) + theta[..., 1].unsqueeze(-1) * xi)
    log_likelihood = - (1 / 2) * torch.log(torch.tensor([2 * torch.pi * sigma**2])) - (1 / (2 * sigma**2)) * (residuals**2)
    power_likelihood = torch.exp(c * log_likelihood)
    return power_likelihood

def gamma_divergence(xi, y, theta, sigma=2.0, gamma_d=1.01):
    residuals = y - (theta[..., 0].unsqueeze(-1) + theta[..., 1].unsqueeze(-1) * xi)
    log_prior_noise_factor = - (1 / 2) * torch.log(torch.tensor([2 * torch.pi * sigma**2]))
    log_obs_factor = - (1 / (2 * sigma**2)) * (residuals**2)
    #log_likelihood = - (1 / 2) * torch.log(torch.tensor([2 * torch.pi * sigma**2])) - (1 / (2 * sigma**2)) * (residuals**2)
    sig_gamma = (2 * torch.pi * sigma**2) ** (- gamma_d * (1 / 2))
    integral_term = (1.0 + gamma_d) ** (- 3 / 2) * sig_gamma
    log_lkl = -(log_prior_noise_factor + (1.0 / gamma_d) * sig_gamma * torch.exp(gamma_d * log_obs_factor) / (integral_term ** (gamma_d / (1.0 + gamma_d))))
    return log_lkl.squeeze()

def beta_divergence_knoblauch(xi, y, theta, sigma=2.0, beta_d=1.01):
    #log_likelihood = - (1 / 2) * torch.log(torch.tensor([2 * torch.pi * sigma**2])) - (1 / (2 * sigma**2)) * (residuals**2)
    residuals = y - (theta[..., 0].unsqueeze(-1) + theta[..., 1].unsqueeze(-1) * xi)
    log_prior_noise_factor = - (1 / 2) * torch.log(torch.tensor([2 * torch.pi * sigma**2]))
    log_obs_factor = - (1 / 2) * (residuals)**2 / sigma**2
    sig_beta = (2 * torch.pi * sigma**2) ** (- beta_d * (1 / 2))
    integral_term = (1.0 + beta_d) ** (- 3 / 2) * sig_beta
    log_lkl = (log_prior_noise_factor - integral_term + (1.0 / beta_d) * sig_beta * torch.exp(beta_d * log_obs_factor))
    return log_lkl.squeeze()

def gamma_divergence_knoblauch(xi, y, theta, sigma=2.0, gamma_d=1.01):
    #log_likelihood = - (1 / 2) * torch.log(torch.tensor([2 * torch.pi * sigma**2])) - (1 / (2 * sigma**2)) * (residuals**2)
    residuals = y - (theta[..., 0].unsqueeze(-1) + theta[..., 1].unsqueeze(-1) * xi)
    log_prior_noise_factor = - (1 / 2) * torch.log(torch.tensor([2 * torch.pi * sigma**2]))
    log_obs_factor = - (1 / 2) * (residuals)**2 / sigma**2
    sig_gamma = (2 * torch.pi * sigma**2) ** (- gamma_d * (1 / 2))
    integral_term = (1.0 + gamma_d) ** (- 3 / 2) * sig_gamma
    log_lkl = (log_prior_noise_factor + (1.0 / gamma_d) * sig_gamma * torch.exp(gamma_d * log_obs_factor) / (integral_term ** (gamma_d / (1.0 + gamma_d))))
    return torch.exp(log_lkl).squeeze()
    
class score_matching_regression:
    def __init__(self, w, mean_vec, covariance_mat, std_y):
        self.w = w
        self.mean_vec = mean_vec
        self.covariance_mat = covariance_mat # user inputs covariance matrix, not precision matrix
        self.std_y = std_y

    def grad_r(self, x, y):
        try:
            return torch.tensor([[1.0, x]]) / (self.std_y ** 2)
        except Exception:
            return self.grad_r_eig(x, y)
    
    def grad_r_eig(self, x, y):
        ones = torch.ones_like(x)
        grad = torch.cat([ones, x], dim=-1)
        return grad / (self.std_y ** 2)
    
    def grad_b(self, x, y):
        try:
            return torch.tensor([[-y]]) / (self.std_y ** 2)
        except Exception:
            return self.grad_b_eig(x, y)
    
    def grad_b_eig(self, x, y):
        return (-y * torch.ones_like(y)) / (self.std_y ** 2)
    
    def A(self, x, y):
        g = self.grad_r(x, y)
        return g.T @ g

    def V(self, x, y):
        return self.grad_r(x, y).T @ self.grad_b(x, y)

    def update_params(self, x_list, y_list):
        As = [self.A(x, y) for x, y in zip(x_list, y_list)]
        Vs = [self.V(x, y) for x, y in zip(x_list, y_list)]

        A_T = torch.stack(As).mean(dim=0)
        V_T = 2 * torch.stack(Vs).mean(dim=0)

        inv_cov = torch.linalg.inv(self.covariance_mat)
        new_covariance_mat = inv_cov + 2 * self.w * len(y_list) * A_T # actually the precision matrix (inverse covariance matrix), but code still works
        inv_new_cov = torch.linalg.inv(new_covariance_mat)
        
        new_mean_vec = inv_new_cov @ (inv_cov @ self.mean_vec - self.w * len(y_list) * V_T)

        self.mean_vec = new_mean_vec
        self.covariance_mat = new_covariance_mat

        return new_mean_vec, new_covariance_mat # returns the precision matrix, not the covariance matrix
    
    def dm_final(self, x, y, theta, predictive=None, tau=None, c_squared=None):
        if isinstance(y, dict):
            y = y["y"]
        if isinstance(theta, dict):
            theta = theta["beta"]
        return ((torch.sum(self.grad_r(x, y) * theta, dim=-1) - (y.squeeze() / (self.std_y ** 2))) ** 2) + (2 * (-1 / (self.std_y ** 2)))
    
    def dm_general(self, x, y, theta, predictive=None, tau=None, c_squared=None):
        if isinstance(y, dict):
            y = y["y"]
        if isinstance(theta, dict):
            theta = theta["beta"]
        return (((torch.sum(x * theta, dim=-1) - y.squeeze()) / (self.std_y ** 2)) ** 2) + (2 * (-1 / (self.std_y ** 2)))
    
    def dm_final_weighted(self, x, y, theta, predictive, tau=1, c_squared=None):
        if isinstance(y, dict):
            y = y["y"]
        if isinstance(theta, dict):
            theta = theta["beta"]
        mean = torch.mean(predictive, dim=0)
        #mean = torch.quantile(predictive, q=0.5, dim=0, interpolation="midpoint")
        
        if c_squared is None:
            c_squared = torch.var(predictive, dim=0)
            #c_squared = iqr(predictive, dim=0) ** 2
        
        imq_squared = (((self.w) * ((1 + tau * (((y - mean) ** 2) / c_squared)) ** (-1 / 2))) ** 2)
        imq_squared_deriv = -(self.w ** 2) * ((1 + tau * (((y - mean) ** 2) / c_squared)) ** -2) * (2 * tau * (y - mean) / c_squared)
        #imq_squared_deriv = -(self.w ** 2) * ((1 + (((y - mean) ** 2) / c_squared)) ** -2) * (2 * (y - mean) / c_squared)
        model = (torch.sum(self.grad_r(x, y) * theta, dim=-1) - (y.squeeze() / (self.std_y ** 2)))
        model_deriv = (-1 / (self.std_y ** 2))
        return imq_squared.squeeze() * (model ** 2) + 2 * (imq_squared.squeeze() * model_deriv + model * imq_squared_deriv.squeeze())
    
    def negative_log_likelihood(self, xi, y, theta, predictive=None, tau=None, c_squared=None):
        if isinstance(theta, dict):
            theta = theta["beta"]
        if isinstance(y, dict):
            y = y["y"]
        residuals = y - (theta[..., 0].unsqueeze(-1) + theta[..., 1].unsqueeze(-1) * xi)
        nll = (1 / 2) * torch.log(torch.tensor([2 * torch.pi * self.std_y**2])) + (1 / (2 * self.std_y**2)) * (residuals**2)
        return nll.squeeze()
    
class score_matching_ces:
    def __init__(self, w, obs_sd):
        self.w = w
        self.obs_sd = obs_sd

    def update_params(self, x_list, y_list):
        return NotImplementedError("This method is not implemented in the class. Please implement it in a subclass if needed.")
    
    def dm_final(self, x, y, theta, predictive=None, tau=None, c_squared=None):
        if isinstance(y, dict):
            y = y["y"]
        slope = theta["slope"]
        rho = 0.01 + 0.99 * theta["rho"].select(-1, 0)
        alpha = theta["alpha"]
        rho, slope = rexpand(rho, x.shape[-2]), rexpand(slope, x.shape[-2])
        d1, d2 = x[..., 0:3], x[..., 3:6]
        U1rho = (rmv(d1.pow(rho.unsqueeze(-1)), alpha)).pow(1./rho)
        U2rho = (rmv(d2.pow(rho.unsqueeze(-1)), alpha)).pow(1./rho)
        mu = slope * (U1rho - U2rho)
        sigma = slope * self.obs_sd * (1 + torch.norm(d1 - d2, dim=-1, p=2))

        print("mu", mu.shape, "sigma", sigma.shape, "y", y.shape)

        logit_y = torch.log(y / (1 - y))
        return ((((2 * y) - 1) / (y * (1 - y)) + (- logit_y + mu) / (sigma ** 2 * y * (1 - y))) ** 2) + (2 * (((2 * y ** 2 - 2 * y + 1) / ((y ** 2) * (1 - y) ** 2)) + ((-1 - (1 - 2 * y) * (-logit_y + mu)) / (sigma ** 2 * (y ** 2) * (1 - y) ** 2))))
    
    def dm_general(self, x, y, theta, predictive=None, tau=None, c_squared=None):
        return NotImplementedError("This method is not implemented in the class. Please implement it in a subclass if needed.")
    
    def dm_final_weighted(self, x, y, theta, predictive, tau=1, c_squared=None):
        if isinstance(y, dict):
            y = y["y"]
        slope = theta["slope"]
        rho = 0.01 + 0.99 * theta["rho"].select(-1, 0)
        alpha = theta["alpha"]
        rho, slope = rexpand(rho, x.shape[-2]), rexpand(slope, x.shape[-2])
        d1, d2 = x[..., 0:3], x[..., 3:6]
        U1rho = (rmv(d1.pow(rho.unsqueeze(-1)), alpha)).pow(1./rho)
        U2rho = (rmv(d2.pow(rho.unsqueeze(-1)), alpha)).pow(1./rho)
        mu = slope * (U1rho - U2rho)
        sigma = slope * 0.005 * (1 + torch.norm(d1 - d2, dim=-1, p=2))

        logit_y = torch.log(y.squeeze() / (1 - y.squeeze()))

        mean = torch.mean(predictive, dim=0)
        #mean = torch.quantile(predictive, q=0.5, dim=0, interpolation="midpoint")
        
        if c_squared is None:
            c_squared = torch.var(predictive, dim=0)
            #c_squared = iqr(predictive, dim=0) ** 2
        
        imq_squared = (((self.w) * ((1 + tau * (((y - mean) ** 2) / c_squared)) ** (-1 / 2))) ** 2)
        imq_squared_deriv = -(self.w ** 2) * ((1 + tau * (((y - mean) ** 2) / c_squared)) ** -2) * (2 * tau * (y - mean) / c_squared)
        #imq_squared_deriv = -(self.w ** 2) * ((1 + (((y - mean) ** 2) / c_squared)) ** -2) * (2 * (y - mean) / c_squared)
        model = (((2 * y.squeeze()) - 1) / (y.squeeze() * (1 - y.squeeze())) + (- logit_y + mu) / (sigma ** 2 * y.squeeze() * (1 - y.squeeze())))
        model_deriv = (((2 * y.squeeze() ** 2 - 2 * y.squeeze() + 1) / ((y.squeeze() ** 2) * (1 - y.squeeze()) ** 2)) + ((-1 - (1 - 2 * y.squeeze()) * (-logit_y + mu)) / (sigma ** 2 * (y.squeeze() ** 2) * (1 - y.squeeze()) ** 2)))
        return imq_squared.squeeze() * (model ** 2) + 2 * (imq_squared.squeeze() * model_deriv + model * imq_squared_deriv.squeeze())
    
    def negative_log_likelihood(self, xi, y, theta, predictive=None, tau=None, c_squared=None):
        return NotImplementedError("This method is not implemented in the class. Please implement it in a subclass if needed.")
    
class score_matching_location:
    def __init__(self, w, obs_sd, b, m, a):
        self.w = w
        self.obs_sd = obs_sd
        self.b = b
        self.m = m
        self.a = a

    def update_params(self, x_list, y_list):
        return NotImplementedError("This method is not implemented in the class. Please implement it in a subclass if needed.")
    
    def dm_normal(self, x, y, theta, model=None, predictive=None, tau=None, c_squared=None):
        if isinstance(y, dict):
            y = y["y"]
        theta = theta["theta"]

        distance = torch.square(theta - x).sum(dim=-1)
        ratio = self.a / (self.m + distance)
        mu = self.b + ratio.sum(dim=-1, keepdims=True)

        model = ((torch.log(mu) - y) / (self.obs_sd ** 2))
        model_deriv = (-1 / (self.obs_sd ** 2))

        return ((model ** 2) + (2 * model_deriv)).squeeze()

    def dm_weighted(self, x, y, theta, model, predictive, tau=1, c_squared=None):
        if isinstance(y, dict):
            y = y["y"]
        theta = theta["theta"]

        distance = torch.square(theta - x).sum(dim=-1)
        ratio = self.a / (self.m + distance)
        mu = self.b + ratio.sum(dim=-1, keepdims=True)

        mean = torch.mean(predictive, dim=0)
        #mean = torch.quantile(predictive, q=0.5, dim=0, interpolation="midpoint")
        if c_squared is None:
            c_squared = torch.var(predictive, dim=0)
            #c_squared = iqr(predictive, dim=0) ** 2
        
        imq_squared = (((self.w) * ((1 + tau * (((y - mean) ** 2) / c_squared)) ** (-1 / 2))) ** 2)
        imq_squared_deriv = -(self.w ** 2) * ((1 + tau * (((y - mean) ** 2) / c_squared)) ** -2) * (2 * tau * (y - mean) / c_squared)
        #imq_squared_deriv = -(self.w ** 2) * ((1 + (((y - mean) ** 2) / c_squared)) ** -2) * (2 * (y - mean) / c_squared)
        model = ((torch.log(mu) - y) / (self.obs_sd ** 2))
        model_deriv = (-1 / (self.obs_sd ** 2))
        return (imq_squared * (model ** 2) + 2 * (imq_squared * model_deriv + model * imq_squared_deriv)).squeeze()

    #def dm_general(self, x, y, theta, model=None, predictive=None, tau=None, c_squared=None):
    #    return NotImplementedError("This method is not implemented in the class. Please implement it in a subclass if needed.")
    
    def dm_autograd_normal(self, x, y, theta, model, predictive=None, tau=None, c_squared=None):
        if isinstance(y, dict):
            y = y["y"]
        grad1, grad2 = derivatives_log_likelihood_wrt_y(model, x, y, theta)
        #print(grad1, grad2)
        return (grad1 ** 2 + 2 * grad2).squeeze()

    def dm_autograd_weighted(self, x, y, theta, model, predictive, tau=1, c_squared=None):
        if isinstance(y, dict):
            y = y["y"]
        mean = torch.mean(predictive, dim=0)
        #mean = torch.quantile(predictive, q=0.5, dim=0, interpolation="midpoint")
        
        if c_squared is None:
            c_squared = torch.var(predictive, dim=0)
            #c_squared = iqr(predictive, dim=0) ** 2

        grad1, grad2 = derivatives_log_likelihood_wrt_y(model, x, y, theta)

        imq_squared = (((self.w) * ((1 + tau * (((y - mean) ** 2) / c_squared)) ** (-1 / 2))) ** 2)
        imq_squared_deriv = -(self.w ** 2) * ((1 + tau * (((y - mean) ** 2) / c_squared)) ** -2) * (2 * tau * (y - mean) / c_squared)

        return (imq_squared * (grad1 ** 2) + 2 * (imq_squared * grad2 + grad1 * imq_squared_deriv)).squeeze()

    def negative_log_likelihood(self, x, y, theta, model, predictive=None, tau=None, c_squared=None):
        log_like, y_var = log_likelihood(model, x, y, theta)
        return -log_like