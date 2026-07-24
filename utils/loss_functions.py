import torch
import pyro
from pyro.contrib.util import rexpand, rmv
from pyro import poutine
from censored_sigmoid_normal import CensoredSigmoidNormal # used for CES only

def iqr(tensor, dim=None, keepdim=False):
    q75 = torch.quantile(tensor, 0.75, dim=dim, keepdim=keepdim, interpolation="midpoint")
    q25 = torch.quantile(tensor, 0.25, dim=dim, keepdim=keepdim, interpolation="midpoint")
    return q75 - q25
    
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
    
    def dm_normal(self, x, y, theta, predictive=None, tau=None, c_squared=None):
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
    
    def dm_weighted(self, x, y, theta, predictive, tau=1, c_squared=None):
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
        #print("imq_squared", imq_squared.shape)
        imq_squared_deriv = -(self.w ** 2) * ((1 + tau * (((y - mean) ** 2) / c_squared)) ** -2) * (2 * tau * (y - mean) / c_squared)
        #imq_squared_deriv = -(self.w ** 2) * ((1 + (((y - mean) ** 2) / c_squared)) ** -2) * (2 * (y - mean) / c_squared)
        model = (torch.sum(self.grad_r(x, y) * theta, dim=-1) - (y.squeeze() / (self.std_y ** 2)))
        #print("model", model.shape)
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
        if isinstance(theta, dict):
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
        if isinstance(theta, dict):
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

        model = ((torch.log(mu) - y) / (self.obs_sd ** 2))
        model_deriv = (-1 / (self.obs_sd ** 2))
        
        return (imq_squared * (model ** 2) + 2 * (imq_squared * model_deriv + model * imq_squared_deriv)).squeeze()

    def dm_general(self, x, y, theta, model=None, predictive=None, tau=None, c_squared=None):
        return NotImplementedError("This method is not implemented in the class. Please implement it in a subclass if needed.")

    def negative_log_likelihood(self, x, y, theta, model, predictive=None, tau=None, c_squared=None):
        if isinstance(y, dict):
            y = y["y"]
        if isinstance(theta, dict):
            theta = theta["theta"]

        distance = torch.square(theta - x).sum(dim=-1)
        ratio = self.a / (self.m + distance)
        mu = self.b + ratio.sum(dim=-1, keepdims=True)

        residuals = (y - torch.log(mu))
        nll = (1 / 2) * torch.log(torch.tensor([2 * torch.pi * self.obs_sd ** 2])) + (1 / (2 * self.obs_sd ** 2)) * (residuals ** 2)
        return nll.squeeze()
    
class score_matching_pharmacokinetic:
    def __init__(self, w, D_v, epsilon_scale, nu_scale):
        self.w = w
        self.D_v = D_v
        self.epsilon_scale = epsilon_scale
        self.nu_scale = nu_scale

    def update_params(self, x_list, y_list):
        return NotImplementedError("This method is not implemented in the class. Please implement it in a subclass if needed.")
    
    def dm_normal(self, x, y, theta, model=None, predictive=None, tau=None, c_squared=None):
        if isinstance(y, dict):
            y = y["y"]
        if isinstance(theta, dict):
            theta = theta["log_theta"]
        theta = theta.exp()

        k_a, k_e, V = [theta[..., [i]] for i in range(3)]
        assert (k_a > k_e).all()
        # compute concentration at time t=xi
        # shape of mean is [batch, n] where n is number of obs per design
        mu = ((self.D_v / V) * (k_a / (k_a - k_e)) * (
                    torch.exp(-torch.einsum("...ijk, ...ik->...ij", x, k_e))
                    - torch.exp(-torch.einsum("...ijk, ...ik->...ij", x, k_a))))
        sd = torch.sqrt((mu * self.epsilon_scale) ** 2 + self.nu_scale ** 2)

        model = ((mu - y) / (sd ** 2))
        model_deriv = (-1 / (sd ** 2))

        return ((model ** 2) + (2 * model_deriv)).squeeze()

    def dm_weighted(self, x, y, theta, model, predictive, tau=1, c_squared=None):
        if isinstance(y, dict):
            y = y["y"]
        if isinstance(theta, dict):
            theta = theta["log_theta"]
        theta = theta.exp()

        k_a, k_e, V = [theta[..., [i]] for i in range(3)]
        assert (k_a > k_e).all()
        # compute concentration at time t=xi
        # shape of mean is [batch, n] where n is number of obs per design
        mu = ((self.D_v / V) * (k_a / (k_a - k_e)) * (
                    torch.exp(-torch.einsum("...ijk, ...ik->...ij", x, k_e))
                    - torch.exp(-torch.einsum("...ijk, ...ik->...ij", x, k_a))))
        sd = torch.sqrt((mu * self.epsilon_scale) ** 2 + self.nu_scale ** 2)

        mean = torch.mean(predictive, dim=0)
        #mean = torch.quantile(predictive, q=0.5, dim=0, interpolation="midpoint")
        if c_squared is None:
            c_squared = torch.var(predictive, dim=0)
            #c_squared = iqr(predictive, dim=0) ** 2
        
        imq_squared = (((self.w) * ((1 + tau * (((y - mean) ** 2) / c_squared)) ** (-1 / 2))) ** 2)
        imq_squared_deriv = -(self.w ** 2) * ((1 + tau * (((y - mean) ** 2) / c_squared)) ** -2) * (2 * tau * (y - mean) / c_squared)

        model = ((mu - y) / (sd ** 2))
        model_deriv = (-1 / (sd ** 2))

        return (imq_squared * (model ** 2) + 2 * (imq_squared * model_deriv + model * imq_squared_deriv)).squeeze()

    def dm_general(self, x, y, theta, model=None, predictive=None, tau=None, c_squared=None):
        return NotImplementedError("This method is not implemented in the class. Please implement it in a subclass if needed.")

    def negative_log_likelihood(self, x, y, theta, model, predictive=None, tau=None, c_squared=None):
        if isinstance(y, dict):
            y = y["y"]
        if isinstance(theta, dict):
            theta = theta["log_theta"]
        theta = theta.exp()

        k_a, k_e, V = [theta[..., [i]] for i in range(3)]
        assert (k_a > k_e).all()
        # compute concentration at time t=xi
        # shape of mean is [batch, n] where n is number of obs per design
        mu = ((self.D_v / V) * (k_a / (k_a - k_e)) * (
                    torch.exp(-torch.einsum("...ijk, ...ik->...ij", x, k_e))
                    - torch.exp(-torch.einsum("...ijk, ...ik->...ij", x, k_a))))
        sd = torch.sqrt((mu * self.epsilon_scale) ** 2 + self.nu_scale ** 2)

        residuals = (y - mu)
        nll = (1 / 2) * torch.log(2 * torch.pi * sd ** 2) + (1 / (2 * sd ** 2)) * (residuals ** 2)
        return nll.squeeze()
    
class score_matching_ces:
    def __init__(self, w, obs_sd):
        self.w = w
        self.obs_sd = obs_sd

    def update_params(self, x_list, y_list):
        return NotImplementedError("This method is not implemented in the class. Please implement it in a subclass if needed.")
    
    def dm_final(self, x, y, theta, predictive=None, tau=None, c_squared=None):
        return NotImplementedError("This method is not implemented in the class. Please implement it in a subclass if needed.")
        
    def dm_general(self, x, y, theta, predictive=None, tau=None, c_squared=None):
        return NotImplementedError("This method is not implemented in the class. Please implement it in a subclass if needed.")
    
    def dm_final_weighted(self, x, y, theta, predictive, tau=1, c_squared=None):
        return NotImplementedError("This method is not implemented in the class. Please implement it in a subclass if needed.")
    
    def negative_log_likelihood(self, x, y, theta, predictive=None, tau=None, c_squared=None):
        if isinstance(y, dict):
            y = y["y"]

        epsilon = torch.tensor(2**-22)

        slope = theta["slope"]
        rho = 0.01 + 0.99 * theta["rho"].select(-1, 0)
        alpha = theta["alpha"]
        rho, slope = rexpand(rho, x.shape[-2]), rexpand(slope, x.shape[-2])
        d1, d2 = x[..., 0:3], x[..., 3:6]
        U1rho = (rmv(d1.pow(rho.unsqueeze(-1)), alpha)).pow(1./rho)
        U2rho = (rmv(d2.pow(rho.unsqueeze(-1)), alpha)).pow(1./rho)
        mu = slope * (U1rho - U2rho)
        sigma = slope * self.obs_sd * (1 + torch.norm(d1 - d2, dim=-1, p=2))

        distribution = CensoredSigmoidNormal(mu, sigma, 1 - epsilon, epsilon).to_event(1)

        distribution_log_prob = distribution.log_prob(y)
        return -distribution_log_prob.squeeze()