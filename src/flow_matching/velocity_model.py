import tqdm
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from src.flow_matching.path.scheduler import CondOTScheduler
from src.flow_matching.path import AffineProbPath
from src.flow_matching.solver import Solver, ODESolver

import logging
logger = logging.getLogger(__name__)

PRED_TYPES = ['data', 'noise', 'velocity']
LOSS_TYPES = ['data', 'noise', 'velocity']

def spatial_gaussian_log_density(x: torch.Tensor) -> torch.Tensor:
    # x: (B, C, H, W)
    return Normal(torch.zeros_like(x), torch.ones_like(x)).log_prob(x).sum(dim=1)

def build_scheduler(scheduler_name: str, scheduler_params: dict) -> CondOTScheduler:
    """Build scheduler for flow matching.

    Args:
        scheduler_name (str): Name of the scheduler.
        scheduler_params (dict): Parameters for the scheduler.

    Returns:
        CondOTScheduler: Configured scheduler.
    """
    if scheduler_name == 'affine_prob':
        scheduler = CondOTScheduler(**scheduler_params)
        return AffineProbPath(scheduler)
    else:
        raise ValueError(f'Unknown scheduler name: {scheduler_name}')
    
def build_solver(model: nn.Module, solver_name: str, solver_params: dict) -> Solver:
    """Build solver for flow matching.

    Args:
        model (nn.Module): The velocity model.
        solver_name (str): Name of the solver.
        solver_params (dict): Parameters for the solver.

    Returns:
        Solver: Configured solver.
    """
    return ODESolver(model, **solver_params)

def logit_t(bs: int, device: torch.device, mu=0., sigma=1.) -> torch.Tensor:
    # -- sample t from gaussian
    t = torch.randn(bs, device=device) * sigma + mu
    # -- apply sigmoid
    t = torch.sigmoid(t)
    return t
    
class WrappedModel(nn.Module):
    """A wrapper for the velocity model to handle additional conditioning information."""
    def __init__(self, model: nn.Module, path, pred_type: str, cfg_interval=[0.1, 1.0], cfg_scale=1.0, eps=0.05):
        super(WrappedModel, self).__init__()
        self.model = model
        self.path = path
        self.pred_type = pred_type
        self.cfg_interval = cfg_interval
        self.cfg_scale = cfg_scale
        self.eps = eps
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, y=None, **extras) -> torch.Tensor:
        
        assert self.cfg_scale == 1.0 or y is not None, "Classifier-free guidance requires conditioning information y."
        
        if self.cfg_scale != 1.0:
            extras.update({'cfg_scale': self.cfg_scale, 'cfg_interval': self.cfg_interval})
            model_out = self.model.forward_with_cfg(x, t, y=y, **extras)
        else:
            model_out = self.model(x, t, y=y, **extras)
        
        if self.pred_type == 'data':
            out = self.data_to_velocity(x, t, model_out)
        elif self.pred_type == 'noise':
            out = self.noise_to_velocity(x, t, model_out)
        else:  # velocity
            out = model_out
        return out

    def noise_to_velocity(self, x_t: torch.Tensor, t: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Convert noise prediction to velocity field.
        Args:
            x_t (torch.Tensor): The intermediate points along the path.
            t (torch.Tensor): The time steps.
            out (torch.Tensor): The noise prediction from the model.
        Returns:
            torch.Tensor: The velocity field at x_t.
        """
        div = t.expand_as(x_t)
        div = torch.clamp(div, min=self.eps)
        v_t = (x_t - out) / div
        return v_t
    
    def data_to_velocity(self, x_t: torch.Tensor, t: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Convert data prediction to velocity field.
        Args:
            x_t (torch.Tensor): The intermediate points along the path.
            t (torch.Tensor): The time steps.
            out (torch.Tensor): The data prediction from the model.
        Returns:
            torch.Tensor: The velocity field at x_t.
        """
        div = torch.clamp(1-t, min=self.eps)
        div = div.expand_as(x_t)
        v_t = (out - x_t) / div
        return v_t

class VelocityField(nn.Module):
    def __init__(self, model: nn.Module, input_sz: tuple, scheduler_name: str,  solver_name: str, \
        pred_type:str='velocity', loss_type:str='velocity', loss_fn:str='mse', train_steps: int = -1, 
        t_scheduler_train: str = 'linear', t_scheduler_infer: str = 'linear', t_mu: float=0.0, t_sigma: float=1.0, \
        cfg_interval: list = [0.1, 1.0], cfg_scale: float = 1.0, div_eps: float=0.05, scheduler_params: dict=None, solver_params: dict = None
    ):
        """Velocity field module for flow matching.
        Args:
            model (nn.Module): The neural network model representing the velocity field.
            input_sz (tuple): The size of the input data.
            scheduler_name (str): The name of the scheduler to use.
            solver_name (str): The name of the solver to use.
            pred_type (str, optional): Type of prediction ('data', 'noise', 'velocity'). Defaults to 'velocity'.
            loss_type (str, optional): Type of loss ('data', 'noise', 'velocity'). Defaults to 'velocity'.
            train_steps (int, optional): Number of training steps. Defaults to -1.
            t_scheduler_train (str, optional): Name of the t scheduler for training. Defaults to 'linear'.
            t_scheduler_infer (str, optional): Name of the t scheduler for inference. Defaults to 'linear'.
            t_mu (float, optional): Mean of the Gaussian distribution for sampling t. Defaults to 0.0.
            t_sigma (float, optional): Standard deviation of the Gaussian distribution for sampling t. Defaults to 1.0.
            cfg_interval (list, optional): Time interval which use classifier-free guidance. Defaults to [0.1, 1.0].
            cfg_scale (int): Cfg scale value. 
            div_eps (float, optional): Epsilon value for numerical stability in velocity calculation. Defaults to 0.05.
            scheduler_params (dict, optional): Additional parameters for the scheduler. Defaults to None.
            solver_params (dict, optional): Additional parameters for the solver. Defaults to None.
        Usage: 
            - initialization
            vf = VelocityField(model, input_sz, scheduler_name, solver_name, scheduler_params, solver_params)
            - training
            loss = vf(x1, y)
            - sampling
            x0 = torch.randn(batch_size, *input_sz).to(device)
            x1 = vf.sample(x0, y, steps=10)
            - inversion
            x0 = vf.invert(x1, y, steps=10)
            - density estimation
            log_prob = vf.log_prob(x1, y, steps=10, solver='euler', exact=False, hte_acc=10)
        Assumptions:
            - The model takes input of shape (batch_size, *input_sz) and returns output of the same shape.
            - The scheduler and solver are implemented elsewhere and are compatible with this module.
        """
        assert pred_type in PRED_TYPES, f"pred_type must be one of {PRED_TYPES}"
        assert loss_type in LOSS_TYPES, f"loss_type must be one of {LOSS_TYPES}"
        super(VelocityField, self).__init__()

        self.model = model
        self.input_sz = input_sz
        self.pred_type = pred_type
        self.loss_type = loss_type
        self.loss_fn = loss_fn
        self.cfg_interval = cfg_interval
        self.cfg_scale = cfg_scale
        self.div_eps = div_eps
        self.t_scheduler_train = t_scheduler_train
        self.t_scheduler_infer = t_scheduler_infer
        self.t_mu = t_mu
        self.t_sigma = t_sigma
        self.train_steps = train_steps  
        self.path = build_scheduler(scheduler_name, scheduler_params or {})
        self.solver = build_solver(model, solver_name, solver_params or {})
        
    def compute_loss(self, x0, x1, path_sample, y=None):
        """Compute the flow matching loss based on the specified loss type.
        """
        # Get the model prediction
        xt = path_sample.x_t
        t = path_sample.t
        v = path_sample.dx_t
        
        out = self.model(xt, t, y=y)
        
        t = t.view(-1, 1, 1, 1)
        recip_one_minus_t = 1.0 / torch.clamp(1 - t, min=self.div_eps)
        recip_t = 1.0 / torch.clamp(t, min=self.div_eps)
        if self.loss_type == 'velocity':
            target = v
            if self.pred_type == 'velocity':
                pred = out
            elif self.pred_type == 'data':
                pred = (out - xt) * recip_one_minus_t
            elif self.pred_type == 'noise':
                pred = (xt - out) * recip_t
        elif self.loss_type == 'data':
            target = x1
            if self.pred_type == 'velocity':
                pred = (1- t) * out + xt
            elif self.pred_type == 'data':
                pred = out
            elif self.pred_type == 'noise':
                pred = (xt - (1 - t) * out) * recip_t
        elif self.loss_type == 'noise':
            target = x0
            if self.pred_type == 'velocity':
                pred = xt - t * out
            elif self.pred_type == 'data':
                pred = (xt - t * out) * recip_one_minus_t
            elif self.pred_type == 'noise':
                pred = out
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented.")
        
        # - compute loss
        if self.loss_fn == 'mse':
            loss = torch.pow(pred - target, 2).mean()
        else:
            raise NotImplementedError(f"Loss function {self.loss_fn} not implemented.")
        return loss
        
    def forward(self, x1, y=None):
        """Compute the loss for flow matching.
        Args:
            x0 (torch.Tensor): The starting points.
            x1 (torch.Tensor): The ending points.
            y (torch.Tensor, optional): Additional conditioning information. Defaults to None.
        Returns:
            torch.Tensor: The computed loss.
        """
        
        bs = x1.shape[0]
        device = x1.device
        
        # - sample timesteps and initial points
        x0 = torch.randn_like(x1).to(device)
        if self.t_scheduler_train == 'linear':
            cand_t = torch.rand(bs, device=device) if self.train_steps < 0 else torch.linspace(0., 1., self.train_steps).to(device)
        elif self.t_scheduler_train == 'logistic':
            cand_t = logit_t(bs, device, self.t_mu, self.t_sigma)
        else:
            raise NotImplementedError(f"t_scheduler_name {self.t_scheduler_train} not implemented.")
        
        if self.train_steps > 0:
            t = cand_t[torch.randint(0, self.train_steps, (bs,))].to(device)
        else:
            t = cand_t.to(device)
        
        # - sample intermediate points along the path
        path_sample = self.path.sample(t=t, x_0=x0, x_1=x1)
        
        # - flow matching loss (l2)
        loss = self.compute_loss(x0, x1, path_sample, y=y)
        return loss
    
    def sample(self, x0, y, steps: int, return_intermediate: bool=False, solver_name: str='euler', solver_params: dict=None, start_t: float=0.0):
        """Sample from the velocity field using the specified solver.
        Args:
            x0 (torch.Tensor): The starting points.
            y (torch.Tensor): Additional conditioning information.
            steps (int): Number of steps for the solver.
            return_intermediate (bool, optional): Whether to return intermediate results. Defaults to False.
            solver_name (str, optional): The name of the solver to use. Defaults to 'euler'.
            solver_params (dict, optional): Additional parameters for the solver. Defaults to None.
            start_t (float, optional): The starting time for sampling. Defaults to 0.0.
        Returns:
            torch.Tensor: The sampled points.
        Usage: 
            x0 = torch.randn(batch_size, *input_sz).to(device)
            x1 = vf.sample(x0, y, steps=10)  # (batch_size, *input_sz)
            x1_inter = vf.sample(x0, y, steps=10, return_intermediate=True)  # List of intermediate results
        """
        
        # - build solver
        model = WrappedModel(self.model, self.path, self.pred_type, self.div_eps)
        solver = build_solver(model, solver_name, solver_params or {})

        # - define timesteps
        timesteps = torch.linspace(start_t, 1., steps).to(x0.device)
        step_size = 1.0 / steps
        
        # - sample with solver
        # NOTE: if you pass step_size to solver, it will prioritize step_size over timegrid.
        samples = solver.sample(
            time_grid=timesteps,
            x_init=x0,
            method=solver_name,
            step_size=step_size,
            return_intermediate=True,
            y=y,
        )  
        if return_intermediate:
            return samples  # List of intermediate results
        else:
            if isinstance(samples, list):
                return samples[-1]  # Final result
            else:
                return samples
    
    def invert(self, x1, y, steps: int, solver_name: str='euler', solver_params: dict=None, return_intermediate: bool=False, stop_time: float=0.0):
        """Invert the velocity field using the specified solver.
        Args:
            x1 (torch.Tensor): The ending points.
            y (torch.Tensor): Additional conditioning information.
            steps (int): Number of steps for the solver.
            solver_name (str, optional): The name of the solver to use. Defaults to 'euler'.
            solver_params (dict, optional): Additional parameters for the solver. Defaults to None.
            return_intermediate (bool, optional): Whether to return intermediate results. Defaults to False.
            stop_time (float, optional): The stopping time for inversion. Defaults to 0.0.
        Returns:
            torch.Tensor: The inverted points.
        Usage: 
            x1 = torch.randn(batch_size, *input_sz).to(device)
            x0 = vf.invert(x1, y, steps=10)  # (batch_size, *input_sz)
        """
        # - build solver
        model = WrappedModel(self.model, self.path, self.pred_type, self.div_eps)
        solver = build_solver(model, solver_name, solver_params or {})

        # - define timesteps
        timesteps = torch.tensor([1., stop_time]).to(x1.device)
        step_size = 1.0 / steps
        
        # - invert with solver
        samples = solver.sample(
            time_grid=timesteps,
            x_init=x1,
            method=solver_name,
            step_size=step_size,
            return_intermediate=True,
            y=y,
        )
        if return_intermediate:
            return samples  # List of intermediate results
        else:
            if isinstance(samples, list):
                return samples[-1]  # Final result
            else:
                return samples
    
    def density(self, x1, y, steps: int, solver_name: str='euler', solver_params: dict=None, exact_divergence: bool=False, hutchinson_samples: int=10, hutchinson_bs: int=20):
        """Estimate the log-probability of the data points using the velocity field.
        Args:
            x1 (torch.Tensor): The data points.
            y (torch.Tensor): Additional conditioning information.
            steps (int): Number of steps for the solver.
            solver_name (str, optional): The name of the solver to use. Defaults to 'euler'.
            solver_params (dict, optional): Additional parameters for the solver. Defaults to None.
            exact_divergence (bool, optional): Whether to use exact computation. Defaults to False.
            hutchinson_samples (int, optional): Accuracy parameter for Hutchinson's trace estimator. Defaults to 10.
        Returns:
            torch.Tensor: The estimated log-probabilities.
        Usage: 
            log_prob = vf.log_prob(x1, y, steps=10)  # (batch_size,)
        """
        # -- build solver
        model = WrappedModel(self.model, self.path, self.pred_type, self.div_eps)
        solver = build_solver(model, solver_name, solver_params or {})
        
        # -- define prior log-probability
        gaussian_log_density = spatial_gaussian_log_density
        if not exact_divergence:
            # -- do hutchinson trace estimator in a si
            hutchinson_bs = min(hutchinson_bs, hutchinson_samples)
            hutchinson_step = hutchinson_samples // hutchinson_bs
            
            log_p_acc = 0
            for i in tqdm.tqdm(range(hutchinson_step)):
                Hb = hutchinson_bs
                B, _, h, w = x1.shape
                x1_rep = x1.repeat_interleave(Hb, dim=0)
                y_rep = y.repeat_interleave(Hb, dim=0) if y is not None else None
                _, log_p_rep = solver.compute_likelihood(x_1=x1_rep, method=solver_name, step_size=1.0/steps, exact_divergence=exact_divergence, log_p0=gaussian_log_density, y=y_rep)
                log_p = log_p_rep.view(B, Hb, h, w).mean(dim=1)  
                log_p_acc += log_p
            log_p = log_p_acc / hutchinson_step  # unbiased estimator
        else:
            _, log_p = solver.compute_likelihood(x_1=x1, method=solver_name, step_size=1.0/steps, exact_divergence=exact_divergence, log_p0=gaussian_log_density, y=y)
        return log_p
    
        
        
        
        