import tqdm
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from src.flow_matching.path.scheduler import CondOTScheduler
from src.flow_matching.path import AffineProbPath
from src.flow_matching.solver import Solver, ODESolver

import logging
logger = logging.getLogger(__name__)

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

class VelocityField(nn.Module):
    def __init__(self, model: nn.Module, input_sz: tuple, scheduler_name: str,  solver_name: str, \
        scheduler_params: dict=None, solver_params: dict = None
    ):
        """Velocity field module for flow matching.
        Args:
            model (nn.Module): The neural network model representing the velocity field.
            input_sz (tuple): The size of the input data.
            scheduler_name (str): The name of the scheduler to use.
            solver_name (str): The name of the solver to use.
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
        super(VelocityField, self).__init__()

        self.model = model
        self.input_sz = input_sz
        self.path = build_scheduler(scheduler_name, scheduler_params or {})
        self.solver = build_solver(model, solver_name, solver_params or {})
        
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
        t = torch.rand(bs).to(device)
        
        # - sample intermediate points along the path
        path_sample = self.path.sample(t=t, x_0=x0, x_1=x1)
        
        # - flow matching loss (l2)
        u_t = self.model(path_sample.x_t, path_sample.t, y=y)
        loss = torch.pow(u_t - path_sample.dx_t, 2).mean()
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
        solver = build_solver(self.model, solver_name, solver_params or {})

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
        solver = build_solver(self.model, solver_name, solver_params or {})

        # - define timesteps
        timesteps = torch.tensor([1.0, stop_time]).to(x1.device)
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
        device = x1.device
        # -- build solver
        solver = build_solver(self.model, solver_name, solver_params or {})
        
        # -- define prior log-probability
        gaussian_log_density = spatial_gaussian_log_density
        if not exact_divergence:
            # -- do hutchinson trace estimator in a si
            hutchinson_bs = min(hutchinson_bs, hutchinson_samples)
            hutchinson_step = hutchinson_samples // hutchinson_bs
            
            log_p_acc = 0
            for i in range(hutchinson_step):
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
    
        
        
        
        