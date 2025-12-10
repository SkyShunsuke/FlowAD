import tqdm
import torch
import torch.nn as nn

from src.flow_matching.path.scheduler import CondOTScheduler
from src.flow_matching.path import AffineProbPath
from src.flow_matching.solver import Solver, ODESolver

import logging
logger = logging.getLogger(__name__)

def build_scheduler(scheduler_name: str, scheduler_params: dict) -> CondOTScheduler:
    """Build scheduler for flow matching.

    Args:
        scheduler_name (str): Name of the scheduler.
        scheduler_params (dict): Parameters for the scheduler.

    Returns:
        CondOTScheduler: Configured scheduler.
    """
    logger.info(f'Building scheduler: {scheduler_name} with params: {scheduler_params}')
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
    logger.info(f'Building solver: {solver_name} with params: {solver_params}')
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
    
    def sample(self, x0, y, steps: int, return_intermediate: bool=False, solver_name: str='euler', solver_params: dict=None):
        """Sample from the velocity field using the specified solver.
        Args:
            x0 (torch.Tensor): The starting points.
            y (torch.Tensor): Additional conditioning information.
            steps (int): Number of steps for the solver.
            return_intermediate (bool, optional): Whether to return intermediate results. Defaults to False.
            solver_name (str, optional): The name of the solver to use. Defaults to 'euler'.
            solver_params (dict, optional): Additional parameters for the solver. Defaults to None.
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
        timesteps = torch.linspace(0., 1., steps).to(x0.device)
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
            return samples[-1]  # Final result
    
    def invert(self, x1, y, steps: int, solver_name: str='euler', solver_params: dict=None, return_intermediate: bool=False):
        """Invert the velocity field using the specified solver.
        Args:
            x1 (torch.Tensor): The ending points.
            y (torch.Tensor): Additional conditioning information.
            steps (int): Number of steps for the solver.
            solver_name (str, optional): The name of the solver to use. Defaults to 'euler'.
            solver_params (dict, optional): Additional parameters for the solver. Defaults to None.
            return_intermediate (bool, optional): Whether to return intermediate results. Defaults to False.
        Returns:
            torch.Tensor: The inverted points.
        Usage: 
            x1 = torch.randn(batch_size, *input_sz).to(device)
            x0 = vf.invert(x1, y, steps=10)  # (batch_size, *input_sz)
        """
        # - build solver
        solver = build_solver(self.model, solver_name, solver_params or {})

        # - define timesteps
        timesteps = torch.tensor([1.0, 0.0]).to(x1.device)
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
    
    def log_prob(self, x1, y, steps: int, solver_name: str='euler', solver_params: dict=None, exact: bool=False, hte_acc: int=10):
        """Estimate the log-probability of the data points using the velocity field.
        Args:
            x1 (torch.Tensor): The data points.
            y (torch.Tensor): Additional conditioning information.
            steps (int): Number of steps for the solver.
            solver_name (str, optional): The name of the solver to use. Defaults to 'euler'.
            solver_params (dict, optional): Additional parameters for the solver. Defaults to None.
            exact (bool, optional): Whether to use exact computation. Defaults to False.
            hte_acc (int, optional): Accuracy parameter for Hutchinson's trace estimator. Defaults to 10.
        Returns:
            torch.Tensor: The estimated log-probabilities.
        Usage: 
            log_prob = vf.log_prob(x1, y, steps=10)  # (batch_size,)
        """
        # XXX: 
        return 


    
    
        
        
        
        