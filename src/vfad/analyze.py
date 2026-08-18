import torch
import torch.nn.functional as F
from typing import *
from collections import OrderedDict
from tqdm import tqdm

from src.flow_matching import VelocityField


import os

import torch
import torch.distributed as dist
from torch import nn
import numpy as np
from scipy.special import logsumexp

import yaml
import logging
import pandas as pd

from src.models import init_model
from src.datasets import build_dataset
from src.backbones import get_backbone, get_backbone_feature_shape, get_normalization_func, VAEWrapper
from src.flow_matching import VelocityField
from src.utils.distributed import init_distributed_mode, get_rank, get_world_size
from src.utils.distributed import is_main_process as is_main
from src.utils.log import setup_logging, get_logger
from src.utils.opt.optimizer import load_model_only

from src.utils.adeval.eval_utils import divide_by_class, calculate_px_metrics
from typing import Optional, List, Dict, Tuple, Any

import logging
logger = logging.getLogger(__name__)

def main(params, args):
    """XXX for VFAD analysis.
    We use following terms in the code:
    - 
    - 
    - 
    """
    # init distributed mode
    init_distributed_mode(args)

    rank = get_rank()
    world_size = get_world_size()
    device = torch.device('cuda:%s'%args.gpu)
    
    # -- setup logging
    setup_logging(rank, world_size)
    logger = get_logger()
    logger.info(f"Using device: {device}, rank: {rank}, world_size: {world_size}")

    # -- make logging stuff
    if is_main():
        log_dir = params['logging']['log_dir']
        
        # save config file
        config_save_path = os.path.join(log_dir, 'analyze_config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(params, f)
    else:
        log_dir = params['logging']['log_dir']
    
    # set seed
    seed = params['meta']['global_seed'] + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    logger.info(f"Building datasets... with config: \\ {params['data']}")
    test_bs = params['data'].get('test_batch_size', 8)
    train_dataset = build_dataset(train=True, **params['data'])
    test_dataset = build_dataset(train=False, **params['data'])
    
    num_classes = len(test_dataset.datasets)
    logger.info(f"Number of classes: {num_classes}")
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=test_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=8,
        pin_memory=params['data'].get('pin_memory', True),
        num_workers=params['data'].get('num_workers', 4),
        persistent_workers=params['data'].get('persistent_workers', True),
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=test_bs,
        pin_memory=params['data'].get('pin_memory', True),
        num_workers=params['data'].get('num_workers', 4),
        persistent_workers=params['data'].get('persistent_workers', True),
        drop_last=False,
    )
    logger.info(f"Data loaders built. Number of evaluation samples: {len(test_dataset)}, "
                f"Number of test samples: {len(test_dataset)}, ")
    
    # build model    
    feat_sz = get_backbone_feature_shape(model_name=params['model']['backbone']['model_name'],)
    fe = get_backbone(**params['model']['backbone'])
    feat_norm_method = params['model']['backbone'].get('normalization', None)

    norm_fn = get_normalization_func(feat_norm_method)
    fe.to(device).eval()
    logger.info(f"Backbone {params['model']['backbone']['model_name']} initialized.")

    logger.info(f"Using input shape {feat_sz} for the flow matching.")
    pred_type, loss_type = params['flow_matching'].get('pred_type', 'velocity'), params['flow_matching'].get('loss_type', 'velocity')
    train_steps = params['flow_matching'].get('train_steps', -1)
    t_scheduler_train = params['flow_matching']['scheduler'].get('t_scheduler_train', 'linear')
    t_scheduler_infer = params['flow_matching']['scheduler'].get('t_scheduler_infer', 'linear')
    t_mu = params['flow_matching']['scheduler'].get('t_mu', 0.0)
    t_sigma = params['flow_matching']['scheduler'].get('t_sigma', 1.0)
    div_eps = params['flow_matching'].get('div_eps', 0.05)
    logger.info(f"Flow Matching Settings: t_scheduler_train: {t_scheduler_train}, t_scheduler_infer: {t_scheduler_infer}, t_mu: {t_mu}, t_sigma: {t_sigma}, div_eps: {div_eps}")
    logger.info(f"Flow Matching Prediction Type: {pred_type}, Loss Type: {loss_type}")
    logger.info(f"Using partial time sampling with {train_steps} training steps." if train_steps > 0 else "Using full time sampling.")
    model = init_model(input_sz=feat_sz, num_classes=num_classes, **params['model']).to(device)
    vf = VelocityField(
        model=model,
        input_sz=feat_sz,
        scheduler_name=params['flow_matching']['scheduler']['name'],
        solver_name=params['flow_matching']['solver']['name'],
        loss_type=loss_type,
        pred_type=pred_type,
        train_steps=train_steps,
        t_scheduler_train=t_scheduler_train,
        t_scheduler_infer=t_scheduler_infer,
        t_mu=t_mu,
        t_sigma=t_sigma,
        div_eps=div_eps,
        scheduler_params=params['flow_matching']['scheduler'].get('params', None),
        solver_params=params['flow_matching']['solver'].get('params', None),
    )
    logger.info(f"Velocity Field Model {params['model']['model_name']} has been initialized.")

    if args.distributed:
        vf = torch.nn.parallel.DistributedDataParallel(vf, static_graph=True)
    
    # -- resume training if needed
    resume_path = params['resume']['resume_path']
    if resume_path is None:
        logger.warning("No resume path specified. This analysis go ahead without loading a checkpoint. ")
    
    if resume_path is not None:
        vf = load_model_only(
            resume_path, vf
        )
        logger.info(f"Resumed from checkpoint: {resume_path}")
    
    # -- analysis setup
    analysis_params = params['logging']['analysis']
    vf.eval()

    # -- evaluate
    logger.info(f"Starting analysis...")
    
    num_noise_per_sample = analysis_params.get('num_noise_per_sample', 16)
    all_latents = []
    all_labels = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(train_loader, desc="Extracting Training(Normal) Features", disable=not is_main())):
            current_idx = step * train_loader.batch_size * get_world_size()
            
            imgs_local, clslabels_local = batch["img"], batch["clslabel"]    # (B, C, H, W), (B,)
            imgs_local, clslabels_local = imgs_local.to(device, non_blocking=True), clslabels_local.to(device, non_blocking=True)
            
            # -- extract features
            if isinstance(fe, VAEWrapper):
                z1_dist = fe.encode(imgs_local)
                z1_mu, z1_sigma = z1_dist.mean_std()
           
                # get M samples from the latent distribution
                z1_mu = z1_mu.repeat_interleave(num_noise_per_sample, dim=0)  # (B * M, C, H, W)
                z1_sigma = z1_sigma.repeat_interleave(num_noise_per_sample, dim=0)  # (B * M, C, H, W)
                z1 = z1_mu + z1_sigma * torch.randn_like(z1_sigma)
            else:
                z1 = fe(imgs_local)[0]  # (B, C, H, W)
            
            all_labels.append(clslabels_local.cpu().repeat_interleave(num_noise_per_sample))  # (B * M,)
            all_latents.append(z1.cpu())
    
    all_latents = torch.cat(all_latents, dim=0)  # (N * M, C, H, W)
    all_labels = torch.cat(all_labels, dim=0)  # (N * M)
    logger.info(f"Analysis completed. Latent features shape: {all_latents.shape}, Labels shape: {all_labels.shape}")
    analysis_results = {}
    
    all_latents = divide_by_class(all_latents, all_labels)
    
    # collate test samples
    test_anom_masks = []
    test_cls_labels = []
    test_latents = []
    for step, batch in enumerate(tqdm(test_loader, desc="Extracting Test Features", disable=not is_main())):
        imgs, cls_labels, anom_masks = batch["img"], batch["clslabel"], batch["mask"]
        imgs, cls_labels, anom_masks = imgs.to(device, non_blocking=True), cls_labels.to(device, non_blocking=True), anom_masks.to(device, non_blocking=True)
        with torch.no_grad():
            if isinstance(fe, VAEWrapper):
                z1 = fe.encode(imgs).sample()
            else:
                z1 = fe(imgs)[0]
            
        test_latents.append(z1.cpu())
        test_cls_labels.append(cls_labels.cpu())
        test_anom_masks.append(anom_masks.cpu())
    test_latents = torch.cat(test_latents, dim=0)
    test_cls_labels = torch.cat(test_cls_labels, dim=0)
    test_anom_masks = torch.cat(test_anom_masks, dim=0)
    logger.info(f"Test features extracted. Latent features shape: {test_latents.shape}, Labels shape: {test_cls_labels.shape}, Anomaly masks shape: {test_anom_masks.shape}")
    
    test_latents_all = divide_by_class(test_latents, test_cls_labels)
    test_anom_masks_all = divide_by_class(test_anom_masks, test_cls_labels)
        
    gmm_val_ratio = analysis_params.get('gmm_val_ratio', 0.2)
    gmm_K = analysis_params.get('gmm_K', 4)
    gmm_cov_type = analysis_params.get('gmm_cov_type', 'diag')
    
    pca_dim = analysis_params.get('pca_dim', 16)
    num_traj_points = analysis_params.get('num_traj_points', 101)
    epsilon = analysis_params.get('epsilon', 0.05)
    delta = analysis_params.get('delta', 0.05)
    
    # one-step optimality
    traj_num_steps = analysis_params.get('traj_num_steps', 10)
    noise_time = analysis_params.get('noise_time', 0.8)
    
    for clsid, latents in all_latents.items():
        logger.info(f"Class {clsid}: Latent features shape: {latents.shape}")
        cls_latents = latents.permute(0, 2, 3, 1).reshape(-1, latents.shape[1]).numpy()  # (N * H * W, C)
        if cls_latents.shape[-1] > 64:
            logger.info(f"Class {clsid}: Latent features have high dimension {cls_latents.shape[-1]}. Performing PCA to reduce to {pca_dim} dimensions.")
            cls_latents, pca_components = dim_reduce_pca_orthogonal(cls_latents, n_components=pca_dim)
            logger.info(f"Class {clsid}: PCA completed. Reduced latent features shape: {cls_latents.shape}")
        
        train_size = int(cls_latents.shape[0] * (1 - gmm_val_ratio))
        train_z = cls_latents[:train_size]
        val_z = cls_latents[train_size:]
        gmm, val_scores = fit_gmm(train_z, val_z, n_components=gmm_K, cov_type=gmm_cov_type)
        logger.info(f"Class {clsid}: GMM fitted with {gmm_K} components and {gmm_cov_type} covariance. Validation scores: {val_scores.mean():.4f} ± {val_scores.std():.4f}")
        
        traj_timesteps = np.linspace(0, 1, num_traj_points)  # (num_traj_points,)
        syn_latents_np, syn_ids = gmm.sample(n_samples=train_size)  # (N * H * W, C)
        syn_latents_mus = gmm.means_ # (K, C)
        syn_latents = torch.from_numpy(syn_latents_np).to(torch.float32).to(device)
        
        logger.info(f"Constructing ClosedFM for class {clsid}")
        cfm = LocalizedClosedFM(
            train_samples=torch.from_numpy(syn_latents_mus).to(device),
            query_batch_size=4096,
            ref_batch_size=4096,
            store_dtype=torch.float32,
            bandwidth_min=0.0,
        ).to(device)
        vf = vf.module
        
        # with torch.no_grad():
        #     x = torch.from_numpy(cls_latents).float()
        #     sub = x[torch.randperm(x.shape[0])[:2048]]
        #     d2 = torch.cdist(sub, sub).pow(2)
        #     sigma0 = float((0.5 * d2[d2 > 0].median()).sqrt())  # 典型ペアで exp(-1) 程度

        # cfm = RidgeLocalizedClosedFM(
        #     train_samples=torch.from_numpy(cls_latents).to(device),
        #     query_batch_size=4096,
        #     ref_batch_size=4096,
        #     ridge_lambda=1.0,
        #     ridge_num_anchors=1024,
        #     ridge_kernel_scale=sigma0,
        #     ridge_seed=0,
        # )
        
        test_latents = test_latents_all[clsid].permute(0, 2, 3, 1).reshape(-1, test_latents_all[clsid].shape[1])
        test_latents_np = test_latents.cpu().numpy()  # (N * H * W, C)
        # (N * H * W, C)
        N, C, H, W = test_latents_all[clsid].shape
        traj_noise = torch.randn(test_latents.size(-1), device=device).repeat(test_latents.shape[0], 1)  # (N * H * W, C)
        perturbed_test_latents = cfm.perturb(test_latents.to(device), t=noise_time, noise=traj_noise.to(device))
        perturbed_test_latents = perturbed_test_latents.reshape(N, H, W, C).permute(0, 3, 1, 2).contiguous() # (N, C, H, W)
        clslabels = torch.full((N,), clsid, dtype=torch.long, device=device)
        # recon_latents = cfm.sample(perturbed_test_latents.to(device), steps=traj_num_steps, start_t=noise_time, return_intermediate=False)
        with torch.no_grad():
            recon_latents = vf.sample(perturbed_test_latents.to(device), clslabels, steps=traj_num_steps, start_t=noise_time, mu_all=torch.from_numpy(syn_latents_mus).float().to(device), return_intermediate=False)
        recon_latents = recon_latents.permute(0, 2, 3, 1).reshape(-1, C)  # (N * H * W, C)
        
        recon_mse = F.mse_loss(recon_latents, test_latents.to(device), reduction='none').sum(dim=-1).reshape(test_latents_all[clsid].shape[0], 16, 16)
        recon_mse = F.interpolate(
            recon_mse.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze(1).cpu().numpy()  # (N, H, W)
        
        test_anom_masks = test_anom_masks_all[clsid].squeeze(1)  # (N, H, W)
        test_anom_masks = F.interpolate(
            test_anom_masks.unsqueeze(1).float(), size=(224, 224), mode='nearest'
        ).squeeze(1).long().numpy().astype(np.uint8)  # (N, H, W)
        px_metrics = calculate_px_metrics(
            gt_masks=test_anom_masks,
            pred_scores=recon_mse,
            metrics=["px_aupro", "px_auroc", "px_f1", "px_ap"]
        )
        logger.info(f"Class {clsid}: Performance: {px_metrics}")
        
        # compute parameters of xi-distribution of the normal and abnormal score
        normal_scores = recon_mse[test_anom_masks < 0.5]
        abnormal_scores = recon_mse[test_anom_masks > 0.5]
        logger.info(f"Class {clsid}: Normal scores: mean={normal_scores.mean():.4f}, std={normal_scores.std():.4f}, min={normal_scores.min():.4f}, max={normal_scores.max():.4f}")
        logger.info(f"Class {clsid}: Abnormal scores: mean={abnormal_scores.mean():.4f}, std={abnormal_scores.std():.4f}, min={abnormal_scores.min():.4f}, max={abnormal_scores.max():.4f}")
        
        timesteps = np.linspace(0, 1, num_traj_points) # (num_traj_points,)
        traj_noise = np.random.randn(len(cls_latents), gmm.means_.shape[1])  # (N*H*W, C)
        
        rng = np.random.default_rng(seed)
        base_noise = rng.standard_normal(size=test_latents_np.shape)

        t_sp, trajectory_tsp, coverage = estimate_persistent_entry_tsp(
            z1=test_latents_np,
            base_noise=base_noise,
            gmm=gmm,
            t_grid=timesteps,
            epsilon=epsilon,
            delta=delta,
        )
        logger.info(f"Class {clsid}: Persistent speciation time t_sp: {t_sp:.4f}, Coverage: {coverage}")
    
    # -- save results
    if is_main():
        results_save_path = os.path.join(log_dir, 'analysis_results.yaml')
        with open(results_save_path, 'w') as f:
            yaml.dump(analysis_results, f)
        logger.info(f"Saved analysis results to {results_save_path}")
        
    # -- close distributed process
    dist.barrier()
    dist.destroy_process_group()

    # -- end of main
    logger.info(f"Analysis completed. Analysis results are saved at {log_dir}")
    
def estimate_persistent_entry_tsp(
    z1: np.ndarray,
    base_noise: np.ndarray,
    gmm,
    t_grid: np.ndarray,
    epsilon: float = 0.1,
    delta: float = 0.05,
):
    """
    Estimate persistent posterior-speciation times along

        Z_t = t Z_1 + (1-t) E.

    A trajectory is considered speciated from t onward when:
      1. the argmax cluster equals its endpoint cluster;
      2. the posterior probability of that endpoint cluster is >= 1-epsilon;
      3. both conditions hold at every later grid point.

    Returns
    -------
    t_sp:
        Dataset-level (1-delta)-coverage speciation time.
    trajectory_tsp:
        Per-trajectory persistent speciation time. inf means failure.
    coverage:
        coverage[j] = P_hat[T <= t_grid[j]].
    """
    z1 = np.asarray(z1, dtype=np.float64)
    base_noise = np.asarray(base_noise, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)

    if z1.shape != base_noise.shape:
        raise ValueError("z1 and base_noise must have identical shapes.")

    if z1.ndim != 2:
        raise ValueError(f"Expected z1 with shape [N, d], got {z1.shape}.")

    if np.any(np.diff(t_grid) <= 0):
        raise ValueError("t_grid must be strictly increasing.")

    if not np.isclose(t_grid[-1], 1.0):
        raise ValueError("t_grid must end at t=1.")

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie in (0, 1).")

    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1).")

    n = len(z1)
    sample_indices = np.arange(n)

    # Endpoint cluster k_i^*.
    posterior_end = compute_cluster_posterior_diag(z1, 1.0, gmm)
    endpoint_clusters = np.argmax(posterior_end, axis=1)
    endpoint_probabilities = posterior_end[
        sample_indices,
        endpoint_clusters,
    ]

    # A trajectory whose endpoint is not concentrated can never satisfy
    # the persistent condition.
    active = endpoint_probabilities >= 1.0 - epsilon

    trajectory_tsp = np.full(n, np.inf, dtype=np.float64)
    trajectory_tsp[active] = 1.0

    # Scan backward from the endpoint.
    for t in t_grid[-2::-1]:
        z_t = t * z1 + (1.0 - t) * base_noise
        posterior = compute_cluster_posterior_diag(z_t, float(t), gmm)

        current_clusters = np.argmax(posterior, axis=1)
        endpoint_cluster_probabilities = posterior[
            sample_indices,
            endpoint_clusters,
        ]

        valid_at_t = (
            (current_clusters == endpoint_clusters)
            & (endpoint_cluster_probabilities >= 1.0 - epsilon)
        )

        # Once a later-time violation has occurred, an earlier point
        # cannot be a persistent entry point.
        active &= valid_at_t
        trajectory_tsp[active] = t

    # This is monotone in t by construction.
    coverage = np.asarray(
        [(trajectory_tsp <= t).mean() for t in t_grid],
        dtype=np.float64,
    )

    valid_indices = np.flatnonzero(coverage >= 1.0 - delta)

    if valid_indices.size == 0:
        t_sp = np.inf
    else:
        t_sp = float(t_grid[valid_indices[0]])

    return t_sp, trajectory_tsp, coverage
    
def compute_cluster_posterior_diag(
    z: np.ndarray,
    t: float,
    gmm,
) -> np.ndarray:
    """
    Compute P(K=k | Z_t=z) for

        Z_t = t Z_1 + (1-t) E,
        Z_1 | K=k ~ N(mu_k, diag(cov_k)),
        E ~ N(0, I).

    Parameters
    ----------
    z:
        Array of shape [N, d].
    t:
        Scalar in [0, 1].
    gmm:
        Fitted sklearn GaussianMixture with covariance_type="diag".

    Returns
    -------
    posterior:
        Array of shape [N, K].
    """
    if gmm.covariance_type != "diag":
        raise ValueError(
            "compute_cluster_posterior_diag only supports covariance_type='diag'."
        )

    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be in [0, 1], got {t}.")

    z = np.asarray(z, dtype=np.float64)
    means = np.asarray(gmm.means_, dtype=np.float64)             # [K, d]
    covariances = np.asarray(gmm.covariances_, dtype=np.float64) # [K, d]
    weights = np.asarray(gmm.weights_, dtype=np.float64)         # [K]

    # Z_t | K=k ~ N(t mu_k, t^2 Sigma_k + (1-t)^2 I)
    means_t = t * means
    covariances_t = (
        t**2 * covariances
        + (1.0 - t)**2
    )

    if np.any(covariances_t <= 0.0):
        raise ValueError("Non-positive time-dependent covariance detected.")

    diff = z[:, None, :] - means_t[None, :, :]  # [N, K, d]

    log_joint = (
        np.log(np.clip(weights, np.finfo(np.float64).tiny, None))[None, :]
        - 0.5
        * np.sum(
            np.log(2.0 * np.pi * covariances_t)[None, :, :]
            + diff**2 / covariances_t[None, :, :],
            axis=-1,
        )
    )

    log_normalizer = logsumexp(log_joint, axis=1, keepdims=True)
    posterior = np.exp(log_joint - log_normalizer)

    return posterior

def dim_reduce_pca_orthogonal(X, n_components=2):
    """Perform PCA on the data X and return the reduced data and the PCA components.
    X: (N, D) data matrix
    n_components: number of components to keep
    Returns:
        X_reduced: (N, n_components) reduced data
        components: (n_components, D) PCA components
    """
    
    # Compute covariance matrix
    cov_matrix = np.cov(X, rowvar=False)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select top n_components
    components = sorted_eigenvectors[:, :n_components]
    
    # Project data onto the new basis
    X_reduced = np.dot(X, components)
    
    return X_reduced, components
    

def fit_gmm(train_z, val_z, n_components=4, cov_type='diagonal', max_iter=100, tol=1e-3):
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, max_iter=max_iter, tol=tol)
    gmm.fit(train_z)
    val_scores = gmm.score_samples(val_z)
    return gmm, val_scores

def compute_stoch_interpolation(z, t, noise):
    return t * z + (1 - t) * noise

def compute_fm_posterior(t, sigma, mu):
    """Compute the FM posterior mean and covariance given the cluster
    X_t | K=k ~ N(t*mu_k, t^2*sigma_k^2 + (1-t)^2*I)
    """
    cov = t**2 * sigma**2 + (1-t)**2
    mean = t * mu
    return mean, cov

def compute_cluster_posterior(z, t, cov_all, mu_all, class_priors):
    """Compute the posterior probability of each cluster given the latent z and time t.
    \gamma(z, t) = P(K=k | z, t),
    z: (N, C) latent features
    t: (,)
    cov_all: (K, C) cluster variance
    mu_all: (K, C) cluster mean
    class_priors: (K,) prior probability of each cluster
    """
    
    diff = z[:, None, :] - mu_all[None, :, :]  # (N, K, C)
    mahalanobis_sq = np.sum(
        diff ** 2 / cov_all[None, :, :],
        axis=-1
    )  # (N, K)
    log_det_cov = np.sum(np.log(cov_all), axis=-1) # (K,)
    log_joint = (
        np.log(class_priors)[None, :]
        - 0.5 
        * (
            z.shape[1] * np.log(2 * np.pi)
            + log_det_cov[None, :]
            + mahalanobis_sq
        )
    )
    
    log_posterior = (
        log_joint
        - logsumexp(log_joint, axis=-1, keepdims=True)
    )
    return np.exp(log_posterior)

class LocalizedClosedFM(nn.Module):
    def __init__(
        self,
        train_samples,
        query_batch_size=4096,
        ref_batch_size=4096,
        store_dtype=None,
        bandwidth_min=0.,
    ):
        super().__init__()

        # assert train_samples.ndim == 4
        # N, C, H, W = train_samples.shape

        # # Important:
        # # (N, C, H, W) -> (N, H, W, C) -> (N*H*W, C)
        # train_flat = (
        #     train_samples
        #     .detach()
        #     .permute(0, 2, 3, 1)
        #     .contiguous()
        #     .view(-1, C)
        # )
        train_flat = train_samples.detach()

        if store_dtype is not None:
            train_flat = train_flat.to(store_dtype)

        # model.to(device) で一緒に移動する
        self.register_buffer("train_samples", train_flat, persistent=False)

        self.query_batch_size = query_batch_size
        self.ref_batch_size = ref_batch_size
        
        self.bandwidth_min = bandwidth_min
        
    @staticmethod
    def perturb(x, t, noise):
        """
        Computes

            x_t = t x + (1-t) noise
        """
        return t * x + (1 - t) * noise

    @torch.no_grad()
    def get_optimal_v(self, xt, t):
        """
        Computes

            v(x_t, t) = ( E_w[x_ref] - x_t ) / (1 - t)

        where

            w_i ∝ exp( -||x_t - t x_ref_i||^2 / (2(1-t)^2) )

        without materializing w of shape (B*H*W, N*H*W).
        """

        assert xt.ndim == 4
        B, C, H, W = xt.shape
        device = xt.device

        assert torch.all(t == t[0]), \
            "All t values must be the same for computing weights."

        t0 = t.flatten()[0].to(device=device, dtype=torch.float32)
        one_minus_t = 1.0 - t0

        if one_minus_t <= 0:
            raise ValueError("t must be strictly smaller than 1.")

        temp = 2.0 * (one_minus_t ** 2)

        # (B, C, H, W) -> (B*H*W, C)
        x_query_all = (
            xt
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, C)
        )

        x_ref_all = self.train_samples
        if x_ref_all.device != device:
            raise RuntimeError(
                "train_samples is on a different device. "
                "Call model.to(xt.device) before sampling."
            )

        Q = x_query_all.shape[0]
        R = x_ref_all.shape[0]

        out = torch.empty((Q, C), device=device, dtype=xt.dtype)

        for q_start in range(0, Q, self.query_batch_size):
            q_end = min(q_start + self.query_batch_size, Q)

            q = x_query_all[q_start:q_end].to(torch.float32)
            q_size = q.shape[0]

            # Online log-sum-exp state
            # m: running max logit
            # z: running sum exp(logit - m)
            # s: running weighted sum of x_ref
            m = torch.full(
                (q_size, 1),
                -torch.inf,
                device=device,
                dtype=torch.float32,
            )
            z = torch.zeros(
                (q_size, 1),
                device=device,
                dtype=torch.float32,
            )
            s = torch.zeros(
                (q_size, C),
                device=device,
                dtype=torch.float32,
            )

            q_norm = (q * q).sum(dim=1, keepdim=True)

            for r_start in range(0, R, self.ref_batch_size):
                r_end = min(r_start + self.ref_batch_size, R)

                r = x_ref_all[r_start:r_end].to(torch.float32)

                r_norm = (r * r).sum(dim=1).unsqueeze(0)

                # score = -||q - t r||^2 / temp
                #
                # ||q - t r||^2
                # = ||q||^2 - 2t q r^T + t^2 ||r||^2
                #
                # score
                # = (-||q||^2 + 2t q r^T - t^2 ||r||^2) / temp
                scores = (
                    2.0 * t0 * (q @ r.T)
                    - q_norm
                    - t0.square() * r_norm
                ) / temp

                chunk_max = scores.max(dim=1, keepdim=True).values
                m_new = torch.maximum(m, chunk_max)

                old_scale = torch.exp(m - m_new)

                # scores becomes exp(scores - m_new)
                scores.sub_(m_new)
                scores.exp_()

                z = z * old_scale + scores.sum(dim=1, keepdim=True)
                s = s * old_scale + scores @ r

                m = m_new

                # optional: release references earlier
                del r, r_norm, scores, chunk_max, m_new, old_scale

            mean_ref = s / z.clamp_min(1e-30)

            v = (mean_ref - q) / one_minus_t
            out[q_start:q_end] = v.to(dtype=xt.dtype)

            del q, q_norm, m, z, s, mean_ref, v

        v_opt = (
            out
            .view(B, H, W, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return v_opt

    @torch.no_grad()
    def sample(
        self,
        x0,
        steps: int,
        start_t: float = 0.0,
        return_intermediate: bool = False,
        **model_kwargs,
    ):
        timesteps = torch.linspace(
            start_t,
            1.0,
            steps + 1,
            device=x0.device,
            dtype=x0.dtype,
        )

        step_size = (1.0 - start_t) / steps

        x_in = x0
        v_traj = []

        for i in range(steps):
            t_in = timesteps[i].expand(x_in.shape[0])
            opt_v = self.get_optimal_v(x_in, t_in)

            x_in = x_in + opt_v * step_size

            if return_intermediate:
                v_traj.append(opt_v)

        if return_intermediate:
            return x_in, v_traj

        return x_in
    
class RidgeLocalizedClosedFM(nn.Module):
    """
    Localized closed-form flow matching with optional KRR regularization
    of the velocity field itself.

    At a fixed t, fixed anchor patches Z_t={z_a(t)} are drawn once from
    the empirical path by

        z_a(t) = t * x_ref[i_a] + (1-t) * eps_a.

    Their exact closed-form velocities Y_t=u*(Z_t,t) are computed using
    all reference patches. The regularized field is then

        u_lambda(q,t)
            = k(q,Z_t) [K(Z_t,Z_t) + M lambda_t I]^{-1} Y_t,

    where M is the number of anchors. This is finite-sample KRR for the
    average-loss objective

        (1/M) sum_a ||f(z_a)-Y_a||^2 + lambda_t ||f||_H^2.

    ridge_lambda=None disables KRR and exactly recovers the original
    implementation. ridge_lambda=0.0 uses the unregularized kernel
    interpolant on the fixed anchors (plus numerical jitter), which is
    generally not identical to the original field away from the anchors.
    """

    def __init__(
        self,
        train_samples: torch.Tensor,
        query_batch_size: int = 4096,
        ref_batch_size: int = 4096,
        store_dtype: Optional[torch.dtype] = None,
        bandwidth_min: float = 0.0,
        *,
        ridge_lambda: Optional[float] = None,
        ridge_num_anchors: int = 512,
        ridge_anchor_batch_size: int = 2048,
        ridge_kernel_bandwidth: Optional[float] = None,
        ridge_kernel_scale: float = 1.0,
        ridge_time_power: float = 0.0,
        ridge_jitter: float = 1e-6,
        ridge_seed: int = 0,
        ridge_cache_size: int = 0,
        ridge_distance_normalize: bool = False,
        ridge_kernel_bias: float = 0.0,
        ridge_feature_map: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if train_samples.ndim != 2:
            raise ValueError(
                "train_samples must have shape (num_reference_patches, patch_dim). "
                f"Got {tuple(train_samples.shape)}."
            )

        train_flat = train_samples.detach()
        if store_dtype is not None:
            train_flat = train_flat.to(store_dtype)
        self.register_buffer("train_samples", train_flat, persistent=False)

        if query_batch_size <= 0 or ref_batch_size <= 0:
            raise ValueError("query_batch_size and ref_batch_size must be positive.")
        if bandwidth_min < 0:
            raise ValueError("bandwidth_min must be non-negative.")
        if ridge_lambda is not None and ridge_lambda < 0:
            raise ValueError("ridge_lambda must be non-negative or None.")
        if ridge_num_anchors <= 0:
            raise ValueError("ridge_num_anchors must be positive.")
        if ridge_anchor_batch_size <= 0:
            raise ValueError("ridge_anchor_batch_size must be positive.")
        if ridge_kernel_bandwidth is not None and ridge_kernel_bandwidth <= 0:
            raise ValueError("ridge_kernel_bandwidth must be positive or None.")
        if ridge_kernel_scale <= 0:
            raise ValueError("ridge_kernel_scale must be positive.")
        if ridge_time_power < 0:
            raise ValueError("ridge_time_power must be non-negative.")
        if ridge_jitter <= 0:
            raise ValueError("ridge_jitter must be positive.")
        if ridge_cache_size < 0:
            raise ValueError("ridge_cache_size must be non-negative.")
        if ridge_kernel_bias < 0:
            raise ValueError("ridge_kernel_bias must be non-negative.")

        self.query_batch_size = int(query_batch_size)
        self.ref_batch_size = int(ref_batch_size)
        self.bandwidth_min = float(bandwidth_min)

        self.ridge_lambda = None if ridge_lambda is None else float(ridge_lambda)
        self.ridge_num_anchors = int(ridge_num_anchors)
        self.ridge_anchor_batch_size = int(ridge_anchor_batch_size)
        self.ridge_kernel_bandwidth = (
            None if ridge_kernel_bandwidth is None
            else float(ridge_kernel_bandwidth)
        )
        self.ridge_kernel_scale = float(ridge_kernel_scale)
        self.ridge_time_power = float(ridge_time_power)
        self.ridge_jitter = float(ridge_jitter)
        self.ridge_seed = int(ridge_seed)
        self.ridge_cache_size = int(ridge_cache_size)
        self.ridge_distance_normalize = bool(ridge_distance_normalize)
        self.ridge_kernel_bias = float(ridge_kernel_bias)
        self.ridge_feature_map = (
            nn.Identity() if ridge_feature_map is None else ridge_feature_map
        )

        num_refs, patch_dim = train_flat.shape
        if num_refs == 0:
            raise ValueError("train_samples must contain at least one reference patch.")

        # Fixed Monte Carlo anchors. Sampling with replacement allows more than
        # one noise realization per reference patch and permits M > num_refs.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.ridge_seed)
        anchor_indices = torch.randint(
            low=0,
            high=num_refs,
            size=(self.ridge_num_anchors,),
            generator=generator,
            dtype=torch.long,
        )
        anchor_noise = torch.randn(
            self.ridge_num_anchors,
            patch_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self.register_buffer(
            "ridge_anchor_indices",
            anchor_indices,
            persistent=False,
        )
        self.register_buffer(
            "ridge_anchor_noise",
            anchor_noise,
            persistent=False,
        )

        # Maps a rounded scalar t to (anchor_features, alpha). These tensors are
        # intentionally not buffers, so the cache is cleared whenever .to(),
        # .cuda(), .cpu(), etc. are applied.
        self._ridge_cache: OrderedDict[
            tuple[str, int, float], tuple[torch.Tensor, torch.Tensor]
        ] = OrderedDict()

    def _apply(self, fn):
        self.clear_ridge_cache()
        return super()._apply(fn)

    def clear_ridge_cache(self) -> None:
        self._ridge_cache.clear()

    @staticmethod
    def perturb(x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Computes x_t = t x + (1-t) noise."""
        return t * x + (1.0 - t) * noise

    def _path_bandwidth(self, one_minus_t: torch.Tensor) -> torch.Tensor:
        if self.bandwidth_min == 0.0:
            return one_minus_t
        return one_minus_t.clamp_min(self.bandwidth_min)

    def _effective_ridge(self, one_minus_t: torch.Tensor) -> torch.Tensor:
        if self.ridge_lambda is None:
            raise RuntimeError("KRR is disabled because ridge_lambda is None.")

        lam = torch.as_tensor(
            self.ridge_lambda,
            device=one_minus_t.device,
            dtype=torch.float32,
        )
        if self.ridge_time_power != 0.0:
            lam = lam * one_minus_t.to(torch.float32).pow(self.ridge_time_power)
        return lam

    def _ridge_bandwidth(self, one_minus_t: torch.Tensor) -> torch.Tensor:
        if self.ridge_kernel_bandwidth is not None:
            return torch.as_tensor(
                self.ridge_kernel_bandwidth,
                device=one_minus_t.device,
                dtype=torch.float32,
            )
        return (
            self._path_bandwidth(one_minus_t.to(torch.float32))
            * self.ridge_kernel_scale
        )

    def _ridge_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.ridge_feature_map(x)
        if features.ndim != 2 or features.shape[0] != x.shape[0]:
            raise ValueError(
                "ridge_feature_map must map (N, patch_dim) to (N, feature_dim). "
                f"Input={tuple(x.shape)}, output={tuple(features.shape)}."
            )
        return features.to(torch.float32)

    def _gaussian_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        sigma2 = sigma.to(torch.float32).square().clamp_min(
            torch.finfo(torch.float32).tiny
        )

        x_norm = (x * x).sum(dim=1, keepdim=True)
        y_norm = (y * y).sum(dim=1).unsqueeze(0)
        dist2 = x_norm + y_norm - 2.0 * (x @ y.T)
        dist2.clamp_min_(0.0)

        if self.ridge_distance_normalize:
            dist2.div_(max(x.shape[1], 1))

        kernel = torch.exp(-0.5 * dist2 / sigma2)
        if self.ridge_kernel_bias != 0.0:
            kernel.add_(self.ridge_kernel_bias)
        return kernel

    def _solve_spd(
        self,
        matrix: torch.Tensor,
        rhs: torch.Tensor,
        max_tries: int = 6,
    ) -> torch.Tensor:
        n = matrix.shape[0]
        eye = torch.eye(n, device=matrix.device, dtype=matrix.dtype)
        jitter = self.ridge_jitter

        for _ in range(max_tries):
            chol, info = torch.linalg.cholesky_ex(
                matrix + jitter * eye,
                check_errors=False,
            )
            if int(info.max().item()) == 0:
                return torch.cholesky_solve(rhs, chol)
            jitter *= 10.0

        raise RuntimeError(
            "KRR Gram matrix was not numerically positive definite. "
            f"Cholesky still failed at jitter={jitter / 10.0:.3e}. "
            "Increase ridge_lambda, ridge_jitter, or the kernel bandwidth."
        )

    @torch.no_grad()
    def _closed_form_flat(
        self,
        x_query_all: torch.Tensor,
        t0: torch.Tensor,
    ) -> torch.Tensor:
        """Original localized closed-form velocity for a flat (Q,C) query."""
        if x_query_all.ndim != 2:
            raise ValueError("x_query_all must have shape (Q, patch_dim).")

        device = x_query_all.device
        one_minus_t = 1.0 - t0
        sigma_path = self._path_bandwidth(one_minus_t)
        temp = 2.0 * sigma_path.square()

        x_ref_all = self.train_samples
        if x_ref_all.device != device:
            raise RuntimeError(
                "train_samples is on a different device. "
                "Call model.to(query.device) before evaluation."
            )
        if x_ref_all.shape[1] != x_query_all.shape[1]:
            raise ValueError(
                "Reference and query patch dimensions differ: "
                f"{x_ref_all.shape[1]} vs {x_query_all.shape[1]}."
            )

        q_total, patch_dim = x_query_all.shape
        r_total = x_ref_all.shape[0]
        out = torch.empty(
            q_total,
            patch_dim,
            device=device,
            dtype=torch.float32,
        )

        for q_start in range(0, q_total, self.query_batch_size):
            q_end = min(q_start + self.query_batch_size, q_total)
            q = x_query_all[q_start:q_end].to(torch.float32)
            q_size = q.shape[0]

            m = torch.full(
                (q_size, 1),
                -torch.inf,
                device=device,
                dtype=torch.float32,
            )
            z = torch.zeros(
                (q_size, 1),
                device=device,
                dtype=torch.float32,
            )
            s = torch.zeros(
                (q_size, patch_dim),
                device=device,
                dtype=torch.float32,
            )
            q_norm = (q * q).sum(dim=1, keepdim=True)

            for r_start in range(0, r_total, self.ref_batch_size):
                r_end = min(r_start + self.ref_batch_size, r_total)
                r = x_ref_all[r_start:r_end].to(torch.float32)
                r_norm = (r * r).sum(dim=1).unsqueeze(0)

                scores = (
                    2.0 * t0 * (q @ r.T)
                    - q_norm
                    - t0.square() * r_norm
                ) / temp

                chunk_max = scores.max(dim=1, keepdim=True).values
                m_new = torch.maximum(m, chunk_max)
                old_scale = torch.exp(m - m_new)

                scores.sub_(m_new).exp_()
                z = z * old_scale + scores.sum(dim=1, keepdim=True)
                s = s * old_scale + scores @ r
                m = m_new

            mean_ref = s / z.clamp_min(1e-30)
            out[q_start:q_end] = (mean_ref - q) / one_minus_t

        return out

    def _cache_key(self, t0: torch.Tensor) -> tuple[str, int, float]:
        device = t0.device
        device_index = -1 if device.index is None else int(device.index)
        # Sampling timesteps are usually generated by linspace; rounding avoids
        # separate entries caused only by float representation noise.
        t_value = round(float(t0.detach().cpu().item()), 12)
        return (device.type, device_index, t_value)

    @torch.no_grad()
    def _build_ridge_state(
        self,
        t0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        one_minus_t = 1.0 - t0
        lambda_t = self._effective_ridge(one_minus_t)
        sigma = self._ridge_bandwidth(one_minus_t)

        ref = self.train_samples.index_select(0, self.ridge_anchor_indices.to(device=t0.device))
        ref = ref.to(torch.float32)
        noise = self.ridge_anchor_noise.to(device=t0.device, dtype=torch.float32)
        anchors = t0 * ref + one_minus_t * noise

        targets = self._closed_form_flat(anchors, t0)
        anchor_features = self._ridge_features(anchors)

        gram = self._gaussian_kernel(anchor_features, anchor_features, sigma)
        reg = anchors.shape[0] * lambda_t
        gram.diagonal().add_(reg)
        alpha = self._solve_spd(gram, targets)

        return anchor_features, alpha

    @torch.no_grad()
    def _get_ridge_state(
        self,
        t0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ridge_cache_size == 0:
            return self._build_ridge_state(t0)

        key = self._cache_key(t0)
        state = self._ridge_cache.get(key)
        if state is not None:
            self._ridge_cache.move_to_end(key)
            return state

        state = self._build_ridge_state(t0)
        self._ridge_cache[key] = state
        self._ridge_cache.move_to_end(key)

        while len(self._ridge_cache) > self.ridge_cache_size:
            self._ridge_cache.popitem(last=False)
        return state

    @torch.no_grad()
    def _predict_ridge_flat(
        self,
        x_query_all: torch.Tensor,
        t0: torch.Tensor,
    ) -> torch.Tensor:
        anchor_features, alpha = self._get_ridge_state(t0)
        sigma = self._ridge_bandwidth(1.0 - t0)

        q_total = x_query_all.shape[0]
        patch_dim = alpha.shape[1]
        out = torch.empty(
            q_total,
            patch_dim,
            device=x_query_all.device,
            dtype=torch.float32,
        )

        for q_start in range(0, q_total, self.query_batch_size):
            q_end = min(q_start + self.query_batch_size, q_total)
            q = x_query_all[q_start:q_end].to(torch.float32)
            q_features = self._ridge_features(q)

            pred = torch.zeros(
                q.shape[0],
                patch_dim,
                device=q.device,
                dtype=torch.float32,
            )
            for a_start in range(
                0,
                anchor_features.shape[0],
                self.ridge_anchor_batch_size,
            ):
                a_end = min(
                    a_start + self.ridge_anchor_batch_size,
                    anchor_features.shape[0],
                )
                k_qa = self._gaussian_kernel(
                    q_features,
                    anchor_features[a_start:a_end],
                    sigma,
                )
                pred.add_(k_qa @ alpha[a_start:a_end])

            out[q_start:q_end] = pred

        return out

    @torch.no_grad()
    def get_optimal_v(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if xt.ndim != 4:
            raise ValueError(f"xt must have shape (B,C,H,W), got {tuple(xt.shape)}.")

        batch, channels, height, width = xt.shape
        t_flat = t.reshape(-1)
        if t_flat.numel() == 0:
            raise ValueError("t must contain at least one value.")
        if not torch.all(t_flat == t_flat[0]):
            raise ValueError("All t values must be identical.")

        t0 = t_flat[0].to(device=xt.device, dtype=torch.float32)
        if float((1.0 - t0).item()) <= 0.0:
            raise ValueError("t must be strictly smaller than 1.")

        x_query_all = (
            xt.permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, channels)
        )

        if self.ridge_lambda is None:
            v_flat = self._closed_form_flat(x_query_all, t0)
        else:
            v_flat = self._predict_ridge_flat(x_query_all, t0)

        return (
            v_flat.to(dtype=xt.dtype)
            .view(batch, height, width, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor,
        steps: int,
        start_t: float = 0.0,
        return_intermediate: bool = False,
        **model_kwargs,
    ):
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if not (0.0 <= start_t < 1.0):
            raise ValueError("start_t must satisfy 0 <= start_t < 1.")

        timesteps = torch.linspace(
            start_t,
            1.0,
            steps + 1,
            device=x0.device,
            dtype=x0.dtype,
        )
        step_size = (1.0 - start_t) / steps

        x_in = x0
        v_traj = []
        for i in range(steps):
            t_in = timesteps[i].expand(x_in.shape[0])
            opt_v = self.get_optimal_v(x_in, t_in)
            x_in = x_in + opt_v * step_size
            if return_intermediate:
                v_traj.append(opt_v)

        if return_intermediate:
            return x_in, v_traj
        return x_in