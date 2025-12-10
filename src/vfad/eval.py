import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.utils.dist import concat_all_gather
from src.flow_matching import VelocityField

from sklearn.metrics import roc_auc_score, average_precision_score
from src.utils.adeval.adeval import EvalAccumulatorCuda, metrics_dist

import logging
logger = logging.getLogger(__name__)

SUPPORTED_METRICS = [
    'img_auroc', 'img_aupr', 'img_f1max', 'img_ap',
    'px_auroc', 'px_aupr', 'px_f1max', 'px_ap', 'px_pro'
]

@torch.no_grad()
def f1_max_gpu_hist(scores: torch.Tensor,
                    labels: torch.Tensor,
                    n_bins: int = 1001,
                    eps: float = 1e-8,
                    distributed: bool = False):
    """
    Memory-efficient F1-max on GPU.
    scores : (N,)  float32/float16,  already in [0,1]
    labels : (N,)  bool / {0,1} tensor   (1=anomaly)
    n_bins : number of threshold bins (≥2)
    eps    : numerical stabiliser
    distributed : if True, gather scores/labels from all processes
    """
    # min-max normalize scores to [0, 1]
    scores = (scores - scores.min()) / (scores.max() - scores.min() + eps)  # (N,)
    scores = torch.clamp(scores, 0.0, 1.0 - eps)
    bin_idx = (scores * (n_bins - 1)).long()

    pos_per_bin = torch.zeros(n_bins, device=scores.device, dtype=torch.int64)
    neg_per_bin = torch.zeros_like(pos_per_bin)

    labels_bool = labels.bool()
    pos_per_bin.scatter_add_(0, bin_idx[labels_bool], torch.ones_like(bin_idx[labels_bool]))
    neg_per_bin.scatter_add_(0, bin_idx[~labels_bool], torch.ones_like(bin_idx[~labels_bool]))

    tp_cum = pos_per_bin.flip(0).cumsum(0).flip(0).to(torch.float32)
    fp_cum = neg_per_bin.flip(0).cumsum(0).flip(0).to(torch.float32)

    total_pos = tp_cum[0]                          
    fn_cum   = total_pos - tp_cum

    denom = 2 * tp_cum + fp_cum + fn_cum + eps
    f1 = (2 * tp_cum) / denom                     # (T,)

    best = torch.argmax(f1)
    thr  = best / (n_bins - 1)                    

    return f1[best], thr

@torch.no_grad()
def extract_features(
    fe: torch.nn.Module,
    imgs: torch.Tensor,
    device: torch.device,
):
    """Extract features using the feature extractor.
    Args:
        fe: Feature extractor model.
        imgs: Input images tensor of shape (B, C, H, W).
        device: Device to perform computation on.
    Returns:
        torch.Tensor: Extracted features tensor.
    """
    imgs = imgs.to(device, non_blocking=True)
    features = fe(imgs)
    return features

def aggregate_px_values(
    agg_method: str,
    px_values: np.ndarray,
):
    """Aggregate pixel-wise values to a single image-level score.
    Args:
        agg_method: Aggregation method ('diff', 'max', 'mean', 'median').
        px_values: Pixel-wise values of shape (N, H, W).
    Returns:
        np.ndarray: Aggregated image-level scores of shape (N,).
    """
    if agg_method == 'diff':
        scores_min = px_values.reshape(px_values.shape[0], -1).min(axis=(1, 2))
        scores_max = px_values.reshape(px_values.shape[0], -1).max(axis=(1, 2))
        scores = scores_max - scores_min
    elif agg_method == 'sum':
        scores = px_values.reshape(px_values.shape[0], -1).sum(axis=(1, 2))
    elif agg_method == 'max':
        scores = px_values.reshape(px_values.shape[0], -1).max(axis=(1, 2))
    elif agg_method == 'mean':
        scores = px_values.reshape(px_values.shape[0], -1).mean(axis=(1, 2))
    elif agg_method == 'median':
        scores = np.median(px_values.reshape(px_values.shape[0], -1), axis=(1, 2))
    elif agg_method == 'diff+sum':
        scores_min = px_values.reshape(px_values.shape[0], -1).min(axis=(1, 2))
        scores_max = px_values.reshape(px_values.shape[0], -1).max(axis=(1, 2))
        scores_diff = scores_max - scores_min
        scores_sum = px_values.reshape(px_values.shape[0], -1).sum(axis=(1, 2))
        normalized_diff = (scores_diff - scores_diff.min()) / (scores_diff.max() - scores_diff.min() + 1e-8)
        normalized_sum = (scores_sum - scores_sum.min()) / (scores_sum.max() - scores_sum.min() + 1e-8)
        scores = normalized_diff + normalized_sum
    else:
        raise ValueError(f"Unknown aggregation method: {agg_method}")
    return scores

def calculate_img_metrics(
    gt_labels: np.ndarray,
    pred_scores: np.ndarray,
    metrics: list,
):
    """Calculate image-level anomaly detection metrics.
    Args:
        gt_labels: Ground truth labels of shape (N,).
        pred_scores: Predicted anomaly scores of shape (N,).
        metrics: List of metrics to compute ('auroc', 'aupr', 'f1').
    Returns:
        dict: Dictionary containing computed metrics.
    """
    results_dict = {}
    if 'img_auroc' in metrics:
        auroc = roc_auc_score(gt_labels, pred_scores)
        results_dict['img_auroc'] = auroc
    if 'img_aupr' in metrics or 'img_ap' in metrics:
        aupr = average_precision_score(gt_labels, pred_scores)
        if 'img_aupr' in metrics:
            results_dict['img_aupr'] = aupr
        if 'img_ap' in metrics:
            results_dict['img_ap'] = aupr
    if 'img_f1' in metrics:
        scores_tensor = torch.from_numpy(pred_scores).to(torch.float32).cuda()
        labels_tensor = torch.from_numpy(gt_labels).to(torch.float32).cuda()
        f1, _ = f1_max_gpu_hist(scores_tensor, labels_tensor)
        results_dict['img_f1max'] = f1.item()

def calculate_px_metrics(
    gt_masks: np.ndarray,
    pred_scores: np.ndarray,
    metrics: list,
    device: torch.device = torch.device('cuda'),
    distributed: bool = False,
    accum_size: int = 10000
):
    """Calculate pixel-level anomaly detection metrics.
    Args:
        gt_masks: Ground truth masks of shape (N, H, W).
        pred_scores: Predicted anomaly scores of shape (N, H, W).
        metrics: List of metrics to compute ('auroc', 'aupr', 'f1').
    Returns:
        dict: Dictionary containing computed metrics.
    """
    results_dict = {}
    gt_masks_flat = gt_masks.reshape(-1)
    pred_scores_flat = pred_scores.reshape(-1)
    score_min, score_max = pred_scores_flat.min(), pred_scores_flat.max()

    # -- use adeval for AUROC and AUPRO, AUPR
    nb = len(gt_masks_flat) // accum_size + (1 if len(gt_masks_flat) % accum_size != 0 else 0)
    evaluator = EvalAccumulatorCuda(score_min, score_max, score_min, score_max)
    for i in range(0, nb):
        start_idx = i * accum_size
        end_idx = min((i + 1) * accum_size, len(gt_masks_flat))
        evaluator.add_anomap_batch(
            torch.from_numpy(pred_scores[start_idx:end_idx]).to(device),
            torch.from_numpy(gt_masks[start_idx:end_idx]).to(device)
        )
    results = evaluator.summary()
    if 'px_auroc' in metrics:
        results_dict['px_auroc'] = results['p_auroc']
    if 'px_aupro' in metrics:
        results_dict['px_aupro'] = results['p_aupro']
    if 'px_aupr' in metrics:
        results_dict['px_aupr'] = results['p_aupr']
    
    # -- use skleran for AP
    if 'px_ap' in metrics:
        ap = average_precision_score(gt_masks_flat.astype(int), pred_scores_flat)
        results_dict['px_ap'] = ap
        
    # -- use f1_max_gpu_hist for F1-max
    if 'px_f1max' in metrics:
        scores_tensor = torch.from_numpy(pred_scores_flat).to(torch.float32).to(device)
        labels_tensor = torch.from_numpy(gt_masks_flat).to(torch.float32).to(device)
        f1, _ = f1_max_gpu_hist(scores_tensor, labels_tensor, distributed=distributed)
        results_dict['px_f1max'] = f1.item()
        
    return results_dict

def divide_by_class(array, labels):
    """Divide array by class labels.
    Args:
        array: Numpy array of (N, ...).
        labels: Numpy array of class labels (N,).
    Returns:
        dict: Dictionary mapping class label to corresponding array.
    """
    class_dict = {}
    for label in np.unique(labels):
        class_dict[label] = array[labels == label]
    return class_dict

@torch.no_grad()
def evaluate(
    vf: VelocityField, 
    fe: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device, 
    img_sz: tuple,
    verbose: bool = True,
    use_bfloat16: bool = False,
    distributed: bool = False,
    eval_params: dict = None,
):
    """Evaluate on the given dataset, returning AD metrics.
    Args:
        vf: VelocityField model for evaluation.
        dataloader: DataLoader providing the evaluation dataset.
        device: Device to perform evaluation on.
        img_sz: Size of the input images (H, W).
        verbose: Whether to display progress bar.
        use_bfloat16: Whether to use bfloat16 precision during evaluation.
        distributed: Whether to use distributed evaluation.
        eval_params (dict, optional): Additional evaluation parameters. Defaults to None.
    Returns:
        dict: Dictionary containing evaluation metrics (e.g., AUROC, AUPR, F1-score) for each class.
    """
    was_train = vf.model.training
    vf.model.eval()
    
    # -- evaluation configuration
    steps = eval_params.get('steps', 1) if eval_params is not None else 1
    solver_name = eval_params.get('solver_name', 'euler') if eval_params is not None else 'euler'
    solver_params = eval_params.get('solver_params', {}) if eval_params is not None else {}
    img_score_agg = eval_params.get('img_score_agg', 'diff') if eval_params is not None else 'diff'
    eval_metrics = eval_params.get('metrics', ['img_auroc', 'px_auroc']) if eval_params is not None else ['img_auroc', 'px_auroc']
    assert all([met in SUPPORTED_METRICS for met in eval_metrics]), f"Some evaluation metrics are not supported. Supported metrics: {SUPPORTED_METRICS}"

    logger.info(f"Starting evaluation with {steps} steps using {solver_name} solver.")
    
    # -- evaluation loop
    N = len(dataloader.dataset)
    masks_all = np.zeros((N, *img_sz), dtype=np.uint8)
    clslabels_all = np.zeros((N,), dtype=np.uint8)
    anom_labels_all = np.zeros((N,), dtype=np.uint8)
    vel_residual_all = np.zeros((N, *img_sz), dtype=np.float32)
    
    logger.info("Extracting features and computing anomaly scores...")
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not verbose):
        
        # -- prepare data
        imgs, clslabels_local = batch["img"], batch["clslabel"]    # (B, C, H, W), (B,)
        imgs, clslabels_local = imgs.to(device, non_blocking=True), clslabels_local.to(device, non_blocking=True)
        anom_labels_local, anom_masks_local = batch['clsname'], batch['mask'] # (B,), (B, H, W)
        anom_labels_local, anom_masks_local = anom_labels_local.to(device, non_blocking=True), anom_masks_local.to(device, non_blocking=True)
        
        # -- extract features
        z1 = extract_features(fe, imgs, device)  # (B, c, h, w)
        
        # -- inversion through the velocity field
        if use_bfloat16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z0_local = vf.sample(z1, clslabels_local, steps=steps, solver_name=solver_name, solver_params=solver_params)# (B, c, h, w)
        else:
            z0_local = vf.sample(z1, clslabels_local, steps=steps, solver_name=solver_name, solver_params=solver_params)  # (B, c, h, w)
        
        # -- share results across GPUs
        if distributed:
            z0 = concat_all_gather(z0_local)
            anom_labels = concat_all_gather(anom_labels_local)
            cls_labels = concat_all_gather(clslabels_local)
            anom_masks = concat_all_gather(anom_masks_local)
        else:
            z0 = z0_local
            anom_labels = anom_labels_local
            cls_labels = clslabels_local
            anom_masks = anom_masks_local
        
        # -- compute anomaly scores
        z0 = torch.norm(z0, dim=1)  # (N, h, w)
        z0 = F.interpolate(
            z0.unsqueeze(1), size=img_sz, mode='bilinear', align_corners=False
        ).squeeze(1)  # (N, H, W)
        vel_residual = z0.cpu().numpy()
        
        # -- store results
        bs = vel_residual.shape[0]
        vel_residual_all[step * bs: step * bs + bs] = vel_residual
        clslabels_all[step * bs: step * bs + bs] = cls_labels.cpu().numpy().astype(np.uint8)
        masks_all[step * bs: step * bs + bs] = anom_masks.cpu().numpy().astype(np.uint8)
        anom_labels_all[step * bs: step * bs + bs] = (anom_labels.cpu().numpy() > 0).astype(np.uint8)
        
    # -- compute anomaly scores
    px_scores = vel_residual_all  # (N, H, W)
    img_scores = aggregate_px_values(
        agg_method=img_score_agg,
        px_values=px_scores
    )  # (N,)
    px_gts = masks_all  # (N, H, W)
    img_gts = anom_labels_all  # (N,)
    
    # -- divide by class
    img_scores_by_class = divide_by_class(img_scores, clslabels_all)
    img_gts_by_class = divide_by_class(img_gts, clslabels_all)
    px_scores_by_class = divide_by_class(px_scores, clslabels_all)
    px_gts_by_class = divide_by_class(px_gts, clslabels_all)
    
    # -- compute image-level metrics
    logger.info("Calculating image-level metrics...")
    clsname_map = dataloader.dataset.labels_to_names
    eval_results = {v : {} for v in clsname_map.values()}
    img_metrics = [met for met in eval_metrics if met.startswith('img_')]
    for cls_label in img_scores_by_class.keys():
        cls_img_scores = img_scores_by_class[cls_label]
        cls_img_gts = img_gts_by_class[cls_label]
        cls_img_metrics = calculate_img_metrics(
            gt_labels=cls_img_gts,
            pred_scores=cls_img_scores,
            metrics=img_metrics,
        )
        cls_name = clsname_map[cls_label]
        for met_name, met_value in cls_img_metrics.items():
            eval_results[cls_name][met_name] = met_value
    
    # -- compute pixel-level metrics
    logger.info("Calculating pixel-level metrics...")
    px_metrics = [met for met in eval_metrics if met.startswith('px_')]
    for cls_label in px_scores_by_class.keys():
        cls_px_scores = px_scores_by_class[cls_label]
        cls_px_gts = px_gts_by_class[cls_label]
        cls_px_metrics = calculate_px_metrics(
            gt_masks=cls_px_gts,
            pred_scores=cls_px_scores,
            metrics=px_metrics,
        )
        cls_name = clsname_map[cls_label]
        for met_name, met_value in cls_px_metrics.items():
            eval_results[cls_name][met_name] = met_value
    
    logger.info(f"Evaluation completed. \\ Results: {eval_results}")
    return eval_results
    
    
    
    
    
    


        
        
        
        
        
        
        
    
    # -- compute metrics
    eval_metrics = {}
    
    # -- restore training state
    if was_train:
        vf.model.train()
    return eval_metrics
    