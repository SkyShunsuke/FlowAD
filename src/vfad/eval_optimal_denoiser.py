import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import os

import torch
import torch.distributed as dist
import numpy as np

import yaml
import logging
import pandas as pd

from src.models import init_model
from src.datasets import build_dataset
from src.backbones import get_backbone, get_backbone_feature_shape, get_normalization_func
from src.flow_matching import LocalizedClosedFM
from src.utils.distributed import init_distributed_mode, get_rank, get_world_size, concat_all_gather
from src.utils.distributed import is_main_process as is_main
from src.utils.log import setup_logging, get_logger

from src.utils.adeval.eval_utils import (
    calculate_img_metrics,
    calculate_px_metrics,
    divide_by_class,
    extract_features,
    aggregate_px_values,
    SUPPORTED_METRICS
)
from src.vfad.visualize import save_anomaly_maps, denormalize_image


import logging
logger = logging.getLogger(__name__)

def main(params, args):
    # init distributed mode
    init_distributed_mode(args)

    rank = get_rank()
    world_size = get_world_size()
    device = torch.device('cuda:%s'%args.gpu)
    
    # -- setup logging
    setup_logging(rank, world_size)
    logger = get_logger()
    logger.info(f"Using device: {device}, rank: {rank}, world_size: {world_size}")

    os.makedirs(params['logging']['log_dir'], exist_ok=True)
    
    # -- make logging stuff
    if is_main():
        log_dir = params['logging']['log_dir']
        
        # save config file
        config_save_path = os.path.join(log_dir, 'eval_config.yaml')
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
    logger.info(f"Extract latents from the backbone")
    
    all_latents = []
    for batch in tqdm(train_loader, desc="Extracting latents"):
        imgs_local, clslabels_local = batch["img"], batch["clslabel"]    # (B, C, H, W), (B,)
        imgs_local, clslabels_local = imgs_local.to(device, non_blocking=True), clslabels_local.to(device, non_blocking=True)
        with torch.no_grad():
            feats_local, _ = fe(imgs_local)    # (B, C, H', W')
            feats_local = norm_fn(feats_local)
            
        feats = concat_all_gather(feats_local)    # (B_total, C, H', W')
        all_latents.append(feats.cpu())
    all_latents = torch.cat(all_latents, dim=0)    # (N_total, C, H', W')
        
    vf = LocalizedClosedFM(all_latents).to(device)
    eval_params = params['logging']['eval']

    # -- evaluate
    logger.info(f"Starting evaluation...")
    ablation_steps = eval_params.get('ablation_steps', [-1])
    do_ablation = ablation_steps != [-1]
    
    if do_ablation:
        logger.info(f"Performing ablation study with steps: {ablation_steps}")
        for step in ablation_steps:
            logger.info(f"Evaluating with {step} steps...")
            eval_params["recon_steps"] = step
            eval_results = eval_recon(
                vf=vf,
                fe=fe,
                norm_fn=norm_fn,
                dataloader=test_loader,
                device=device,
                img_sz=(params['data']['img_size'], params['data']['img_size']),
                verbose=True,
                use_bfloat16=params['meta']['use_bfloat16'],
                distributed=args.distributed,
                save_anomaps=eval_params.get('save_anomaps', False),
                save_dir=eval_params.get('save_dir', None),
                eval_params=eval_params,
                noise_shape=feat_sz,
            )
            csv_path = os.path.join(eval_params.get('save_dir', '.'), f'eval_results_{step}.csv')
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            df = pd.DataFrame.from_dict(eval_results, orient='index')
            df.index.name = 'category'
            df.reset_index(inplace=True)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved evaluation results to {csv_path}")
    else:
        eval_results = eval_recon(
            vf=vf,
            fe=fe,
            norm_fn=norm_fn,
            dataloader=test_loader,
            device=device,
            img_sz=(params['data']['img_size'], params['data']['img_size']),
            verbose=True,
            use_bfloat16=params['meta']['use_bfloat16'],
            distributed=args.distributed,
            save_anomaps=eval_params.get('save_anomaps', False),
            save_dir=eval_params.get('save_dir', None),
            eval_params=eval_params,
            noise_shape=feat_sz,
        )
    
    # -- save results
    if is_main():
        results_save_path = os.path.join(log_dir, 'eval_results.yaml')
        with open(results_save_path, 'w') as f:
            yaml.dump(eval_results, f)
        logger.info(f"Saved evaluation results to {results_save_path}")
        
    # -- close distributed process
    dist.barrier()
    dist.destroy_process_group()

    # -- end of main
    logger.info(f"Evaluation completed. Evaluation results are saved at {log_dir}")
    
def eval_recon(
    vf, fe, norm_fn, dataloader, device, img_sz, verbose=True, use_bfloat16=False,
    distributed=False, save_anomaps=False, save_dir=None, eval_params=None, noise_shape=None,
):

    img_score_agg = eval_params.get('img_score_agg', 'diff') if eval_params is not None else 'diff'
    eval_metrics = eval_params.get('metrics', ['img_auroc', 'px_auroc']) if eval_params is not None else ['img_auroc', 'px_auroc']
    
    recon_steps = eval_params.get('recon_steps', 1) if eval_params is not None else 1
    stop_time = eval_params.get('stop_time', 0.0) if eval_params is not None else 0.0
    
    assert all([met in SUPPORTED_METRICS for met in eval_metrics]), f"Some evaluation metrics are not supported. Supported metrics: {SUPPORTED_METRICS}"
    
    logger.info(f"Starting evaluation with reconstruction steps: {recon_steps}.")
    logger.info(f"Reconstruction stop time: {stop_time}")

    # -- evaluation loop
    N = len(dataloader.dataset)
    masks_all = np.zeros((N, *img_sz), dtype=np.uint8)
    clslabels_all = np.zeros((N,), dtype=np.uint8)
    anom_types_all = np.zeros((N,), dtype=np.uint8)
    anom_labels_all = np.zeros((N,), dtype=np.uint8)
    mse_all = np.zeros((N, *img_sz), dtype=np.float32)
    org_imgs_all = np.zeros((N, img_sz[0], img_sz[1], 3), dtype=np.uint8)
    recon_imgs_all = np.zeros((N, img_sz[0], img_sz[1], 3), dtype=np.uint8)
        
    print("▶️Closed-form velocity fields constructed for each class.")
    
    logger.info("Extracting features and computing anomaly scores...")
    noise = torch.randn((1, *noise_shape), device=device)
    perturb_t = torch.tensor([stop_time], device=device)
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not verbose):
        
        # -- prepare data
        imgs_local, clslabels_local = batch["img"], batch["clslabel"]    # (B, C, H, W), (B,)
        imgs_local, clslabels_local = imgs_local.to(device, non_blocking=True), clslabels_local.to(device, non_blocking=True)
        anom_labels_local, anom_masks_local = batch['label'], batch['mask'] # (B,), (B, H, W)
        anom_labels_local, anom_masks_local = anom_labels_local.to(device, non_blocking=True), anom_masks_local.to(device, non_blocking=True)
        idx_local = batch["index"].to(device, non_blocking=True)  # (B,)
        
        # -- extract features
        z1, _ = extract_features(fe, imgs_local, device)  # (B, c, h, w)
        z1 = norm_fn(z1)
        
        # -- pertubation
        zt = vf.perturb(z1, noise.repeat(len(z1), 1, 1, 1), perturb_t.repeat(len(z1))) # (B, c, h, w)
        
        # -- reconstruction through the velocity field
        if use_bfloat16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pass
                z1_local = vf.sample(zt, steps=recon_steps, start_t=stop_time)
        else:
            z1_local = vf.sample(zt, steps=recon_steps, start_t=stop_time)
            
        # -- compute reconstruction error as anomaly score
        mse_local = F.mse_loss(z1_local, z1, reduction='none').sum(dim=1)  # (B, h, w)
        imgs_local = denormalize_image(imgs_local) * 255.0
        
        # -- share results across GPUs
        if distributed:
            mse = concat_all_gather(mse_local)
            anom_labels = concat_all_gather(anom_labels_local)
            cls_labels = concat_all_gather(clslabels_local)
            anom_masks = concat_all_gather(anom_masks_local)
            imgs = concat_all_gather(imgs_local)
            idx = concat_all_gather(idx_local)
        else:
            mse = mse_local
            anom_labels = anom_labels_local
            cls_labels = clslabels_local
            anom_masks = anom_masks_local
            imgs = imgs_local
            idx = idx_local
        
        # -- compute anomaly scores
        mse = F.interpolate(
            mse.unsqueeze(1), size=img_sz, mode='bilinear', align_corners=False
        ).squeeze(1)  # (N, H, W)
        mse_np = mse.cpu().numpy()
        
        # -- reshape masks
        anom_masks = anom_masks.squeeze(1)  # (N, H, W)
        anom_masks = F.interpolate(
            anom_masks.unsqueeze(1).float(), size=img_sz, mode='nearest'
        ).squeeze(1).long()  # (N, H, W)
        
        # -- store results
        idx_np = idx.cpu().numpy()
        mse_all[idx_np] = mse_np
        clslabels_all[idx_np] = cls_labels.cpu().numpy().astype(np.uint8)
        masks_all[idx_np] = anom_masks.cpu().numpy().astype(np.uint8)
        anom_labels_all[idx_np] = (anom_labels.cpu().numpy() > 0).astype(np.uint8)
    
    if distributed:
        torch.distributed.barrier()
    
    # -- compute anomaly scores
    px_scores = mse_all  # (N, H, W)
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

    clsname_map = dataloader.dataset.datasets[0].labels_to_names

    # -- save anomaly maps if required
    if save_anomaps and save_dir is not None and get_rank() == 0:
        save_anomaly_maps(
            save_dir=save_dir,
            img_scores_by_class=img_scores_by_class,
            img_gts_by_class=img_gts_by_class,
            px_scores_by_class=px_scores_by_class,
            px_gts_by_class=px_gts_by_class,
            org_img_by_class=org_imgs_by_class,
            class_map=clsname_map,
            recon_imgs_by_class=recon_imgs_by_class,
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


    logger.info("Calculating image-level metrics...")
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
        cls_px_gts = px_gts_by_class[cls_label]  # (N_cls, H, W)
        
        cls_px_metrics = {
            k: float(v)
            for k, v in calculate_px_metrics(
                gt_masks=cls_px_gts,
                pred_scores=cls_px_scores,
                metrics=px_metrics,
            ).items()
        }
        cls_name = clsname_map[cls_label]
        for met_name, met_value in cls_px_metrics.items():
            eval_results[cls_name][met_name] = met_value
        
    # -- average over classes
    eval_results['average'] = {}
    for cls_name in eval_results.keys():
        for met_name in eval_results[cls_name].keys():
            if cls_name == 'average':
                continue
            if met_name not in eval_results['average']:
                eval_results['average'][met_name] = 0.0
            eval_results['average'][met_name] += eval_results[cls_name][met_name]
    num_classes = len(dataloader.dataset.datasets)
    for met_name in eval_results['average'].keys():
        eval_results['average'][met_name] /= num_classes

    # -- calculate averaged metrics (mad) for all classes
    del_classes = []
    for cls_name in eval_results.keys():
        mad_sum = 0.0
        for met_name, met_value in eval_results[cls_name].items():
            mad_sum += met_value
        if len(eval_results[cls_name]) > 0:
            eval_results[cls_name]['mad'] = mad_sum / len(eval_results[cls_name])
        else:
            del_classes.append(cls_name)
    
    for cls_name in del_classes:
        del eval_results[cls_name]
    
    logger.info(f"Evaluation completed. \\ Results: {eval_results}")
    return eval_results