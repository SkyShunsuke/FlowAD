import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.utils.distributed import concat_all_gather
from src.flow_matching import VelocityField

from sklearn.metrics import roc_auc_score, average_precision_score
from src.utils.adeval import EvalAccumulatorCuda, f1_max_gpu_hist
from src.utils.adeval.eval_utils import (
    calculate_img_metrics,
    calculate_px_metrics,
    divide_by_class,
    extract_features,
    aggregate_px_values,
    SUPPORTED_METRICS
)

import os

import torch
import torch.distributed as dist
import numpy as np

import yaml
import logging

from src.models import init_model
from src.datasets import build_dataset
from src.backbones import get_backbone, get_backbone_feature_shape
from src.flow_matching import VelocityField
from src.utils.distributed import init_distributed_mode, get_rank, get_world_size
from src.utils.distributed import is_main_process as is_main
from src.utils.log import setup_logging, get_logger
from src.utils.opt.optimizer import load_model_only
from src.vfad.eval_inversion import evaluate_inv
from src.vfad.eval_density import evaluate_density
from src.vfad.eval_recon import evaluate_recon

import logging
logger = logging.getLogger(__name__)

def main(params, args):
    """XXX for VFAD evaluation.
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
    test_dataset = build_dataset(train=False, **params['data'])
    
    if isinstance(test_dataset, torch.utils.data.ConcatDataset):
        num_classes = len(test_dataset.datasets)
    else:
        num_classes = 1
    logger.info(f"Number of classes: {num_classes}")
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=test_dataset,
        num_replicas=world_size,
        rank=rank
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
    fe.to(device).eval()
    logger.info(f"Backbone {params['model']['backbone']['model_name']} initialized.")

    logger.info(f"Using input shape {feat_sz} for the flow matching.")
    model = init_model(input_sz=feat_sz, num_classes=num_classes, **params['model']).to(device)
    vf = VelocityField(
        model=model,
        input_sz=feat_sz,
        scheduler_name=params['flow_matching']['scheduler']['name'],
        solver_name=params['flow_matching']['solver']['name'],
        scheduler_params=params['flow_matching']['scheduler'].get('params', None),
        solver_params=params['flow_matching']['solver'].get('params', None),
    )
    logger.info(f"Velocity Field Model {params['model']['model_name']} has been initialized.")

    if args.distributed:
        vf = torch.nn.parallel.DistributedDataParallel(vf, static_graph=True)
    
    # -- resume training if needed
    resume_path = params['resume']['resume_path']
    assert resume_path is not None, "Please specify the checkpoint path for evaluation."
    assert os.path.isfile(resume_path), f"Resume path {resume_path} not found!, Please check the path."
    
    vf = load_model_only(
        resume_path, vf
    )
    logger.info(f"Resumed from checkpoint: {resume_path}")
    
    # -- evaluation setup
    eval_params = params['logging']['eval']
    eval_strategy = eval_params.get('eval_strategy', 'inversion') if eval_params is not None else 'inversion'
    if eval_strategy == 'inversion':
        logger.info("Using inversion-based evaluation strategy.")
        eval_func = evaluate_inv
    elif eval_strategy == 'density':
        logger.info("Using density-estimation-based evaluation strategy.")
        eval_func = evaluate_density
    elif eval_strategy == 'recon':
        logger.info("Using reconstruction-based evaluation strategy.")
        eval_func = evaluate_recon
    else:
        raise ValueError(f"Unknown evaluation strategy: {eval_strategy}")
    
    vf.eval()

    # -- evaluate
    logger.info(f"Starting evaluation...")
    eval_results = eval_func(
        vf=vf,
        fe=fe,
        dataloader=test_loader,
        device=device,
        img_sz=(params['data']['img_size'], params['data']['img_size']),
        verbose=True,
        use_bfloat16=params['meta']['use_bfloat16'],
        distributed=args.distributed,
        eval_params=eval_params
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