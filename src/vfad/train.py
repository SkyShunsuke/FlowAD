
import os
import sys

import torch
import torch.distributed as dist
import numpy as np

import argparse
import yaml
import logging
from torch.cuda import Event as CUDAEvent

from src.models import init_model
from src.datasets import build_dataset
from src.backbones import get_backbone, get_backbone_feature_shape
from src.flow_matching import VelocityField
from src.utils.distributed import init_distributed_mode, get_rank, get_world_size
from src.utils.distributed import is_main_process as is_main
from src.utils.log import setup_logging, get_logger, AverageMeter
from src.utils.opt.optimizer import build_optimizer, save_checkpoint, load_checkpoint
from src.utils.opt.scheduler import get_cosine_wd_scheduler, get_warmup_cosine_lr_scheduler
from src.utils.opt.scaler import get_gradient_scaler
from src.vfad.eval import evaluate as eval

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser(description="InvAD Training")
    
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the config file')
    parser.add_argument(
        "--devices", type=str, nargs="+", default=["cuda:0"],
    )
    parser.add_argument(
        "--port", type=int, default=29500,
    )
    args = parser.parse_args()
    return args

def main(params, args):
    """XXX for VFAD training.
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
        use_wandb = params['logging']['wandb'].get('use_wandb', False)
        use_tensorboard = params['logging'].get('use_tensorboard', False)
        use_csv = params['logging'].get('use_csv', False)
        tb_logdir = os.path.join(log_dir, 'tb_logs')
        ckpt_logdir = os.path.join(log_dir, 'checkpoints')
        vis_logdir = os.path.join(log_dir, 'visualizations')
        # this experiment
        os.makedirs(log_dir, exist_ok=True)
        # directory for tensorboard logs
        os.makedirs(tb_logdir, exist_ok=True)
        # directory for saving models
        os.makedirs(ckpt_logdir, exist_ok=True)
        # directory for visualizations
        os.makedirs(vis_logdir, exist_ok=True)
        if use_wandb:
            from src.utils.log import WandbLogger
            wandb_logger = WandbLogger(
                project_name=params['logging']['wandb']['project_name'],
                run_name=params['logging']['wandb']['run_name'],
                entity=params['logging']['wandb']['entity'],
                config=params,
                rank=rank
            )
        if use_tensorboard:
            from src.utils.log import TensorboardLogger
            tensorboard_logger = TensorboardLogger(
                log_dir=tb_logdir,
                rank=rank
            )
        if use_csv:
            from src.utils.log import CSVLogger
            csv_logger = CSVLogger(
                log_dir=log_dir,
                rank=rank
            )
        
        # save config file
        config_save_path = os.path.join(log_dir, 'config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(params, f)    
    else:
        use_csv = False
        use_tensorboard = False
        use_wandb = False
    
    # set seed
    seed = params['opt']['global_seed'] + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    logger.info(f"Building datasets... with config: \\ {params['data']}")
    train_bs = params['data']['train_batch_size']
    test_bs = params['data'].get('test_batch_size', train_bs)
    train_dataset = build_dataset(train=True, **params['data'])
    test_dataset = build_dataset(train=False, **params['data'])
    
    if isinstance(train_dataset, torch.utils.data.ConcatDataset):
        num_classes = len(train_dataset.datasets)
    else:
        num_classes = 1
    logger.info(f"Number of classes: {num_classes}")
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=test_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_bs,
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
    ipe = len(train_loader)
    logger.info(f"Data loaders built. Number of training samples: {len(train_dataset)}, "
                f"Number of test samples: {len(test_dataset)}, "
                f"Iterations per epoch: {ipe}")
    
    # build model
    feat_sz = get_backbone_feature_shape(model_name=params['model']['backbone']['model_name'],)
    fe = get_backbone(**params['model']['backbone'])
    fe.to(device).eval()
    logger.info(f"Backbone {params['model']['backbone']['model_name']} initialized. "
                f"Model summary: {fe}")

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
    logger.info(f"Velocity Field Model {params['model']['model_name']} has been initialized.", 
                f"Model summary: {vf}")
    
    # optimizer, scheduler, scaler
    opt_params = params['opt']
    num_epochs = opt_params['epochs']
    optimizer = build_optimizer(
        model=vf,
        optimizer_name=opt_params['name'],
        bias_decay=opt_params['bias_decay'],
        norm_decay=opt_params['norm_decay'],
    )
    lr_scheduler = get_warmup_cosine_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=ipe*opt_params['lr_warmup_epochs'],
        start_lr=opt_params['lr_start'],
        ref_lr=opt_params['lr_warmup'],
        T_max=ipe*num_epochs,
        final_lr=opt_params['lr_end'],
        fix_lr_thres=opt_params['fix_lr_thres'],
        fix_strategy=opt_params['fix_lr_strategy'],
    )
    wd_scheduler = get_cosine_wd_scheduler(
        optimizer=optimizer,
        ref_wd=opt_params['wd_start'],
        T_max=ipe*num_epochs,
        final_wd=opt_params['wd_end'],
        fix_wd_thres=opt_params['fix_wd_thres'],
        fix_strategy=opt_params['fix_wd_strategy'],
    )
    scaler = get_gradient_scaler(
        use_bf16=opt_params['use_bfloat16'],
        device=device
    )

    if args.distributed:
        vf = torch.nn.parallel.DistributedDataParallel(vf, static_graph=True)
    
    vf.train()
    start_epoch = 1
    # -- resume training if needed
    resume_path = params['resume']['resume_path']
    if resume_path is not None:
        assert os.path.isfile(resume_path), f"Resume path {resume_path} not found!, Please check the path."
        vf, optimizer, scaler, start_epoch = load_checkpoint(
            resume_path, vf, optimizer, scaler, lr_scheduler, wd_scheduler
        )
        logger.info(f"Resumed from checkpoint: {resume_path} at epoch {start_epoch}", 
                    "model and optimizer states are loaded.")

    # -- evaluation setup
    eval_freq = params['logging']['eval']['eval_epoch_freq']
    eval_params = params['logging']['eval']

    # -- initial update for schedulers
    wd_scheduler.step()
    lr_scheduler.step()

    logger.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{num_epochs}")

        # -- set epoch for sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        loss_meter = AverageMeter()
        fe_time_meter = AverageMeter()
        vf_time_meter = AverageMeter()
        
        for step, batch in enumerate(train_loader):

            # -- load images and labels
            imgs, labels = batch["img"], batch["clslabel"]    # (B, C, H, W), (B,)
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # -- feature extraction (sample z1)
            start, end = CUDAEvent(enable_timing=True), CUDAEvent(enable_timing=True)
            start.record()
            with torch.no_grad():
                z1, _ = fe(imgs)  # (B, c, h, w)
            end.record()
            
            torch.cuda.synchronize()
            fe_time = start.elapsed_time(end)  # [ms]
            fe_time_meter.step(fe_time)

            # -- fwd pass of VF
            with torch.amp.autocast("cuda", enabled=opt_params['use_bfloat16']):
                start.record()
                loss = vf(z1, labels)  
                end.record()
            torch.cuda.synchronize()
            vf_time = start.elapsed_time(end)  # [ms]
            vf_time_meter.step(vf_time)
            
            # -- backward pass and optimization step
            optimizer.zero_grad()
            if opt_params['use_bfloat16']:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if opt_params['clip_grad'] > 0.0:
                    torch.nn.utils.clip_grad_norm_(vf.parameters(), opt_params['clip_grad'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if opt_params['clip_grad'] > 0.0:
                    torch.nn.utils.clip_grad_norm_(vf.parameters(), opt_params['clip_grad'])
                optimizer.step()

            # -- update meters
            loss = loss.item()
            loss_meter.step(loss)

            log_step = epoch * ipe + step
            if step % params['logging']['log_step_freq'] == 0:
                logger.info(f"Epoch [{epoch}/{num_epochs}] Step [{step}/{ipe}] Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                            f"FE Time: {fe_time_meter.val:.4f}ms ({fe_time_meter.avg:.4f}ms) "
                            f"VF Time: {vf_time_meter.val:.4f}ms ({vf_time_meter.avg:.4f}ms) ")
                # csv log
                if use_csv:
                    log_dict = {
                        'epoch': epoch,
                        'step': log_step,
                        'loss': loss_meter.val,
                        'loss_avg': loss_meter.avg,
                        'fe_time': fe_time_meter.val,
                        'fe_time_avg': fe_time_meter.avg,
                        'vf_time': vf_time_meter.val,
                        'vf_time_avg': vf_time_meter.avg,
                        'lr': lr_scheduler.get_current_lr(),
                        'wd': wd_scheduler.get_current_wd(),
                    }
                    csv_logger.log_metrics(log_dict, step=log_step)

                if use_tensorboard:
                    tb_log_dict = {
                        'Loss/step': loss_meter.val,
                        'Loss/avg': loss_meter.avg,
                        'FE_Time/step': fe_time_meter.val,
                        'FE_Time/avg': fe_time_meter.avg,
                        'VF_Time/step': vf_time_meter.val,
                        'VF_Time/avg': vf_time_meter.avg,
                        'LR/lr': lr_scheduler.get_current_lr(),
                        'WD/wd': wd_scheduler.get_current_wd(),
                    }
                    tensorboard_logger.log_metrics(tb_log_dict, step=log_step)

                if use_wandb:
                    wandb_log_dict = {
                        'Loss/step': loss_meter.val,
                        'Loss/avg': loss_meter.avg,
                        'FE_Time/step': fe_time_meter.val,
                        'FE_Time/avg': fe_time_meter.avg,
                        'VF_Time/step': vf_time_meter.val,
                        'VF_Time/avg': vf_time_meter.avg,
                        'LR/lr': lr_scheduler.get_current_lr(),
                        'WD/wd': wd_scheduler.get_current_wd(),
                    }
                    wandb_logger.log_metrics(wandb_log_dict, step=log_step)
            
            # -- step schedulers
            lr_scheduler.step()
            wd_scheduler.step()
            
        logger.info(f"Epoch [{epoch}/{num_epochs}] completed. "
                    f"Avg Loss: {loss_meter.avg:.4f}, "
                    f"Avg FE Time: {fe_time_meter.avg:.4f}ms, "
                    f"Avg VF Time: {vf_time_meter.avg:.4f}ms.")
        
        # -- evaluate
        if epoch % eval_freq == 0:
            eval_results = eval(
                vf=vf,
                fe=fe,
                dataloader=test_loader,
                device=device,
                img_sz=(params['data']['img_size'], params['data']['img_size']),
                verbose=True,
                use_bfloat16=opt_params['use_bfloat16'],
                distributed=args.distributed,
                eval_params=eval_params
            )
            
            # -- log eval results
            if is_main():
                log_step = epoch * ipe
                if use_csv:
                    eval_log_dict = {'epoch': epoch}
                    eval_log_dict.update(eval_results)
                    csv_logger.log_metrics(eval_log_dict, step=log_step)
                
                if use_tensorboard:
                    tensorboard_logger.log_metrics(eval_results, step=log_step)
                
                if use_wandb:
                    wandb_logger.log_metrics(eval_results, step=log_step)
        
        # -- save checkpoint
        if epoch % params['logging']['ckpt']['ckpt_epoch_freq'] == 0 and is_main():
            save_checkpoint(
                save_path=os.path.join(ckpt_logdir, f'ckpt_epoch_{epoch}.pth'),
                epoch=epoch,
                model=vf,
                opt=optimizer,
                scaler=scaler,
                lr_scheduler=lr_scheduler,
                wd_scheduler=wd_scheduler,
            )
        elif params['logging']['ckpt']['save_latest'] and is_main():
            save_checkpoint(
                save_path=os.path.join(ckpt_logdir, 'checkpoint_latest.pth'),
                epoch=epoch,
                model=vf,
                opt=optimizer,
                scaler=scaler,
                lr_scheduler=lr_scheduler,
                wd_scheduler=wd_scheduler,
            )
        else:
            pass
    
    # -- close loggers
    if is_main():
        if use_wandb:
            wandb_logger.close()
        if use_tensorboard:
            tensorboard_logger.close()
        if use_csv:
            csv_logger.close()
        
    # -- close distributed process
    dist.barrier()
    dist.destroy_process_group()

    # -- end of main
    logger.info("Training completed. Training results are saved in %s", log_dir)
        
    #     if (epoch + 1) % config["evaluation"]["eval_interval"] == 0:
    #         all_results = {}
    #         categories = [ds.category for ds in anom_dataset.datasets]
    #         for anom_loader, normal_loader in zip(anom_loaders, normal_loaders):
    #             logger.info(f"Evaluating on {anom_loader.dataset.category} dataset")
    #             metrics_dict = evaluate.evaluate_dist(
    #                 vf,
    #                 fe,
    #                 anom_loader,
    #                 normal_loader,
    #                 config, 
    #                 diff_in_sh,
    #                 epoch + 1,
    #                 config["evaluation"]["eval_step"],
    #                 device,
    #                 world_size=world_size,
    #                 rank=rank,
    #             )
    #             if rank == 0:
    #                 all_results.update(metrics_dict)
    #             dist.barrier()  # wait for all processes to finish evaluation
            
    #         # Compute average AUC across all categories
    #         avg_results = {}
    #         keys = ["I-AUROC", "I-AP", "I-F1Max", "P-AUROC", "P-AP", "P-F1Max", "PRO", "mAD"]
    #         for key in keys:
    #             avg_results[key] = np.mean([all_results[cat][key] for cat in all_results.keys()])
    #         logger.info(f"Average results: {avg_results}")
            
    #         if rank == 0:
    #             current_auc = avg_results["I-AUROC"]
    #             if current_auc > best_auc:
    #                 best_auc = current_auc
    #                 save_path = save_dir / f"model_best.pth"
    #                 torch.save(vf.state_dict(), save_path)
    #                 logger.info(f"Model is saved at {save_dir}")

    #             if use_wandb:
    #                 for cat in categories:
    #                     wandb.log({
    #                         f"{cat}/I-AUROC": all_results[cat]["I-AUROC"],
    #                         f"{cat}/I-AP": all_results[cat]["I-AP"],
    #                         f"{cat}/I-F1Max": all_results[cat]["I-F1Max"],
    #                         f"{cat}/P-AUROC": all_results[cat]["P-AUROC"],
    #                         f"{cat}/P-AP": all_results[cat]["P-AP"],
    #                         f"{cat}/P-F1Max": all_results[cat]["P-F1Max"],
    #                         f"{cat}/PRO": all_results[cat]["PRO"],
    #                         f"{cat}/mAD": all_results[cat]["mAD"]
    #                     })
                    
    #                 wandb.log({
    #                     "I-AUROC": current_auc,
    #                     "I-AP": avg_results["I-AP"],
    #                     "I-F1Max": avg_results["I-F1Max"],
    #                     "P-AUROC": avg_results["P-AUROC"],
    #                     "P-AP": avg_results["P-AP"],
    #                     "P-F1Max": avg_results["P-F1Max"],
    #                     "PRO": avg_results["PRO"],
    #                     "mAD": avg_results["mAD"]
    #                 })
    #             logger.info(f"AUC: {current_auc} at epoch {epoch}")
            
    #         dist.barrier()  # wait for all processes to finish evaluation
    # logger.info("Training is done!")
    
    # # save model
    # save_path = save_dir / "model_latest.pth"
    # torch.save(vf.state_dict(), save_path)
    # save_path = save_dir / "model_ema_latest.pth"
    # torch.save(model_ema.state_dict(), save_path)
    # logger.info(f"Model is saved at {save_dir}")


    
    
    
        
      
            
    
        
            
            
            
            
            
            
            
            
    
    
