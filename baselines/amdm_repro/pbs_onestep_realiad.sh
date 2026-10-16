#!/bin/bash
#PBS -P gch51615
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -m n
# FlowAD one-step recon plugged into the AMDM (OmiAD teacher) RealIAD checkpoint (epoch 300, EMA):
# native t_init=960 / 40-step DDIM vs one x0 prediction at t_init in {100,...,960}
# (configs/amdm_realiad_onestep.yaml: eval_variants). Submit from the repo root:
#   qsub -N amdm_1step baselines/amdm_repro/pbs_onestep_realiad.sh
cd ${PBS_O_WORKDIR:-$HOME/projects/FlowAD}
export PATH=$PWD/flowad/bin:$PATH
mkdir -p baselines/amdm_repro/out/onestep_amdm_realiad_0922
torchrun --rdzv-backend c10d --rdzv-endpoint localhost:0 --nnodes=1 --nproc-per-node=8 \
  baselines/amdm_repro/train_amdm.py --config_file baselines/amdm_repro/configs/amdm_realiad_onestep.yaml \
  --eval_only --eval_ckpt baselines/amdm_repro/out/amdm_mc_unet_enet_realiad_0922/checkpoints/ckpt_epoch_300.pth \
  > baselines/amdm_repro/out/onestep_amdm_realiad_0922/eval.log 2>&1
