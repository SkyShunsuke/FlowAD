#!/bin/bash
#PBS -P gch51615
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=5:00:00
#PBS -N vpdm_multi
#PBS -j oe
#PBS -m abe
# Batch version of run_multi_300ep_x0cond.sh (VisA is the long pole: ~2 h
# training + ~1.3 h evaluation). Submit from the repo root:
#   qsub baselines/vpdm_repro/pbs_multi_300ep_x0cond.sh
cd ${PBS_O_WORKDIR:-$HOME/projects/FlowAD}
bash baselines/vpdm_repro/run_multi_300ep_x0cond.sh
