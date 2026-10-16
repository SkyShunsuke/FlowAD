#!/bin/bash
#PBS -P gch51615
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -m n
# ReFP-AD (EfficientNet-B4, UNIFIED, seed 42) on MVTec-AD with and without the FlowAD one-step plug-in:
#   native      : flow encode = RK4 x 10 (40 NFE), ReFP-AD as is
#   euler1      : flow encode = 1 Euler step (1 NFE) for the EBM training buffer AND test scoring
#   euler1_eval : native checkpoint, 1-step encode at test time only (EBM trained on RK4 latents)
# Submit from the repo root: qsub -N refpad_1step baselines/refpad_repro/pbs_onestep_mvtec.sh
cd ${PBS_O_WORKDIR:-$HOME/projects/FlowAD}/baselines/refpad_repro
PY=$HOME/projects/FlowAD/flowad/bin/python
DATA=$HOME/projects/FlowAD/data/mvtec_ad
OUT=$PWD/out
mkdir -p logs
run() {  # gpu tag encode [extra args]
  CUDA_VISIBLE_DEVICES=$1 $PY run_repro_enet.py --dataset MVTec --data_path $DATA \
    --out_dir $OUT/mvtec_unified_enet_$2 --cache_dir $PBS_LOCALDIR/cache_$2 --encode $3 "${@:4}" \
    > logs/mvtec_enet_$2.log 2>&1
}
run 0 native native &
run 1 euler1 euler1 &
wait
run 0 euler1_eval euler1 --eval_only --ckpt $OUT/mvtec_unified_enet_native/unified_checkpoint.pt
