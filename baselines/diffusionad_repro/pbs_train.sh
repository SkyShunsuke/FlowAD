#!/bin/bash
#PBS -P gch51615
#PBS -q rt_HF
#PBS -l select=1
#PBS -j oe
#PBS -m n
# DiffusionAD multi-class DDP run on one 8xH200 node. Submit from the repo root:
#   qsub -N diffad_mvtec -l walltime=3:00:00 -v RUN=mvtec_300ep baselines/diffusionad_repro/pbs_train.sh
# RUN = config name in configs/ (<dataset>_<budget>). Resumes from the running
# checkpoint if one exists, so the same command continues a run after walltime.
cd ${PBS_O_WORKDIR:-$HOME/projects/FlowAD}
PY=$HOME/venv/diffad/bin/python
R=baselines/diffusionad_repro
CFG=$R/configs/$RUN.json
DS=$(${PY} -c "import json;print(json.load(open('$CFG'))['dataset'])")
OUT=$R/out/$RUN
CKPT=/groups/gch51615/acg17237wy/diffusionad_repro/$RUN   # home is at quota
mkdir -p $OUT $CKPT
if [ "$DS" = realiad ]; then
  # ~40 GB cache: build on node-local NVMe (takes a few minutes)
  export DIFFAD_CACHE=$PBS_LOCALDIR/diffad_cache
  $PY -m torch.distributed.run --standalone --nproc-per-node=8 $R/prepare_cache.py --datasets realiad \
    >> $OUT/prepare_cache.log 2>&1 || exit 1
fi
$PY -m torch.distributed.run --standalone --nproc-per-node=8 $R/train_ddp.py \
  --config $CFG --out $OUT --ckpt_dir $CKPT >> $OUT/stdout.log 2>&1
