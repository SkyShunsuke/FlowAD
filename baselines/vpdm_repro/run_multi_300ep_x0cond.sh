#!/bin/bash
# VPDM + input-feature path with training-time jitter (best MVTec variant:
# out/vpdm300x_mvtec_jitter_s2), seed 2, 300-epoch multi-class protocol on
# MVTec-AD / MPDD / VisA. One GPU per run. Run from the FlowAD repo root.
# The MVTec run is a rerun of jitter_s2 with the fixed collapse guard (the
# original used the old guard, which matters only for consecutive rollbacks).
PY=~/venv/jit/bin/python
CFG=baselines/vpdm_repro/configs/vpdm_mvtec_300ep_x0cond.yaml
OUT=baselines/vpdm_repro/out
launch() {  # gpu tag dataset_name data_root
  CUDA_VISIBLE_DEVICES=$1 nohup $PY baselines/vpdm_repro/train_vpdm.py --config_file $CFG \
    --override x0_cond.mode=jitter opt.global_seed=2 data.dataset_name=$3 data.data_root=$4 \
    logging.log_dir=$OUT/vpdm300x_$2_jitter_s2/ > $OUT/vpdm300x_$2_jitter_s2.log 2>&1 &
}
launch 0 visa   visa_all     data/visa
launch 1 mpdd   mpdd_all     data/mpdd
launch 2 mvtecR mvtec_ad_all data/mvtec_ad
wait
