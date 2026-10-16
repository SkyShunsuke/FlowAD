#!/bin/bash
#PBS -P gch51615
#PBS -q rt_HF
#PBS -l select=1
#PBS -j oe
# DiT-G + FM + 1-step recon + EfficientNet-b4 (0829 setting), extra seeds on MVTec AD / VisA / MPDD.
# Runs the given configs (names in configs/train/ditG_seeds, no .yaml) sequentially on 8 GPUs.
# Finished runs (logs/ditG_seeds/<name>/DONE) are skipped, so re-submitting resumes.
#   qsub -l walltime=14:00:00 -v RUNS="visa_ditG_enet_0829_seed10" scripts/train_ditG_seeds.sh
#   RUNS="mpdd_ditG_enet_0829_seed10 mpdd_ditG_enet_0829_seed20" bash scripts/train_ditG_seeds.sh
cd /home/acg17237wy/projects/FlowAD
export PATH=$HOME/.pyenv/versions/invad/bin:$PATH
for r in $RUNS; do
  d=logs/ditG_seeds/$r
  [ -f $d/DONE ] && continue
  mkdir -p $d
  echo "start $r $(date '+%F %T') $(hostname)"
  torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    main.py --config_file configs/train/ditG_seeds/$r.yaml --task train > $d/train.out 2>&1
  rc=$?
  [ $rc -eq 0 ] && touch $d/DONE
  echo "end $r rc=$rc $(date '+%F %T')"
done
