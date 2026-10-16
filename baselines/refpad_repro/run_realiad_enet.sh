#!/bin/bash
# ReFP-AD (EfficientNet-B4 backbone, controlled setting) on RealIAD:
#   1) one-time MVTec-style symlink layout from the official realiad_jsons
#   2) parallel train-token cache pre-extraction (6 GPUs x 5 categories)
#   3) single training run: ALL epochs (flow early stopping disabled),
#      then ONE final eval over the full 114,585-image test set.
# No intermediate results; metrics appear only at the end in
#   out/realiad_unified_enet/eval_results_flowad_256.csv (+ eval_results_full.json)
set -e
cd "$(dirname "$0")"
PY=~/projects/FlowAD/flowad/bin/python
ROOT=/home/acg17237wy/projects/FlowAD
DATA=$ROOT/data/realiad_1cls
CACHE=$ROOT/baselines/refpad_repro/cache_tokens_enet
OUT=$ROOT/baselines/refpad_repro/out/realiad_unified_enet
mkdir -p logs

# ---- 1) dataset layout (idempotent, symlinks only) ----
$PY prepare_realiad_1cls.py

# ---- 2) parallel token-cache pre-extraction ----
SHARDS=(
  "audiojack,bottle_cap,button_battery,end_cap,eraser"
  "fire_hood,mint,mounts,pcb,phone_battery"
  "plastic_nut,plastic_plug,porcelain_doll,regulator,rolled_strip_base"
  "sim_card_set,switch,tape,terminalblock,toothbrush"
  "toy,toy_brick,transistor1,u_block,usb"
  "usb_adaptor,vcpill,wooden_beads,woodstick,zipper"
)
GPUS=(0 1 2 3 4 5 6 7)
for i in "${!SHARDS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} nohup $PY pre_extract_enet.py \
    --dataset RealIAD --data_path "$DATA" --cache_dir "$CACHE" \
    --categories "${SHARDS[$i]}" > "logs/pre_realiad_gpu${GPUS[$i]}.log" 2>&1 &
done
wait
if grep -L "ALL DONE" logs/pre_realiad_gpu*.log | grep -q .; then
  echo "pre-extract FAILED in:"; grep -L "ALL DONE" logs/pre_realiad_gpu*.log; exit 1
fi

# ---- 3) train (all epochs) + single final eval, one GPU ----
CUDA_VISIBLE_DEVICES=0, nohup $PY run_repro_enet_realiad.py \
  --data_path "$DATA" --out_dir "$OUT" --cache_dir "$CACHE" \
  > logs/full_realiad_enet.log 2>&1 &
echo "Launched training. tail -f logs/full_realiad_enet.log"

# ---- optional: shard the final eval over 6 GPUs instead ----
# If the single-GPU eval is too slow, kill it after unified_checkpoint.pt is
# saved (training itself is done at that point) and run:
# for i in "${!SHARDS[@]}"; do
#   CUDA_VISIBLE_DEVICES=${GPUS[$i]} nohup $PY run_repro_enet_realiad.py \
#     --data_path "$DATA" --out_dir "$OUT" --cache_dir "$CACHE" \
#     --eval_only --categories "${SHARDS[$i]}" \
#     > "logs/eval_realiad_gpu${GPUS[$i]}.log" 2>&1 &
# done; wait
# $PY run_repro_enet_realiad.py --data_path "$DATA" --out_dir "$OUT" --cache_dir "$CACHE" --merge_only
