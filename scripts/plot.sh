RUNNAME=mulsenad_mc_ditB_enet_041815
GRIDSIZE=4x4
PLOTDIR=./logs/$RUNNAME/plots
mkdir -p $PLOTDIR

python plot_eval_metric.py \
  --folder ./logs/$RUNNAME \
  --metric mad \
  --all-categories \
  --output-dir $PLOTDIR/mad

python plot_eval_metric.py \
    --folder ./logs/$RUNNAME \
    --metric px_auroc \
    --all-categories \
    --output-dir $PLOTDIR/px_auroc

python plot_eval_metric.py \
    --folder ./logs/$RUNNAME \
    --metric img_auroc \
    --all-categories \
    --output-dir $PLOTDIR/img_auroc

python plot_grid.py $PLOTDIR/mad $GRIDSIZE -o $PLOTDIR/mad_grid.png
python plot_grid.py $PLOTDIR/px_auroc $GRIDSIZE -o $PLOTDIR/px_auroc_grid.png
python plot_grid.py $PLOTDIR/img_auroc $GRIDSIZE -o $PLOTDIR/img_auroc_grid.png



