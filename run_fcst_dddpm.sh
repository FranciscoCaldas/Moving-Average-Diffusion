#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT_DIR"
gpu=0
model_config=(DIFFWAVE_fcst)

save_dir=save_dir
num_train=5
seq_len=96
pred_len=(96 192 336 720)

data_pth=ettm2

# Train
for mc in "${model_config[@]}"; do
  for i in "${pred_len[@]}"; do
    echo $data_pth $i $mc

    python -u scripts/train_fcst.py \
      -dc $data_pth \
      -mc $mc \
      --save_dir $save_dir \
      --seq_len $seq_len \
      --pred_len $i \
      --gpu $gpu --num_train $num_train --batch_size 64 --condition fcst

    python scripts/sample_fcst.py -dc $data_pth \
      --model_name "${mc}_bs64_condfcst" \
      --num_train $num_train \
      --save_dir $save_dir \
      --condition fcst \
      --w_cond 1 \
      --n_sample 100 \
      --deterministic \
      --gpu $gpu \
      --seq_len $seq_len \
      --pred_len $i --fast_sample

  done
done