#!/bin/bash

#!/bin/bash
#
#SBATCH -p phd
#SBATCH --job-name=DiffWave
#SBATCH --output=outputs/ettm2/slurm_output_%j.out
#SBATCH --gpus=2g.20g:1
#SBATCH -c 8
#SBATCH --export=ALL

# SET MINICONDA_PATH HERE
MINICONDA_PATH=/data/f.caldas/miniconda3 #EX:/home/<USER>/miniconda3/ (without any leading or trailing spaces)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH}"
if [ -z "$MINICONDA_PATH" ]
then
	self=$(basename "$0")
	echo "JOB SUBMISSION FAILED. PLEASE SET MINICONDA_PATH on $self"
else

  gpu=0
  model_config=(DIFFWAVE_fcst)

  save_dir=save_dir
  num_train=5
  seq_len=96
  pred_len=96
  #pred_len=(96 192 336 720)

  data_pth=ettm2

	source "$MINICONDA_PATH"/etc/profile.d/conda.sh
	echo "Activating conda environment: fftenv"
	conda activate fftenv
	
	# Run the python script
  echo "Running python script: $2"
  for mc in "${model_config[@]}"; do
    for i in "${pred_len[@]}"; do
      echo $data_pth $i $mc

      srun --export=ALL python -u scripts/train_fcst.py \
        -dc $data_pth \
        -mc $mc \
        --save_dir $save_dir \
        --seq_len $seq_len \
        --pred_len $i \
        --gpu $gpu --num_train $num_train --batch_size 64 --condition fcst
      echo "Finished training for $mc with pred_len $i"
      
      srun --export=ALL python scripts/sample_fcst.py -dc $data_pth \
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
fi





