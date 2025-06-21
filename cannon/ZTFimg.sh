#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 2-00:30
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o ./logs/ZTFimagedaep.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/ZTFimagedaep.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=yshen99@mit.edu
module load python/3.10.13-fasrc01
source activate torch

python ZTFimg.py --epoch 200 --lr 0.00025 --bottlenecklen 5 --bottleneckdim 2 --batch 256 --regularize 0.0001 --patch 3 --save_every 5