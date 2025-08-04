#!/bin/bash

#SBATCH -J train_galah_20k
#SBATCH -o train_galah_20k_%j.out
#SBATCH -e train_galah_20k_%j.err
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=10:00:00

# Load CUDA module (needed for some dependencies)
module load cuda/11.8

# Initialize conda from miniforge3 (don't load system anaconda3)
source /nobackup/users/allisone/miniforge3/bin/activate
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate daep

# Verify activation
echo "Active conda environment: $CONDA_DEFAULT_ENV"

# Fix for cpuinfo architecture error on Satori cluster (POWER9 ppc64le)
export CPUINFO_ARCH_NAME=ppc64le
export CPUINFO_LOG_LEVEL=0
export CPUINFO_LOG_LEVEL_NAME=0

# Set environment variables for CPU optimization (for pandas/numpy operations)
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export VECLIB_MAXIMUM_THREADS=$(nproc)
export BLAS_NUM_THREADS=$(nproc)
export LAPACK_NUM_THREADS=$(nproc)

# Pandas-specific optimizations
export PANDAS_NUM_THREADS=$(nproc)
export PANDAS_USE_NUMBA=1
export PANDAS_USE_PYARROW=1

# Set custom paths for checkpoints and logs (optional)
export CKPT_DIR="/nobackup/users/allisone/UROP_2025_Summer/vastclammm/ckpt"
export LOGS_DIR="/nobackup/users/allisone/UROP_2025_Summer/vastclammm/logs"

# Print system info for debugging
echo "=== System Information ==="
echo "CPU cores available: $(nproc)"
echo "Memory available: $(free -h | grep Mem | awk '{print $2}')"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "PANDAS_NUM_THREADS: $PANDAS_NUM_THREADS"
echo "CPUINFO_ARCH_NAME: $CPUINFO_ARCH_NAME"

# Run the dataset creation script
python training/GALAHspectra_train.py 