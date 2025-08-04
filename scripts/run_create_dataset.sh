#!/bin/bash

#SBATCH -J create_dataset_tess_20k
#SBATCH -o create_dataset_tess_20k_%j.out
#SBATCH -e create_dataset_tess_20k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=03:00:00

# Load CUDA module (needed for some dependencies)
module load cuda/11.8

# Initialize conda from miniforge3 (don't load system anaconda3)
source /nobackup/users/allisone/miniforge3/bin/activate
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate daep

# Verify activation
echo "Active conda environment: $CONDA_DEFAULT_ENV"

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

# Print system info for debugging
echo "=== System Information ==="
echo "CPU cores available: $(nproc)"
echo "Memory available: $(free -h | grep Mem | awk '{print $2}')"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "PANDAS_NUM_THREADS: $PANDAS_NUM_THREADS"

# Run the dataset creation script
python ../scripts/daep/datasets/TESSlightcurve_dataset.py --test_name tess_20k