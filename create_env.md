cd package
pip install -e .
conda install pytorch -c https://ftp.osuosl.org/pub/open-ce/current
conda install tqdm matplotlib h5py astropy scikit-learn torchvision -y
pip install fire

<!-- module load cuda/11.8.0
module avail
conda install conda-forge::datasets

conda install datasets -c https://opence.mit.edu -->
