
conda install python=3.11
conda install pytorch -c https://ftp.osuosl.org/pub/open-ce/current
conda install torchvision -c https://ftp.osuosl.org/pub/open-ce/current
conda install pytorch-lightning -c https://ftp.osuosl.org/pub/open-ce/current
conda install tqdm matplotlib h5py astropy scikit-learn -y
pip install fire
pip install -U 'tensorboard'

cd package
pip install -e .

<!-- module load cuda/11.8.0
module avail
conda install conda-forge::datasets

conda install datasets -c https://opence.mit.edu -->
