cd /home/altair_above/UROP_2025_Summer_Linux/vastclammm



rsync -r -av --progress cannon allisone@satori-login-001.mit.edu:/home/allisone/UROP_2025_Summer/Perceiver-diffusion-autoencoder
rsync -r -av --progress package allisone@satori-login-001.mit.edu:/home/allisone/UROP_2025_Summer/Perceiver-diffusion-autoencoder
rsync -r -av --progress scripts allisone@satori-login-001.mit.edu:/home/allisone/UROP_2025_Summer/Perceiver-diffusion-autoencoder
rsync -r -av --progress models allisone@satori-login-002.mit.edu:/nobackup/users/allisone/UROP_2025_Summer/Perceiver-diffusion-autoencoder
rsync -r -av --progress data/lightcurves_raw/tessv1 allisone@satori-login-002.mit.edu:/nobackup/users/allisone/UROP_2025_Summer/Perceiver-diffusion-autoencoder/data/lightcurves_raw


rsync -r -av --progress allisone@satori-login-001.mit.edu:/home/allisone/UROP_2025_Summer/Perceiver-diffusion-autoencoder/cannon .
rsync -r -av --progress allisone@satori-login-001.mit.edu:/home/allisone/UROP_2025_Summer/Perceiver-diffusion-autoencoder/package .
rsync -r -av --progress allisone@satori-login-001.mit.edu:/home/allisone/UROP_2025_Summer/Perceiver-diffusion-autoencoder/scripts .
rsync -r -av --progress allisone@satori-login-002.mit.edu:/nobackup/users/allisone/UROP_2025_Summer/Perceiver-diffusion-autoencoder/models .
rsync -r -av --progress allisone@satori-login-002.mit.edu:/nobackup/users/allisone/UROP_2025_Summer/Perceiver-diffusion-autoencoder/data .

rsync -r -av --progress allisone@satori-login-002.mit.edu:/nobackup/users/allisone/UROP_2025_Summer/Perceiver-diffusion-autoencoder/data/spectra data
rsync -r -av --progress allisone@satori-login-002.mit.edu:/nobackup/users/allisone/UROP_2025_Summer/Perceiver-diffusion-autoencoder/data/lightcurves data




