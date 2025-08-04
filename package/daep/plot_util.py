import matplotlib.pyplot as plt
import numpy as np

def plot_lsst_lc(photoband, photomag, phototime, photomask, ax = None, label = False, s = 5, lw = 2, flip = True):
    lsst_bands = ["u", "g", "r", "i", "z", "y"]
    colors = ["purple", "blue", "darkgreen", "lime", "orange", "red"]
    photoband = photoband[~photomask]
    photomag = photomag[~photomask]
    phototime = phototime[~photomask]
    if ax is None:
        fig, ax = plt.subplots()
    for bnd in range(6):
        idx = np.where(photoband == bnd)[0]
        if len(idx) > 0:
            if label:
                ax.scatter(phototime[idx], photomag[idx], label=lsst_bands[bnd], s=s, color=colors[bnd])
            else:
                ax.scatter(phototime[idx], photomag[idx], s=s, color=colors[bnd])
            ax.plot(phototime[idx], photomag[idx], color=colors[bnd], alpha=0.5, lw = lw)
    if flip:
        ax.invert_yaxis()
    if ax is None:
        return fig


def plot_ztf_lc(photoband, photomag, phototime, photomask, ax = None, label = False, s = 5, lw = 2):
    lsst_bands = ["g", "r"]
    colors = ["darkgreen", "red"]
    #breakpoint()
    photoband = photoband[~photomask]
    photomag = photomag[~photomask]
    phototime = phototime[~photomask]
    if ax is None:
        fig, ax = plt.subplots()
    for bnd in range(2):
        idx = np.where(photoband == bnd)[0]
        if len(idx) > 0:
            if label:
                ax.scatter(phototime[idx], photomag[idx], label=lsst_bands[bnd], s=s, color=colors[bnd])
            else:
                ax.scatter(phototime[idx], photomag[idx], s=s, color=colors[bnd])
            ax.plot(phototime[idx], photomag[idx], color=colors[bnd], alpha=0.5, lw = lw)
    ax.invert_yaxis()
    if ax is None:
        return fig