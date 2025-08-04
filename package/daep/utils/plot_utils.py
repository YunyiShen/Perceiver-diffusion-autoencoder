# Preamble
import numpy as np
import os
# import wget
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from pathlib import Path

working_directory = Path('/home/altair/Documents/UROP/2025_Summer/vastclammm/SpectraFM/dataset_galah/')

def load_dr3_lines():
    """
    Load the list of important lines from the DR3 file
    """
    important_lines = dict()

    important_lines[1] = []
    important_lines[2] = []
    important_lines[3] = []
    important_lines[4] = []

    important_molecules = dict()
    important_molecules[1] = [[4710,4740,'Mol. C2']]
    important_molecules[2] = [[]]
    important_molecules[3] = [[]]
    important_molecules[4] = [[7594,7695,'Mol. O2 (tell.)']]
    
    mode_dr3_path = Path('/home/altair/Documents/UROP/2025_Summer/vastclammm/SpectraFM/dataset_galah/chem_abundance_lines.txt')

    try:
        line, wave = np.loadtxt(mode_dr3_path,usecols=(0,1),unpack=True,dtype=str, comments=';')

        for each_index in range(len(line)):
            if line[each_index] != 'Sp':
                if (float(wave[each_index]) > 4710) & (float(wave[each_index]) < 4905):
                    if len(line[each_index]) < 5:
                        important_lines[1].append([float(wave[each_index]), line[each_index], line[each_index]])
                    else:
                        important_lines[1].append([float(wave[each_index]), line[each_index][:-4], line[each_index]])
                if (float(wave[each_index]) > 5645) & (float(wave[each_index]) < 5877.5):
                    if len(line[each_index]) < 5:
                        important_lines[2].append([float(wave[each_index]), line[each_index], line[each_index]])
                    else:
                        important_lines[2].append([float(wave[each_index]), line[each_index][:-4], line[each_index]])
                if (float(wave[each_index]) > 6472.5) & (float(wave[each_index]) < 6740):
                    if len(line[each_index]) < 5:
                        important_lines[3].append([float(wave[each_index]), line[each_index], line[each_index]])
                    else:
                        important_lines[3].append([float(wave[each_index]), line[each_index][:-4], line[each_index]])
                if (float(wave[each_index]) > 7580) & (float(wave[each_index]) < 7890):
                    if len(line[each_index]) < 5:
                        important_lines[4].append([float(wave[each_index]), line[each_index], line[each_index]])
                    else:
                        important_lines[4].append([float(wave[each_index]), line[each_index][:-4], line[each_index]])
        important_lines[1].sort()
        important_lines[2].sort()
        important_lines[3].sort()
        important_lines[4].sort()
    except:
        print('Could not read in list of elements run as part of DR3')
        
    return(important_lines,important_molecules)

def extract_spectrum(sobject_id, test_name):
    """
    Read in all four CCD spectra for each sobject_id
    """
    datadir = working_directory / test_name
    # Example filepaths: '1311180029013071.fits', '1311180029013072.fits', etc. 
    spectrum_length = 4096
    fluxes = []
    wavelengths = []
    uncertainties = []
    for ccd_num in [1,2,3,4]:
        spectrum_path = datadir / f'{sobject_id}{ccd_num}.fits'
        if spectrum_path.exists():
            hdul = pyfits.open(spectrum_path)
            fluxes_partial = np.array(hdul[4].data)
            fluxes.append(fluxes_partial[:spectrum_length])
            
            start_wavelength = hdul[4].header["CRVAL1"]
            dispersion       = hdul[4].header["CDELT1"]
            nr_pixels        = hdul[4].header["NAXIS1"]
            reference_pixel  = hdul[4].header["CRPIX1"]
            if reference_pixel == 0:
                reference_pixel=1
            wavelengths_partial = ((np.arange(0,nr_pixels)--reference_pixel+1)*dispersion+start_wavelength)
            wavelengths.append(wavelengths_partial[:spectrum_length])
            uncertainties_partial = np.array(hdul[1].data)
            uncertainties.append(uncertainties_partial[:spectrum_length])
            hdul.close()
        else:
            fluxes.append(np.full(spectrum_length, np.nan)) # If no spectrum, fill with nans
            wavelengths.append(np.full(spectrum_length, np.nan))
            uncertainties.append(np.full(spectrum_length, np.nan))
        
    fluxes = np.concatenate(fluxes)
    wavelengths = np.concatenate(wavelengths)
    uncertainties = np.concatenate(uncertainties)
    
    if np.isnan(fluxes).all():
        print(f"Warning: No spectra found for {sobject_id}")
    
    return fluxes, uncertainties, wavelengths


# USE THIS ONE TOO
def plot_spectra_simple(sobject_id, fluxes, wavelengths, uncertainties=None,
                        plot_elements=True, savefig=False, showfig=True, save_dir='',
                        fig=None, axes=None):
    """
    Plot the spectra for a given object in a 4x1 panel figure, one panel per CCD.

    Parameters
    ----------
    sobject_id : str or int
        Identifier for the spectrum object to be plotted.
    fluxes : np.ndarray
        Array of flux values for the spectrum, concatenated across all CCDs.
    wavelengths : np.ndarray
        Array of wavelength values corresponding to the fluxes, concatenated across all CCDs.
    uncertainties : np.ndarray, optional
        Array of uncertainty values for the fluxes. If None, uncertainties are not shown.
    plot_elements : bool, default=True
        Whether to overlay spectral line markers and element labels.
    savefig : bool, default=False
        Whether to save the figure to disk.
    showfig : bool, default=True
        Whether to display the figure interactively.
    save_dir : str, default=''
        Directory path to save the figure if `savefig` is True.
    fig : matplotlib.figure.Figure, optional
        Existing figure object to plot on. If provided, `axes` must also be provided.
    axes : np.ndarray, optional
        Existing axes array to plot on. If provided, `fig` must also be provided.

    Returns
    -------
    f : matplotlib.figure.Figure or None
        The matplotlib Figure object if `showfig` is False, otherwise None.
    ccds : np.ndarray or None
        Array of Axes objects for each CCD panel if `showfig` is False, otherwise None.

    Notes
    -----
    This function creates a 4-panel plot, with each panel corresponding to a different CCD wavelength range.
    Optionally, it overlays element lines and labels, and can save or display the figure.
    If `fig` and `axes` are provided, the function will plot on the existing axes instead of creating new ones.
    When using existing axes, spectral lines are only added if `plot_elements=True` and no spectral lines
    are already present on the axes.
    """
    # Check if using existing figure/axes or creating new ones
    if fig is not None and axes is not None:
        f, ccds = fig, axes
        use_existing = True
    elif fig is None and axes is None:
        f, ccds = plt.subplots(4,1,figsize=(11.69, 8.27))
        use_existing = False
    else:
        raise ValueError("Both `fig` and `axes` must be provided together, or both must be None")
    
    if not use_existing:
        kwargs_sob = dict(c = 'k', lw=0.5, label='Flux', rasterized=True)
        kwargs_error_spectrum = dict(color = 'grey', label='Flux error', rasterized=True)
    else:
        kwargs_sob = dict(color='cyan', label='Predicted Flux', lw=0.5, rasterized=True)
        kwargs_error_spectrum = dict(color='blue', alpha=0.2, label='Predicted Flux Error', rasterized=True)

    for each_ccd in [1,2,3,4]:
        ax=ccds[each_ccd-1]
        
        # Plot the uncertainty as grey background
        if uncertainties is not None:
            ax.fill_between(
                wavelengths,
                fluxes - uncertainties,
                fluxes + uncertainties,
                **kwargs_error_spectrum
                )
        
        # Overplot observed spectrum a bit thicker
        ax.plot(
            wavelengths,
            fluxes,
            **kwargs_sob
            )
        
        # Only set title, labels, and limits if not using existing axes
        if not use_existing:
            if each_ccd == 1:
                ax.set_title(str(sobject_id))
            ax.set_ylabel('Flux [norm.]')
            ax = adjust_axis_lims_galah(ax, each_ccd)

        # Only add spectral lines if requested and not using existing axes
        # (or if using existing axes but no spectral lines are present yet)
        if plot_elements and (not use_existing or len(ax.get_lines()) <= 1):
            ax = plot_spectral_lines_galah(ax, each_ccd)
            
    # Only call tight_layout if not using existing figure
    if not use_existing:
        plt.tight_layout()
    
    if savefig:
        if len(save_dir) > 0:
            plt.savefig(Path(save_dir) / f'{sobject_id}.png',bbox_inches='tight',dpi=200)
        else:
            print('No save directory provided, so not saving figure')
    if showfig:
        plt.show()
        plt.close()
        return None, None
    else:
        return f, ccds

def adjust_axis_lims_galah(ax, each_ccd):
    ax.set_ylim(-0.1,1.3)
    if each_ccd == 1:
        ax.set_xlim(4710,4905)
    if each_ccd == 2:
        ax.set_xlim(5645,5877.5)
    if each_ccd == 3:
        ax.set_xlim(6472.5,6740)
    if each_ccd == 4:
        ax.set_xlim(7580,7890)
        ax.set_xlabel('Wavelength [Å]')
    if each_ccd == 4:
        ax.legend(loc='lower left')
    return ax

def plot_spectral_lines_galah(ax, each_ccd):
    important_lines, important_molecules = load_dr3_lines()
    
    if each_ccd==1:
        ax.axvline(4861.3230,lw=0.5,ls='dashed',c='r')
        ax.text(4861.3230,1.15,r'H$_\beta$',fontsize=10,ha='center',color='k')
    if each_ccd==3:
        ax.axvline(6562.7970,lw=0.5,ls='dashed',c='r')
        ax.text(6562.7970,1.15,r'H$_\alpha$',fontsize=10,ha='center',color='k')
    for each_index, each_element in enumerate(important_lines[each_ccd]):
        offset = 0.1*(each_index%3)
        ax.axvline(each_element[0],lw=0.2,ls='dashed',c='r')
        if each_element[1] in ['Li','C','O']:
            ax.text(each_element[0],offset,each_element[1],fontsize=10,ha='center',color='pink')
        elif each_element[1] in ['Mg','Si','Ca','Ti','Ti2']:
            ax.text(each_element[0],offset,each_element[1],fontsize=10,ha='center',color='b')
        elif each_element[1] in ['Na','Al','K']:
            ax.text(each_element[0],offset,each_element[1],fontsize=10,ha='center',color='orange')
        elif each_element[1] in ['Sc','V', 'Cr','Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']:
            ax.text(each_element[0],offset,each_element[1],fontsize=10,ha='center',color='brown')
        elif each_element[1] in ['Rb', 'Sr', 'Y', 'Zr', 'Ba', 'La', 'Ce','Mo','Ru', 'Nd', 'Sm','Eu']:
            ax.text(each_element[0],offset,each_element[1],fontsize=10,ha='center',color='purple')
    if each_ccd in [1,4]:
        for each_molecule in important_molecules[each_ccd]:
            ax.axvspan(each_molecule[0],each_molecule[1],color='y',alpha=0.05)
            ax.text(0.5*(each_molecule[0]+each_molecule[1]),1.15,each_molecule[2],fontsize=10,ha='center',color='k')                
    return ax

# USE THIS ONE
def galah_plot_simple(sobject_id, test_name, savefig=False, save_dir=''):
    """
    Plot the spectra of a given GALAH object.

    Parameters
    ----------
    sobject_id : str or int
        The GALAH object identifier.
    test_name : str
        The name of the test or dataset context.
    savefig : bool, optional
        Whether to save the generated figure to disk. Default is False.
    save_dir : str, optional
        Directory path to save the figure if `savefig` is True. Default is '' (current directory).

    Returns
    -------
    f : matplotlib.figure.Figure
        The matplotlib Figure object containing the plotted spectra.

    Notes
    -----
    This function extracts the spectrum, uncertainties, and wavelengths for the specified GALAH object,
    optionally attempts to download the data if not found, and plots the spectra with element lines and labels.
    """
    spectra, uncertainties, wavelengths = extract_spectrum(sobject_id, test_name)
    f = plot_spectra_simple(sobject_id, spectra, wavelengths, uncertainties, plot_elements=True, savefig=savefig, save_dir=save_dir)
    return f
