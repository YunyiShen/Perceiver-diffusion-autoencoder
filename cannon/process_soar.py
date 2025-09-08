import os
import glob
import numpy as np
from tqdm import tqdm
import astropy.io.fits as afits
from scipy.signal import medfilt


class PhotoSpecData:
    def __init__(self, data_dir="/n/holylabs/LABS/iaifi_lab/Lab/mmvae_sims/gemini", 
                 data_root="SASSAFRAS_train_", 
                 min_photo_counts = 10,
                 min_spectra_counts = 20,
                 midfiltsize=3, centering=False, 
                 standardize=False, verbose=False):
        self.data_dir = data_dir
        self.data_root = data_root
        self.midfiltsize = midfiltsize
        self.centering = centering
        self.standardize = standardize
        self.verbose = verbose
        self.min_photo_counts = min_photo_counts
        self.min_spectra_counts = min_spectra_counts

        # raw buffers
        # spectra & photometry
        self._raw_spec_waves = []
        self._raw_spec_fluxes = []
        self._raw_spec_phases = []
        self._raw_phot_fluxes = []
        self._raw_phot_masks = []
        self._raw_phot_bands = []
        self._raw_phot_phases = []
        self._raw_phot_error = []
        self._raw_spec_error = []
        # SN metadata
        self.snids = []
        self.sntypes = []
        # redshift
        self.redshift_helio = []
        self.redshift_helio_errors = []
        self.redshift_final = []
        self.redshift_final_errors = []

        # scale for computing lupitudes
        self.zero_point = 27.5
        self.m5_sig = {'u': 23.9, 'g': 25.0, 'r': 24.7, 'i': 24.0, 'z': 23.3, 'y': 22.1}


    def load_data(self):
        files = glob.glob(os.path.join(self.data_dir, self.data_root + '*', '*HEAD.FITS.gz'))

        for head_file in tqdm(files, disable=not self.verbose):
            print(head_file)
            # open HEAD
            with afits.open(head_file) as head_hdu:
                head_meta = head_hdu[1].data

            peak_mjds = head_meta['SIM_PEAKMJD']  # peak MJD for each SN
            phot_min, phot_max = head_meta['PTROBS_MIN'], head_meta['PTROBS_MAX']  # photometry index ranges
            snid_array = head_meta['SNID']
            types = head_meta['SIM_TYPE_NAME']  # SN types
            z_helio = head_meta['REDSHIFT_HELIO']
            z_helio_errors = head_meta['REDSHIFT_HELIO_ERR']
            z_final = head_meta['REDSHIFT_FINAL']
            z_final_errors = head_meta['REDSHIFT_FINAL_ERR']

            # open PHOT & SPEC
            with afits.open(head_file.replace('HEAD','PHOT')) as phot_hdu:
                phot_data = phot_hdu[1].data
            with afits.open(head_file.replace('HEAD','SPEC')) as spec_hdu:
                spec_meta = spec_hdu[1].data
                spec_flux_table = spec_hdu[2].data

            spec_snid = spec_meta['SNID']  # SNID for each spectral segment
            spec_min, spec_max = spec_meta['PTRSPEC_MIN'], spec_meta['PTRSPEC_MAX']  # spectra index ranges
            spec_mjds = spec_meta['MJD']

            # loop over each SN in this HEAD file
            for j, sn in enumerate(snid_array):
                # photometry
                t0, t1 = phot_min[j] - 1, phot_max[j]
                ptime = phot_data['MJD'][t0:t1]  # photometry observation times
                pphase = ptime - peak_mjds[j]  # photometry phase relative to peak MJD
                pflux = phot_data['FLUXCAL'][t0:t1]
                perr = phot_data['FLUXCALERR'][t0:t1]
                raw_bands = phot_data['BAND'][t0:t1]  # photometry bands
                # decode bytes to ascii strings
                pbands = np.array([b.decode('ascii') if isinstance(b, (bytes, bytearray)) else str(b) for b in raw_bands], dtype=object)
                # filter by signal-to-noise ratio > 2
                snr_mask = np.isfinite(pflux) & np.isfinite(perr) & (np.abs(pflux/perr) > 2.0)
                if np.sum(snr_mask) < self.min_photo_counts:  # skip if < 10 detections
                    continue
                pphase, pflux, perr, pbands = pphase[snr_mask], pflux[snr_mask], perr[snr_mask], pbands[snr_mask]
                pmask = np.ones_like(pphase, dtype=bool)
                # convert photometry flux to lupitude scale
                f0 = 10 ** (0.4 * self.zero_point)
                pbands_temp = np.array([b[-1].lower() for b in pbands])
                b = np.array([(1/5) * 10**(-0.4 * (self.m5_sig[band] - self.zero_point)) for band in pbands_temp]) / f0
                pflux = -2.5 / np.log(10) * (np.arcsinh((pflux / f0) / (2 * b)) + np.log(b))
                # print how many pflux values are negative
                if np.any(pflux < 0):
                    print(f"Warning: Negative pflux values found for SN {sn} in {head_file}. This may affect the analysis.")

                # spectra
                mask = (spec_snid == sn)  # select spectra rows for this SN
                mins, maxs = spec_min[mask], spec_max[mask]  # index ranges for this spectrum
                if mins.size == 0:  # skip if no spectra for this SN
                    continue
                
                # go through each spectral segment for this SN
                for i, (w0, w1) in enumerate(zip(mins, maxs)):
                    # wavelength as midpoint of bins
                    wave = 0.5 * (spec_flux_table['LAMMIN'][w0:w1] + spec_flux_table['LAMMAX'][w0:w1])
                    rawf = spec_flux_table['FLAM'][w0:w1]
                    flux_err = spec_flux_table['FLAMERR'][w0:w1]
                    # keep only finite flux and wavelength within a range
                    valid = np.where(np.isfinite(rawf) & (3500 <= wave) & (wave <= 10000))[0]
                    
                    if valid.size < 20:  # skip if too few valid points
                        continue
                    #breakpoint()
                    sw = wave[valid]
                    sf = rawf[valid].copy()
                    flux_error = flux_err[valid].copy()
                    # filter by signal-to-noise ratio > 4, remove negative fluxes
                    snr_spec_mask = (np.isfinite(sf) & np.isfinite(flux_err[valid]) & (np.abs(sf/flux_err[valid]) > 4.0) & (sf > 0))
                    if np.sum(snr_spec_mask) < self.min_spectra_counts:
                        continue
                    
                    sf, sw = sf[snr_spec_mask], sw[snr_spec_mask]
                    sf = np.log10(sf)  # flux in log scale
                    sf = medfilt(sf, self.midfiltsize)  # median filter
                    if self.centering:
                        sf -= sf.mean()
                    if self.standardize:
                        sw = (sw - sw.mean()) / sw.std()

                    spec_phase = spec_mjds[mask][i] - peak_mjds[j]  # spectra phase relative to peak MJD

                    # store data
                    self._raw_spec_waves.append(sw)
                    self._raw_spec_fluxes.append(sf)
                    self._raw_spec_phases.append(spec_phase)
                    self._raw_spec_error.append(flux_error)
                    self._raw_phot_fluxes.append(pflux)
                    self._raw_phot_error.append(perr/pflux)
                    self._raw_phot_masks.append(pmask)
                    self._raw_phot_bands.append(pbands)
                    self._raw_phot_phases.append(pphase)
                    sid = f"{sn}_{i}"
                    self.snids.append(sid)
                    self.sntypes.append(types[j])
                    self.redshift_helio.append(z_helio[j])
                    self.redshift_helio_errors.append(z_helio_errors[j])
                    self.redshift_final.append(z_final[j])
                    self.redshift_final_errors.append(z_final_errors[j])

        # map bands to idx
        all_bands = np.unique(np.concatenate(self._raw_phot_bands))
        band_to_idx = {b: i for i, b in enumerate(all_bands)}
        self._raw_phot_wavelengths = [np.array([band_to_idx[b] for b in bands], float) for bands in self._raw_phot_bands]
        
        # pad & stack for spectra
        max_spec_len = max(len(x) for x in self._raw_spec_fluxes)  # longest spectrum length
        spec_waves = np.stack([np.pad(w, (0, max_spec_len-len(w)), constant_values=0.) for w in self._raw_spec_waves])
        spec_fluxes = np.stack([np.pad(f, (0, max_spec_len-len(f)), constant_values=0.) for f in self._raw_spec_fluxes])
        spec_errors = np.stack([np.pad(f, (0, max_spec_len-len(f)), constant_values=0.) for f in self._raw_spec_error])
        spec_phases = np.array(self._raw_spec_phases, dtype=np.float32)
        spec_masks = np.stack([np.pad(np.ones(len(w), int), (0, max_spec_len-len(w)), constant_values=0) for w in self._raw_spec_waves])
        spec_masks = (spec_masks == 0).astype(np.int64)
        # pad & stack for photometry
        max_phot_len = max(len(x) for x in self._raw_phot_fluxes)  # longest photometry length
        phot_fluxes = np.stack([np.pad(f, (0, max_phot_len-len(f)), constant_values=0.) for f in self._raw_phot_fluxes]).astype(np.float32)
        phot_masks = np.stack([np.pad(m.astype(int), (0, max_phot_len-len(m)), constant_values=0) for m in self._raw_phot_masks])
        phot_errors = np.stack([np.pad(f, (0, max_phot_len-len(f)), constant_values=0.) for f in self._raw_phot_error]).astype(np.float32)
        phot_masks = (phot_masks == 0).astype(np.float32)
        phot_wavelengths = np.stack([np.pad(idx, (0, max_phot_len-len(idx)), constant_values=0) for idx in self._raw_phot_wavelengths]).astype(np.float32)
        phot_phases = np.stack([np.pad(ph, (0, max_phot_len-len(ph)), constant_values=0.) for ph in self._raw_phot_phases]).astype(np.float32)
        # metadata
        snids = np.array(self.snids, dtype=f'<U{max(len(s) for s in self.snids)}')
        sntypes = np.array(self.sntypes, dtype=f'<U{max(len(t) for t in self.sntypes)}')
        # redshift
        redshift_helio = np.array(self.redshift_helio)
        redshift_helio_errors = np.array(self.redshift_helio_errors)
        redshift_final = np.array(self.redshift_final)
        redshift_final_errors = np.array(self.redshift_final_errors)

        return (
            spec_waves, spec_fluxes, spec_masks, spec_phases, spec_errors,
            phot_fluxes, phot_masks, phot_wavelengths, phot_phases, phot_errors,
            snids, sntypes,
            redshift_helio, redshift_helio_errors, redshift_final, redshift_final_errors
        )


if __name__=='__main__':
    which_data = "soar"
    loader = PhotoSpecData(data_dir=f"/n/holylabs/LABS/iaifi_lab/Lab/mmvae_sims/{which_data}", 
                           centering=False, standardize=False,
                           min_photo_counts = 20,
                            min_spectra_counts = 80,
                           )
    sw, sf, sm, sp, s_error,pf, pm, pw, pph, phot_err, snid, sntype, z_helio, z_helio_errors, z_final, z_final_errors = loader.load_data()

    # get train / test splits
    D = sf.shape[0]
    train_idx = np.random.choice(D, int(0.8 * D), replace=False)
    test_idx = np.setdiff1d(np.arange(D), train_idx)

    # compute statistics only on non-masked entries
    sw_valid, sf_valid = sw[sm==0], sf[sm==0]
    pf_valid, pph_valid, pw_valid = pf[pm==0], pph[pm==0], pw[pm==0]
    eps = 1e-6
    sw_mean, sw_std = sw_valid.mean(), sw_valid.std()
    sf_mean, sf_std = sf_valid.mean(), sf_valid.std()
    sp_mean, sp_std = sp.mean(), sp.std()
    pf_mean, pf_std = pf_valid.mean(), pf_valid.std()
    pph_mean, pph_std = pph_valid.mean(), pph_valid.std()

    # normalize data
    sw_norm, sf_norm = sw.copy(), sf.copy()
    sp_norm = (sp - sp_mean) / (sp_std + eps)
    pf_norm, pph_norm = pf.copy(), pph.copy()
    sw_norm[sm == 0] = (sw[sm == 0] - sw_mean) / (sw_std + eps)
    sf_norm[sm == 0] = (sf[sm == 0] - sf_mean) / (sf_std + eps)
    pf_norm[pm == 0] = (pf[pm == 0] - pf_mean) / (pf_std + eps)
    pph_norm[pm == 0] = (pph[pm == 0] - pph_mean) / (pph_std + eps)

    # save
    np.savez(
        f'../data/{which_data}_dataset_full_minphot{loader.min_photo_counts}_minspec{loader.min_spectra_counts}.npz',
        wavelength=sw_norm, flux=sf_norm, mask=sm, phase=sp_norm, 
        spec_error = s_error,
        photoflux=pf_norm, photomask=pm, photowavelength=pw, photophase=pph_norm, 
        photoerror = phot_err,
        wavelength_mean=sw_mean, wavelength_std=sw_std, 
        flux_mean=sf_mean, flux_std=sf_std, 
        phase_mean=sp_mean, phase_std=sp_std, 
        photoflux_mean=pf_mean, photoflux_std=pf_std,
        photowavelength_mean=np.mean(pw_valid), photowavelength_std=np.std(pw_valid),
        photophase_mean=pph_mean, photophase_std=pph_std,
        training_idx=train_idx, testing_idx=test_idx,
        snid=snid, sntype=sntype,
        redshift_helio=z_helio, redshift_helio_err=z_helio_errors, redshift_final=z_final, redshift_final_err=z_final_errors
    )
    print(f'saved at ../data/{which_data}_dataset_full_minphot{loader.min_photo_counts}_minspec{loader.min_spectra_counts}.npz')

    # print shapes of all keys
    from pprint import pprint
    data = np.load(f'../data/{which_data}_dataset_full_minphot{loader.min_photo_counts}_minspec{loader.min_spectra_counts}.npz', allow_pickle=True)
    for key in data.keys():
        arr = data[key]
        print(f"{key:15s} → shape {arr.shape}, dtype {arr.dtype}")
        if key == 'photowavelength':
            unique_types = np.unique(arr)
            print(f"  Unique types ({unique_types.size}): {unique_types}")
        if 'mean' in key or 'std' in key:
            print(f"  {key} value: {arr}")
    
    # breakdown of data points per SN type
    # unique_types, counts = np.unique(sntype, return_counts=True)
    # print("\nDatapoints per SN type:")
    # for typ, cnt in zip(unique_types, counts):
    #     print(f"{typ}: {cnt}")

    # train_types, train_counts = np.unique(sntype[train_idx], return_counts=True)
    # print("\nTraining-set datapoints per SN type:")
    # for typ, cnt in zip(train_types, train_counts):
    #     print(f"{typ}: {cnt}")