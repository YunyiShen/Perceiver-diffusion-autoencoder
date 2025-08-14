
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Optional, List
from astropy.time import Time
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d
from daep.utils.general_utils import detect_env, convert_to_native_byte_order, set_paths


ENV = detect_env()
BASE_PATH, MODEL_PATH, DATA_PATH, RAW_DATA_PATH = set_paths(ENV, 'lightcurve')
TESS_XMATCH_CATALOG_PATH = RAW_DATA_PATH + '/id_catalog_gt_1800.fits'

TEST_NAME = 'tess_1k'

class TESSDataset(Dataset):
    
    def __init__(self, data_dir: Path, train: bool, extract: bool = True,
                 raw_data_dir: Optional[Path] = None, tess_xmatch_catalog_path: Optional[Path] = None,
                 targets_path: Optional[Path] = None):
        self.data_dir = data_dir
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
        self.train = train
        # Size of GALAH lightcurve
        self.seq_len = 1171
        self._set_legacy_data_paths()
        if extract:
            self.extract_data(raw_data_dir, tess_xmatch_catalog_path, targets_path) 
        elif self._check_legacy_exists():
            print(f"Found legacy data -- loading from {self.data_dir}")
            self._load_legacy_data()
        else:            
            raise FileNotFoundError(f"Extracted data not found at {data_dir} -- set extract=True to extract data")
        
        self._convert_starclass_to_one_hot()
    
    def __len__(self):
        return len(self.fluxes)
    
    def __getitem__(self, idx):
        raise NotImplementedError("__getitem__ not implemented: use TESSDatasetProcessed instead")

    def _set_legacy_data_paths(self):
        self.fluxes_path = self.data_dir / f"fluxes{'_train' if self.train else '_test'}.npy"
        self.fluxes_errs_path = self.data_dir / f"fluxes_errs{'_train' if self.train else '_test'}.npy"
        self.labels_path = self.data_dir / f"labels{'_train' if self.train else '_test'}.npy"
        self.label_errs_path = self.data_dir / f"label_errs{'_train' if self.train else '_test'}.npy"
        self.times_path = self.data_dir / f"times{'_train' if self.train else '_test'}.npy"
        self.ids_path = self.data_dir / f"ids{'_train' if self.train else '_test'}.npy"
        self.catalog_path = self.data_dir / f"catalog_{'train' if self.train else 'test'}.csv"
    
    def _check_legacy_exists(self):
        data_paths = [self.fluxes_path, self.fluxes_errs_path, self.labels_path,
                    self.label_errs_path, self.times_path, self.ids_path, self.catalog_path]
        all_exist = np.all([data_path.exists() for data_path in data_paths])
        return all_exist
    
    def _load_legacy_data(self):
        self.fluxes = np.load(self.fluxes_path, allow_pickle=True)
        self.fluxes_errs = np.load(self.fluxes_errs_path, allow_pickle=True)
        self.times = np.load(self.times_path, allow_pickle=True)
        self.labels = np.load(self.labels_path, allow_pickle=True)
        self.label_errs = np.load(self.label_errs_path, allow_pickle=True)
        self.ids = np.load(self.ids_path, allow_pickle=True)
        self.catalog = pd.read_csv(self.catalog_path)
        self.catalog = convert_to_native_byte_order(self.catalog)
    
    def _convert_starclass_to_one_hot(self):
        """
        Convert the first column of self.labels (starclass integer) to a one-hot encoded array.

        Returns
        -------
        np.ndarray
            One-hot encoded array of shape (num_samples, num_starclasses).
        """
        # Define the mapping from starclass names to integers
        # self.starclass_name_to_int = {'INSTRUMENT': 0, 'APERIODIC': 1, 'CONSTANT': 2, 'CONTACT_ROT': 3,
        #                              'DSCT_BCEP': 4, 'ECLIPSE': 5, 'GDOR_SPB': 6, 'RRLYR_CEPHEID': 7,
        #                              'SOLARLIKE': 8}
        self.starclass_names = ['APERIODIC', 'CONSTANT', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE', 'GDOR_SPB', 'RRLYR_CEPHEID', 'SOLARLIKE']
        self.starclass_name_to_int = {name: i for i, name in enumerate(self.starclass_names)}
        self.starclass_int_to_name = {v: k for k, v in self.starclass_name_to_int.items()}
        self.num_starclasses = len(self.starclass_name_to_int)
        
        # Extract starclass integer labels from the first column of self.labels
        starclass_names = self.labels[:, 0].astype(str)
        starclass_ints = []
        idx_with_starclass = []
        for idx, name in enumerate(starclass_names):
            if name in self.starclass_name_to_int:
                starclass_ints.append(self.starclass_name_to_int[name])
                idx_with_starclass.append(idx)
        starclass_ints = np.array(starclass_ints)
        idx_with_starclass = np.array(idx_with_starclass)

        # Create one-hot encoded array
        if len(idx_with_starclass) > 0:
            one_hot = np.zeros((self.fluxes.shape[0], self.num_starclasses), dtype=np.float32)
            one_hot[idx_with_starclass, starclass_ints] = 1.0
            self.starclass = one_hot
        else:
            self.starclass = np.zeros((self.fluxes.shape[0], self.num_starclasses), dtype=np.float32)
    
    def extract_data(self, raw_data_dir, tess_xmatch_catalog_path, targets_path=None):
        """"
        Extract data from the raw data directory and save it to the data directory.
        """
        if raw_data_dir is None:
            raise ValueError("raw_data_dir must be provided if extract is True")
        if tess_xmatch_catalog_path is None:
            raise ValueError("tess_xmatch_catalog_path must be provided if extract is True")
        
        # Find all directories inside raw_data_dir
        subset_dirs = [d for d in raw_data_dir.iterdir() if d.is_dir()]
        print(f"Raw fits files will be extracted from {raw_data_dir}")
    
        print("Step 1: Loading label data from TESS catalog...")
        with fits.open(tess_xmatch_catalog_path) as hdul:    
            tess_xmatch_df = pd.DataFrame(hdul[1].data) #type: ignore
            tess_xmatch_df = convert_to_native_byte_order(tess_xmatch_df)
            tess_xmatch_df['TIC'] = tess_xmatch_df['TIC'].astype(str)
            
        all_targets_df = pd.read_csv(targets_path)
        all_targets_df = convert_to_native_byte_order(all_targets_df)
        all_targets_df = all_targets_df.rename(columns={'starname': 'TIC', 'lightcurve': 'path'})
        all_targets_df['TIC'] = all_targets_df['TIC'].astype(str)
        if 'path' not in all_targets_df.columns:
            all_targets_df['path'] = np.full(len(all_targets_df), None)
        if 'starclass' not in all_targets_df.columns:
            all_targets_df['starclass'] = np.full(len(all_targets_df), None)
    
        # Merge on the now-matching types
        full_catalog = all_targets_df.merge(
            tess_xmatch_df,
            left_on='TIC',
            right_on='TIC',
            how='left',
            suffixes=('', '_duplicate')
        )
    
        ids_columns = ['obs_type', 'survey_name', 'TIC', 'GAIA', 'GALAH', 'date_obs_jd']
        label_columns = ['starclass', 'TEFF', 'LOGG']
        err_columns = ['e_starclass', 'e_TEFF', 'e_LOGG']
        
        # Add a column to the matched_df to indicate that the data is from the lightcurve survey
        full_catalog['obs_type'] = np.full(len(full_catalog), 'lightcurve')
        full_catalog['survey_name'] = np.full(len(full_catalog), 'TESS')
        full_catalog['date_obs_jd'] = np.full(len(full_catalog), None)
        
        full_catalog['TEFF'] = np.full(len(full_catalog), np.nan)
        full_catalog['LOGG'] = np.full(len(full_catalog), np.nan)

        full_catalog['e_TEFF'] = np.full(len(full_catalog), np.nan)
        full_catalog['e_LOGG'] = np.full(len(full_catalog), np.nan)
        full_catalog['e_starclass'] = np.full(len(full_catalog), None)
        
        # Reorder the columns of full_catalog: ids_columns, label_columns, err_columns, then all remaining columns.
        ordered_cols = ids_columns + label_columns + err_columns
        remaining_cols = [col for col in full_catalog.columns if col not in ordered_cols]
        full_catalog = full_catalog[ordered_cols + remaining_cols]

        # Step 2: Train/Test Split
        print("Step 2: Performing train/test split...")
        test_size = 0.1
        ticids_train, ticids_test = train_test_split(
            full_catalog['TIC'].values, test_size=test_size, random_state=42)
        
        if self.train:
            ticids = ticids_train
        else:
            ticids = ticids_test
        
        train_or_test_catalog = full_catalog[full_catalog['TIC'].isin(ticids)].copy()
        num_lightcurves = len(train_or_test_catalog)
        
        # Ensure the catalog is sorted by TIC ID for consistent ordering
        train_or_test_catalog = train_or_test_catalog.sort_values('TIC').reset_index(drop=True)
        
        # Step 3: Extract and save lightcurve
        print(f"Step 3: Saving {num_lightcurves} lightcurves...")
        fluxes_array = np.full((num_lightcurves, self.seq_len), np.nan)
        times_array = np.full((num_lightcurves, self.seq_len), np.nan)
        uncertainties_array = np.full((num_lightcurves, self.seq_len), np.nan)
        # dates_array = np.full(num_lightcurves, np.nan)
        
        for idx, row in enumerate(tqdm(train_or_test_catalog.itertuples(index=False), total=len(train_or_test_catalog),
                                       desc="Extracting lightcurves", unit="lightcurves", mininterval=1.0)):
            
            def update_data_arrays(extracted_lc):
                lc_size = min(extracted_lc['flux'].shape[0], extracted_lc['time'].shape[0], self.seq_len)
                fluxes_array[idx, :lc_size] = extracted_lc['flux'][:lc_size]
                times_array[idx, :lc_size] = extracted_lc['time'][:lc_size]
                uncertainties_array[idx, :lc_size] = extracted_lc['flux_err'][:lc_size]
                train_or_test_catalog.loc[train_or_test_catalog['TIC'] == ticid, 'date_obs_jd'] = extracted_lc['date_obs_jd']
                train_or_test_catalog.loc[train_or_test_catalog['TIC'] == ticid, 'TEFF'] = extracted_lc['teff']
                train_or_test_catalog.loc[train_or_test_catalog['TIC'] == ticid, 'LOGG'] = extracted_lc['logg']
            
            ticid = row.TIC
            try:
                if row.path is not None:
                    if row.path.endswith('.txt'):
                        path_to_txt = raw_data_dir / row.path
                        extracted_lc = self.extract_txt_lightcurve(path_to_txt)
                    else:
                        path_to_fits = raw_data_dir / row.path
                        extracted_lc = self.extract_qlp_lightcurve(path_to_fits)
                    update_data_arrays(extracted_lc)
                else:
                    for subset_dir in subset_dirs:
                        subset_subdir = subset_dir / 'mastDownload' / 'HLSP'
                        matching_folders = list(subset_subdir.glob(f"*{ticid}_tess*"))
                        if not matching_folders:
                            continue
                        # Only using the first matching folder for now
                        for matching_folder in matching_folders[:1]:
                            path_to_fits = list(matching_folder.glob("*.fits"))[0]
                            extracted_lc = self.extract_qlp_lightcurve(path_to_fits)
                            update_data_arrays(extracted_lc)
                
            except Exception as e:
                print(f"Error extracting lightcurve for {ticid}: {e}")
                continue
        
         # Ensure all columns in train_or_test_catalog with big-endian byte order are byteswapped to native order
        train_or_test_catalog = convert_to_native_byte_order(train_or_test_catalog)
        
        # Extract labels, label errors, and IDs from the already filtered catalog
        # Since train_or_test_catalog is already filtered by ticids, we can extract directly
        labels = train_or_test_catalog.loc[:, label_columns].values
        label_errs = train_or_test_catalog.loc[:, err_columns].values
        ids = train_or_test_catalog.loc[:, ids_columns].values
        
        # Remove stars without any lightcurve
        # Now that arrays and DataFrame are properly aligned, we can safely use the same indices
        stars_wo_lc = np.where(np.isnan(fluxes_array).all(axis=1))
        print(f"Deleting {len(stars_wo_lc[0])} stars without any lightcurve")
        
        # Remove from all arrays and DataFrame using the same indices
        labels = np.delete(labels, stars_wo_lc, axis=0)
        label_errs = np.delete(label_errs, stars_wo_lc, axis=0)
        ids = np.delete(ids, stars_wo_lc, axis=0)
        fluxes_array = np.delete(fluxes_array, stars_wo_lc, axis=0)
        times_array = np.delete(times_array, stars_wo_lc, axis=0)
        uncertainties_array = np.delete(uncertainties_array, stars_wo_lc, axis=0)
        train_or_test_catalog = train_or_test_catalog.drop(index=stars_wo_lc[0]).reset_index(drop=True)
        
        # Print the shapes of all relevant arrays for debugging and documentation purposes
        print(f"ids shape: {ids.shape}")
        print(f"labels shape: {labels.shape}")
        print(f"label_errs shape: {label_errs.shape}")
        print(f"fluxes_array shape: {fluxes_array.shape}")
        print(f"times_array shape: {times_array.shape}")
        print(f"uncertainties_array shape: {uncertainties_array.shape}")
        
        np.save(self.ids_path, ids)
        np.save(self.labels_path, labels)
        np.save(self.label_errs_path, label_errs)
        np.save(self.fluxes_path, fluxes_array)
        np.save(self.times_path, times_array)
        np.save(self.fluxes_errs_path, uncertainties_array)
        train_or_test_catalog.to_csv(self.data_dir / f'catalog_{"train" if self.train else "test"}.csv', index=False)
        print(f"Data extracted to {self.data_dir}")
        
        self.fluxes = fluxes_array
        self.fluxes_errs = uncertainties_array
        self.times = times_array
        self.labels = labels
        self.label_errs = label_errs
        self.ids = ids
    
    def extract_qlp_lightcurve(self, path):
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")
        with fits.open(path, mode="readonly") as hdulist:
            ticid = hdulist[0].header['TICID']
            teff = hdulist[0].header['TEFF']
            logg = hdulist[0].header['LOGG']
            date_obs_jd = Time(hdulist[1].header['TSTART'] + 2457000.0, format='jd', scale='tdb').jd
            
            # read time, sap flux, quality flag
            tess_bjds = hdulist[1].data['TIME']
            sap_fluxes = hdulist[1].data['SAP_FLUX']
            sap_flux_errs = hdulist[1].data['KSPSAP_FLUX_ERR']
            qual_flags = hdulist[1].data['QUALITY']
            
            # # remove flagged data
            dont_exclude = [0,64,256,1024,2048,8192]
            where_gt0 = np.where(np.isin(qual_flags,dont_exclude))
            tess_bjds = tess_bjds[where_gt0]
            sap_fluxes = sap_fluxes[where_gt0]
            sap_flux_errs = sap_flux_errs[where_gt0]
            qual_flags = qual_flags[where_gt0]
            
            return {
                'time': tess_bjds,
                'flux': sap_fluxes,
                'flux_err': sap_flux_errs,
                'quality': qual_flags,
                'ticid': ticid,
                'teff': teff,
                'logg': logg,
                'date_obs_jd': date_obs_jd
            }
    def extract_txt_lightcurve(self, txt_path):
        txt_path = Path(txt_path)
        if not txt_path.exists():
            raise FileNotFoundError(f"File not found: {txt_path}")
        with open(txt_path, 'r') as f:
            # Extract time, flux, and flux_err columns from the txt file
            data = np.loadtxt(f)
            time = data[:, 0]
            flux = data[:, 1]
            flux_err = data[:, 2]

            return {
                'time': time,
                'flux': flux,
                'flux_err': flux_err,
                'quality': np.full(len(time), 0),
                'ticid': txt_path.name,
                'teff': np.nan,
                'logg': np.nan,
                'date_obs_jd': np.nan
            }

class TESSDatasetProcessed(TESSDataset):
    def __init__(self, data_dir: Path, train: bool, extract: bool = False,
                 raw_data_dir: Optional[Path] = None, tess_xmatch_catalog_path: Optional[Path] = None,
                 targets_path: Optional[Path] = None):
        super().__init__(data_dir, train, extract, raw_data_dir, tess_xmatch_catalog_path, targets_path)
        
        # Process the lightcurves (normalize w/ mean and std)
        self.process_lightcurves()
        self.to_tensor()
        self._total_flux_mean = np.nanmean(self._fluxes_medians)
    
    def __len__(self):
        return len(self.fluxes)
    
    def __getitem__(self, idx):
        idx = idx % len(self.fluxes)
        
        arr = self.fluxes_normalized[idx].cpu().numpy() 
        if np.all(np.isnan(arr)):
            print(f"All NaN lightcurve at index {idx}")
        
        res = {"flux": self.fluxes_normalized[idx],
               "flux_err": self.fluxes_errs_normalized[idx],
               "time": self.times_normalized[idx],
               'starclass': torch.tensor(self.starclass[idx]),
            #    "mask": ~torch.isnan(self.fluxes_normalized[idx]),   # THIS BREAKS THE TRAINING -> loss NaN on epoch 1
               "idx": torch.tensor(idx)}
        
        return res
    
    def get_actual_lightcurve(self, idx):
        return {
                "flux": self.fluxes[idx],
                "flux_errs": self.fluxes_errs[idx],
                "time": self.times[idx],
                "ids": self.ids[idx],
                "labels": self.labels[idx],
                "label_errs": self.label_errs[idx],
                "starclass": self.labels[idx][0],
                "idx": idx
                }
    
    def get_actual_lightcurve_from_ticid(self, ticid):
        idx = np.where(self.ids[:, 2] == ticid)[0][0]
        return self.get_actual_lightcurve(idx)
    
    def process_lightcurves(self):
        """
        Efficiently preprocess all lightcurves in the dataset without using lightkurve.
        - Removes outliers (sigma=10)
        - Normalizes time to start at 0.0001
        - Detrends flux using Gaussian filter (sigma=61)
        - Normalizes flux to zero mean and unit std
        """
        print("Processing lightcurves...")
        
        # Initialize arrays for processed data
        num_lightcurves = len(self.fluxes)
        self.fluxes_normalized = np.full_like(self.fluxes, np.nan)
        self.fluxes_errs_normalized = np.full_like(self.fluxes_errs, np.nan)
        self.times_normalized = np.full_like(self.times, np.nan)
        
        # Arrays to store normalization parameters for each lightcurve
        self._fluxes_medians = np.full(num_lightcurves, np.nan)
        self._fluxes_stds = np.full(num_lightcurves, np.nan)
        
        for idx in range(num_lightcurves):
            # Ensure time, flux, and flux_err are finite and of type float64
            time_arr = np.asarray(self.times[idx], dtype=np.float64)
            flux_arr = np.asarray(self.fluxes[idx], dtype=np.float64)
            flux_err_arr = np.asarray(self.fluxes_errs[idx], dtype=np.float64)
            
            # Remove any non-finite values
            mask = np.isfinite(time_arr) & np.isfinite(flux_arr) #& np.isfinite(flux_err_arr)
            time_arr = time_arr[mask]
            flux_arr = flux_arr[mask]
            flux_err_arr = flux_err_arr[mask]
            
            # Truncate to sequence length if needed
            if len(time_arr) > self.seq_len:
                time_arr = time_arr[:self.seq_len]
                flux_arr = flux_arr[:self.seq_len]
                flux_err_arr = flux_err_arr[:self.seq_len]
            
            # Remove outliers using sigma clipping (sigma=10)
            # Calculate median and MAD (Median Absolute Deviation) for robust statistics
            flux_median = np.nanmedian(flux_arr)
            flux_mad = np.nanmedian(np.abs(flux_arr - flux_median))
            # Convert MAD to standard deviation approximation (MAD ≈ 0.6745 * std for normal distribution)
            flux_std_approx = flux_mad / 0.6745
            # Create outlier mask (points beyond 10 sigma)
            outlier_mask = np.abs(flux_arr - flux_median) <= 10 * flux_std_approx
            # Apply outlier mask
            time_clean = time_arr[outlier_mask]
            flux_clean = flux_arr[outlier_mask]
            flux_err_clean = flux_err_arr[outlier_mask]
            # Skip if no valid data points remain
            if len(time_clean) == 0:
                continue
                
            # Normalize time to start at 0.0001
            t_normalized = time_clean - time_clean[0] + 0.0001
            
            # Detrend flux using Gaussian filter (sigma=61)
            flux_detrended = flux_clean - gaussian_filter1d(flux_clean, 61)
            # flux_detrended = flux_clean
            
            # Normalize flux to zero mean and unit std
            self._fluxes_medians[idx] = np.nanmedian(flux_detrended)
            self._fluxes_stds[idx] = np.nanstd(flux_detrended)

            # Add safety check for very small standard deviations
            if self._fluxes_stds[idx] < 1e-8:
                self._fluxes_stds[idx] = 1.0  # Use unit std if too small
                # print(f"Warning: Very small std detected for lightcurve {idx}, using unit std")

            flux_normalized = (flux_detrended - self._fluxes_medians[idx]) / self._fluxes_stds[idx]
            
            # Store processed data
            self.fluxes_normalized[idx, :len(flux_normalized)] = flux_normalized
            self.times_normalized[idx, :len(t_normalized)] = t_normalized
            
            # Also normalize flux errors (scale by the same std)
            if len(flux_err_clean) > 0:
                flux_err_normalized = flux_err_clean / self._fluxes_stds[idx]
                self.fluxes_errs_normalized[idx, :len(flux_err_normalized)] = flux_err_normalized
        
        print(f"Processed {num_lightcurves} lightcurves")
    
    def unprocess_lightcurves(self, idx, time, flux, flux_err=None):
        """
        Reverse the preprocessing steps applied to lightcurves.
        
        Parameters
        ----------
        idx : int, optional
            Index of the lightcurve in the dataset
        time : np.ndarray or torch.Tensor
            Normalized time array (starts at 0.0001)
        flux : np.ndarray or torch.Tensor
            Normalized flux array (zero mean, unit std)
        flux_err : np.ndarray or torch.Tensor, optional
            Normalized flux error array
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'time': Original time array
            - 'flux': Original flux array (approximate, trend information is lost)
            - 'flux_err': Original flux error array (if provided)
        """
        # Convert to numpy if needed
        if torch.is_tensor(time):
            time = time.cpu().numpy()
        if torch.is_tensor(flux):
            flux = flux.cpu().numpy()
        if flux_err is not None and torch.is_tensor(flux_err):
            flux_err = flux_err.cpu().numpy()
        
        # Reverse normalization: flux_orig = flux_norm * std + mean
        actual_fluxes = self.fluxes[idx]
        
        stds = self._fluxes_stds[idx] #np.nanstd(actual_fluxes)
        medians = self._fluxes_medians[idx] #np.nanmedian(actual_fluxes)
        
        if stds.ndim == 1 and flux.ndim == 2 and stds.shape[0] == flux.shape[0]:
            stds = stds[:, np.newaxis]
            medians = medians[:, np.newaxis]
        
        # Reverse normalization
        flux_orig = flux * stds + medians
        
        # Reverse detrending
        flux_orig = flux_orig + gaussian_filter1d(actual_fluxes, 61)
        
        # Reverse time normalization: time_orig = time_norm - 0.0001
        time_orig = time - 0.0001
        
        # Reverse flux error normalization if provided
        flux_err_orig = None
        if flux_err is not None:
            flux_err_orig = flux_err * self._fluxes_stds[idx]
                
        return {
            'time': time_orig,
            'flux': flux_orig,
            'flux_err': flux_err_orig
        }
    
    def to_tensor(self):
        """
        Convert all relevant numpy arrays in the dataset to PyTorch tensors.
        """
        self.fluxes_normalized = torch.tensor(self.fluxes_normalized, dtype=torch.float32)
        self.fluxes_errs_normalized = torch.tensor(self.fluxes_errs_normalized, dtype=torch.float32)
        self.times_normalized = torch.tensor(self.times_normalized, dtype=torch.float32)


class TESSDatasetProcessedSubset(TESSDatasetProcessed):
    def __init__(self, num_lightcurves: int, data_dir: Path, train: bool, extract: bool = False,
                 raw_data_dir: Optional[Path] = None, tess_xmatch_catalog_path: Optional[Path] = None,
                 targets_path: Optional[Path] = None):
        super().__init__(data_dir, train, extract, raw_data_dir, tess_xmatch_catalog_path, targets_path)
        
        if num_lightcurves <= 0:
            num_lightcurves = len(self.fluxes) - 1
            print(f"Using all {num_lightcurves} lightcurves")
        
        # Set the random seed for reproducibility of the subset selection
        np.random.seed(42)
        subset_indices = np.random.choice(len(self.fluxes), num_lightcurves, replace=False)
        
        self.fluxes = self.fluxes[subset_indices]
        self.times = self.times[subset_indices]
        self.ids = self.ids[subset_indices]
        self.labels = self.labels[subset_indices]
        self.label_errs = self.label_errs[subset_indices]
        self.fluxes_errs = self.fluxes_errs[subset_indices]
        
        # Also truncate the normalization arrays to match the subset
        if hasattr(self, '_fluxes_medians'):
            self._fluxes_medians = self._fluxes_medians[subset_indices]
        if hasattr(self, '_fluxes_stds'):
            self._fluxes_stds = self._fluxes_stds[subset_indices]
        if hasattr(self, 'fluxes_normalized'):
            self.fluxes_normalized = self.fluxes_normalized[subset_indices]
        if hasattr(self, 'fluxes_errs_normalized'):
            self.fluxes_errs_normalized = self.fluxes_errs_normalized[subset_indices]


def download_qlp_data_from_catalog(
    test_name: str,
    catalog_path: Path,
    tess_ref_catalog_path: Path,
    save_dir: Path,
    num_lightcurves: int = 0
):
    """
    Download QLP data for TIC IDs matched via GAIA IDs between an input catalog and a TESS reference catalog.

    Parameters
    ----------
    test_name : str
        Name of the test dataset.
    catalog_path : Path
        Path to the input catalog CSV.
    tess_ref_catalog_path : Path
        Path to the TESS reference catalog (FITS or CSV).
    save_dir : Path
        Directory to save downloaded QLP data.
    num_lightcurves : int, optional
        Number of lightcurves to process (default is 0, meaning all).

    Notes
    -----
    Uses pandas merge for concise matching of GAIA IDs to TIC IDs.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load TESS reference catalog (FITS or CSV)
    if tess_ref_catalog_path.suffix == '.fits':
        with fits.open(tess_ref_catalog_path, mode="readonly") as hdulist:
            tess_ref_df = pd.DataFrame(hdulist[1].data)
    else:
        tess_ref_df = pd.read_csv(tess_ref_catalog_path)

    # Load input catalog and determine GAIA ID column
    catalog_df = pd.read_csv(catalog_path)
    if num_lightcurves > 0:
        catalog_df = catalog_df.head(num_lightcurves)
    gaia_col = next((col for col in ['dr2_source_id', 'GAIA'] if col in catalog_df.columns), None)
    if gaia_col is None:
        raise ValueError(f"Catalog {catalog_path} does not contain a column for GAIA or dr2_source_id IDs")
    
    # Convert both the input catalog and the TESS reference catalog DataFrames to native endianness
    # to avoid ValueError: Big-endian buffer not supported on little-endian compiler.
    catalog_df = convert_to_native_byte_order(catalog_df)
    tess_ref_df = convert_to_native_byte_order(tess_ref_df)

    # Use pandas merge for concise matching
    merged_df = pd.merge(
        catalog_df,
        tess_ref_df,
        left_on=gaia_col,
        right_on='GAIA',
        how='inner'
    )
    
    # Save the merged DataFrame for documentation and reproducibility purposes
    merged_df_path = Path(RAW_DATA_PATH) / test_name / "catalog.csv"
    merged_df.to_csv(merged_df_path, index=False)

    tic_ids = merged_df['TIC'].astype(str).unique().tolist()
    
    # Download QLP data in batches of 500 TIC IDs at a time for efficiency and to avoid overloading the server.
    # This also helps avoid issues with too many IDs in a single query.
    batch_size = 500
    for i in tqdm(range(0, len(tic_ids), batch_size), desc="Downloading QLP data", unit="batch"):
        subset_dir = save_dir / f"batch_{i//batch_size + 1}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        batch_tic_ids = tic_ids[i:i+batch_size]
        print(f"Downloading QLP data for TIC IDs batch {i//batch_size + 1}: {batch_tic_ids[:3]}... (total {len(batch_tic_ids)})")
        try:
            download_qlp_data(subset_dir, batch_tic_ids)
        except Exception as e:
            print(f"Error downloading QLP data for batch {i//batch_size + 1}: {e}")
            continue


def download_qlp_data(save_dir: Path, ticids_list: List[str]):
    from astroquery.mast import Observations
    
    # print(f"Downloading QLP data for {len(ticids_list)} TIC IDs")
    # print(f"TIC IDs: {ticids_list}")

    # obsTable = Observations.query_criteria(provenance_name="QLP", target_name=ticids_list, sequence_number=26)
    obsTable = Observations.query_criteria(
            project='TESS',
            dataproduct_type='TIMESERIES',
            provenance_name='QLP',
            t_exptime=[1799, 1801],
            target_name=ticids_list
        )

    print(f"Found {len(obsTable)} QLP products")
    # print(obsTable)
    
    data = Observations.get_product_list(obsTable)

    download_lc = Observations.download_products(data, download_dir=save_dir)


def download_galah_20k_tess_qlp_data():
    catalog_path = Path(RAW_DATA_PATH).parent / "spectra" / "galah_1k" / "catalog.csv"
    tess_ref_catalog_path = Path(TESS_XMATCH_CATALOG_PATH)
    save_dir = Path(RAW_DATA_PATH) / TEST_NAME
    download_qlp_data_from_catalog(test_name=TEST_NAME, catalog_path=catalog_path, tess_ref_catalog_path=tess_ref_catalog_path, save_dir=save_dir)

def create_and_extract_dataset(test_name: str = TEST_NAME, data_path: str = DATA_PATH, raw_data_path: str = RAW_DATA_PATH,
         tess_xmatch_catalog_path: str = TESS_XMATCH_CATALOG_PATH, targets_path: str = None):
    data_dir = Path(data_path) / test_name
    raw_data_dir = Path(raw_data_path) / test_name
    tess_xmatch_catalog_path = Path(tess_xmatch_catalog_path)
    if targets_path is not None:
        targets_path = Path(targets_path)
    else:
        targets_path = raw_data_dir / 'catalog.csv'
    
    dataset = TESSDataset(data_dir=data_dir, raw_data_dir=raw_data_dir, tess_xmatch_catalog_path=tess_xmatch_catalog_path, targets_path=targets_path, train=True, extract=False)
    # dataset_test = TESSDataset(data_dir=data_dir, raw_data_dir=raw_data_dir, tess_xmatch_catalog_path=tess_xmatch_catalog_path, targets_path=targets_path, train=False, extract=True)

import fire

if __name__ == "__main__":
    fire.Fire(create_and_extract_dataset)
    # download_galah_20k_tess_qlp_data()