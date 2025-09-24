
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import mad_std
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Optional
from astropy.time import Time
import torch
from torch.utils.data import Dataset
from daep.utils.general_utils import detect_env, convert_to_native_byte_order, set_paths


ENV = detect_env()
BASE_PATH, MODEL_PATH, DATA_PATH, RAW_DATA_PATH = set_paths(ENV, 'spectra')
GALAH_CATALOG_PATH = RAW_DATA_PATH + "/GALAH_DR3_main_allstar_v2.fits"

TEST_NAME = 'galah_1k'

class GALAHDataset(Dataset):
    
    def __init__(self, data_dir: Path, train: bool, extract: bool = True,
                 raw_data_dir: Optional[Path] = None, galah_catalog_path: Optional[Path] = None,
                 ids_path: Optional[Path] = None):
        self.data_dir = data_dir
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
        self.train = train
        # Size of GALAH spectra
        self.num_ccds = 4
        self.ccd_len = 4096
        self.spectra_size = self.num_ccds * self.ccd_len
        
        self._set_legacy_data_paths()
        if extract:
                self.extract_data(raw_data_dir, galah_catalog_path, ids_path)
        elif self._check_legacy_exists():
            print(f"Found legacy data -- loading from {self.data_dir}")
            self._load_legacy_data()
        else:
            raise FileNotFoundError(f"Extracted data not found at {data_dir} -- set extract=True to extract data")
    
    def __len__(self):
        return len(self.fluxes)
    
    def __getitem__(self, idx):
        raise NotImplementedError("__getitem__ not implemented: use GALAHDatasetProcessed instead")

    def _set_legacy_data_paths(self):
        self.fluxes_path = self.data_dir / f"fluxes{'_train' if self.train else '_test'}.npy"
        self.fluxes_errs_path = self.data_dir / f"fluxes_errs{'_train' if self.train else '_test'}.npy"
        self.labels_path = self.data_dir / f"labels{'_train' if self.train else '_test'}.npy"
        self.label_errs_path = self.data_dir / f"label_errs{'_train' if self.train else '_test'}.npy"
        self.wavelengths_path = self.data_dir / f"wavelengths{'_train' if self.train else '_test'}.npy"
        self.ids_path = self.data_dir / f"ids{'_train' if self.train else '_test'}.npy"
        self.catalog_path = self.data_dir / f"catalog_{'train' if self.train else 'test'}.csv"
    
    def _check_legacy_exists(self):
        data_paths = [self.fluxes_path, self.fluxes_errs_path, self.labels_path,
                    self.label_errs_path, self.wavelengths_path, self.ids_path, self.catalog_path]
        all_exist = np.all([data_path.exists() for data_path in data_paths])
        return all_exist
    
    def _load_legacy_data(self):
        self.fluxes = np.load(self.fluxes_path, allow_pickle=True)
        self.fluxes_errs = np.load(self.fluxes_errs_path, allow_pickle=True)
        self.wavelengths = np.load(self.wavelengths_path, allow_pickle=True)
        self.labels = np.load(self.labels_path, allow_pickle=True)
        self.label_errs = np.load(self.label_errs_path, allow_pickle=True)
        self.ids = np.load(self.ids_path, allow_pickle=True)
        self.catalog = pd.read_csv(self.catalog_path)
        self.catalog = convert_to_native_byte_order(self.catalog)
    
    def extract_data(self, raw_data_dir, galah_catalog_path, ids_path):
        """"
        Extract data from the raw data directory and save it to the data directory.
        """
        if raw_data_dir is None:
            raise ValueError("raw_data_dir must be provided if extract is True")
        if galah_catalog_path is None:
            raise ValueError("galah_catalog_path must be provided if extract is True")
        if ids_path is None:
            raise ValueError("ids_path must be provided if extract is True")
        
        # Find all directories inside raw_data_dir
        subset_dirs = [d for d in raw_data_dir.iterdir() if d.is_dir()]
        print(f"Raw fits files will be searched in the following directories: {subset_dirs}")
    
        print("Step 1: Loading label data from GALAH catalog...")
        # TODO: change to the path of the data to path/to/allStar_file.fits
        # Currently extracting data from GALAH catalog for all stars in crossmatched_ids
        with fits.open(galah_catalog_path) as hdul:    
            galah_full_df = pd.DataFrame(hdul[1].data) #type: ignore
        with fits.open(ids_path) as hdul:    # type: ignore
            crossmatched_ids_df = pd.DataFrame(hdul[1].data) #type: ignore
        
        try:
            matched_df = galah_full_df[galah_full_df['sobject_id'].isin(crossmatched_ids_df['GALAH'])]
            galah_key = 'sobject_id'
        except KeyError:
            try:
                matched_df = galah_full_df[galah_full_df['galah_id'].isin(crossmatched_ids_df['GALAH'])]
                galah_key = 'galah_id'
            except KeyError:
                try:
                    matched_df = galah_full_df[galah_full_df['galah_id'].isin(crossmatched_ids_df['sobject_id'])]
                    galah_key = 'galah_id'
                except KeyError:
                    matched_df = galah_full_df[galah_full_df['sobject_id'].isin(crossmatched_ids_df['sobject_id'])]
                    galah_key = 'sobject_id'
        
        # Print all column names of matched_df for documentation and debugging purposes
        print("Columns in matched_df:", matched_df.columns.tolist())
        
        # Add a column to the matched_df to indicate that the data is from the spectra survey
        # Use .loc to assign the 'obs_type' column for all rows to 'spectra'
        matched_df.loc[:, 'obs_type'] = 'spectra'

        old_ids_columns = [
            'obs_type', 'survey_name', galah_key, 'dr2_source_id', 'field_id', 
        ]
        old_label_columns = [
            'teff', 'logg', 'O_fe', 'Mg_fe', 'fe_h'
        ]
        old_err_columns = [
            'e_teff', 'e_logg', 'e_O_fe', 'e_Mg_fe', 'e_fe_h'
        ]
        label_columns = [col.upper() for col in old_label_columns]
        err_columns = [f'{col}_ERR' for col in label_columns]
        ids_columns = ['obs_type', 'survey_name', 'GALAH', 'GAIA', 'field_id']

        extracted_data = {new_name: matched_df[name] for name, new_name in zip(old_ids_columns + old_label_columns + old_err_columns,
                                                                               ids_columns + label_columns + err_columns)}
        full_catalog = pd.DataFrame(extracted_data)
        full_catalog = convert_to_native_byte_order(full_catalog)
        
        # Step 2: Train/Test Split
        print("Step 2: Performing train/test split...")
        test_size = 0.1
        galah_ids_train, galah_ids_test = train_test_split(
            full_catalog['GALAH'].values, test_size=test_size, random_state=42)
        
        # Step 3: Extract and save spectra
        print(f"Step 3: Saving spectra...")
        if self.train:
            galah_ids = galah_ids_train
        else:
            galah_ids = galah_ids_test
        
        # Extract only the labels, label errors, and IDs for the selected (train or test) galah_ids
        # Use a boolean mask to select rows where full_catalog['GALAH'] is in galah_ids
        train_or_test_mask = full_catalog['GALAH'].isin(galah_ids)
        
        train_or_test_catalog = full_catalog[full_catalog['GALAH'].isin(galah_ids)].copy()
        num_spectra = len(train_or_test_catalog)
        
        # Ensure the catalog is sorted by GALAH ID to match the order of galah_ids
        train_or_test_catalog = train_or_test_catalog.sort_values('GALAH').reset_index(drop=True)
        
        # Create a mapping from galah_ids to catalog indices for proper alignment
        galah_id_to_catalog_idx = {galah_id: idx for idx, galah_id in enumerate(train_or_test_catalog['GALAH'])}
        
        labels = train_or_test_catalog.loc[:, label_columns].values
        label_errs = train_or_test_catalog.loc[:, err_columns].values
        ids = train_or_test_catalog.loc[:, ids_columns].values
        
        fluxes_array = np.full((num_spectra, self.spectra_size), np.nan)
        wavelengths_array = np.full((num_spectra, self.spectra_size), np.nan)
        uncertainties_array = np.full((num_spectra, self.spectra_size), np.nan)
        dates_array = np.full(num_spectra, np.nan)
        
        # Search all subset directories for each spectrum & extract
        for idx, galah_id in enumerate(tqdm(galah_ids, desc=f"Extracting spectra", unit="spectra", mininterval=1.0)):
            # Get the catalog index for this galah_id
            catalog_idx = galah_id_to_catalog_idx.get(galah_id)
            if catalog_idx is None:
                tqdm.write(f"Warning: GALAH ID {galah_id} not found in catalog")
                continue
                
            spectrum_found = False
            for subset_dir in subset_dirs:
                try:
                    fluxes, wavelengths, uncertainties, date_obs_jd = self.extract_fits_data(galah_id, subset_dir)
                    # Check if we got valid data (not all NaN)
                    if not np.all(np.isnan(fluxes)):
                        fluxes_array[catalog_idx, :] = fluxes
                        wavelengths_array[catalog_idx, :] = wavelengths
                        uncertainties_array[catalog_idx, :] = uncertainties
                        dates_array[catalog_idx] = date_obs_jd
                        spectrum_found = True
                        break  # Found the spectrum, no need to check other subset_dirs
                except Exception as e:
                    tqdm.write(f"Error extracting spectrum for {galah_id} from {subset_dir}: {e}")
                    continue
            if not spectrum_found:
                tqdm.write(f"Warning: No valid spectrum found for {galah_id} in any subset_dir")
        
        # Concatenate dates_array as the last column of ids for future reference of observation dates
        ids = np.concatenate([ids, dates_array[:, None]], axis=1)
        ids_columns.append('date_obs_jd')
        train_or_test_catalog.loc[:, 'date_obs_jd'] = dates_array
        
        # Remove stars without any labels or all NaN spectra
        # Now that arrays and DataFrame are properly aligned, we can safely use the same indices
        stars_wo_labels = np.where(np.isnan(labels).all(axis=1))[0]
        stars_wo_spectra = np.where(np.isnan(fluxes_array).all(axis=1))[0]
        exclude_stars = np.union1d(stars_wo_labels, stars_wo_spectra)
        print(f"Deleting {len(exclude_stars)} stars without any labels or spectra")
        
        # Remove from all arrays and DataFrame using the same indices
        labels = np.delete(labels, exclude_stars, axis=0)
        label_errs = np.delete(label_errs, exclude_stars, axis=0)
        ids = np.delete(ids, exclude_stars, axis=0)
        fluxes_array = np.delete(fluxes_array, exclude_stars, axis=0)
        wavelengths_array = np.delete(wavelengths_array, exclude_stars, axis=0)
        uncertainties_array = np.delete(uncertainties_array, exclude_stars, axis=0)
        train_or_test_catalog = train_or_test_catalog.drop(index=exclude_stars).reset_index(drop=True)
        
        # Print the shapes of all relevant arrays for debugging and documentation purposes
        print(f"ids shape: {ids.shape}")
        print(f"labels shape: {labels.shape}")
        print(f"label_errs shape: {label_errs.shape}")
        print(f"fluxes_array shape: {fluxes_array.shape}")
        print(f"wavelengths_array shape: {wavelengths_array.shape}")
        print(f"uncertainties_array shape: {uncertainties_array.shape}")
        
        np.save(self.ids_path, ids)
        np.save(self.labels_path, labels)
        np.save(self.label_errs_path, label_errs)
        np.save(self.fluxes_path, fluxes_array)
        np.save(self.wavelengths_path, wavelengths_array)
        np.save(self.fluxes_errs_path, uncertainties_array)
        train_or_test_catalog.to_csv(self.data_dir / f'catalog_{"train" if self.train else "test"}.csv', index=False)
        print(f"Data extracted to {self.data_dir}")
        
        self.fluxes = fluxes_array
        self.fluxes_errs = uncertainties_array
        self.wavelengths = wavelengths_array
        self.labels = labels
        self.label_errs = label_errs
        self.ids = ids
    
    def extract_fits_data(self, galah_id: int, raw_data_dir: Path):
        """
        Read in all four CCD spectra for a given galah_id
        """
        # Example filepaths: '1311180029013071.fits', '1311180029013072.fits', etc. 
        fluxes = np.full((self.num_ccds, self.ccd_len), np.nan)
        uncertainties = np.full((self.num_ccds, self.ccd_len), np.nan)
        wavelengths = np.full((self.num_ccds, self.ccd_len), np.nan)
        date_obs_jd = None
        for ccd_num in [1,2,3,4]:
            spectrum_path = raw_data_dir / f'{galah_id}{ccd_num}.fits'
            if spectrum_path.exists():
                hdul = fits.open(spectrum_path)
                fluxes_partial = np.array(hdul[4].data) #type: ignore
                fluxes[ccd_num-1, :] = fluxes_partial[:self.ccd_len]
                
                if ccd_num != 4:
                    uncertainties_partial = np.array(hdul[4].data * hdul[1].data) #type: ignore
                else:
                    uncertainties_partial = np.array(hdul[4].data * hdul[1].data)[-len(fluxes_partial):] #type: ignore
                uncertainties[ccd_num-1, :] = uncertainties_partial[:self.ccd_len]
                
                start_wavelength = hdul[4].header["CRVAL1"] #type: ignore
                dispersion       = hdul[4].header["CDELT1"] #type: ignore
                nr_pixels        = hdul[4].header["NAXIS1"] #type: ignore
                reference_pixel  = hdul[4].header["CRPIX1"] #type: ignore
                
                # Extract the observation date from the FITS header and convert to Julian date
                date_obs = hdul[0].header.get("DATE", None)
                date_obs_jd = Time(date_obs.strip(), format='isot').jd
                # The variable `date_obs_jd` now contains the Julian date or None
                
                if reference_pixel == 0:
                    reference_pixel=1
                wavelengths_partial = ((np.arange(0,nr_pixels)--reference_pixel+1)*dispersion+start_wavelength)
                wavelengths[ccd_num-1, :] = wavelengths_partial[:self.ccd_len]
                
                hdul.close()

        fluxes = np.concatenate(fluxes, axis=0)
        wavelengths = np.concatenate(wavelengths, axis=0)
        uncertainties = np.concatenate(uncertainties, axis=0)
        
        return fluxes, wavelengths, uncertainties, date_obs_jd

class GALAHDatasetProcessed(GALAHDataset):
    def __init__(self, data_dir: Path, train: bool, extract: bool = False,
                 raw_data_dir: Optional[Path] = None, galah_catalog_path: Optional[Path] = None,
                 ids_path: Optional[Path] = None):
        super().__init__(data_dir, train, extract, raw_data_dir, galah_catalog_path, ids_path)
        
        # Process the spectra (normalize w/ mean and std)
        self.process_spectra()
        self.to_tensor()
        self._total_flux_mean = np.nanmean(self._fluxes_means)
    
    def __len__(self):
        return len(self.fluxes)
    
    def __getitem__(self, idx):
        idx = idx % len(self.fluxes)
        
        res = {"flux": self.fluxes_normalized[idx], #torch.log10(self.fluxes_normalized[idx]) + torch.tensor(self._total_flux_mean),
               "flux_err": self.fluxes_errs_normalized[idx],
               "wavelength": self.wavelengths[idx], 
               "phase": torch.tensor(0.),
            #    "mask": ~np.isnan(self.fluxes_normalized[idx])
               "idx": torch.tensor(idx)}

        return res
    
    def get_actual_spectrum(self, idx):
        return {
                "flux": self.fluxes[idx],
                "flux_errs": self.fluxes_errs[idx],
                "wavelength": self.wavelengths[idx].cpu().numpy(),
                "phase": 0,
                "ids": self.ids[idx],
                "labels": self.labels[idx],
                "label_errs": self.label_errs[idx],
                "idx": idx
                }
    
    def get_actual_spectrum_from_galah_id(self, galah_id):
        """
        Get the actual spectrum from the GALAH ID.
        """
        idx = np.where(self.ids[:, 2] == galah_id)[0][0]
        return self.get_actual_spectrum(idx)
    
    def process_spectra(self):
         # Vectorized calculation of means and stds for all tokens at once using pandas groupby
        print(f"Processing {len(self.fluxes)} spectra")
        self._fluxes_means = np.nanmean(self.fluxes, axis=1)
        self._fluxes_stds = mad_std(self.fluxes, axis=1)
        
        # Handle zero stds by setting them to a small value to avoid division by zero
        zero_std_mask = self._fluxes_stds == 0
        if np.any(zero_std_mask):
            print(f"Warning: {np.sum(zero_std_mask)} spectra have zero std, setting to 1e-6")
            self._fluxes_stds[zero_std_mask] = 1e-6
        
        # Reshape means and stds for broadcasting so each mean/std is subtracted/divided from all 1600 fluxes in its row
        means_2d = self._fluxes_means[:, np.newaxis]  # shape (900, 1)
        stds_2d = self._fluxes_stds[:, np.newaxis]    # shape (900, 1)

        # Standardize all spectra using broadcasting
        fluxes_normalized = (self.fluxes - means_2d) / stds_2d
        fluxes_errs_normalized = self.fluxes_errs / stds_2d

        # Each mean is subtracted from all 1600 fluxes in its row (broadcasted), fixing shape issues.
        print(f"Processing complete for {len(self.fluxes)} spectra")

        # Replace values outside thresholds with NaN
        threshold_low = -10
        threshold_high = 10
        fluxes_normalized = np.where((fluxes_normalized < threshold_low) | (fluxes_normalized > threshold_high), np.nan, fluxes_normalized)
        nan_idx = np.isnan(fluxes_normalized)
        fluxes_errs_normalized = np.where(nan_idx, 0, fluxes_errs_normalized)
        
        self.fluxes_normalized = fluxes_normalized
        self.fluxes_errs_normalized = fluxes_errs_normalized
    
    def unprocess_spectra(self, flux, idx):
        """
        Convert log-normalized flux back to actual flux values.
        
        Parameters:
            flux (np.ndarray): Log-normalized flux values
            idx (np.ndarray): Batch indices for the spectra
            
        Returns:
            np.ndarray: Actual flux values
        """
        # Ensure we have the normalization arrays
        if not hasattr(self, '_fluxes_stds') or not hasattr(self, '_fluxes_means'):
            raise ValueError("Normalization arrays not found. Make sure process_spectra() was called.")
        
        # Handle batch indexing - idx should be a batch of indices
        if isinstance(idx, (int, np.integer)):
            idx = np.array([idx])
        
        # Get the corresponding stds and means for this batch
        stds = self._fluxes_stds[idx][..., None]  # Shape: (batch_size, 1)
        means = self._fluxes_means[idx][..., None]  # Shape: (batch_size, 1)
        
        return flux * stds + means #10**(flux - self._total_flux_mean) * stds + means
        
    def to_tensor(self):
        """
        Convert all relevant numpy arrays in the dataset to PyTorch tensors.
        """
        self.fluxes_normalized = torch.tensor(self.fluxes_normalized, dtype=torch.float32)
        self.fluxes_errs_normalized = torch.tensor(self.fluxes_errs_normalized, dtype=torch.float32)
        self.wavelengths = torch.tensor(self.wavelengths, dtype=torch.float32)


class GALAHDatasetProcessedSubset(GALAHDatasetProcessed):
    def __init__(self, num_spectra: int, data_dir: Path, train: bool, extract: bool = False,
                 raw_data_dir: Optional[Path] = None, galah_catalog_path: Optional[Path] = None,
                 ids_path: Optional[Path] = None):
        super().__init__(data_dir, train, extract, raw_data_dir, galah_catalog_path, ids_path)
        
        if num_spectra <= 0:
            num_spectra = len(self.fluxes)
        
        # Set the random seed for reproducibility of the subset selection
        np.random.seed(42)
        subset_indices = np.random.choice(len(self.fluxes), num_spectra, replace=False)
        
        self.fluxes = self.fluxes[subset_indices]
        self.wavelengths = self.wavelengths[subset_indices]
        self.ids = self.ids[subset_indices]
        self.labels = self.labels[subset_indices]
        self.label_errs = self.label_errs[subset_indices]
        self.fluxes_errs = self.fluxes_errs[subset_indices]
        
        # Also truncate the normalization arrays to match the subset
        if hasattr(self, '_fluxes_means'):
            self._fluxes_means = self._fluxes_means[subset_indices]
        if hasattr(self, '_fluxes_stds'):
            self._fluxes_stds = self._fluxes_stds[subset_indices]
        if hasattr(self, 'fluxes_normalized'):
            self.fluxes_normalized = self.fluxes_normalized[subset_indices]
        if hasattr(self, 'fluxes_errs_normalized'):
            self.fluxes_errs_normalized = self.fluxes_errs_normalized[subset_indices]


def create_and_extract_dataset(test_name: str = TEST_NAME, data_path: str = DATA_PATH, raw_data_path: str = RAW_DATA_PATH, galah_catalog_path: str = GALAH_CATALOG_PATH):
    data_dir = Path(data_path) / test_name
    raw_data_dir = Path(raw_data_path) / test_name
    galah_catalog_path = Path(galah_catalog_path)
    ids_path = raw_data_dir / f"{test_name}_ids.fits"
    
    dataset = GALAHDataset(data_dir=data_dir, raw_data_dir=raw_data_dir, galah_catalog_path=galah_catalog_path, ids_path=ids_path, train=True, extract=True)
    dataset_test = GALAHDataset(data_dir=data_dir, raw_data_dir=raw_data_dir, galah_catalog_path=galah_catalog_path, ids_path=ids_path, train=False, extract=True)

import fire

if __name__ == "__main__":
    fire.Fire(create_and_extract_dataset)