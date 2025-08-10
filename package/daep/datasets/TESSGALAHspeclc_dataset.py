
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional
from daep.utils.general_utils import detect_env, set_paths

TEST_NAME = 'tess_20k'
ENV = detect_env()

BASE_PATH, MODEL_PATH, DATA_PATH, RAW_DATA_PATH = set_paths(ENV, 'lightcurve')

# TARGETS_PATH = RAW_DATA_PATH + "/" + TEST_NAME + "/targets_qlp.csv"
# TESS_XMATCH_CATALOG_PATH = RAW_DATA_PATH + "/" + TEST_NAME + "/id_catalog_gt_1800.fits"

from torch.utils.data import Dataset


class TESSGALAHDataset(Dataset):
    
    def __init__(self, lightcurve_dataset: Optional[Path] = None,
                 spectra_dataset: Optional[Path] = None):
        self.lightcurve_dataset = lightcurve_dataset
        self.spectra_dataset = spectra_dataset
        
        print(f"Crossmatching lightcurve and spectra datasets")
        self.catalog = self.crossmatch()
    
    def __len__(self):
        return len(self.catalog)
    
    def __getitem__(self, idx):
        raise NotImplementedError("__getitem__ not implemented: use TESSDatasetProcessed instead")
    
    def crossmatch(self):
        
        lightcurve_catalog = self.lightcurve_dataset.catalog
        spectra_catalog = self.spectra_dataset.catalog
        
        print(f"Lightcurve catalog length: {len(lightcurve_catalog)}")
        print(f"Spectra catalog length: {len(spectra_catalog)}")
        
        # Add index columns to both catalogs for traceability after merging
        lightcurve_catalog['lightcurve_idx'] = lightcurve_catalog.index.copy()
        spectra_catalog['spectra_idx'] = spectra_catalog.index.copy()
        
        if 'dr2_source_id' in spectra_catalog.columns:
            catalog = pd.merge(lightcurve_catalog, spectra_catalog, left_on='GAIA', right_on='dr2_source_id', how='inner', suffixes=('_lc', '_galah'))
        else:
            catalog = pd.merge(lightcurve_catalog, spectra_catalog, left_on='GAIA', right_on='GAIA', how='inner', suffixes=('_lc', '_galah'))        
        
        print(f"Catalog length: {len(catalog)}")
        return catalog

    def idx_to_lightcurve_idx(self, idx):
        return self.catalog.iloc[idx]['lightcurve_idx']
    
    def idx_to_spectra_idx(self, idx):
        return self.catalog.iloc[idx]['spectra_idx']

class TESSGALAHDatasetProcessed(TESSGALAHDataset):
    def __init__(self, lightcurve_dataset: Optional[Path] = None, spectra_dataset: Optional[Path] = None):
        super().__init__(lightcurve_dataset, spectra_dataset)
    
    def __len__(self):
        return len(self.catalog)
    
    def __getitem__(self, idx):
        idx = int(idx) % len(self.catalog)
        
        spectra_idx = self.idx_to_spectra_idx(idx)
        lightcurve_idx = self.idx_to_lightcurve_idx(idx)
        
        res = {"flux": self.spectra_dataset.fluxes_normalized[spectra_idx],
               "flux_err": self.spectra_dataset.fluxes_errs_normalized[spectra_idx],
               "wavelength": self.spectra_dataset.wavelengths[spectra_idx], 
               "phase": torch.tensor(0.),
               "spectra_idx": torch.tensor(spectra_idx),
               "speclc_idx": torch.tensor(idx)}
        
        photores = {"flux": self.lightcurve_dataset.fluxes_normalized[lightcurve_idx],
                    "flux_err": self.lightcurve_dataset.fluxes_errs_normalized[lightcurve_idx],
                    "time": self.lightcurve_dataset.times_normalized[lightcurve_idx], 
                    "lightcurve_idx": torch.tensor(lightcurve_idx),
                    "speclc_idx": torch.tensor(idx)}
        
        return {"spectra": res, "photometry": photores}
    
    def get_actual_data(self, idx):
        return {
                "lightcurve": self.lightcurve_dataset.get_actual_lightcurve(self.idx_to_lightcurve_idx(idx)),
                "spectrum": self.spectra_dataset.get_actual_spectrum(self.idx_to_spectra_idx(idx))
                }
    
    def get_actual_data_from_gaia_id(self, gaia_id):
        idx = self.catalog[self.catalog['GAIA'] == gaia_id].index[0]
        return self.get_actual_data(idx)


#TODO: FIX THIS -- spectra_idx is 711, out of bounds for num_samples = 30
# slice crossmatched catalog to num_samples??
class TESSGALAHDatasetProcessedSubset(TESSGALAHDatasetProcessed):
    def __init__(self, num_samples: int, lightcurve_dataset: Optional[Path] = None, spectra_dataset: Optional[Path] = None,
                 ):
        super().__init__(lightcurve_dataset, spectra_dataset)
        
        if num_samples <= 0:
            num_samples = len(self.catalog)
        
        # Set the random seed for reproducibility of the subset selection
        np.random.seed(42)
        subset_indices = np.random.choice(len(self.catalog), num_samples, replace=False)
        print(f"Subset of catalog taken: {len(self.catalog)} indices ({subset_indices})")
        
        self.catalog = self.catalog.iloc[subset_indices].reset_index(drop=True)
        self.catalog['pre_subset_idx'] = subset_indices
            
    def idx_to_lightcurve_idx(self, idx):
        # Use the current subset index to get the lightcurve_idx from the subset catalog
        return self.catalog[self.catalog.index == idx]['lightcurve_idx'].values[0]
    
    def idx_to_spectra_idx(self, idx):
        # Use the current subset index to get the spectra_idx from the subset catalog
        return self.catalog[self.catalog.index == idx]['spectra_idx'].values[0]

    def pre_subset_idx_to_idx(self, idx):
        return self.catalog[self.catalog['pre_subset_idx'] == idx].index[0]

    def get_actual_data(self, idx):        
        return {
                "lightcurve": self.lightcurve_dataset.get_actual_lightcurve(self.idx_to_lightcurve_idx(idx)),
                "spectrum": self.spectra_dataset.get_actual_spectrum(self.idx_to_spectra_idx(idx))
                }
    
    def get_actual_data_from_gaia_id(self, gaia_id):
        idx = self.catalog[self.catalog['GAIA'] == gaia_id].index[0]
        return self.get_actual_data(idx)