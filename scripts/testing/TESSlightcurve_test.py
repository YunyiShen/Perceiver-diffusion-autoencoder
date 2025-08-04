from daep.pipelines.testing import run_tests
import fire

def main(config_path: str = "configs/config_test_tess.json", **kwargs):
    # config_path = "/home/altair/Documents/UROP/2025_Summer/Perceiver-diffusion-autoencoder/scripts/training/configs/config_train_tess_local.json"
    run_tests(config_path, 'lightcurves', use_saved_results=False, **kwargs)

if __name__ == '__main__':
    fire.Fire(main)
