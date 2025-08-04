from daep.pipelines.testing import run_tests
import fire

def main(config_path: str = "configs/configs_test_galah.json", **kwargs):
    run_tests(config_path, 'spectra', **kwargs)

if __name__ == '__main__':
    fire.Fire(main)
