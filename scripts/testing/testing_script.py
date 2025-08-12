from daep.pipelines.testing import run_tests
import fire

def main(config_path: str = "configs/config_test.yaml", **kwargs):
    run_tests(config_path, plot_from_saved=False, **kwargs)

if __name__ == '__main__':
    fire.Fire(main)
