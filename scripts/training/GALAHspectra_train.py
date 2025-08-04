from daep.pipelines.training import train
import fire

def main(config_path: str = "configs/config_train_galah.json", **kwargs):
    train(config_path, 'spectra', **kwargs)

if __name__ == '__main__':
    fire.Fire(main)
