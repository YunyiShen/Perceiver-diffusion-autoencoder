from daep.pipelines.training import train
import fire

def main(config_path: str = "configs/config_train_tess.json", **kwargs):
    train(config_path, 'lightcurves', **kwargs)

if __name__ == '__main__':
    fire.Fire(main)
