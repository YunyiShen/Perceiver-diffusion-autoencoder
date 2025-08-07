from daep.pipelines.training_classifier import train_classifier
import fire

def main(config_path: str = "configs/config_train_classifier.json", **kwargs):
    train_classifier(config_path, 'lightcurves', **kwargs)

if __name__ == '__main__':
    fire.Fire(main)
