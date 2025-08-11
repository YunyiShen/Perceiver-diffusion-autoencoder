from daep.pipelines.training import train
import fire

def main(config_path: str = "configs/config_classifier.yaml", **kwargs):
    train(config_path, model_type='classifier', **kwargs)

if __name__ == '__main__':
    fire.Fire(main)
