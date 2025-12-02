import hydra
from omegaconf import DictConfig
from jirvis.train.core.runner.runner import *


@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(config: DictConfig):
    if config.mode == "train":
        train_module(config)
    elif config.mode == "test":
        test_module(config)


if __name__ == "__main__":
    main()
