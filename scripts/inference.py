import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="inference", version_base="1.2")
def main(config: DictConfig):
    pass


if __name__ == "__main__":
    main()
