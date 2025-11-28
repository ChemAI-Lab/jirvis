import torch
import pytorch_lightning as pl
import os

# from core.tasks.classification import CFTask
# from core.system import *
import hydra
from core.system import *
from core.data import *


def set_seed(seed):
    pl.seed_everything(seed)


def set_device(device_config):
    # set the global cuda device
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_config.cuda_visible_devices)
    torch.cuda.set_device(device_config.cuda)
    torch.set_float32_matmul_precision("medium")
    # warnings.filterwarnings("always")


def set_processtitle(cfg):
    # set process title
    import setproctitle

    setproctitle.setproctitle(cfg.process_title)


def init_experiment(cfg, **kwargs):
    cfg = cfg

    print("config:")
    for k, v in cfg.items():
        print(k, v)
    print("=" * 20)

    print("kwargs:")
    for k, v in kwargs.items():
        print(k, v)
    print("=" * 20)

    # set seed
    set_seed(cfg.seed)
    # set device
    set_device(cfg.device)

    # set process title
    set_processtitle(cfg)


def train_module(cfg):
    init_experiment(cfg)
    system_cls = systems[cfg.system.name]
    system = system_cls(cfg)

    datamodule = system.get_datamodule()

    dataloader = datamodule.train_dataloader()
    print(len(dataloader))
    print("batch size", dataloader.batch_size)

    trainer = hydra.utils.instantiate(cfg.system.train.trainer, num_sanity_val_steps=0)
    trainer.fit(system, datamodule=datamodule, ckpt_path=cfg.load_system_checkpoint)
    trainer.test(system, datamodule=datamodule)

    return {}


def test_module(cfg):
    pass
