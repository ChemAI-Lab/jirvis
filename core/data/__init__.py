from .ir_datamodule import *
from .nmr_datamodule import *
from .multimodal_datamodule import *

datasets = {
    "sim_ir": IRData,
    "sim_nmr": NMRData,
    "sim_spectro": SpectroData,
}
