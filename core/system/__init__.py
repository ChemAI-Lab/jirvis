from .jirvis import *
from .nmr_enc import *
from .decoder import *
from .spectro import *

systems = {
    "jirvis": jirvis,
    "decoder": MoleculeDecoder,
    "nmr_enc": NMREncoder,
    "spectro": Spectro,
}
