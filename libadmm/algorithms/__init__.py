"""
ADMM-based optimization algorithms
"""

from .comp_loss import comp_loss
from .l1 import l1
from .l1R import l1R
from .elasticnet import elasticnet
from .elasticnetR import elasticnetR
from .fusedl1 import fusedl1
from .fusedl1R import fusedl1R
from .groupl1 import groupl1
from .groupl1R import groupl1R
from .ksupport import ksupport
from .ksupportR import ksupportR
from .tracelasso import tracelasso
from .tracelassoR import tracelassoR
from .lrr import lrr
from .lrmc import lrmc
from .lrmcR import lrmcR
from .latlrr import latlrr
from .lrsr import lrsr
from .rpca import rpca
from .rmsc import rmsc
from .igc import igc
from .trpca_tnn import trpca_tnn
from .trpca_snn import trpca_snn
from .lrtc_tnn import lrtc_tnn
from .lrtc_snn import lrtc_snn
from .lrtcR_tnn import lrtcR_tnn
from .lrtcR_snn import lrtcR_snn
from .lrtr_Gaussian_tnn import lrtr_Gaussian_tnn
from .sparsesc import sparsesc

__all__ = [
    'comp_loss',
    'l1',
    'l1R',
    'elasticnet',
    'elasticnetR',
    'fusedl1',
    'fusedl1R',
    'groupl1',
    'groupl1R',
    'ksupport',
    'ksupportR',
    'tracelasso',
    'tracelassoR',
    'lrr',
    'lrmc',
    'lrmcR',
    'latlrr',
    'lrsr',
    'rpca',
    'rmsc',
    'igc',
    'trpca_tnn',
    'trpca_snn',
    'lrtc_tnn',
    'lrtc_snn',
    'lrtcR_tnn',
    'lrtcR_snn',
    'lrtr_Gaussian_tnn',
    'sparsesc',
]
