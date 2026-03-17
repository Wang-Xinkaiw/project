"""
Proximal operators for various regularization penalties
"""

from .prox_l1 import prox_l1
from .prox_elasticnet import prox_elasticnet
from .prox_gl1 import prox_gl1
from .prox_ksupport import prox_ksupport
from .prox_l21 import prox_l21
from .prox_nuclear import prox_nuclear
from .prox_tnn import prox_tnn
from .project_box import project_box
from .project_simplex import project_simplex
from .project_fantope import project_fantope

__all__ = [
    'prox_l1',
    'prox_elasticnet',
    'prox_gl1',
    'prox_ksupport',
    'prox_l21',
    'prox_nuclear',
    'prox_tnn',
    'project_box',
    'project_simplex',
    'project_fantope',
]
