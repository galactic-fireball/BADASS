import sys

from . import badass
from .badass import run_BADASS

from .components.spectral_lines.line_lists import common_lines, coronal_lines

sys.modules[__name__ + '.common_lines'] = common_lines
sys.modules[__name__ + '.coronal_lines'] = coronal_lines

__all__ = ['common_lines', 'coronal_lines']
