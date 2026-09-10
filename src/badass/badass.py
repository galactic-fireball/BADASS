"""Bayesian AGN Decomposition Analysis for SDSS Spectra (BADASS3)

BADASS is an open-source spectral analysis tool designed for detailed decomposition
of Sloan Digital Sky Survey (SDSS) spectra, and specifically designed for the
fitting of Type 1 ("broad line") Active Galactic Nuclei (AGN) in the optical.
The fitting process utilizes the Bayesian affine-invariant Markov-Chain Monte
Carlo sampler emcee for robust parameter and uncertainty estimation, as well
as autocorrelation analysis to access parameter chain convergence.
"""

__author__ = 'Remington O. Sexton (USNO), Sara M. Doan (GMU), Michael A. Reefe (GMU), William Matzko (GMU), Nicholas Darden (UCR)'
__copyright__ = 'Copyright (c) 2023 Remington Oliver Sexton'
__credits__ = ['Remington O. Sexton (GMU/USNO)', 'Sara Doan (GMU)', 'Michael A. Reefe (GMU)', 'William Matzko (GMU)', 'Nicholas Darden (UCR)']
__license__ = 'MIT'
__version__ = '11.0.0'
__maintainer__ = 'Sara Doan'
__email__ = 'sdoan2@gmu.edu'
__status__ = 'Release'


# TODO: fix warnings
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning) 
warnings.filterwarnings('ignore', category=UserWarning) 


from badass.utils.config import BadassConfig
from badass.input.input import BadassSpec
from badass.runner.pipeline import BadassPipeline


def run_BADASS(inputs, **kwargs):
    cfg = BadassConfig.get_config_from_args(kwargs)
    source = BadassSpec.get_inputs(inputs, cfg)

    pipeline = BadassPipeline.init(source, cfg)
    results = pipeline.run()
    pipeline.finalize()
    return results


def target_check(inputs, **kwargs):
    cfg = BadassConfig.get_config_from_args(kwargs)
    targets = BadassSpec.get_inputs(inputs, cfg)
    print('Fitting %d targets'%len(targets))
