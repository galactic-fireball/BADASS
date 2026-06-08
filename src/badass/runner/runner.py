from dataclasses import dataclass
import logging
import numpy as np
import os
import pathlib
import shutil
import time
from typing import Any

from badass.components.params import ParameterRegistry
from badass.components.blobs import BlobRegistry
from badass.components.templates.common import initialize_templates
from badass.components.spectral_lines.spectral_line import SpectralLine
from badass.input.input import BadassSpec
from badass.utils.config import BadassConfig


# TODO: move to BadassLogger class
def make_logger(name, log_file=None):
    log = logging.getLogger('badass.%s'%name)
    log.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.addHandler(sh)

    if not log_file is None:
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setFormatter(formatter)
        log.addHandler(fh)

    return log


class BadassResult:
    OUT_NAME = 'badass_result'

    def __init__(self, ctx, name):
        self.name = name
        self.out_dir = ctx.cfg.io.output_dir.joinpath(BadassResult.OUT_NAME)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def compile_results(self, ctx):
        pass

    def dump_results(self, ctx):
        pass

    def output(self, ctx):
        pass


@dataclass
class BadassRunContext:
    result_cls = BadassResult

    source: BadassSpec = None
    cfg: BadassConfig = None
    # log: BadassLogger = None
    outdir: pathlib.Path = None

    # TODO: type should by numpy arrays?
    fit_wave: Any = None
    fit_flux: Any = None
    fit_err: Any = None


    def __post_init__(self):
        self.start_time = time.time()

        if self.outdir is None:
            if not self.cfg.io.output_dir is None:
                self.outdir = self.cfg.io.output_dir
            elif not source.file is None:
                self.outdir = source.file.with_suffix('')
            else:
                self.outdir = pathlib.Path(os.getcwd()).resolve().joinpath(self.source.name)
        if not self.outdir.is_absolute():
            self.outdir = pathlib.Path(os.getcwd()).resolve().joinpath(self.outdir)

        # TODO: implement fit status files
        if self.outdir.joinpath('results', 'mc_result', 'par_table.fits').exists():
            if self.cfg.io.overwrite:
                # TODO: set up tmp logger
                print('Removing old output directory: [%s]'%str(self.outdir))
                shutil.rmtree(str(self.outdir))
            else:
                self.source.valid = False
                self.source.err_log = 'Output directory [%s] already exists, not overwriting'%str(self.outdir)
                print(self.err_log)
                return

        self.outdir.mkdir(parents=True, exist_ok=True)
        log_dir = self.outdir.joinpath('log')
        log_dir.mkdir(parents=True, exist_ok=True) # TODO: 'log' mkdir eventually happens in separate output class

        self.log = make_logger(self.source.name, log_file=log_dir.joinpath('log.txt'))
        self.source.log = self. log # TODO: separate logger for source?

        self.source.postinit()
        if not self.source.valid:
            return

        # The spectral data currently being fit
        if not hasattr(self, 'fit_wave'):
            self.fit_wave = self.target.wave.copy()
        if not hasattr(self, 'fit_spec'):
            self.fit_spec = self.target.spec.copy()
        if not hasattr(self, 'fit_noise'):
            self.fit_noise = self.target.noise.copy()

        max_flux = np.nanmax(self.fit_spec)*1.5
        median_flux = np.nanmedian(self.fit_spec)

        # For use in parameter/hyperpar expressions
        component_args = {
            'median_flux':median_flux, 'max_flux':max_flux,
            'min_wave':np.min(self.fit_wave), 'max_wave':np.max(self.fit_wave),
        }

        self.param_reg = ParameterRegistry(self)
        self.blob_reg = BlobRegistry(self)

        self.templates = initialize_templates(self)
        self.line_list = SpectralLine.initialize_spectral_lines(self, [line.dict() for line in self.cfg.user_lines])

        self.param_reg.init_values(component_args)
        self.param_reg.validate_constraints()

        self.param_reg.dump_parameters()
        self.blob_reg.dump_blobs()

        # current model components
        self.comps = {}
        self.model = np.zeros_like(self.fit_spec)

        self.result = self.result_cls(self, target.name)


    def lnprob_wrapper(self, fit_vals):
        if any([np.isnan(v) for v in fit_vals]):
            return np.inf

        self.param_reg.update_vals(fit_vals)
        return -(self.lnprob()[0]) # only care about the first returned value


    def lnprob(self):
        # Log-probability function

        ll = self.lnlike()
        lp = self.param_reg.get_lnpriors()
        if not np.isfinite(lp):
            return -np.inf, ll

        # return log-prob and log-like:
        # bootstrap mode will ignore the latter, mcmc will return it as a blob
        return lp + ll, ll


    def lnlike(self):
        # Log-likelihood function

        self.fit_model()
        fit_mask = self.target.fit_mask
        fit_stat = self.cfg.fit.fit_stat

        data = self.fit_spec[fit_mask]
        model = self.model[fit_mask]
        noise = self.fit_noise[fit_mask]

        if fit_stat == 'ML':
            return -0.5*np.sum(((data-model)**2/noise**2) + np.log(2*np.pi*noise**2), axis=0)

        if fit_stat == 'OLS':
            return -np.sum((data - model)**2, axis=0)


    def fit_model(self):
        host_model = np.copy(self.fit_spec)

        self.comps = {}
        extra_comps = {}
        for line in self.line_list:
            host_model = line.add_components(self.comps, host_model, extra_comps)

        for template in self.templates.values():
            host_model = template.add_components(self.comps, host_model)

        # The final model
        self.model = np.sum(list(self.comps.values()), axis=0)

        # Add extra comps after we've computed the model
        self.comps.update(extra_comps)


