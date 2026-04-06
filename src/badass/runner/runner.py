from dataclasses import dataclass
import logging
import numpy as np
import time

from badass.components.params import ParameterRegistry
from badass.components.blobs import BlobRegistry
from badass.components.templates.common import initialize_templates
from badass.components.spectral_lines.spectral_line import SpectralLine


def make_logger(name):
    log = logging.getLogger('badass.%s'%name)
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())
    return log


class BadassResult:
    OUT_NAME = 'badass_result'

    def __init__(self, ctx):
        self.out_dir = ctx.cfg.io.output_dir.joinpath(BadassResult.OUT_NAME)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def compile_results(self, ctx):
        pass

    def dump_results(self, ctx):
        pass

    def output(self, ctx):
        pass


class BadassRunContext:

    result_cls = BadassResult

    def __init__(self, target, **kwargs):
        self.start_time = time.time()
        self.target = target
        self.log = make_logger(kwargs.get('name', self.target.name))
        self.target.log = self.log
        self.target.postinit()

        self.__dict__.update(kwargs)

        if not hasattr(self, 'cfg'):
            self.cfg = target.cfg

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

        self.result = self.result_cls(self)


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

        for line in self.line_list:
            host_model = line.add_components(self.comps, host_model)

        for template in self.templates.values():
            host_model = template.add_components(self.comps, host_model)

        # The final model
        self.model = np.sum(list(self.comps.values()), axis=0)

