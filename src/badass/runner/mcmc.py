from astropy.io import fits
from dataclasses import asdict, dataclass, field
import emcee
import numpy as np
import pandas as pd
import pathlib
from scipy import stats
from typing import Callable, List, Union

from badass.badass_utils import badass_test_suite
from badass.runner import BadassResult, BadassRunContext, ParamResult
from badass.utils import plotting
import badass.utils.utils as ba_utils


@dataclass
class ConfidenceInterval:
    conf: float
    lo: float = np.nan
    hi: float = np.nan


@dataclass
class MCMCParamResult(ParamResult):
    ci_68: ConfidenceInterval
    ci_95: ConfidenceInterval
    mean: float
    med_abs_dev: float
    post_max: float


    @classmethod
    def from_chain(cls, name, chain):
        best_fit = np.nanmedian(chain)

        ci68 = ConfidenceInterval(0.68)
        lo, hi = ba_utils.compute_HDI(chain, ci68.conf)
        ci68.lo, ci68.hi = best_fit - lo, hi - best_fit

        ci95 = ConfidenceInterval(0.95)
        lo, hi = ba_utils.compute_HDI(chain, ci95.conf)
        ci95.lo, ci95.hi = best_fit - lo, hi - best_fit

        # TODO: this sometimes fails if the values in the chain are too close
        #   to create adequate bins. Another way to handle this case?
        try:
            hist, bin_edges = np.histogram(chain, bins='doane', density=False)
            post_max = bin_edges[hist.argmax()]
        except:
            post_max = np.nan

        # TODO: flags
        return cls(name, best_fit, np.nanstd(chain), 0, ci68, ci95, np.nanmean(chain), stats.median_abs_deviation(chain), post_max)



@dataclass
class MCMCResult(BadassResult):
    OUT_NAME = 'mcmc_result'
    PLOT_FUNC = plotting.plot_mcmc_results

    backend_file: pathlib.Path = None
    backend: emcee.backends.HDFBackend = None


    def __post_init__(self):
        super().__post_init__()

        self.backend_file = self.out_dir.joinpath('mcmc_chains.h5')
        self.backend = emcee.backends.HDFBackend(self.backend_file)


    def set_final_theta(self):
        chains = self.ctx.sampler.get_chain(discard=self.ctx.burn_in, flat=True).T
        self.final_theta = [np.nanmedian(c) for c in chains]


    def collect_final_parameters(self):
        param_chains = self.ctx.sampler.get_chain(discard=self.ctx.burn_in, flat=True)
        blob_chains = self.ctx.sampler.get_blobs(discard=self.ctx.burn_in, flat=True)

        fp_chain = param_chains.T
        param_chains = self.ctx.param_reg.evaluate_chains(fp_chain)

        for pname, chain in param_chains.items():
            self.final_params[pname] = MCMCParamResult.from_chain(pname, chain)

        for idx, bname in enumerate(self.ctx.blob_order):
            self.final_params[pname] = MCMCParamResult.from_chain(bname, blob_chains[bname])


@dataclass(kw_only=True)
class MCMCRunner(BadassRunContext):
    result_cls = MCMCResult

    initial_theta: np.ndarray
    blob_order: list = field(default_factory=list)

    taus: List[np.ndarray] = field(default_factory=list)
    tolerances: List[np.ndarray] = field(default_factory=list)
    prev_tau: np.ndarray = None

    min_samp: int = 0
    ncor_times: int = 0
    conv_type: Union[str,tuple] = ''
    conv_func: Callable = None
    conv_tau: np.ndarray = None

    stop_iter: int = 0
    burn_in: int = 0
    converged: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not self.source.valid:
            return

        for k,v in self.cfg.mcmc.model_dump().items():
            setattr(self,k,v)

        self.param_reg.update(self.initial_theta)

        ndim = self.param_reg.free_count
        self.nwalkers = max(self.nwalkers, 2*ndim)

        # The blob_dtypes order needs to match what is returned from lnprob_wrapper
        self.blob_order = ['LOG_LIKE', 'R_SQUARED', 'RCHI2_RATIO'] + self.blob_reg.get_blob_names()
        blob_dtypes = [(bname, np.float32) for bname in self.blob_order]
        self.sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.lnprob_wrapper, blobs_dtype=blob_dtypes, backend=self.result.backend)

        self.setup_autocorr()


    def setup_autocorr(self):
        if not self.auto_stop:
            return

        self.prev_tau = np.full(self.param_reg.free_count, np.inf)

        conv_types = {
            'mean': self.mean_conv,
            'median': self.median_conv,
            'all': self.all_conv,
        }

        if isinstance(self.conv_type,tuple):
            self.conv_func = self.param_conv
            self.conv_idx = np.array([i for i, key in enumerate(self.cur_params.keys()) if key in self.conv_type])
        elif self.conv_type in conv_types:
            self.conv_func = conv_types[self.conv_type]
        else:
            self.conv_func = self.all_conv

        self.stop_iter = self.max_iter


    def lnprob_wrapper(self, fit_vals):
        if any([np.isnan(v) for v in fit_vals]):
            return np.inf

        self.param_reg.update(fit_vals)

        lp, ll = self.lnprob()

        # TODO: do we really want to compute all of these every time?
        self.blob_reg.compute_all()
        blobs_dict = self.blob_reg.get_blobs_dict()
        blobs_dict['LOG_LIKE'] = ll

        blobs_dict['R_SQUARED'] = badass_test_suite.r_squared(self.fit_flux, self.model)
        blobs_dict['RCHI2_RATIO'] = badass_test_suite.r_chi_squared(self.fit_flux, self.model, self.fit_err, self.param_reg.free_count)

        blobs = [blobs_dict[bname] for bname in self.blob_order]

        # return values after the first need to be in the order specified in blob_dtypes passed to EnsembleSampler
        return lp + ll, *blobs


    def initialize_walkers(self):
        # Initializes the MCMC walkers within bounds and soft constraints

        free_params = self.param_reg.free_params.values()
        cur_pvals = [p.value for p in free_params]
        walkers = cur_pvals + 1e-3 * np.random.randn(self.nwalkers, len(free_params))

        for param in free_params:
            for w in range(self.nwalkers):
                while (walkers[w][param.idx] < param.plim[0]) or (walkers[w][param.idx] > param.plim[1]):
                    walkers[w][param.idx] = param.value + 1e-3 * np.random.randn(1)[0]

        # TODO: soft constraints

        return walkers


    def run(self):
        pos = self.initialize_walkers()

        for result in self.sampler.sample(pos, iterations=self.max_iter):
            it = self.sampler.iteration
            if (it >= self.write_thresh) and (it % self.write_iter == 0):
                self.log.info('MCMC iteration: %d' % it)
                if self.auto_stop and self.check_convergence():
                    break


    def mean_conv(self, sampler, tau, tol):
        par_conv = np.array([x for x in range(len(tau)) if round(tau[x],1) > 1.0]) # TODO: print converged params
        return (par_conv.size > 0) and (sampler.iteration > (np.nanmean(tau[par_conv]) * self.ncor_times) and (np.nanmean(tol[par_conv]) < self.autocorr_tol))

    def median_conv(self, sampler, tau, tol):
        par_conv = np.array([x for x in range(len(tau)) if round(tau[x],1) > 1.0]) # TODO: print converged params
        return (par_conv.size > 0) and (sampler.iteration > (np.nanmedian(tau[par_conv]) * self.ncor_times) and (np.nanmedian(tol[par_conv]) < self.autocorr_tol))

    def all_conv(self, sampler, tau, tol):
        return (all(sampler.iteration > tau*self.ncor_times)) and (all(tau > 1.0)) and (all(tol < self.autocorr_tol))

    def param_conv(self, sampler, tau, tol):
        return (all(sampler.iteration > tau[self.conv_idx]*self.ncor_times)) and (all(tau[self.conv_idx] > 1.0)) and (all(tol[self.conv_idx] < self.autocorr_tol))


    def check_convergence(self):
        it = self.sampler.iteration
        self.past_miniter = ((it >= self.write_thresh) and (it >= self.min_iter))
        if not self.past_miniter:
            return False

        tau = self.sampler.get_autocorr_time(quiet=True)
        tol = (np.abs(tau-self.prev_tau)/self.prev_tau) * 100
        print(tol)
        self.taus.append(tau)
        self.tolerances.append(tol)

        if (not self.converged) and (self.conv_func(self.sampler, tau, tol)):
            self.log.info('Converged at %d iterations\nPerforming %d iterations of sampling'%(it, self.min_samp))
            self.burn_in = it
            self.stop_iter = it+self.min_samp
            self.conv_tau = tau
            self.converged = True

        elif (self.converged) and (not self.conv_func(self.sampler, tau, tol)):
            self.log.info('Iteration: %d - Jumped out of convergence, resetting burn_in and max_iter'%it)
            self.burn_in = self.cfg.mcmc.burn_in
            self.stop_iter = self.cfg.mcmc.max_iter
            self.converged = False

        if it == self.stop_iter:
            return True

        self.prev_tau = tau
        return False

