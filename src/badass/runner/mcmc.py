from astropy.io import fits
from dataclasses import dataclass, field
import emcee
import numpy as np
import pandas as pd
import pathlib
from scipy import stats
from typing import Callable, List, Union

from badass.badass_utils import badass_test_suite
from badass.runner import BadassResult, BadassRunContext
from badass.utils import plotting
import badass.utils.utils as ba_utils


@dataclass
class MCMCResult(BadassResult):
    OUT_NAME = 'mcmc_result'
    PLOT_FUNC = plotting.plot_mcmc_results

    result_attrs = ['best_fit', 'ci_68_low', 'ci_68_upp', 'ci_95_low', 'ci_95_upp',
                    'mean', 'std_dev', 'median', 'med_abs_dev', 'flag']


    chain_df: pd.DataFrame = None
    chain_file: pathlib.Path = None
    mcmc_result_chains: dict = field(default_factory=dict)
    mcmc_results_dict: dict = field(default_factory=dict)


    def __post_init__(self):
        super().__post_init__()

        self.chain_df = pd.DataFrame(columns=['iter']+list(self.ctx.param_reg.params.keys()))
        chain_dict = {'iter': 0}
        chain_dict.update(self.ctx.param_reg.get_param_dict())
        self.chain_df.loc[len(self.chain_df)] = chain_dict
        self.chain_file = self.out_dir.joinpath('MCMC_chain.csv')

        # TODO: do we need both of these?
        self.mcmc_result_chains['chains'] = {}
        self.mcmc_result_chains['flat_chains'] = {}


    def add_chain(self, ctx, sampler):
        chain_dict = {'iter': sampler.iteration}
        last_iter = self.chain_df.iter.values[-1]
        chain_vals = {param.name:np.nanmedian(sampler.chain[:,last_iter:,i]) for i, param in enumerate(ctx.param_reg.free_params.values())}
        chain_dict.update(chain_vals)

        self.chain_df.loc[len(self.chain_df)] = chain_dict
        self.chain_df.to_csv(self.chain_file, index=False)


    def calc_mcmc_blob(self, ctx):
        # TODO: do we really want to compute all of these every time?
        ctx.blob_reg.compute_all()

        blob_dict = {}
        for blob in ctx.blob_reg.get_blobs():
            if isinstance(blob.cur_val, dict):
                for key, val in blob.cur_val.items():
                    blob_dict[key] = val
            else:
                blob_dict[blob.name] = blob.cur_val

        # TODO: add these as blob params?
        blob_dict['R_SQUARED'] = badass_test_suite.r_squared(ctx.fit_flux, ctx.model)
        blob_dict['RCHI_SQUARED'] = badass_test_suite.r_chi_squared(ctx.fit_flux, ctx.model, ctx.fit_err, ctx.param_reg.free_count)

        return blob_dict


    def collect_mcmc_results(self):
        ctx = self.ctx
        chain = ctx.sampler.chain
        nwalkers, niters, nparams = chain.shape
        if self.ctx.burn_in >= niters: ctx.burn_in = int(niters/2)

        def flatten_chain(chain):
            # TODO: zero-trim if converged before max iters
            chain[~np.isfinite(chain)] = 0
            return chain[:,ctx.burn_in:].flatten()


        def get_key_chain(chain, param):
            # Loop through each iteration of the chain and grab the parameter value
            with np.nditer([chain, None], flags=['refs_ok', 'multi_index', 'buffered'], op_flags=[['readonly'], ['writeonly', 'allocate', 'no_broadcast']]) as it:
                for x, y in it:
                    y[...] = x.item()[param]
                return it.operands[1]

        for param in ctx.param_reg.free_params.values():
            self.mcmc_result_chains['chains'][param.name] = ctx.sampler.chain[:,:,param.idx]
            self.mcmc_result_chains['flat_chains'][param.name] = flatten_chain(ctx.sampler.chain[:,:,param.idx])

        full_blob = np.swapaxes(ctx.sampler.get_blobs()['full_blob'],0,1)
        keys = full_blob[0][0].keys()

        # self.mcmc_result_chains['chains'].update(
        #     {key: np.array([[sample[key] for sample in row] for row in full_blob]) for key in keys}
        # )
        # for pname, chain in self.mcmc_result_chains['chains'].items():
        #     self.mcmc_result_chains['flat_chains'][pname] = flatten_chain(chain)

        for key in keys:
            val = get_key_chain(full_blob, key).astype(float)
            self.mcmc_result_chains['chains'][key] = val
            self.mcmc_result_chains['flat_chains'][key] = flatten_chain(val)

        # TODO: create a flag_behavior function in the Parameter class
        for key, chain, in self.mcmc_result_chains['flat_chains'].items():
            if len(chain) == 0:
                self.mcmc_results_dict[key] = {k.np.nan for k in MCMCResult.result_attrs}
                continue

            par_results = {}

            # if key.split('_')[-1] == 'AMP':
            #     chain *= ctx.source.fit_norm

            post_med = np.nanmedian(chain)
            par_results['best_fit'] = post_med

            # 68% confidence interval
            lo, hi = ba_utils.compute_HDI(chain, 0.68)
            par_results['ci_68_low'] = post_med - lo
            par_results['ci_68_upp'] = hi - post_med

            # 95% confidence interval
            lo, hi = ba_utils.compute_HDI(chain, 0.95)
            par_results['ci_95_low'] = post_med - lo
            par_results['ci_95_upp'] = hi - post_med

            # TODO: this sometimes fails if the values in the chain are too close
            #   to create adequate bins. Another way to handle this case?
            try:
                hist, bin_edges = np.histogram(chain, bins='doane', density=False)
                par_results['post_max'] = bin_edges[hist.argmax()]
            except:
                par_results['post_max'] = np.nan

            par_results['mean'] = np.nanmean(chain)
            par_results['std_dev'] = np.nanstd(chain)
            par_results['median'] = post_med
            par_results['med_abs_dev'] = stats.median_abs_deviation(chain)
            par_results['flat_chain'] = chain

            par_results['flag'] = 0

            self.mcmc_results_dict[key] = par_results

        # TODO: hack for now, fix!
        self.params = self.mcmc_results_dict
        self.line_list = ctx.line_list

        # update params for final model fit
        med_values = [v['best_fit'] for p,v in self.params.items() if ctx.param_reg.is_free(p)]
        ctx.param_reg.update(med_values)
        ctx.fit_model()

        ctx.param_reg.dump_parameters()

        self.components = {k:comp*ctx.source.fit_norm for k,comp in ctx.comps.items()}

        self.meta_components['wave'] = ctx.fit_wave.copy()
        meta_comps_dict = {'data':ctx.fit_flux.copy(),'noise':ctx.fit_err.copy(),'model':ctx.model.copy(),}
        for comp, comp_arr in meta_comps_dict.items():
            self.meta_components[comp] = comp_arr * ctx.source.fit_norm
        self.meta_components['resid'] = (ctx.fit_flux-ctx.model) * ctx.source.fit_norm
        self.meta_components['mask'] = ctx.source.fit_mask.copy()

        self.mcmc_output(ctx)


    def mcmc_output(self, ctx):
        # Write chains
        # if self.cfg.out.write_chain:
        #     cols = []
        #     for key, chain in self.mcmc_result_chains['chains'].items():
        #         cols.append(fits.Column(name=key, format='%dD'%(chain.shape[0]*chain.shape[1]), dim='(%d,%d)'%(chain.shape[1],chain.shape[0]), array=[chain]))
        #     cols = fits.ColDefs(cols)
        #     hdu = fits.BinTableHDU.from_columns(cols)
        #     hdu.writeto(self.target.outdir.joinpath('log', 'MCMC_chains.fits'), overwrite=True)
        #     hdu.close()


        # TODO: remove redundancy with ml bmc.fits
        # Write best-fit components
        cols = []
        for key, value in self.components.items():
            cols.append(fits.Column(name=key.upper(), format='E', array=value))
        for key, value in self.meta_components.items():
            cols.append(fits.Column(name=key.upper(), format='E', array=value))

        cols = fits.ColDefs(cols)
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.writeto(self.out_dir.joinpath('best_model_components.fits'), overwrite=True)


        # TODO: remove redundancy with ml pt.fits
        # Write parameter table
        hdr = fits.Header()
        hdr['z'] = ctx.source.target.z
        hdr['med_noise'] = np.nanmedian(ctx.fit_err)
        hdr['velscale'] = ctx.source.velscale
        hdr['fit_norm'] = ctx.source.fit_norm
        hdr['flux_norm'] = ctx.source.flux_norm
        primary = fits.PrimaryHDU(header=hdr)

        cols_dict = {'parameter': []}
        cols_dict.update({k:[] for k in MCMCResult.result_attrs})
        for key, result_dict in self.mcmc_results_dict.items():
            cols_dict['parameter'].append(key)
            for attr in MCMCResult.result_attrs:
                cols_dict[attr].append(result_dict[attr])

        cols = []
        for key, values in cols_dict.items():
            fmt = 'E' if key != 'parameter' else '30A'
            cols.append(fits.Column(name=key, format=fmt, array=values))
        cols = fits.ColDefs(cols)
        table = fits.BinTableHDU.from_columns(cols)

        hdu = fits.HDUList([primary, table])
        hdu.writeto(self.out_dir.joinpath('par_table.fits'), overwrite=True)
        hdu.close()


@dataclass(kw_only=True)
class MCMCRunner(BadassRunContext):
    result_cls = MCMCResult

    initial_theta: np.ndarray

    times: List[np.ndarray] = field(default_factory=list)
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

        dtype = [('full_blob',dict),]
        self.sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.lnprob_wrapper, blobs_dtype=dtype)#, backend=backend)

        if self.auto_stop:
            self.prev_tau = np.full(len(self.ctx.param_reg.params), np.inf)
            self.conv_tau = np.full(len(self.ctx.param_reg.params), np.inf)

            conv_types = {
                'mean': self.mean_conv,
                'median': self.median_conv,
                'all': self.all_conv,
            }

            if isinstance(self.conv_type,tuple):
                self.conv_func = self.param_conv
                self.conv_idx = np.array([i for i, key in enumerate(self.ctx.cur_params.keys()) if key in self.conv_type])
            elif self.conv_type in conv_types:
                self.conv_func = conv_types[self.conv_type]
            else:
                self.conv_func = self.all_conv

            self.stop_iter = self.max_iter


    def run(self):
        self.log.info('MCMCRunner run')
        self.run_mcmc()


    def finalize(self):
        self.result.collect_mcmc_results()


    def lnprob_wrapper(self, fit_vals):
        if any([np.isnan(v) for v in fit_vals]):
            return np.inf

        self.param_reg.update(fit_vals)

        lp, ll = self.lnprob()
        blob_dict = self.result.calc_mcmc_blob(self)
        blob_dict['LOG_LIKE'] = ll
        return lp, blob_dict


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


    def run_mcmc(self):
        pos = self.initialize_walkers()

        for result in self.sampler.sample(pos, iterations=self.max_iter):
            it = self.sampler.iteration
            if (it >= self.write_thresh) and (it % self.write_iter == 0):
                self.log.info('MCMC iteration: %d' % it)
                self.result.add_chain(self, self.sampler)

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

        tau = autocorr_convergence(self.sampler.chain) # autocorr time for each parameter
        self.times.append(tau)
        tol = (np.abs(tau-self.prev_tau)/self.prev_tau) * 100 # tolerances
        self.tolerances.append(tol)

        if (not self.converged) and (self.conv_func(self.sampler, tau, tol)):
            self.ctx.log.info('Converged at %d iterations\nPerforming %d iterations of sampling'%(it, self.min_samp))
            self.burn_in = it
            self.stop_iter = it+self.min_samp
            self.conv_tau = tau
            self.converged = True

        elif (self.converged) and (not self.conv_func(self.sampler, tau, tol)):
            self.ctx.log.info('Iteration: %d - Jumped out of convergence, resetting burn_in and max_iter'%it)
            self.burn_in = self.ctx.cfg.mcmc.burn_in
            self.stop_iter = self.ctx.cfg.mcmc.max_iter
            self.converged = False

        if it == self.stop_iter:
            return True

        self.prev_tau = tau
        return False


def autocorr_convergence(sampler_chain, c=5.0):
    """
    Estimates the autocorrelation times using the
    methods outlined on the Autocorrelation page
    on the emcee website:
    https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
    """

    npar = np.shape(sampler_chain)[2]

    tau_est = np.empty(npar)
    for p in range(npar):
        y = sampler_chain[:,:,p]
        f = np.zeros(y.shape[1])
        for yy in y:
            f += autocorr_func_1d(yy)
        f /= len(y)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = auto_window(taus, c)
        tau_est[p] = taus[window]
    return tau_est


def autocorr_func_1d(x, norm=True):
    # Estimates the 1d autocorrelation function for a chain.

    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError('invalid dimensions for 1D autocorrelation function')
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.nanmean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


def auto_window(taus, c):
    # Automated windowing procedure following Sokal (1989)
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


