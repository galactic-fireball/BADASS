from dataclasses import dataclass
import emcee
import numpy as np
import pandas as pd

from badass.badass_utils import badass_test_suite
from badass.runner import BadassResult, BadassRunContext


class MCMCResult(BadassResult):
    OUT_NAME = 'mcmc_result'


    def __init__(self, ctx, name):
        super().__init__(ctx, name)

        self.chain_df = pd.DataFrame(columns=['iter']+list(ctx.param_reg.param_names))
        cur_params = {param.name:param.value for param in ctx.param_reg.get_free_parameters()}
        chain_dict = {'iter': 0}
        chain_dict.update(cur_params)
        self.chain_df.loc[len(self.chain_df)] = chain_dict
        self.chain_file = self.out_dir.joinpath('MCMC_chain.csv')

        # TODO: do we need both of these?
        self.mcmc_result_chains = {'chains':{}, 'flat_chains':{}}


    def add_chain(self, ctx, sampler):
        chain_dict = {'iter': sampler.iteration}
        last_iter = self.chain_df.iter.values[-1]
        chain_vals = {param.name:np.nanmedian(sampler.chain[:,last_iter:,i]) for i, param in enumerate(ctx.param_reg.get_free_parameters())}
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
        blob_dict['RCHI_SQUARED'] = badass_test_suite.r_chi_squared(ctx.fit_flux, ctx.model, ctx.fit_err, ctx.param_reg.free_param_count)

        return blob_dict


    def collect_mcmc_results(self, ctx, sampler, autocorr):
        nwalkers, niters, nparams = sampler.chain.shape
        burn_in = autocorr.burn_in if autocorr else ctx.burn_in
        if burn_in >= niters: burn_in = int(niters/2)

        def flatten_chain(chain):
            # TODO: zero-trim if converged before max iters
            chain[~np.isfinite(chain)] = 0
            return chain[:,burn_in:].flatten()

        for param in ctx.param_reg.get_free_parameters():
            self.mcmc_result_chains['chains'][param.name] = sampler.chain[:,:,param.idx]

        full_blob = sampler.get_blobs()['full_blob']
        keys = full_blob[0][0].keys()
        self.mcmc_result_chains['chains'].update(
            {key: np.array([[sample[key] for sample in row] for row in full_blob]) for key in keys}
        )

        for pname, chain in self.mcmc_result_chains['chains'].items():
            self.mcmc_result_chains['flat_chains'][pname] = flatten_chain(chain)


@dataclass
class MCMCRunner(BadassRunContext):
    result_cls = MCMCResult

    def __post_init__(self):
        super().__post_init__()
        if not self.source.valid:
            return

        for k,v in self.cfg.mcmc.model_dump().items():
            setattr(self,k,v)

        ndim = self.param_reg.free_param_count
        self.nwalkers = max(self.nwalkers, 2*ndim)

        dtype = [('full_blob',dict),]
        self.sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.lnprob_wrapper, blobs_dtype=dtype)#, backend=backend)

        # TODO
        self.autocorr = None


    def run(self):
        self.log.info('MCMCRunner run')
        self.run_mcmc()


    def finalize(self):
        self.result.collect_mcmc_results(self, self.sampler, self.autocorr)


    def lnprob_wrapper(self, fit_vals):
        if any([np.isnan(v) for v in fit_vals]):
            return np.inf

        self.param_reg.update_vals(fit_vals)

        lp, ll = self.lnprob()
        blob_dict = self.result.calc_mcmc_blob(self)
        blob_dict['LOG_LIKE'] = ll
        return lp, blob_dict


    def initialize_walkers(self):
        # Initializes the MCMC walkers within bounds and soft constraints

        free_params = self.param_reg.get_free_parameters()
        cur_pvals = [p.value for p in free_params]
        walkers = cur_pvals + 1e-3 * np.random.randn(self.nwalkers, len(free_params))

        for param in free_params:
            for w in range(self.nwalkers):
                while (walkers[w][param.idx] < param.plim[0]) or (walkers[w][param.idx] > param.plim[1]):
                    walkers[w][param.idx] = param.value + 1e-3 * np.random.randn(1)

        # TODO: soft constraints

        return walkers


    def run_mcmc(self):
        pos = self.initialize_walkers()

        for result in self.sampler.sample(pos, iterations=self.max_iter):
            it = self.sampler.iteration
            if (it >= self.write_thresh) and (it % self.write_iter == 0):
                self.log.info('MCMC iteration: %d' % it)
                self.result.add_chain(self, self.sampler)

