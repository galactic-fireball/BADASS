"""Bayesian AGN Decomposition Analysis for SDSS Spectra (BADASS3)

BADASS is an open-source spectral analysis tool designed for detailed decomposition
of Sloan Digital Sky Survey (SDSS) spectra, and specifically designed for the
fitting of Type 1 ("broad line") Active Galactic Nuclei (AGN) in the optical.
The fitting process utilizes the Bayesian affine-invariant Markov-Chain Monte
Carlo sampler emcee for robust parameter and uncertainty estimation, as well
as autocorrelation analysis to access parameter chain convergence.
"""

import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
import astropy.units as u
from astroquery.irsa_dust import IrsaDust
import copy
from dataclasses import dataclass, field
import emcee
import json
import multiprocessing as mp
from numbers import Number
import numexpr as ne
import numpy as np
import pandas as pd
import pathlib
import pickle
from prettytable import PrettyTable
from scipy import signal, stats
from scipy.integrate import simpson
from scipy.interpolate import interp1d
import scipy.optimize as op
import sys
import time
from typing import Callable, List, Union

# TODO: fix warnings
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning) 
warnings.filterwarnings('ignore', category=UserWarning) 

# TODO: reorganize
from badass.badass_utils import badass_test_suite

from badass.components.params import ParameterRegistry
from badass.components.blobs import BlobRegistry

from badass.utils.config import BadassConfig
from badass.input.input import BadassInput
from badass.utils.output import ResultWriter
import badass.utils.utils as ba_utils
from badass.components.templates.common import initialize_templates
import badass.utils.constants as ba_consts
import badass.utils.plotting as plotting
# from badass.components.spectral_lines.line_lists.optical_qso import optical_qso_default
# from badass.components.spectral_lines.line_profiles import line_constructor

from badass.components.spectral_lines.spectral_line import SpectralLine


__author__ = 'Remington O. Sexton (USNO), Sara M. Doan (GMU), Michael A. Reefe (GMU), William Matzko (GMU), Nicholas Darden (UCR)'
__copyright__ = 'Copyright (c) 2023 Remington Oliver Sexton'
__credits__ = ['Remington O. Sexton (GMU/USNO)', 'Sara Doan (GMU)', 'Michael A. Reefe (GMU)', 'William Matzko (GMU)', 'Nicholas Darden (UCR)']
__license__ = 'MIT'
__version__ = '11.0.0'
__maintainer__ = 'Sara Doan'
__email__ = 'sdoan2@gmu.edu'
__status__ = 'Release'


# TODO: all init/plim values to config file
# TODO: ability to resume from line test and ml results
# TODO: ability to resume mid run (save status at certain checkpoints?)
# TODO: ability to multiprocess mcmc runs?
# TODO: line type classes? or just a general line class?
# TODO: use rng seed to be able to reproduce fits


def target_check(inputs, **kwargs):
    cfg = BadassConfig.get_config_from_args(kwargs)
    targets = BadassInput.get_inputs(inputs, cfg)
    print('Fitting %d targets'%len(targets))


class FitStage:
    INIT = 1
    BOOTSTRAP = 2
    MCMC = 3


from badass.runner.pipeline import BadassPipeline


def run_BADASS(inputs, **kwargs):
    cfg = BadassConfig.get_config_from_args(kwargs)
    targets = BadassInput.get_inputs(inputs, cfg)

    pipeline = BadassPipeline.init(targets, cfg)
    pipeline.run()
    pipeline.finalize()


    skip_comps = ['DATA','WAVE','MODEL','NOISE','RESID','POWER','HOST_GALAXY','BALMER_CONT','APOLY','MPOLY',]
    tied_target_pars = ['amp', 'disp', 'voff', 'shape', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10',]


    def run_emcee(self):
        self.reweight()

        # TODO: need to re-initalize parameters here?
        self.log.output_free_pars(self.line_list, self.param_dict, self.soft_cons)
        self.cur_params = dict(sorted(self.cur_params.items()))

        nwalkers = self.cfg.mcmc.nwalkers
        if nwalkers < 2*len(self.cur_params):
            self.log.info('Number of walkers < 2 x (# of parameters)! Setting nwalkers = %d' % (2*len(self.cur_params)))
            nwalkers = 2*len(self.cur_params)

        pos = self.initialize_walkers(nwalkers)
        ndim = len(self.cur_params)

        # Keep original burn_in and max_iter to reset convergence if jumps out of convergence
        max_iter = self.cfg.mcmc.max_iter
        min_iter = self.cfg.mcmc.min_iter
        write_iter = self.cfg.mcmc.write_iter
        write_thresh = self.cfg.mcmc.write_thresh

        # TODO: create Backend class that supports objects (pickle-based? npz-based?)
        # backend = emcee.backends.HDFBackend(self.target.outdir.joinpath('log', 'MCMC_chain.h5'))
        # backend.reset(nwalkers, ndim)

        def lnprob_wrapper(p):
            self.cur_params = dict(zip(self.cur_params.keys(), p))
            lp, ll = self.lnprob()
            blob_dict = self.calc_mcmc_blob()
            blob_dict['LOG_LIKE'] = ll
            return lp, blob_dict

        dtype = [('full_blob',dict),]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_wrapper, blobs_dtype=dtype)#, backend=backend)

        autocorr = None
        if self.cfg.mcmc.auto_stop:

            @dataclass
            class AutoCorr:
                ctx: BadassRunContext = None
                times: List[np.ndarray] = field(default_factory=list)
                tolerances: List[np.ndarray] = field(default_factory=list)
                prev_tau: np.ndarray = field(default_factory=lambda: np.full(len(self.cur_params), np.inf))

                max_tol: float = 0.0
                min_samp: int = 0
                ncor_times: int = 0
                conv_type: Union[str,tuple] = ''
                conv_func: Callable = None
                conv_tau: np.ndarray = field(default_factory=lambda: np.full(len(self.cur_params), np.inf))

                stop_iter: int = 0
                burn_in: int = 0
                converged: bool = False

                def mean_conv(self, sampler, tau, tol):
                    par_conv = np.array([x for x in range(len(tau)) if round(tau[x],1) > 1.0]) # TODO: print converged params
                    return (par_conv.size > 0) and (sampler.iteration > (np.nanmean(tau[par_conv]) * self.ncor_times) and (np.nanmean(tol[par_conv]) < self.max_tol))

                def median_conv(self, sampler, tau, tol):
                    par_conv = np.array([x for x in range(len(tau)) if round(tau[x],1) > 1.0]) # TODO: print converged params
                    return (par_conv.size > 0) and (sampler.iteration > (np.nanmedian(tau[par_conv]) * self.ncor_times) and (np.nanmedian(tol[par_conv]) < self.max_tol))

                def all_conv(self, sampler, tau, tol):
                    return (all(sampler.iteration > tau*self.ncor_times)) and (all(tau > 1.0)) and (all(tol < self.max_tol))

                def param_conv(self, sampler, tau, tol):
                    return (all(sampler.iteration > tau[self.conv_idx]*self.ncor_times)) and (all(tau[self.conv_idx] > 1.0)) and (all(tol[self.conv_idx] < self.max_tol))

                def __post_init__(self):
                    self.prev_tau = np.full(len(self.ctx.cur_params), np.inf)
                    self.max_tol = self.ctx.cfg.mcmc.autocorr_tol
                    self.min_samp = self.ctx.cfg.mcmc.min_samp
                    self.ncor_times = self.ctx.cfg.mcmc.ncor_times
                    self.conv_type = self.ctx.cfg.mcmc.conv_type

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

                    self.stop_iter = self.ctx.cfg.mcmc.max_iter
                    self.burn_in = self.ctx.cfg.mcmc.burn_in


                def check_convergence(self, sampler):
                    it = sampler.iteration
                    self.past_miniter = ((it >= write_thresh) and (it >= min_iter))
                    if not self.past_miniter:
                        return

                    tau = autocorr_convergence(sampler.chain) # autocorr time for each parameter
                    self.times.append(tau)
                    tol = (np.abs(tau-self.prev_tau)/self.prev_tau) * 100 # tolerances
                    self.tolerances.append(tol)

                    if (not self.converged) and (self.conv_func(sampler, tau, tol)):
                        self.ctx.log.info('Converged at %d iterations\nPerforming %d iterations of sampling'%(it, self.min_samp))
                        self.burn_in = it
                        self.stop_iter = it+self.min_samp
                        self.conv_tau = tau
                        self.converged = True

                    elif (self.converged) and (not self.conv_func(sampler, tau, tol)):
                        self.ctx.log.info('Iteration: %d - Jumped out of convergence, resetting burn_in and max_iter'%it)
                        self.burn_in = self.ctx.cfg.mcmc.burn_in
                        self.stop_iter = self.ctx.cfg.mcmc.max_iter
                        self.converged = False

                    self.prev_tau = tau

            autocorr = AutoCorr(ctx=self)


        # TODO: do something with
        start_time = time.time()

        # TODO
        # write_log((ndim,nwalkers,auto_stop,conv_type,burn_in,write_iter,write_thresh,min_iter,max_iter),'emcee_options',run_dir)

        # self.add_chain()
        # sampler.run_mcmc(pos, write_thresh)
        # self.add_chain(sampler=sampler)

        # while sampler.iteration < max_iter:
        #     self.log.info('MCMC iteration: %d' % sampler.iteration)
        #     sampler.run_mcmc(pos, min(write_iter, max_iter-sampler.iteration))
        #     self.add_chain(sampler=sampler)
            # TODO: verbose -> print current parameter values


        self.add_chain()
        for result in sampler.sample(pos, iterations=max_iter):
            it = sampler.iteration
            if (it >= write_thresh) and (it % write_iter == 0):
                self.log.info('MCMC iteration: %d' % it)
                self.add_chain(sampler=sampler)
                # TODO: log current parameter values

                if not autocorr:
                    continue

                autocorr.check_convergence(sampler)


        elap_time = (time.time() - start_time)
        run_time = ba_utils.time_convert(elap_time)
        self.log.debug('emcee Runtime = %s' % (run_time))

        # TODO
        # write_log(run_time,'emcee_time',run_dir)

        # TODO: remove excess zeros on convergence

        if autocorr:
            autocorr_times = np.stack(autocorr.times, axis=1)
            autocorr_tols = np.stack(autocorr.tolerances, axis=1)
            autocorr_dict = {}
            for k, pname in enumerate(self.cur_params.keys()):
                autocorr_dict[pname] = {
                    'tau': autocorr_times[k],
                    'tol': autocorr_tols[k],
                }

            # TODO: handle in separate output file
            np.save(self.target.outdir.joinpath('log', 'autocorr_dict.npy'), autocorr_dict)
            tau = autocorr.conv_tau if autocorr.converged else autocorr.prev_tau
            tol = (np.abs(tau-autocorr.prev_tau)/autocorr.prev_tau)
            ptbl = PrettyTable()
            ptbl.field_names = ['Parameter', 'Autocorr. Time', 'Target Autocorr. Time', 'Tolerance', 'Converged?']
            for i, pname in enumerate(self.cur_params.keys()):
                ptbl.add_row([pname, tau[i], autocorr.max_tol, tol[i], autocorr.ncor_times])
            self.log.debug(ptbl)

        # TODO: output files
        self.collect_mcmc_results(sampler, autocorr)

        if self.cfg.plot.html:
            plotting.plotly_best_fit(self)

        if self.cfg.plot.param_hist:
            for key in self.param_dict.keys():
                plotting.posterior_plot(key, self.mcmc_results_dict[key], self.mcmc_result_chains['chains'][key], autocorr.burn_in, self.target.outdir)
            plotting.posterior_plot('LOG_LIKE', self.mcmc_results_dict['LOG_LIKE'], self.mcmc_result_chains['chains']['LOG_LIKE'], autocorr.burn_in, self.target.outdir)

        if self.cfg.plot.corner:
            plotting.corner_plot(self)

        plotting.plot_best_model(self, 'best_fit_model.pdf')

        elap_time = (time.time() - start_time)
        self.log.debug('Total Runtime = %s' % (ba_utils.time_convert(elap_time)))

        # TODO:
        # write_log(elap_time,'total_time',run_dir)

        self.log.info('Done MCMC fitting %s!' % self.target.cfg.io.output_dir)


    def initialize_walkers(self, nwalkers):
        # Initializes the MCMC walkers within bounds and soft constraints

        pos = list(self.cur_params.values()) + 1.e-3 * np.random.randn(nwalkers, len(self.cur_params))
        for i, key in enumerate(self.cur_params.keys()):
            bounds = self.param_dict[key]['plim']
            for walker in range(nwalkers): # iterate through walker
                while (pos[walker][i] < bounds[0]) or (pos[walker][i] > bounds[1]):
                    pos[walker][i] = self.cur_params[key] + 1.e-3*np.random.randn(1)

        return pos


    def add_chain(self, sampler=None):
        if sampler is None:
            self.chain_df = pd.DataFrame(columns=['iter']+list(self.cur_params.keys()))
            chain_dict = {'iter': 0}
            chain_dict.update(self.cur_params)
        else:
            chain_dict = {'iter': sampler.iteration}
            last_iter = self.chain_df.iter.values[-1]
            chain_vals = {key:np.nanmedian(sampler.chain[:,last_iter:,i]) for i, key in enumerate(self.cur_params.keys())}
            chain_dict.update(chain_vals)

        self.chain_df.loc[len(self.chain_df)] = chain_dict
        chain_file = self.target.outdir.joinpath('log', 'MCMC_chain.csv')
        self.chain_df.to_csv(chain_file, index=False)


    def calc_mcmc_blob(self):
        blob_dict = {}
        noise = self.comp_dict['NOISE']
        wave = self.comp_dict['WAVE']
        total_cont, agn_cont, host_cont = get_continuums(self.comp_dict, len(wave))

        for key, val in self.comp_dict.items():
            if key in self.skip_comps:
                continue

            # TODO: better way to integrate?
            blob_dict[key+'_FLUX'] = np.abs(np.trapz(val, self.fit_wave))
            eqwidth = np.trapz(val / total_cont, self.fit_wave)
            blob_dict[key+'_EW'] = eqwidth if np.isfinite(eqwidth) else 0.0

        for line_name, line_dict in {**self.line_list, **self.combined_line_list}.items():
            line_comp = self.comp_dict[line_name]
            blob_dict[line_name+'_FWHM'] = combined_fwhm(wave, np.abs(line_comp), self.target.velscale)
            blob_dict[line_name+'_W80'] = calculate_w80(wave, np.abs(line_comp), line_dict['center'])
            blob_dict[line_name+'_NPIX'] = len(np.where(np.abs(line_comp) > noise)[0])
            blob_dict[line_name+'_SNR'] = np.nanmax(np.abs(line_comp)) / np.nanmean(noise)

            if not line_name in self.combined_line_list:
                continue

            vel = np.arange(len(self.fit_wave))*self.target.velscale - self.blob_pars[line_name+'_LINE_VEL']
            full_profile = np.abs(line_comp)
            norm_profile = full_profile / np.sum(full_profile)
            voff = np.trapz(vel*norm_profile, vel) / simpson(norm_profile, vel)
            blob_dict[line_name+'_VOFF'] = voff if np.isfinite(voff) else 0.0

            disp = np.sqrt(np.trapz(vel**2*norm_profile, vel) / np.trapz(norm_profile, vel) - (voff**2))
            blob_dict[line_name+'_DISP'] = disp if np.isfinite(disp) else 0.0


        cont_types = {
            'TOT': total_cont,
            'AGN': agn_cont,
            'HOST': host_cont,
        }

        for wave in [1350, 3000, 5100]:
            if (wave < self.fit_wave[self.target.fit_mask][0]) or (wave > self.fit_wave[self.target.fit_mask][-1]):
                continue

            for cont_key, cont_val in cont_types.items():
                blob_dict['F_CONT_%s_%d'%(cont_key,wave)] = cont_val[self.blob_pars['INDEX_%d'%wave]]

        for wave in [4000, 7000]:
            if (wave < self.fit_wave[self.target.fit_mask][0]) or (wave > self.fit_wave[self.target.fit_mask][-1]):
                continue

            for cont_key in ['AGN', 'HOST']:
                blob_dict['HOST_FRAC_%d'%wave] = cont_types[cont_key][self.blob_pars['INDEX_%d'%wave]]/total_cont[self.blob_pars['INDEX_%d'%wave]]


        blob_dict['R_SQUARED'] = badass_test_suite.r_squared(self.comp_dict['DATA'], self.comp_dict['MODEL'])
        blob_dict['RCHI_SQUARED'] = badass_test_suite.r_chi_squared(self.comp_dict['DATA'], self.comp_dict['MODEL'], self.comp_dict['NOISE'], len(self.cur_params))
        return blob_dict


    def collect_mcmc_results(self, sampler, autocorr):
        nwalkers, niters, nparams = sampler.chain.shape
        burn_in = autocorr.burn_in if autocorr else self.cfg.mcmc.burn_in
        if burn_in >= niters: burn_in = int(niters/2)

        self.mcmc_result_chains = {'chains':{}, 'flat_chains':{}}

        def flatten_chain(chain):
            # TODO: zero-trim if converged before max iters
            chain[~np.isfinite(chain)] = 0
            return chain[:,burn_in:].flatten()

        for i, param in enumerate(self.cur_params.keys()):
            self.mcmc_result_chains['chains'][param] = sampler.chain[:,:,i]
            self.mcmc_result_chains['flat_chains'][param] = flatten_chain(sampler.chain[:,:,i])

        all_chains = np.swapaxes(sampler.get_blobs()['full_blob'],0,1)
        for key in all_chains[0][0].keys():
            self.mcmc_result_chains['chains'][key] = np.zeros((nwalkers,niters))
            self.mcmc_result_chains['flat_chains'][key] = np.zeros((nwalkers,niters))
            if key.split('_')[-1] == 'FLUX':
                lum_key = key.replace('_FLUX', '_LUM')
                self.mcmc_result_chains['chains'][lum_key] = np.zeros((nwalkers,niters))
                self.mcmc_result_chains['flat_chains'][lum_key] = np.zeros((nwalkers,niters))
            if key[:6] == 'F_CONT':
                lum_key = key.replace('F_CONT', 'L_CONT')
                self.mcmc_result_chains['chains'][lum_key] = np.zeros((nwalkers,niters))
                self.mcmc_result_chains['flat_chains'][lum_key] = np.zeros((nwalkers,niters))


        def get_key_chain(chain, param):
            # Loop through each iteration of the chain and grab the parameter value
            with np.nditer([chain, None], flags=['refs_ok', 'multi_index', 'buffered'], op_flags=[['readonly'], ['writeonly', 'allocate', 'no_broadcast']]) as it:
                for x, y in it:
                    y[...] = x.item()[param]
                return it.operands[1]


        for key in all_chains[0][0].keys():
            val = get_key_chain(all_chains, key).astype(float)

            if (key.split('_')[-1] == 'FLUX') or (key[:6] == 'F_CONT'):
                val = val * self.target.flux_norm * self.target.fit_norm * (1.0+self.target.z)

            elif key.split('_')[-1] == 'EW':
                val = val * (1.0+self.target.z)

            self.mcmc_result_chains['chains'][key] = val
            self.mcmc_result_chains['flat_chains'][key] = flatten_chain(val)

            if key.split('_')[-1] == 'FLUX':
                lum_key = key.replace('_FLUX', '_LUM')
                lum = np.log10(self.flux_to_lum(10**val))
                lum[~np.isfinite(lum)] = 0
                self.mcmc_result_chains['chains'][lum_key] = lum
                self.mcmc_result_chains['flat_chains'][lum_key] = flatten_chain(lum)

            if key[:6] == 'F_CONT':
                lum_key = key.replace('F_CONT', 'L_CONT')
                wave = float(key.split('_')[-1])
                lum = np.log10(self.flux_to_lum(10**val)*wave)
                lum[~np.isfinite(lum)] = 0
                self.mcmc_result_chains['chains'][lum_key] = lum
                self.mcmc_result_chains['flat_chains'][lum_key] = flatten_chain(lum)

            # TODO: move to stellar template?
            if key == 'STEL_VEL':
                zsys = (self.target.z+1) * (1+val/c)-1
                self.mcmc_result_chains['chains']['Z_SYS'] = zsys
                self.mcmc_result_chains['flat_chains']['Z_SYS'] = flatten_chain(zsys)


        self.fit_model()
        self.collect_mcmc_pars(sampler)
        self.mcmc_output()


    result_attrs = ['best_fit', 'ci_68_low', 'ci_68_upp', 'ci_95_low', 'ci_95_upp', 'mean', 'std_dev', 'median', 'med_abs_dev', 'flag']

    def collect_mcmc_pars(self, sampler):
        for key, chain in self.mcmc_result_chains['flat_chains'].items():

            if len(chain) == 0:
                par_results = {k:np.nan for k in self.result_attrs}
                par_results['flat_chain'] = flat
                par_results['flag'] = 1
                self.mcmc_results_dict[key] = par_results
                continue

            par_results = {}

            if key.split('_')[-1] == 'AMP':
                chain *= self.target.fit_norm

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
            if (not np.isfinite(post_med)) or (not np.isfinite(par_results['ci_68_low'])) or (not np.isfinite(par_results['ci_68_upp'])):
                    par_results['flag'] = 1

            if key in self.param_dict.keys():
                plim = self.param_dict[key]['plim']
                if key.split('_')[-1] == 'AMP':
                    plim = [p*self.target.fit_norm for p in plim]
                if (post_med-(1.5*par_results['ci_68_low']) <= plim[0]) or (post_med+(1.5*par_results['ci_68_upp']) >= plim[1]):
                    par_results['flag'] = 1

            elif key.split('_')[-1] == 'FLUX':
                if post_med-(1.5*par_results['ci_68_low']) <= -20:
                    par_results['flag'] = 1

            elif key.split('_')[-1] == 'LUM':
                if post_med-(1.5*par_results['ci_68_low']) <= 30:
                    par_results['flag'] = 1

            elif (key.split('_')[-1] == 'EW') or (key[:6] == 'F_CONT'):
                if post_med-(1.5*par_results['ci_68_low']) <= 0:
                    par_results['flag'] = 1

            elif key == 'Z_SYS':
                if post_med-(3.0*par_results['ci_68_low']) < 0:
                    par_results['flag'] = 1

            self.mcmc_results_dict[key] = par_results

        self.collect_tied_pars()

        for line_name, line_dict in ({**self.line_list, **self.combined_line_list}).items():
            disp_res_par_results = {k:np.nan for k in self.result_attrs}
            disp_res = line_dict['disp_res_kms']
            disp_res_par_results['best_fit'] = disp_res
            self.mcmc_results_dict[line_name+'_DISP_RES'] = disp_res_par_results

            self.mcmc_results_dict[line_name+'_DISP_CORR'] = copy.deepcopy(self.mcmc_results_dict[line_name+'_DISP'])
            self.mcmc_results_dict[line_name+'_DISP_CORR']['best_fit'] = np.nanmax([0.0, np.sqrt(self.mcmc_results_dict[line_name+'_DISP']['best_fit']**2-(disp_res**2))])
            self.mcmc_results_dict[line_name+'_FWHM_CORR'] = copy.deepcopy(self.mcmc_results_dict[line_name+'_FWHM'])
            self.mcmc_results_dict[line_name+'_FWHM_CORR']['best_fit'] = np.nanmax([0.0, np.sqrt(self.mcmc_results_dict[line_name+'_FWHM']['best_fit']**2-(disp_res*2.3548)**2)])
            self.mcmc_results_dict[line_name+'_W80_CORR'] = copy.deepcopy(self.mcmc_results_dict[line_name+'_W80'])
            self.mcmc_results_dict[line_name+'_W80_CORR']['best_fit'] = np.nanmax([0.0, np.sqrt(self.mcmc_results_dict[line_name+'_W80']['best_fit']**2-(2.567*disp_res)**2)])


    def collect_tied_pars(self):
        best_fit_dict = {k:v['best_fit'] for k,v in self.mcmc_results_dict.items()}

        for line_name, line_dict in self.line_list.items():
            for par_name, par_val in line_dict.items():
                if (par_val == 'free') or (not par_name in self.tied_target_pars) or (isinstance(par_val, Number)):
                    continue

                par_results = {}
                expr_vars = [p for p in self.cur_params.keys() if p in par_val]

                # TODO: what we want?
                par_results['init'] = self.param_dict[expr_vars[0]]['init']
                par_results['plim'] = self.param_dict[expr_vars[0]]['plim']

                par_results['best_fit'] = ne.evaluate(par_val, local_dict=best_fit_dict).item()
                # TODO: add to self.mcmc_result_chains instead?
                par_results['chain'] = ne.evaluate(par_val, local_dict=self.mcmc_result_chains['chains'])
                par_results['flat_chain'] = ne.evaluate(par_val, local_dict=self.mcmc_result_chains['flat_chains'])

                for attr in ['ci_68_low', 'ci_68_upp', 'ci_95_low', 'ci_95_upp', 'mean', 'std_dev', 'median', 'med_abs_dev']:
                    par_results[attr] = np.sqrt(np.sum(np.array([self.mcmc_results_dict[k][attr] for k in expr_vars], dtype=float)**2))
                par_results['flag'] = np.sum([self.mcmc_results_dict[k]['flag'] for k in expr_vars])

                self.mcmc_results_dict[line_name+'_'+par_name.upper()] = par_results


    def mcmc_output(self):
        # Write chains
        if self.cfg.out.write_chain:
            cols = []
            for key, chain in self.mcmc_result_chains['chains'].items():
                cols.append(fits.Column(name=key, format='%dD'%(chain.shape[0]*chain.shape[1]), dim='(%d,%d)'%(chain.shape[1],chain.shape[0]), array=[chain]))
            cols = fits.ColDefs(cols)
            hdu = fits.BinTableHDU.from_columns(cols)
            hdu.writeto(self.target.outdir.joinpath('log', 'MCMC_chains.fits'), overwrite=True)
            hdu.close()


        # TODO: remove redundancy with ml bmc.fits
        # Write best-fit components
        cols = []
        for key, value in self.comp_dict.items():
            cols.append(fits.Column(name=key, format='E', array=value))
        cols.append(fits.Column(name='MASK', format='E', array=self.target.fit_mask))
        cols = fits.ColDefs(cols)
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.writeto(self.target.outdir.joinpath('log', 'best_model_components.fits'), overwrite=True)


        # TODO: remove redundancy with ml pt.fits
        # Write parameter table
        hdr = fits.Header()
        hdr['z'] = self.target.z
        hdr['med_noise'] = np.nanmedian(self.target.noise)
        hdr['velscale'] = self.target.velscale
        hdr['fit_norm'] = self.target.fit_norm
        hdr['flux_norm'] = self.target.flux_norm
        primary = fits.PrimaryHDU(header=hdr)

        cols_dict = {'parameter': []}
        cols_dict.update({k:[] for k in self.result_attrs})
        for key, result_dict in self.mcmc_results_dict.items():
            cols_dict['parameter'].append(key)
            for attr in self.result_attrs:
                cols_dict[attr].append(result_dict[attr])

        cols = []
        for key, values in cols_dict.items():
            fmt = 'E' if key != 'parameter' else '30A'
            cols.append(fits.Column(name=key, format=fmt, array=values))
        cols = fits.ColDefs(cols)
        table = fits.BinTableHDU.from_columns(cols)

        hdu = fits.HDUList([primary, table])
        hdu.writeto(self.target.outdir.joinpath('log', 'par_table.fits'), overwrite=True)
        hdu.close()

        # TODO:
        # write_log((par_names,par_best,ci_68_low,ci_68_upp,ci_95_low,ci_95_upp,mean,std_dev,median,med_abs_dev,flags),'emcee_results',run_dir)


def get_continuums(components, size):
    # TODO: store key arrays elsewhere
    total_cont = np.zeros(size)
    for key in ['POWER', 'HOST_GALAXY', 'BALMER_CONT', 'APOLY', 'MPOLY']:
        if not key in components:
            continue
        total_cont += components[key]
    agn_cont = np.zeros(size)
    for key in ['POWER', 'BALMER_CONT', 'APOLY', 'MPOLY']:
        if not key in components:
            continue
        agn_cont += components[key]
    host_cont = np.zeros(size)
    for key in ['HOST_GALAXY', 'APOLY', 'MPOLY']:
        if not key in components:
            continue
        host_cont += components[key]

    return total_cont, agn_cont, host_cont


# Autocorrelation analysis
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


# TODO: to utils
def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


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

