from astropy.io import fits
from dataclasses import dataclass, field
import numpy as np
import scipy.optimize as op
from tabulate import tabulate
from typing import NamedTuple

from badass.badass_utils import badass_test_suite
from badass.runner import BadassResult, BadassRunContext, ParamResult


# intermediate result
class MLState(NamedTuple):
    params: list[float]
    log_like: float


@dataclass
class MLResult(BadassResult):
    OUT_NAME = 'ml_result'

    fp_chain: list[list[float]] = field(default_factory=list)
    ll_chain: list[float] = field(default_factory=list)

    blobs_chain: dict[str,list[float]] = field(default_factory=dict)
    final_theta: np.ndarray = None

    def save_state(self, state):
        self.fp_chain.append(state.params)
        self.ll_chain.append(state.log_like)

        self.ctx.blob_reg.compute_all()
        for blob_name, blob_val in self.ctx.blob_reg.get_blobs_dict().items():
            if not blob_name in self.blobs_chain:
                self.blobs_chain[blob_name] = []
            self.blobs_chain[blob_name].append(blob_val)


    def set_final_theta(self):
        chains = np.array(self.fp_chain).T
        self.final_theta = [np.nanmedian(c) for c in chains]


    def collect_final_parameters(self):
        # transpose so fp_chain[idx] is a chain for a single param
        self.fp_chain = np.array(self.fp_chain).T
        param_chains = self.ctx.param_reg.evaluate_chains(self.fp_chain, finalize=True)

        for param in self.ctx.param_reg.params.values():
            pr = ParamResult.from_chain(param.name, param_chains[param.name])

            if param.is_free:
                if pr.best_fit-pr.sigma <= param.plim.min: pr.flag += 1
                if pr.best_fit+pr.sigma >= param.plim.max: pr.flag += 1

            self.final_params[param.name] = pr

        # BLOBS
        # TODO: calculate all blob chains in finalize as ufuncs
        # TODO: don't calculate blobs if bootstrapping is not the last runner
        for blob in self.ctx.blob_reg.get_blobs_dict().keys():
            self.final_params[blob] = ParamResult.from_chain(blob, self.blobs_chain[blob])


@dataclass
class MLRunner(BadassRunContext):
    result_cls = MLResult

    force_thresh: float = None

    def __post_init__(self):
        super().__post_init__()
        if self.force_thresh is None:
            self.force_thresh = badass_test_suite.root_mean_squared_error(self.fit_flux, np.full_like(self.fit_flux, np.nanmedian(self.fit_flux)))
        if not np.isfinite(self.force_thresh):
            self.force_thresh = np.inf


    def run(self):
        self.log.info('MLStage run')

        if self.param_reg.free_count == 0:
            self.log.warn('No parameters to fit!')
            return

        basinhop_result = self.basinhop()
        self.max_likelihood(basinhop_result)


    def basinhop(self):

        param_constraints = self.param_reg.get_constraints()
        param_bounds = self.param_reg.get_fit_bounds()

        n_basinhop = self.cfg.fit.n_basinhop
        lowest_rmse = badass_test_suite.root_mean_squared_error(self.fit_flux, np.zeros(len(self.fit_flux)))
        callback_ftn = None
        if np.isfinite(self.force_thresh):
            self.log.debug('Required Maximum Likelihood RMSE threshold: %0.4f' % (self.force_thresh))
            force_basinhop = n_basinhop
            # TODO: config
            n_basinhop = 250 # Set to arbitrarily high threshold

            basinhop_count = 0
            accepted_count = 0
            basinhop_value = np.inf

            # x and f are the coordinates and function value of the trial minimum,
            # and accept is whether that minimum was accepted.
            # returning True stops basinhopping routine
            def callback_ftn(x, f, accepted):
                nonlocal basinhop_value, basinhop_count, lowest_rmse, accepted_count

                if f <= basinhop_value:
                    basinhop_value = f
                    basinhop_count = 0 # reset counter
                else:
                    basinhop_count += 1

                if accepted == 1:
                    accepted_count += 1

                self.fit_model()
                rmse = badass_test_suite.root_mean_squared_error(self.fit_flux, self.model)
                lowest_rmse = min(lowest_rmse, rmse)

                accept_thresh = 0.001 # Define an acceptance threshold
                if (basinhop_count > n_basinhop) and (accepted_count >=1) and ((lowest_rmse-accept_thresh > self.force_thresh) or (lowest_rmse > self.force_thresh)):
                    self.log.warn('Warning: basinhopping has exceeded %d attemps to find a new global maximum. Terminating fit...'%n_basinhop)
                    return True

                terminate = False
                if (accepted_count > 1) and (basinhop_count >= force_basinhop) and (((lowest_rmse-accept_thresh) <= self.force_thresh) or (lowest_rmse <= self.force_thresh)):
                    terminate = True

                self.log.debug('\tFit Status: %s\n\tForce threshold: %0.4f\n\tLowest RMSE: %0.4f\n\tCurrent RMSE: %0.4f\n\tAccepted Count: %d\n\tBasinhop Count: %d'%(terminate,self.force_thresh,lowest_rmse,rmse,accepted_count,basinhop_count))
                return terminate


        self.param_reg.dump_parameters()
        self.log.info('Basinhopping')
        minimizer_args = {'method':'SLSQP', 'bounds':param_bounds,'constraints':param_constraints,}
        result = op.basinhopping(func=self.lnprob_wrapper, x0=self.param_reg.fit_vector(), stepsize=1.0, interval=1, niter=2500, minimizer_kwargs=minimizer_args,
                                 disp=False, niter_success=n_basinhop, callback=callback_ftn)

        self.param_reg.dump_parameters()
        self.log.info('Basinhopping complete')

        # TODO: add back in
        # self.reweight()

        return MLState(result['x'], result['fun'])


    def max_likelihood(self, init_state):

        self.result.save_state(init_state)

        max_like_niter = self.cfg.fit.max_like_niter
        if max_like_niter == 0:
            return

        self.log.info('Performing Monte Carlo bootstrapping')

        self.param_reg.update(init_state.params)
        param_constraints = self.param_reg.get_constraints()
        param_bounds = self.param_reg.get_fit_bounds()

        # TODO: option to save all params/blobs as the fitting happens
        # self.result.init_chains(self, max_like_niter)

        orig_fit_flux = self.fit_flux.copy()

        for n in range(1, max_like_niter+1):
            self.log.info('Bootstrap iteration %d'%n)
            # Generate a simulated galaxy spectrum with noise added at each pixel
            sim_flux = np.random.normal(self.fit_flux, np.abs(self.fit_err))
            sim_flux[~np.isfinite(sim_flux)] = np.nanmedian(sim_flux)
            self.fit_flux = sim_flux

            result = op.minimize(fun=self.lnprob_wrapper, x0=self.param_reg.fit_vector(), method='SLSQP',
                                   bounds=param_bounds, constraints=param_constraints, options={'maxiter':1000,'disp': False})
            self.result.save_state(MLState(result['x'], result['fun']))

            # return original spectrum
            self.fit_flux = orig_fit_flux


    def reweight(self):
        if not self.cfg.fit.reweighting:
            return
        self.log.debug('Reweighting noise to achieve a reduced chi-squared ~ 1')
        cur_rchi2 = badass_test_suite.r_chi_squared(self.fit_flux, self.model, self.fit_err, self.param_reg.free_count)
        self.log.debug('\tCurrent reduced chi-squared = %0.5f' % cur_rchi2)
        self.fit_err = self.fit_err*np.sqrt(cur_rchi2)
        new_rchi2 = badass_test_suite.r_chi_squared(self.fit_flux, self.model, self.fit_err, self.param_reg.free_count)
        self.log.debug('\tNew reduced chi-squared = %0.5f' % new_rchi2)

