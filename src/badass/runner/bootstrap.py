from astropy.io import fits
import numpy as np
import scipy.optimize as op
from tabulate import tabulate

from badass.badass_utils import badass_test_suite
from badass.runner import BadassResult, BadassRunContext
from badass.utils import plotting


class BasinhopResult(BadassResult):
    OUT_NAME = 'basinhop_result'

    def __init__(self, ctx):
        self.params = {}
        self.blobs = {}
        self.metrics = {}


class MLResult(BadassResult):

    OUT_NAME = 'mc_result'

    def __init__(self, ctx):
        super().__init__(ctx)
        self.out_dir = self.out_dir.joinpath(self.OUT_NAME)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.bh_result = BasinhopResult(ctx)
        self.bh_result.out_dir = self.out_dir.joinpath(BasinhopResult.OUT_NAME)
        self.bh_result.out_dir.mkdir(parents=True, exist_ok=True)

        self.params_chain = {}
        self.blobs_chain = {}
        self.metrics_chain = {
            'LOG_LIKE': None,
            'R_SQUARED': None,
            'RCHI_SQUARED': None,
        }

        self.comps_chain = {}
        self.meta_comps_chain = {
            'wave': None,
            'data': None,
            'noise': None,
            'model': None,
            'resid': None,
        }

        self.params = {}
        self.components = {}
        self.meta_components = {}


    def init_chains(self, ctx, niter):
        for param in ctx.param_reg.get_param_dict().keys():
            self.params_chain[param] = np.zeros(niter+1)

        for blob in ctx.blob_reg.get_blobs():
            if isinstance(blob.cur_val, dict):
                for key in blob.cur_val.keys():
                    self.blobs_chain[key] = np.zeros(niter+1)
            else:
                self.blobs_chain[blob.name] = np.zeros(niter+1)

        for metric in self.metrics_chain.keys():
            self.metrics_chain[metric] = np.zeros(niter+1)

        for comp, val in ctx.comps.items():
            self.comps_chain[comp] = np.zeros((niter+1,len(val)))
        for comp in self.meta_comps_chain.keys():
            self.meta_comps_chain[comp] = np.zeros((niter+1,len(ctx.fit_wave)))


    def save_iter(self, ctx, i, result):
        ctx.param_reg.update_vals(result['x'])
        ctx.fit_model()
        ctx.blob_reg.compute_all()

        for name, value in ctx.param_reg.get_param_dict().items():
            self.params_chain[name][i] = value
        for blob in ctx.blob_reg.get_blobs():
            if isinstance(blob.cur_val, dict):
                for key, val in blob.cur_val.items():
                    self.blobs_chain[key][i] = val
            else:
                self.blobs_chain[blob.name][i] = blob.cur_val

        self.metrics_chain['LOG_LIKE'][i] = result['fun']
        self.metrics_chain['R_SQUARED'][i] = badass_test_suite.r_squared(ctx.fit_spec, ctx.model)
        self.metrics_chain['RCHI_SQUARED'][i] = badass_test_suite.r_chi_squared(ctx.fit_spec, ctx.model, ctx.fit_noise, len(ctx.param_reg.get_free_parameters()))

        # TODO: copy needed? option to turn off saving these
        for comp, val in ctx.comps.items():
            self.comps_chain[comp][i] = val.copy()

        meta_comps_dict = {'wave':ctx.fit_wave.copy(),'data':ctx.fit_spec.copy(),'noise':ctx.fit_noise.copy(),'model':ctx.model.copy()}
        for comp, comp_arr in meta_comps_dict.items():
            self.meta_comps_chain[comp][i] = comp_arr
        self.meta_comps_chain['resid'][i] = ctx.fit_spec-ctx.model


    def compile_results(self, ctx):
        def add_param_result(key, vals):
            med = np.nanmedian(vals)
            std = np.nanstd(vals)
            if not np.isfinite(med): med = 0.0
            if not np.isfinite(std): std = 0.0
            self.params[key] = {'med':med, 'std':std}
            return med, std

        for key, vals in self.params_chain.items():
            med, std = add_param_result(key, vals)

            param = ctx.param_reg.get_param(key)
            if not param.is_free:
                continue

            flag = 0
            if med-std <= param.plim[0]: flag += 1
            if med+std >= param.plim[1]: flag += 1
            self.params[key]['flag'] = flag

        # update params for final model fit
        ctx.param_reg.update_vals([v['med'] for v in self.params.values()])
        med_values = [v['med'] for p,v in self.params.items() if ctx.param_reg.is_free(p)]
        ctx.param_reg.update_vals(med_values)
        ctx.fit_model()

        for key, vals in self.blobs_chain.items():
            add_param_result(key, vals)

        self.params.update(ctx.blob_reg.get_postfits(self.params))

        for key, vals in self.metrics_chain.items():
            add_param_result(key, vals)

        # Rescale amplitudes
        for pname, param_dict in self.params.items():
            if pname[-4:] != '_AMP':
                continue
            param_dict['med'] *= ctx.target.fit_norm
            param_dict['std'] *= ctx.target.fit_norm

        # updated with final model fit
        for key, comp in ctx.comps.items():
            self.components[key] = comp * ctx.target.fit_norm

        self.meta_components['wave'] = ctx.fit_wave.copy()
        meta_comps_dict = {'data':ctx.fit_spec.copy(),'noise':ctx.fit_noise.copy(),'model':ctx.model.copy()}
        for comp, comp_arr in meta_comps_dict.items():
            self.meta_components[comp] = comp_arr * ctx.target.fit_norm
        self.meta_components['resid'] = (ctx.fit_spec-ctx.model) * ctx.target.fit_norm


    def dump_results(self, ctx):
        headers = ['Name', 'Value', 'STD', 'Flag']
        table = []

        for param, param_dict in self.params.items():
            row = [param, param_dict['med'], param_dict['std'], param_dict.get('flag', '--')]
            table.append(row)
        print(tabulate(table, headers, tablefmt='grid'))


    def output(self, ctx):
        col1 = fits.Column(name='parameter', format='30A', array=list(self.params.keys()))
        col2 = fits.Column(name='best_fit', format='E', array=[v['med'] for v in self.params.values()])
        col3 = fits.Column(name='sigma', format='E', array=[v['std'] for v in self.params.values()])
        cols = fits.ColDefs([col1,col2,col3])
        table_hdu = fits.BinTableHDU.from_columns(cols)

        hdr = fits.Header()
        hdr['z'] = ctx.target.z
        hdr['med_noise'] = np.nanmedian(ctx.fit_noise)
        hdr['velscale'] = ctx.target.velscale
        hdr['fit_norm'] = ctx.target.fit_norm
        hdr['flux_norm'] = ctx.target.flux_norm

        primary = fits.PrimaryHDU(header=hdr)
        hdu = fits.HDUList([primary, table_hdu])
        hdu.writeto(self.out_dir.joinpath('par_table.fits'), overwrite=True)

        cols = []
        for key, val in self.components.items():
            cols.append(fits.Column(name=key.upper(), format='E', array=val))

        for key, val in self.meta_components.items():
            cols.append(fits.Column(name=key.upper(), format='E', array=val))

        mask = np.zeros(len(self.meta_components['wave']), dtype=bool)
        mask[ctx.target.fit_mask] = True
        cols.append(fits.Column(name='MASK', format='E', array=mask))

        cols = fits.ColDefs(cols)
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.writeto(self.out_dir.joinpath('best_model_components.fits'), overwrite=True)

        plot_out = self.out_dir
        plotting.plot_ml_results(self, ctx, plot_out)


class MLRunner(BadassRunContext):

    result_cls = MLResult

    def __init__(self, target, **kwargs):
        super().__init__(target, **kwargs)

        if not hasattr(self, 'force_thresh'):
            self.force_thresh = badass_test_suite.root_mean_squared_error(self.target.spec, np.full_like(self.target.spec,np.nanmedian(self.target.spec)))
        if not np.isfinite(self.force_thresh):
            self.force_thresh = np.inf


    def run(self):
        self.log.info('MLStage run')

        if len(self.param_reg.get_free_parameters()) == 0:
            self.log.warn('No parameters to fit!')
            return

        basinhop_result = self.basinhop()
        self.max_likelihood(basinhop_result)


    def basinhop(self):

        param_constraints = self.param_reg.get_constraints()
        param_bounds = self.param_reg.get_fit_bounds()

        n_basinhop = self.cfg.fit.n_basinhop
        lowest_rmse = badass_test_suite.root_mean_squared_error(self.fit_spec, np.zeros(len(self.fit_spec)))
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
                rmse = badass_test_suite.root_mean_squared_error(self.fit_spec, self.model)
                lowest_rmse = min(lowest_rmse, rmse)

                accept_thresh = 0.001 # Define an acceptance threshold
                if (basinhop_count > n_basinhop) and (accepted_count >=1) and ((lowest_rmse-accept_thresh > self.force_thresh) or (lowest_rmse > self.force_thresh)):
                    self.log.warn('Warning: basinhopping has exceeded %d attemps to find a new global maximum. Terminating fit...'%n_basinhop)
                    return True

                terminate = False
                if (accepted_count > 1) and (basinhop_count >= force_basinhop) and (((lowest_rmse-accept_thresh) <= self.force_thresh) or (lowest_rmse <= self.force_thresh)):
                    terminate = True

                self.log.info('\tFit Status: %s\n\tForce threshold: %0.4f\n\tLowest RMSE: %0.4f\n\tCurrent RMSE: %0.4f\n\tAccepted Count: %d\n\tBasinhop Count: %d'%(terminate,self.force_thresh,lowest_rmse,rmse,accepted_count,basinhop_count))
                return terminate


        self.param_reg.dump_parameters()
        self.log.info('Basinhopping')
        minimizer_args = {'method':'SLSQP', 'bounds':param_bounds,'constraints':param_constraints,'options':{'disp':True,}}
        result = op.basinhopping(func=self.lnprob_wrapper, x0=self.param_reg.fit_vector(), stepsize=1.0, interval=1, niter=2500, minimizer_kwargs=minimizer_args,
                                 disp=False, niter_success=n_basinhop, callback=callback_ftn)

        self.param_reg.update_vals(result['x'])
        self.result.bh_result.params = self.param_reg.get_param_dict().copy()
        self.result.bh_result.blobs = self.blob_reg.compute_all()

        self.param_reg.dump_parameters()
        self.blob_reg.dump_blobs()
        self.log.info('Basinhopping complete')

        self.fit_model()
        self.reweight()
        self.result.bh_result.metrics['LOG_LIKE'] = result['fun']

        return result


    def max_likelihood(self, basinhop_result):

        max_like_niter = self.cfg.fit.max_like_niter
        if max_like_niter == 0:
            return

        self.log.info('Performing Monte Carlo bootstrapping')

        param_constraints = self.param_reg.get_constraints()
        param_bounds = self.param_reg.get_fit_bounds()

        self.result.init_chains(self, max_like_niter)
        self.result.save_iter(self, 0, basinhop_result)

        orig_fit_spec = self.fit_spec.copy()

        for n in range(1, max_like_niter+1):
            self.log.info('Bootstrap iteration %d'%n)
            # Generate a simulated galaxy spectrum with noise added at each pixel
            mcgal = np.random.normal(self.fit_spec, np.abs(self.fit_noise))
            # Get rid of any infs or nan if there are none; this will cause scipy.optimize to fail
            mcgal[~np.isfinite(mcgal)] = np.nanmedian(mcgal)
            self.fit_spec = mcgal

            result = op.minimize(fun=self.lnprob_wrapper, x0=self.param_reg.fit_vector(), method='SLSQP',
                                   bounds=param_bounds, constraints=param_constraints, options={'maxiter':1000,'disp': False})
            self.result.save_iter(self, n, result)

            # return original spectrum
            self.fit_spec = orig_fit_spec


    def reweight(self):
        if not self.cfg.fit.reweighting:
            return
        self.log.debug('Reweighting noise to achieve a reduced chi-squared ~ 1')
        cur_rchi2 = badass_test_suite.r_chi_squared(self.fit_spec, self.model, self.fit_noise, self.param_reg.free_count)
        self.log.debug('\tCurrent reduced chi-squared = %0.5f' % cur_rchi2)
        self.fit_noise = self.fit_noise*np.sqrt(cur_rchi2)
        new_rchi2 = badass_test_suite.r_chi_squared(self.fit_spec, self.model, self.fit_noise, self.param_reg.free_count)
        self.log.debug('\tNew reduced chi-squared = %0.5f' % new_rchi2)


    def finalize(self):
        self.log.info('MLStage finalize')

        self.result.bh_result.compile_results(self)
        self.result.bh_result.dump_results(self)
        self.result.bh_result.output(self)

        self.result.compile_results(self)
        self.result.dump_results(self)
        self.result.output(self)

