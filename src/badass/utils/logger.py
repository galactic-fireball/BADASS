import json
import logging
import numpy as np
import sys
import toml

# TODO: create error file with warning +
#           check err_level option

class BadassLogger:

    _logger = None

    def __new__(cls, ba_ctx):

        if cls._logger:
            return cls._logger

        cls._logger = super().__new__(cls)

        cls._logger.ctx = ba_ctx # BadassContext

        cls._logger.log_dir = ba_ctx.outdir.joinpath('log')
        cls._logger.log_dir.mkdir(parents=True, exist_ok=True)

        # File for useful BADASS output
        cls._logger.log_file_path = cls._logger.log_dir.joinpath('log_file.txt')
        # File for all BADASS logging
        cls._logger.log_out_path = cls._logger.log_dir.joinpath('out_log.txt')

        log_lvl = logging.getLevelName(cls._logger.ctx.options.io_options.log_level.upper())
        log_lvl = log_lvl if isinstance(log_lvl, int) else logging.INFO

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        cls._logger.logger = logging.getLogger('BADASS_log')
        cls._logger.logger.setLevel(log_lvl) # TODO: have a separate log level for default to INFO
        fh = logging.FileHandler(cls._logger.log_file_path)
        cls._logger.logger.addHandler(fh)

        cls._logger.logout = logging.getLogger('BADASS_out')
        cls._logger.logout.setLevel(log_lvl)
        fh = logging.FileHandler(cls._logger.log_out_path)
        fh.setFormatter(formatter)
        cls._logger.logout.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        cls._logger.logout.addHandler(sh)

        cls._logger.verbose = log_lvl < logging.WARN
        cls._logger.log_title()
        return cls._logger


    def debug(self, msg):
        self.logout.debug(msg)

    def info(self, msg):
        self.logout.info(msg)

    def warn(self, msg):
        self.logout.warning(msg)

    def error(self, msg):
        self.logout.error(msg)

    def critical(self, msg):
        self.logout.critical(msg)


    def log_title(self):
        # TODO: get version from central source
        self.logger.info('############################### BADASS v11.0.0 LOGFILE ####################################')


    # TODO: move to input classes
    def log_target_info(self):
        self.logger.info('-----------------------------------------------------------------------------------------------------------------')
        self.logger.info('{0:<30}{1:<30}'.format('name:', self.ctx.name))
        if (isinstance(self.ctx.ra, (float,int))) and (isinstance(self.ctx.dec, (float,int))):
            self.logger.info('{0:<30}{1:<30}'.format('(RA, DEC):', '(%0.6f,%0.6f)' % (self.ctx.ra,self.ctx.dec)))
        else:
            self.logger.info('{0:<30}{1:<30}'.format('(RA, DEC):', '(%s,%s)' % (self.ctx.ra,self.ctx.dec)))
        self.logger.info('{0:<30}{1:<30}'.format('SDSS redshift:', '%0.5f' % self.ctx.z))
        self.logger.info('{0:<30}{1:<30}'.format('fitting region:', '(%d,%d) [A]' % (self.ctx.fit_reg.min,self.ctx.fit_reg.max)))
        self.logger.info('{0:<30}{1:<30}'.format('velocity scale:', '%0.2f [km/s/pixel]' % self.ctx.velscale))
        # self.logger.info('{0:<30}{1:<30}'.format('Galactic E(B-V):', '%0.3f' % self.ctx.ebv)) # TODO
        self.logger.info('{0:<30}{1:<30}'.format('Flux Normalization:', '%0.0e' % self.ctx.flux_norm))
        self.logger.info('{0:<30}{1:<30}'.format('Fit Normalization:', '%0.5f' % self.ctx.fit_norm))

        self.logger.info('\n')
        self.logger.info('{0:<30}'.format('Units:'))
        self.logger.info('{0:<30}'.format('\t- Fluxes are in units of [%0.0e erg/s/cm2/Å]' % (self.ctx.options.fit_options.flux_norm)))
        self.logger.info('{0:<30}'.format('\t- Fiting normalization factor is %0.5f' % (self.ctx.fit_norm)))
        
        self.logger.info('\n')
        self.logger.info(
        """
        \t The flux normalization is usually given in the spectrum FITS header as
        \t BUNIT and is usually dependent on the detector.  For example, SDSS spectra
        \t have a flux normalization of 1.E-17, MUSE 1.E-20, KCWI 1.E-16 etc.

        \t The fit normalization is a normalization of the spectrum internal to BADASS
        \t such that the spectrum that is fit has a maximum of 1.0.  This is done so
        \t all spectra that are fit are uniformly scaled for the various algorithms
        \t used by BADASS.
        """
        )
        self.logger.info('\n')

        self.logger.info('{0:<30}'.format('\t- Velocity, dispersion, and FWHM have units of [km/s]'))
        self.logger.info('{0:<30}'.format('\t- Fluxes and Luminosities are in log-10'))
        self.logger.info('\n')
        self.logger.info('{0:<30}'.format('Cosmology:'))
        self.logger.info('{0:<30}'.format('\t H0 = %0.1f' % self.ctx.options.fit_options.cosmology['H0']))
        self.logger.info('{0:<30}'.format('\t Om0 = %0.2f' % self.ctx.options.fit_options.cosmology['Om0']))
        self.logger.info('\n')
        self.logger.info('-----------------------------------------------------------------------------------------------------------------')


    def log_fit_information(self):
        # TODO: does it make more sense to just pretty print the entire options dict to a file?
        # TODO: use options.<sub_option>.items() to just print all items?
        self.logger.info('### User-Input Fitting Paramters & Options ###')
        self.logger.info('-----------------------------------------------------------------------------------------------------------------')

        self.logger.info(json.dumps(self.ctx.options, default=str, indent=4))


    def pca_information(self, pca_nan_fix=False, pca_exp_var=None):
        self.logger.info('### PCA Options ###')
        self.logger.info('-----------------------------------------------------------------------------------------------------------------')
        self.logger.info('{0:<30}'.format('pca_options:'))
        self.logger.info('{0:>30}{1:<2}{2:<30}'.format('do_pca', ':', str(self.ctx.options.pca_options.do_pca)))
        if self.ctx.options.pca_options.do_pca:
            self.logger.info('{0:>30}{1:<2}{2:<30.8f}'.format('exp_var', ':', pca_exp_var))
            self.logger.info('{0:>30}{1:<2}{2:<30}'.format('pca_nan_fix', ':', str(pca_nan_fix)))
            n_comps = self.ctx.options.pca_options.n_components if self.ctx.options.pca_options.n_components else 'All'
            self.logger.info('{0:>30}{1:<2}{2:<30}'.format('n_components', ':', n_comps))
            self.logger.info('{0:>30}{1:<2}'.format('pca_masks', ':'))
            pca_masks = self.ctx.options.pca_options.pca_masks
            for ind, m in enumerate(pca_masks):
                self.logger.info(', '.join([str(p) for p in pca_masks]))                
        self.logger.info('-----------------------------------------------------------------------------------------------------------------\n') 


    # TODO: change names
    # TODO: move to individual template class
    def update_opt_feii(self):
        self.logger.info('\t* optical FeII templates outside of fitting region and disabled.')

    def update_uv_iron(self):
        self.logger.info('\t* UV iron template outside of fitting region and disabled.')

    def update_balmer(self):
        self.logger.info('\t* Balmer continuum template outside of fitting region and disabled.')


    def log_max_like_fit(self, result_dict, noise_std, resid_std):
        self.logger.info('### Maximum Likelihood Fitting Results ###')
        self.logger.info('-----------------------------------------------------------------------------------------------------------------')
        self.logger.info('{0:<30}{1:<30}{2:<30}{3:<30}'.format('Parameter','Max. Like. Value','+/- 1-sigma', 'Flag') )
        self.logger.info('-----------------------------------------------------------------------------------------------------------------')
        for pname, pdict in result_dict.items():
            self.logger.info('{0:<30}{1:<30.4f}{2:<30.4f}{3:<30}'.format(pname, pdict['med'], pdict['std'], pdict['flag']))
        self.logger.info('{0:<30}{1:<30.4f}'.format('NOISE_STD.', noise_std ))
        self.logger.info('{0:<30}{1:<30.4f}'.format('RESID_STD', resid_std ))
        self.logger.info('-----------------------------------------------------------------------------------------------------------------')


    # TODO: just pretty print line list, soft cons?
    def output_line_list(self, line_list, soft_cons):
        self.logger.info('----------------------------------------------------------------------------------------------------------------------------------------')
        self.logger.info('Line List:')
        nfree = 0 
        self.logger.info('----------------------------------------------------------------------------------------------------------------------------------------')
        for line in sorted(list(line_list)):
            self.logger.info('{0:<30}{1:<30}{2:<30.2}'.format(line, '',''))
            for par in sorted(list(line_list[line])):
                self.logger.info('{0:<30}{1:<30}{2:<30}'.format('', par,str(line_list[line][par])))
                if line_list[line][par] == 'free': nfree+=1
        self.logger.info('----------------------------------------------------------------------------------------------------------------------------------------')
        self.logger.info('Soft Constraints:')
        for con in soft_cons:
            self.logger.info('\n{0:>30}{1:<0}{2:<0}'.format(con[0], ' > ',con[1]))
        self.logger.info('----------------------------------------------------------------------------------------------------------------------------------------')


    # TODO: just pretty print?
    def output_free_pars(self, line_list, par_input, soft_cons):
        self.logger.info('----------------------------------------------------------------------------------------------------------------------------------------')
        self.logger.info('----------------------------------------------------------------------------------------------------------------------------------------')

        self.logger.info('Line List:')
        nfree = 0 
        self.logger.info('----------------------------------------------------------------------------------------------------------------------------------------')
        for line in sorted(list(line_list)):
            self.logger.info('{0:<30}{1:<30}{2:<30.2}'.format(line, '',''))
            for par in sorted(list(line_list[line])):
                self.logger.info('{0:<30}{1:<30}{2:<30}'.format('',par,str(line_list[line][par])))
                if line_list[line][par] == 'free': nfree+=1
        self.logger.info('----------------------------------------------------------------------------------------------------------------------------------------')
        self.logger.info('Number of Free Line Parameters: %d' % nfree)
        self.logger.info('----------------------------------------------------------------------------------------------------------------------------------------')
        self.logger.info('All Free Parameters:')
        self.logger.info('----------------------------------------------------------------------------------------------------------------------------------------')

        nfree = 0
        for par in sorted(list(par_input)):
            self.logger.info('{0:<30}{1:<30}{2:<30.2}'.format(par, '',''))
            nfree+=1
            for hpar in sorted(list(par_input[par])):
                self.logger.info('{0:<30}{1:<30}{2:<30}'.format('', hpar,str(par_input[par][hpar])))
        self.logger.info('----------------------------------------------------------------------------------------------------------------------------------------')
        self.logger.info('Total number of free parameters: %d' % nfree)
        self.logger.info('----------------------------------------------------------------------------------------------------------------------------------------')
        self.logger.info('Soft Constraints:')
        for con in soft_cons:
            self.logger.info('{0:>30}{1:<0}{2:<0}'.format(con[0],' > ',con[1]))
        self.logger.info('----------------------------------------------------------------------------------------------------------------------------------------')
        self.logger.info('----------------------------------------------------------------------------------------------------------------------------------------')


    def output_options(self):
        file_path = self.log_dir.joinpath('fit_options.toml')
        with open(file_path, 'w') as opt_out:
            toml.dump(self.ctx.options.to_dict(), opt_out)

