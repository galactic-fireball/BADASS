import logging
import numpy as np
import sys
import toml

# TODO: create error file with warning +
#           check err_level option

class BadassLogger:
    def __init__(self, ba_ctx):
        self.ctx = ba_ctx # BadassContext

        self.log_dir = ba_ctx.outdir.joinpath('log')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # File for useful BADASS output
        self.log_file_path = self.log_dir.joinpath('log_file.txt')
        # File for all BADASS logging
        self.log_out_path = self.log_dir.joinpath('out_log.txt')

        log_lvl = logging.getLevelName(self.ctx.options.io_options.log_level.upper())
        log_lvl = log_lvl if isinstance(log_lvl, int) else logging.INFO

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        self.logger = logging.getLogger('BADASS_log')
        self.logger.setLevel(log_lvl) # TODO: have a separate log level for default to INFO
        fh = logging.FileHandler(self.log_file_path)
        self.logger.addHandler(fh)

        self.logout = logging.getLogger('BADASS_out')
        self.logout.setLevel(log_lvl)
        fh = logging.FileHandler(self.log_out_path)
        fh.setFormatter(formatter)
        self.logout.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logout.addHandler(sh)

        self.verbose = log_lvl < logging.WARN

        self.log_title()


    def debug(self, msg):
        self.logout.debug(msg)

    def info(self, msg):
        self.logout.info(msg)

    def warning(self, msg):
        self.logout.warning(msg)

    def error(self, msg):
        self.logout.error(msg)

    def critical(self, msg):
        self.logout.critical(msg)


    def log_title(self):
        # TODO: get version from central source
        self.logger.info('############################### BADASS v9.1.1 LOGFILE ####################################')


    # TODO: move to input classes
    def log_target_info(self):
        self.logger.info('-----------------------------------------------------------------------------------------------------------------')
        self.logger.info('{0:<30}{1:<30}'.format('file:', self.ctx.infile.name))
        if (self.ctx.ra is not None) and (self.ctx.dec is not None):
            self.logger.info('{0:<30}{1:<30}'.format('(RA, DEC):', '(%0.6f,%0.6f)' % (self.ctx.ra,self.ctx.dec)))
        self.logger.info('{0:<30}{1:<30}'.format('SDSS redshift:', '%0.5f' % self.ctx.z))
        self.logger.info('{0:<30}{1:<30}'.format('fitting region:', '(%d,%d) [A]' % (self.ctx.fit_reg.min,self.ctx.fit_reg.max)))
        self.logger.info('{0:<30}{1:<30}'.format('velocity scale:', '%0.2f [km/s/pixel]' % self.ctx.velscale))
        # self.logger.info('{0:<30}{1:<30}'.format('Galactic E(B-V):', '%0.3f' % self.ctx.ebv)) # TODO
        self.logger.info('{0:<30}{1:<30}'.format('Flux Normalization:', '%0.0e' % self.ctx.options.fit_options.flux_norm))
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
        self.logger.info('\n### User-Input Fitting Paramters & Options ###')
        self.logger.info('-----------------------------------------------------------------------------------------------------------------')

        # General fit options
        fit_options = self.ctx.options.fit_options
        self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('fit_options:','',''))
        for key in ['fit_reg', 'good_thresh', 'mask_bad_pix', 'n_basinhop', 'test_lines', 'max_like_niter', 'output_pars']:
            self.logger.info('{0:>30}{1:<2}{2:<30}'.format(key,':', str(fit_options[key])))
        self.logger.info('\n')

        # MCMC options
        mcmc_options = self.ctx.options.mcmc_options
        self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('mcmc_options:','',''))
        if mcmc_options.mcmc_fit:
            for key in ['mcmc_fit', 'nwalkers', 'auto_stop', 'conv_type', 'min_samp', 'ncor_times', 'autocorr_tol', 'write_iter', 'write_thresh', 'burn_in', 'min_iter', 'max_iter']:
                self.logger.info('{0:>30}{1:<2}{2:<30}'.format(key,':', str(mcmc_options[key])))
        else:
            self.logger.info('{0:>30}{1:<2}{2:<30}'.format('','','MCMC fitting is turned off.' ))
        self.logger.info('\n')

        # Fit Component options
        comp_options = self.ctx.options.comp_options
        self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('comp_options:','',''))
        for key in ['fit_opt_feii', 'fit_uv_iron', 'fit_balmer', 'fit_losvd', 'fit_host', 'fit_power', 'fit_narrow', 'fit_broad', 'fit_absorp', 'tie_line_disp', 'tie_line_voff']:
            self.logger.info('{0:>30}{1:<2}{2:<30}'.format(key,':', str(comp_options[key])))
        self.logger.info('\n')

        # LOSVD options
        self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('losvd_options:','',''))
        if comp_options.fit_losvd:
            losvd_options = self.ctx.options.losvd_options
            for key in ['library', 'vel_const', 'disp_const']:
                self.logger.info('{0:>30}{1:<2}{2:<30}'.format(key,':', str(losvd_options[key])))
        else:
            self.logger.info('{0:>30}{1:<2}{2:<30}'.format('','','LOSVD fitting is turned off.'))
        self.logger.info('\n')

        # Host Options
        self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('host_options:','',''))
        if comp_options.fit_host:
            host_options = self.ctx.options.host_options
            for key in ['age', 'vel_const', 'disp_const']:
                self.logger.info('{0:>30}{1:<2}{2:<30}'.format(key,':', str(host_options[key])))
        else:
            self.logger.info('{0:>30}{1:<2}{2:<30}'.format('','','Host-galaxy template fitting is turned off.'))
        self.logger.info('\n')

        # Power-law continuum options
        self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('power_options:','',''))
        if comp_options.fit_power:
            power_options = self.ctx.options.power_options
            self.logger.info('{0:>30}{1:<2}{2:<30}'.format('type',':', str(power_options['type'])))
        else:
            self.logger.info('{0:>30}{1:<2}{2:<30}'.format('','','Power Law fitting is turned off.'))
        self.logger.info('\n')

        # Optical FeII fitting options
        self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('opt_feii_options:','',''))
        if comp_options.fit_opt_feii:
            opt_feii_options = self.ctx.options.opt_feii_options
            self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_template',':','type: %s' % str(opt_feii_options['opt_template']['type']) ))
            if opt_feii_options.opt_template.type == 'VC04':
                self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_amp_const',':','bool: %s, br_opt_feii_val: %s, na_opt_feii_val: %s' % (str(opt_feii_options['opt_amp_const']['bool']),str(opt_feii_options['opt_amp_const']['br_opt_feii_val']),str(opt_feii_options['opt_amp_const']['na_opt_feii_val']))))
                self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_disp_const',':','bool: %s, br_opt_feii_val: %s, na_opt_feii_val: %s' % (str(opt_feii_options['opt_disp_const']['bool']),str(opt_feii_options['opt_disp_const']['br_opt_feii_val']),str(opt_feii_options['opt_disp_const']['na_opt_feii_val']))))
                self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_voff_const',':','bool: %s, br_opt_feii_val: %s, na_opt_feii_val: %s' % (str(opt_feii_options['opt_voff_const']['bool']),str(opt_feii_options['opt_voff_const']['br_opt_feii_val']),str(opt_feii_options['opt_voff_const']['na_opt_feii_val']))))
            elif opt_feii_options.opt_template.type =='K10':
                self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_amp_const',':','bool: %s, f_feii_val: %s, s_feii_val: %s, g_feii_val: %s, z_feii_val: %s' % (str(opt_feii_options['opt_amp_const']['bool']),str(opt_feii_options['opt_amp_const']['f_feii_val']),str(opt_feii_options['opt_amp_const']['s_feii_val']),str(opt_feii_options['opt_amp_const']['g_feii_val']),str(opt_feii_options['opt_amp_const']['z_feii_val']))))
                self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_disp_const',':','bool: %s, opt_feii_val: %s' % (str(opt_feii_options['opt_disp_const']['bool']),str(opt_feii_options['opt_disp_const']['opt_feii_val']),)))
                self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_voff_const',':','bool: %s, opt_feii_val: %s' % (str(opt_feii_options['opt_voff_const']['bool']),str(opt_feii_options['opt_voff_const']['opt_feii_val']),)))
                self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_temp_const',':','bool: %s, opt_feii_val: %s' % (str(opt_feii_options['opt_temp_const']['bool']),str(opt_feii_options['opt_temp_const']['opt_feii_val']),)))
        else:
            self.logger.info('{0:>30}{1:<2}{2:<30}'.format('','','Optical FeII fitting is turned off.'))
        self.logger.info('\n')

        # UV Iron options'
        self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('uv_iron_options:','',''))
        if comp_options.fit_uv_iron:
            uv_iron_options = self.ctx.options.uv_iron_options
            for key in ['uv_amp_const', 'uv_disp_const', 'uv_voff_const']:
                self.logger.info('{0:>30}{1:<2}{2:<100}'.format(key,':','bool: %s, uv_iron_val: %s' % (str(uv_iron_options[key]['bool']), str(uv_iron_options[key]['uv_iron_val']))))
        else:
            self.logger.info('{0:>30}{1:<2}{2:<30}'.format('','','UV Iron fitting is turned off.'))
        self.logger.info('\n')

        # Balmer options
        self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('balmer_options:','',''))
        if comp_options.fit_balmer:
            balmer_options = self.ctx.options.balmer_options
            for key in ['R', 'balmer_amp', 'balmer_disp', 'balmer_voff', 'Teff', 'tau']:
                self.logger.info('{0:>30}{1:<2}{2:<100}'.format('%s_const'%key,':','bool: %s, %s_val: %s' % (str(balmer_options['%s_const'%key]['bool']), key, str(balmer_options['%s_const'%key]['%s_val'%key]))))
        else:
            self.logger.info('{0:>30}{1:<2}{2:<30}'.format('','','Balmer pseudo-continuum fitting is turned off.' )) 
        self.logger.info('\n')

         # Plotting options
        self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('plot_options:','',''))
        plot_options = self.ctx.options.plot_options
        for key in ['plot_param_hist', 'plot_pca']:
            self.logger.info('{0:>30}{1:<2}{2:<30}'.format(key,':',str(plot_options[key])))
        self.logger.info('\n')

        # Output options
        self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('output_options:','',''))
        output_options = self.ctx.options.output_options
        for key in ['write_chain', 'verbose']:
            self.logger.info('{0:>30}{1:<2}{2:<30}'.format(key,':',str(output_options[key]) )) 

        # TODO: other options?

        self.logger.info('\n')
        self.logger.info('-----------------------------------------------------------------------------------------------------------------')


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

