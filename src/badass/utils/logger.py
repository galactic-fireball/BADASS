import json
import logging
import sys


class BufferHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self._buffer: list[logging.LogRecord] = []


    def emit(self, record: logging.LogRecord) -> None:
        self._buffer.append(record)


    def flush_to(self, handler: logging.Handler) -> None:
        for record in self._buffer:
            handler.emit(record)
        self._buffer.clear()


    def close(self) -> None:
        self._buffer.clear()
        super().close()


class BadassLogger:
    default_level = logging.INFO
    default_name = 'BADASS-MAIN'
    _loggers: dict[str, 'BadassLogger'] = {}


    def __new__(cls, name=None, level=None):
        if name is None:
            name = BadassLogger.default_name

        if name in cls._loggers:
            return cls._loggers[name]

        logger = super().__new__(cls)
        logger._initialized = False
        cls._loggers[name] = logger
        return logger


    def __init__(self, name=None, level=None):
        if self._initialized:
            return
        self._initialized = True

        self.name = name
        if self.name is None:
            self.name = self.default_name

        level = self.to_log_level(level)
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(level)
        self._logger.propagate = False

        self._formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        self._console_handler = logging.StreamHandler(sys.stdout)
        self._console_handler.setFormatter(self._formatter)
        self._console_handler.setLevel(level)
        self._logger.addHandler(self._console_handler)

        # Create a buffer to write to log file once we know
        # where the output directory is
        self._buffer_handler = BufferHandler()
        self._buffer_handler.setFormatter(self._formatter)
        self._buffer_handler.setLevel(level)
        self._logger.addHandler(self._buffer_handler)

        self._file_handler = None


    @staticmethod
    def to_log_level(level):
        if level is None:
            return BadassLogger.default_level

        if isinstance(level, int):
            return level

        if isinstance(level, str):
            nlvl = logging.getLevelName(level.upper())
            if not isinstance(nlvl, int): nlvl = BadassLogger.default_level
            return nlvl

        return BadassLogger.default_level


    def set_level(self, lvl):
        lvl = self.to_log_level(lvl)
        self._logger.setLevel(lvl)
        for handler in self._logger.handlers:
            handler.setLevel(lvl)


    def set_output(self, outdir):
        log_out_file = outdir.joinpath('log', 'log.txt')
        log_out_file.parent.mkdir(parents=True, exist_ok=True)

        if not self._file_handler is None:
            self._logger.removeHandler(self._file_handler)

        self._file_handler = logging.FileHandler(log_out_file)
        self._file_handler.setFormatter(self._formatter)
        self._file_handler.setLevel(self._logger.level)

        self._buffer_handler.flush_to(self._file_handler)
        self._logger.addHandler(self._file_handler)
        self._logger.removeHandler(self._buffer_handler)
        self._buffer_handler.close()



    def get_runner_logger(self):
        pass


    def debug(self, msg, *args, **kwargs): self._logger.debug(msg, *args, **kwargs)
    def info(self, msg, *args, **kwargs):
        # breakpoint()
        self._logger.info(msg, *args, **kwargs)
    def warn(self, msg, *args, **kwargs): self._logger.warn(msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs): self._logger.error(msg, *args, **kwargs)
    def critical(self, msg, *args, **kwargs): self._logger.critical(msg, *args, **kwargs)


    
# TODO: implement, fix, remove, etc:

# def log_title(self):
#     # TODO: get version from central source
#     self.logger.info('############################### BADASS v11.0.0 LOGFILE ####################################')


# # TODO: move to input classes
# def log_target_info(self):
#     self.logger.info('-----------------------------------------------------------------------------------------------------------------')
#     self.logger.info('{0:<30}{1:<30}'.format('name:', self.ctx.name))
#     if (isinstance(self.ctx.ra, (float,int))) and (isinstance(self.ctx.dec, (float,int))):
#         self.logger.info('{0:<30}{1:<30}'.format('(RA, DEC):', '(%0.6f,%0.6f)' % (self.ctx.ra,self.ctx.dec)))
#     else:
#         self.logger.info('{0:<30}{1:<30}'.format('(RA, DEC):', '(%s,%s)' % (self.ctx.ra,self.ctx.dec)))
#     self.logger.info('{0:<30}{1:<30}'.format('SDSS redshift:', '%0.5f' % self.ctx.z))
#     self.logger.info('{0:<30}{1:<30}'.format('fitting region:', '(%d,%d) [A]' % (self.ctx.fit_reg.min,self.ctx.fit_reg.max)))
#     self.logger.info('{0:<30}{1:<30}'.format('velocity scale:', '%0.2f [km/s/pixel]' % self.ctx.velscale))
#     # self.logger.info('{0:<30}{1:<30}'.format('Galactic E(B-V):', '%0.3f' % self.ctx.ebv)) # TODO
#     self.logger.info('{0:<30}{1:<30}'.format('Flux Normalization:', '%0.0e' % self.ctx.flux_norm))
#     self.logger.info('{0:<30}{1:<30}'.format('Fit Normalization:', '%0.5f' % self.ctx.fit_norm))

#     self.logger.info('\n')
#     self.logger.info('{0:<30}'.format('Units:'))
#     self.logger.info('{0:<30}'.format('\t- Fluxes are in units of [%0.0e erg/s/cm2/Å]' % (self.ctx.flux_norm)))
#     self.logger.info('{0:<30}'.format('\t- Fiting normalization factor is %0.5f' % (self.ctx.fit_norm)))

#     self.logger.info('\n')
#     self.logger.info(
#     """
#     \t The flux normalization is usually given in the spectrum FITS header as
#     \t BUNIT and is usually dependent on the detector.  For example, SDSS spectra
#     \t have a flux normalization of 1.E-17, MUSE 1.E-20, KCWI 1.E-16 etc.

#     \t The fit normalization is a normalization of the spectrum internal to BADASS
#     \t such that the spectrum that is fit has a maximum of 1.0.  This is done so
#     \t all spectra that are fit are uniformly scaled for the various algorithms
#     \t used by BADASS.
#     """
#     )
#     self.logger.info('\n')

#     self.logger.info('{0:<30}'.format('\t- Velocity, dispersion, and FWHM have units of [km/s]'))
#     self.logger.info('{0:<30}'.format('\t- Fluxes and Luminosities are in log-10'))
#     self.logger.info('\n')
#     self.logger.info('{0:<30}'.format('Cosmology:'))
#     self.logger.info('{0:<30}'.format('\t H0 = %0.1f' % self.ctx.cfg.fit.cosmology['H0']))
#     self.logger.info('{0:<30}'.format('\t Om0 = %0.2f' % self.ctx.cfg.fit.cosmology['Om0']))
#     self.logger.info('\n')
#     self.logger.info('-----------------------------------------------------------------------------------------------------------------')


# def log_fit_information(self):
#     # TODO: does it make more sense to just pretty print the entire cfg dict to a file?
#     # TODO: use cfg.<sub_option>.items() to just print all items?
#     self.logger.info('### User-Input Fitting Paramters & Options ###')
#     self.logger.info('-----------------------------------------------------------------------------------------------------------------')

#     self.logger.info(json.dumps(self.ctx.cfg, default=str, indent=4))


# def pca_information(self, pca_nan_fix=False, pca_exp_var=None):
#     self.logger.info('### PCA Options ###')
#     self.logger.info('-----------------------------------------------------------------------------------------------------------------')
#     self.logger.info('{0:<30}'.format('pca_options:'))
#     self.logger.info('{0:>30}{1:<2}{2:<30}'.format('do_pca', ':', str(self.ctx.cfg.pca.do_pca)))
#     if self.ctx.cfg.pca.do_pca:
#         self.logger.info('{0:>30}{1:<2}{2:<30.8f}'.format('exp_var', ':', pca_exp_var))
#         self.logger.info('{0:>30}{1:<2}{2:<30}'.format('pca_nan_fix', ':', str(pca_nan_fix)))
#         n_comps = self.ctx.cfg.pca.n_components if self.ctx.cfg.pca.n_components else 'All'
#         self.logger.info('{0:>30}{1:<2}{2:<30}'.format('n_components', ':', n_comps))
#         self.logger.info('{0:>30}{1:<2}'.format('pca_masks', ':'))
#         pca_masks = self.ctx.cfg.pca.pca_masks
#         for ind, m in enumerate(pca_masks):
#             self.logger.info(', '.join([str(p) for p in pca_masks]))                
#     self.logger.info('-----------------------------------------------------------------------------------------------------------------\n') 


# # TODO: change names
# # TODO: move to individual template class
# def update_opt_feii(self):
#     self.logger.info('\t* optical FeII templates outside of fitting region and disabled.')

# def update_uv_iron(self):
#     self.logger.info('\t* UV iron template outside of fitting region and disabled.')

# def update_balmer(self):
#     self.logger.info('\t* Balmer continuum template outside of fitting region and disabled.')


# def log_max_like_fit(self, result_dict, noise_std, resid_std):
#     self.logger.info('### Maximum Likelihood Fitting Results ###')
#     self.logger.info('-----------------------------------------------------------------------------------------------------------------')
#     self.logger.info('{0:<30}{1:<30}{2:<30}{3:<30}'.format('Parameter','Max. Like. Value','+/- 1-sigma', 'Flag') )
#     self.logger.info('-----------------------------------------------------------------------------------------------------------------')
#     for pname, pdict in result_dict.items():
#         self.logger.info('{0:<30}{1:<30.4f}{2:<30.4f}{3:<30}'.format(pname, pdict['med'], pdict['std'], pdict['flag']))
#     self.logger.info('{0:<30}{1:<30.4f}'.format('NOISE_STD.', noise_std ))
#     self.logger.info('{0:<30}{1:<30.4f}'.format('RESID_STD', resid_std ))
#     self.logger.info('-----------------------------------------------------------------------------------------------------------------')


# def output_cfg(self):
#     file_path = self.log_dir.joinpath('fit_cfg.json')
#     with open(file_path, 'w') as opt_out:
#         opt_out.write(self.ctx.cfg.model_dump_json())

