from cerberus import Validator
from cerberus import rules_set_registry

import utils.constants as consts

# short hands for some common type rules
rules_set_registry.add('bool_false', {'type':'boolean', 'default':False})
rules_set_registry.add('bool_true', {'type':'boolean', 'default':True})
rules_set_registry.add('intfl_min0_d1', {
                                            'type': ['integer', 'float'],
                                            'min': 0.0,
                                            'default': 1.0,
                                        })
rules_set_registry.add('intfl_d0', {
                                        'type': ['integer', 'float'],
                                        'default': 0.0,
                                   })
rules_set_registry.add('line_profile', {'type':'string', 'allowed':consts.LINE_PROFILES, 'default':'gaussian'})
rules_set_registry.add('poly_dict', {
                                        'type': 'dict',
                                        'default': {},
                                        'schema': {
                                            'bool': 'bool_false',
                                            'order': {
                                                'type': ['integer', 'float'],
                                                'default': 3.0,
                                                'min': 0.0,
                                                'max': 99.0,
                                            },
                                        },
                                    })


class DefaultValidator(Validator):
    def _validate_min_ex(self, test_val, field, value):
        if value <= test_val:
            self._error(field, '%s must be strictly greater than %s' % (field, str(test_val)))


    def _validate_max_ex(self, test_val, field, value):
        if value >= test_val:
            self._error(field, '%s must be strictly less than %s' % (field, str(test_val)))


    def _validate_lt_other(self, other, field, value):
        if other not in self.document:
            return False
        if value >= self.document[other]:
            self._error(field, '%s must be less than %s' % (field, other))


    def _validate_gt_other(self, other, field, value):
        if other not in self.document:
            return False
        if value <= self.document[other]:
            self._error(field, '%s must be greater than %s' % (field, other))


    def _validate_le_other(self, other, field, value):
        if other not in self.document:
            return False
        if value > self.document[other]:
            self._error(field, '%s must be less than or equal to %s' % (field, other))


    def _validate_ge_other(self, other, field, value):
        if other not in self.document:
            return False
        if value < self.document[other]:
            self._error(field, '%s must be greater than or equal to %s' % (field, other))


    def _validate_is_lohi(self, constraint, field, value):
        if not constraint:
            return

        if (not isinstance(value, (list,tuple))) or (len(value) != 2) or (value[1] < value[0]):
            self._error(field, '%s must be a list of length 2' % field)


    def _normalize_coerce_nonzero(self, value):
        if value <= 0:
            return 1.e-3
        return value



# NOTE: Any container objects (ie. dicts, lists, etc.) needs to have a default
#       empty object ({}, [], etc.) in order for its child objects/values to
#       have their defaults set

# io_options
DEFAULT_IO_OPTIONS = {
    'infmt': {
        'type': 'string',
        'default': 'sdss_spec',
    },
    'output_dir': {
        'type': 'string',
        'nullable': True,
        'default': None,
    },
    'overwrite': 'bool_false',
    'dust_cache': {
        'type': 'string',
        'nullable': True,
        'default': None,
    },
    'log_level': {
        'type': 'string',
        'default': 'info',
    },
    'err_level': {
        'type': 'string',
        'default': 'warning',
    },
    'multiprocess': 'bool_true',
    'flux_norm': {
        'type': ['integer', 'float'],
        'default': 1e-17,
    }
}


# fit_options
DEFAULT_FIT_OPTIONS = {
    # Fitting region; Note: Indo-US Library=(3460,9464)
    'fit_reg': {
        'type': 'list',
        'minlength': 2,
        'maxlength': 2,
        'schema': {'type': ['integer', 'float']},
        'default': (4400.0, 5500.0),
    },
    # percentage of "good" pixels required in fig_reg for fit.
    'good_thresh': {
        'type': ['integer', 'float'],
        'min': 0,
        'max': 1,
        'default': 0.0,
    },
    # mask pixels SDSS flagged as 'bad' (careful!)
    'mask_bad_pix': 'bool_false',
    # mask emission lines for continuum fitting.
    'mask_emline': 'bool_false',
    # interpolate over metal absorption lines for high-z spectra
    'mask_metal': 'bool_false',
    # fit statistic; ML = Max. Like. , LS = Least Squares
    'fit_stat': {
        'type': 'string',
        'allowed': consts.FIT_STATS,
        'default': 'ML',
    },
    # Number of consecutive basinhopping thresholds before solution achieved
    'n_basinhop': {
        'type': 'integer',
        'min': 0,
        'default': 5,
    },
    # re-weight the noise after initial fit to achieve RCHI2 = 1
    'reweighting': 'bool_true',
    # only test for outflows; stops after test
    'test_lines': 'bool_false',
    # number of maximum likelihood iterations
    'max_like_niter': {
        'type': 'integer',
        'min': 0,
        'default': 10,
    },
    # only output free parameters of fit and stop code (diagnostic)
    'output_pars': 'bool_false',
    'cosmology': {
        'type': 'dict',
        'default': {},
        'schema': {
            'H0': {
                'type': ['integer', 'float'],
                'default': 70.0,
            },
            'Om0': {
                'type': 'float',
                'default': 0.30,
            },
        },
    },
    'flux_norm': {
        'type': 'float',
        'default': 1.0,
    },
}


# test_options
DEFAULT_TEST_OPTIONS = {
    'test_mode': {
        'type': 'string',
        'allowed': ['line', 'config'],
    },
    'lines': {
        'type': ['string', 'list'],
        'default': [],
    },
    'metrics': {
        'type': 'list',
        'default': ['BADASS'],
    },
    'thresholds': {
        'type': 'list',
        'default': [0.95],
    },
    'conv_mode': {
        'type': 'string',
        'allowed': ['any', 'all'],
        'default': 'any',
    },
    'auto_stop': 'bool_true',
    'full_verbose': 'bool_false',
    'plot_tests': 'bool_true',
    'force_best': 'bool_true',
    'continue_fit': 'bool_true',
}


# ifu_options
DEFAULT_IFU_OPTIONS = {
    'z': 'intfl_d0',
    'aperture': {
        'type': 'list',
        'minlength': 4,
        'maxlength': 4,
        'schema': {'type': 'integer',},
    },
    'voronoi_binning': 'bool_true',
    # Target S/N ratio to bin for
    'targetsn': {
        'type': ['integer', 'float'],
        'nullable': True,
        'default': None,
    },
    'cvt': 'bool_true',
    'voronoi_plot': 'bool_true',
    'quiet': 'bool_true',
    'wvt': 'bool_false',
    'maxbins': {
        'type': 'integer',
        'default': 800,
    },
    'snr_threshold': {
        'type': ['integer', 'float'],
        'min': 0,
        'default': 3.0,
    },
    'use_and_mask': 'bool_true',
}


# mcmc_options
DEFAULT_MCMC_OPTIONS = {
    # Perform robust fitting using emcee
    'mcmc_fit': 'bool_false',
    # Number of emcee walkers; min = 2 x N_parameters
    'nwalkers': {
        'type': 'integer',
        'min_ex': 0,
        'default': 100,
    },
    # Automatic stop using autocorrelation analysis
    'auto_stop': 'bool_false',
    'conv_type': {
        'type': ['string', 'list'],
        'allowed': ['mean','median','all'],
        'default': 'median',
    },
    # min number of iterations for sampling post-convergence
    'min_samp': {
        'type': 'integer',
        'min_ex': 0,
        'default': 100,
    },
    # number of autocorrelation times for convergence
    'ncor_times': {
        'type': ['integer', 'float'],
        'min': 0,
        'default': 5.0,
    },
    # percent tolerance between checking autocorr. times
    'autocorr_tol': {
        'type': ['integer', 'float'],
        'min': 0,
        'default': 10.0,
    },
    # write/check autocorrelation times interval
    'write_iter': {
        'type': 'integer',
        'min_ex': 0,
        'lt_other': 'max_iter',
        'default': 100,
    },
    # when to start writing/checking parameters
    'write_thresh': {
        'type': 'integer',
        'min_ex': 0,
        'lt_other': 'max_iter',
        'default': 100,
    },
    # burn-in if max_iter is reached
    'burn_in': {
        'type': 'integer',
        'min_ex': 0,
        'default': 1000,
    },
    # min number of iterations before stopping
    'min_iter': {
        'type': 'integer',
        'min_ex': 0,
        'default': 100,
    },
    # max number of MCMC iterations
    'max_iter': {
        'type': 'integer',
        'min_ex': 0,
        'ge_other': 'min_iter',
        'default': 1500,
    },
}


# comp_options
DEFAULT_COMP_OPTIONS = {
    'fit_opt_feii': 'bool_true', # optical FeII
    'fit_uv_iron': 'bool_false', # UV Iron 
    'fit_balmer': 'bool_false', # Balmer continuum (<4000 A)
    'fit_losvd': 'bool_false', # stellar LOSVD
    'fit_host': 'bool_true', # host template
    'fit_power': 'bool_true', # AGN power-law
    'fit_poly': 'bool_false', # polynomial continuum component
    'fit_narrow': 'bool_true', # narrow lines
    'fit_broad': 'bool_true', # broad lines
    'fit_outflow': 'bool_true', # outflow lines
    'fit_absorp': 'bool_true', # absorption lines
    'tie_line_disp': 'bool_false', # tie line widths
    'tie_line_voff': 'bool_false', # tie line velocity offsets
}


# narrow_options
DEFAULT_NARROW_OPTIONS = {
    'amp_plim': { # line amplitude parameter limits
        'type': 'list',
        'minlength': 1,
        'maxlength': 2,
        'schema': {
            'type': ['integer', 'float'],
            'nullable': True,
            'min': 0,
        },
        'nullable': True,
        'default': None,
    },
    'disp_plim': { # line dispersion parameter limits
        'type': 'list',
        'is_lohi': True,
        'nullable': True,
        'default': [0.001,300.0],
    },
    'voff_plim': { # line velocity offset parameter limits
        'type': 'list',
        'is_lohi': True,
        'nullable': True,
        'default': [-500.0,500.0],
    },
    'line_profile': 'line_profile', # line profile shape
    'n_moments': { # number of higher order Gauss-Hermite moments
        'type': 'integer',
        'min': 2,
        'max': 10,
        'default': 4,
    },
}


# broad_options
DEFAULT_BROAD_OPTIONS = {
    'amp_plim': { # line amplitude parameter limits
        'type': 'list',
        'minlength': 1,
        'maxlength': 2,
        'schema': {
            'type': ['integer', 'float'],
            'nullable': True,
            'min': 0,
        },
        'nullable': True,
        'default': None,
    },
    'disp_plim': { # line dispersion parameter limits
        'type': 'list',
        'is_lohi': True,
        'nullable': True,
        'default': [300.0,3000.0],
    },
    'voff_plim': { # line velocity offset parameter limits
        'type': 'list',
        'is_lohi': True,
        'nullable': True,
        'default': [-1000.0,1000.0],
    },
    'line_profile': 'line_profile', # line profile shape
    'n_moments': { # number of higher order Gauss-Hermite moments
        'type': 'integer',
        'min': 2,
        'max': 10,
        'default': 4,
    }
}


# absorp_options
DEFAULT_ABSORP_OPTIONS = {
    'amp_plim': { # line amplitude parameter limits
        'type': 'list',
        'minlength': 2,
        'maxlength': 2,
        'schema': {
            'type': ['integer', 'float'],
            'nullable': True,
            'max': 0,
        },
        'nullable': True,
        'default': [-1,0],
    },
    'disp_plim': { # line dispersion parameter limits
        'type': 'list',
        'is_lohi': True,
        'nullable': True,
        'default': [0.001,3000.0],
    },
    'voff_plim': { # line velocity offset parameter limits
        'type': 'list',
        'is_lohi': True,
        'nullable': True,
        'default': [-1000.0,1000.0],
    },
    'line_profile': 'line_profile', # line profile shape
    'n_moments': { # number of higher order Gauss-Hermite moments
        'type': 'integer',
        'min': 2,
        'max': 10,
        'default': 4,
    }
}


# pca_options
DEFAULT_PCA_OPTIONS = {
    'do_pca': 'bool_false',
    'n_components': {
        'type': 'integer',
        'nullable': True,
        'min': 1,
        'default': 20,
    },
    'pca_masks': {
        'type': 'list',
        'schema': {
            'is_lohi': True,
        },
    },
}


# user_constraints
DEFAULT_USER_CONSTRAINTS = {
    'type': 'list',
    'minlength': 2,
    'maxlength': 2,
    'schema': {
        'type': 'string',
    },
}


# user_mask
DEFAULT_USER_MASK = {
    'type': 'list',
    'minlength': 2,
    'maxlength': 2,
    'schema': {
        'type': ['integer', 'float'],
    },
    'is_lohi': True,
}


# power_options
DEFAULT_POWER_OPTIONS = {
    'type': {
        'type': 'string',
        'allowed': ['simple', 'broken'],
        'default': 'simple',
    },
}


# losvd_options
DEFAULT_LOSVD_OPTIONS = {
    'library': {
        'type': 'string',
        'allowed': list(consts.LOSVD_LIBRARIES.keys()),
        'default': 'IndoUS'
    },
    'vel_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'val': {
                'type': ['integer', 'float'],
                'min': 0,
                'default': 0.0
            },
        },
    },
    'disp_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'val': {
                'type': ['integer', 'float'],
                'min': 1.e-3,
                'coerce': 'nonzero',
                'default': 100.0,
            },
        },
    },
}


# poly_options
DEFAULT_POLY_OPTIONS = {
    'apoly': 'poly_dict',
    'mpoly': 'poly_dict',
}


# host_options
DEFAULT_HOST_OPTIONS  = {
    'age' : {
        'type': 'list',
        'default': [0.1,1.0,10.0],
        'schema': {
            'type': ['integer', 'float'],
            'allowed': [0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0],
        }
    },
    'vel_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'val': {
                'type': ['integer', 'float'],
                'min': 0,
                'default': 0.0
            },
        },
    },
    'disp_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'val': {
                'type': ['integer', 'float'],
                'min': 1.e-3,
                'coerce': 'nonzero',
                'default': 100.0,
            },
        },
    },
}


DEFAULT_OPT_FEII_VC04_OPTIONS = {
    'opt_template' : {
        'type': 'dict',
        'default': {},
        'schema': {
            'type': {
                'type': 'string',
                'allowed': ['VC04'],
                'default': 'VC04',
            },
        },
    },
    'opt_amp_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'br_opt_feii_val' : 'intfl_min0_d1',
            'na_opt_feii_val' : 'intfl_min0_d1',
        },
    },
    'opt_disp_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'br_opt_feii_val' : {
                'type': ['integer', 'float'],
                'min_ex': 0.0,
                'default': 3000.0,
            },
            'na_opt_feii_val' : {
                'type': ['integer', 'float'],
                'min_ex': 0.0,
                'default': 500.0,
            },
        },
    },
    'opt_voff_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'br_opt_feii_val' : 'intfl_d0',
            'na_opt_feii_val' : 'intfl_d0',
        },
    },
}


DEFAULT_OPT_FEII_K10_OPTIONS = {
    'opt_template' : {
        'type': 'dict',
        'default': {},
        'schema': {
            'type': {
                'type': 'string',
                'allowed': ['K10'],
                'default': 'K10',
            },
        },
    },
    'opt_amp_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'f_feii_val' : 'intfl_min0_d1',
            's_feii_val' : 'intfl_min0_d1',
            'g_feii_val' : 'intfl_min0_d1',
            'z_feii_val' : 'intfl_min0_d1',
        },
    },
    'opt_disp_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'opt_feii_val' : {
                'type': ['integer', 'float'],
                'min_ex': 0.0,
                'default': 1500.0,
            },
        },
    },
    'opt_voff_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'opt_feii_val' : 'intfl_d0',
        },
    },
    'opt_temp_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'opt_feii_val' : {
                'type': ['integer', 'float'],
                'min': 0.0,
                'default': 10000.0,
            },
        },
    },
}


DEFAULT_UV_IRON_OPTIONS = {
    'uv_amp_const' : {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'uv_iron_val': 'intfl_min0_d1',
        },
    },
    'uv_disp_const' : {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'uv_iron_val': {
                'type': ['integer', 'float'],
                'min_ex': 0.0,
                'default': 3000.0,
            },
        },
    },
    'uv_voff_const' : {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'uv_iron_val': 'intfl_d0',
        },
    },
}


DEFAULT_BALMER_OPTIONS = {
    # ratio between balmer continuum and higher-order balmer lines
    'R_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'R_val': {
                'type': ['integer', 'float'],
                'min_ex': 0.0,
                'default': 0.5,
            },
        },
    },
    # amplitude of overall balmer model (continuum + higher-order lines)
    'balmer_amp_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'balmer_amp_val': 'intfl_min0_d1',
        },
    },
    # broadening of higher-order Balmer lines
    'balmer_disp_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'balmer_disp_val': {
                'type': ['integer', 'float'],
                'min_ex': 0.0,
                'default': 5000.0,
            },
        },
    },
    # velocity offset of higher-order Balmer lines
    'balmer_voff_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'balmer_voff_val': 'intfl_d0',
        },
    },
    # effective temperature
    'Teff_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'Teff_val': {
                'type': ['integer', 'float'],
                'min_ex': 0.0,
                'default': 15000.0,
            },
        },
    },
    # optical depth
    'tau_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'tau_val': {
                'type': ['integer', 'float'],
                'min': 0.0,
                'max': 1.0,
                'default': 1.0,
            },
        },
    },
}


DEFAULT_PLOT_OPTIONS = {
    # Plot MCMC histograms and chains for each parameter
    'plot_param_hist': 'bool_true',
    # make interactive plotly HTML best-fit plot
    'plot_HTML': 'bool_false',
    # Plot PCA reconstructed spectrum
    'plot_pca': 'bool_false',
    # Plot corner (parameter covariance) plot
    'plot_corner': 'bool_false',
    'corner_options': {
        'type': 'dict',
        'default': {},
        'schema': {
            'pars': {
                'type': 'list',
                'default': [],
            },
            'labels': {
                'type': 'list',
                'default': [],
            },
        },
    },
}


DEFAULT_OUTPUT_OPTIONS = {
    # Write MCMC chains for all paramters, fluxes, and
    # luminosities to a FITS table We set this to false 
    # because MCMC_chains.FITS file can become very large, 
    # especially  if you are running multiple objects.  
    # You only need this if you want to reconstruct chains 
    # and histograms. 
    'write_chain': 'bool_false',
    'write_options': 'bool_false',
    'res_correct': 'bool_true', # Correct final emission line widths for the intrinsic resolution of the spectrum
    'verbose': 'bool_true',
}


# Combine all options into one schema

dict_options = {
    'io_options': DEFAULT_IO_OPTIONS,
    'fit_options': DEFAULT_FIT_OPTIONS,
    'test_options': DEFAULT_TEST_OPTIONS,
    'ifu_options': DEFAULT_IFU_OPTIONS,
    'mcmc_options': DEFAULT_MCMC_OPTIONS,
    'comp_options': DEFAULT_COMP_OPTIONS,
    'narrow_options': DEFAULT_NARROW_OPTIONS,
    'broad_options': DEFAULT_BROAD_OPTIONS,
    'absorp_options': DEFAULT_ABSORP_OPTIONS,
    'pca_options': DEFAULT_PCA_OPTIONS,
    'power_options': DEFAULT_POWER_OPTIONS,
    'losvd_options': DEFAULT_LOSVD_OPTIONS,
    'poly_options': DEFAULT_POLY_OPTIONS,
    'host_options': DEFAULT_HOST_OPTIONS,
    'uv_iron_options': DEFAULT_UV_IRON_OPTIONS,
    'balmer_options': DEFAULT_BALMER_OPTIONS,
    'plot_options': DEFAULT_PLOT_OPTIONS,
    'output_options': DEFAULT_OUTPUT_OPTIONS,
}

list_options = {
    'user_constraints': DEFAULT_USER_CONSTRAINTS,
    'user_mask': DEFAULT_USER_MASK,
}


DEFAULT_OPTIONS_SCHEMA = {}
for option_name, schema in dict_options.items():
    DEFAULT_OPTIONS_SCHEMA[option_name] = {
        'type': 'dict',
        'default': {},
        'schema': schema,
    }

for option_name, schema in list_options.items():
    DEFAULT_OPTIONS_SCHEMA[option_name] = {
        'type': 'list',
        'default': [],
        'schema': schema,
    }


# Some options need special treatment

# TODO: more validation on user line dict
DEFAULT_OPTIONS_SCHEMA['user_lines'] = {
    'type': 'dict',
    'default': {},
    'keysrules': {'type': 'string'},
    'valuesrules': {'type': 'dict',},
}

DEFAULT_OPTIONS_SCHEMA['combined_lines'] = {
    'type': 'dict',
    'default': {},
    'keysrules': {'type': 'string'},
    'valuesrules': {
        'type': 'list',
        'schema': {'type': 'string'},
    },
}

DEFAULT_OPTIONS_SCHEMA['opt_feii_options'] = {
    'oneof_schema': [DEFAULT_OPT_FEII_VC04_OPTIONS, DEFAULT_OPT_FEII_K10_OPTIONS],
    'default': DefaultValidator().normalized({}, DEFAULT_OPT_FEII_VC04_OPTIONS),
}

