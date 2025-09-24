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
import importlib
import multiprocessing as mp
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

BADASS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(BADASS_DIR))
sys.path.insert(0,str(BADASS_DIR.joinpath('badass_utils'))) # utility functions
sys.path.insert(0,str(BADASS_DIR.joinpath('badass_tools'))) # tool functions

import badass_test_suite  as badass_test_suite
import badass_tools as badass_tools

from utils.options import BadassOptions
from input.input import BadassInput
import utils.utils as ba_utils
from templates.common import initialize_templates
import utils.plotting as plotting
from line_utils.line_lists.optical_qso import optical_qso_default
from line_utils.line_profiles import line_constructor


__author__     = "Remington O. Sexton (USNO), Sara M. Doan (GMU), Michael A. Reefe (GMU), William Matzko (GMU), Nicholas Darden (UCR)"
__copyright__  = "Copyright (c) 2023 Remington Oliver Sexton"
__credits__    = ["Remington O. Sexton (GMU/USNO)", "Sara M. Doan (GMU)", "Michael A. Reefe (GMU)", "William Matzko (GMU)", "Nicholas Darden (UCR)"]
__license__    = "MIT"
__version__    = "10.2.0"
__maintainer__ = "Remington O. Sexton"
__email__      = "remington.o.sexton.civ@us.navy.mil"
__status__     = "Release"


# TODO: create BadassContext class that contains a target, options, parameters, etc. and the following relevant
#       functions are instance functions
# TODO: all print statements to logger
# TODO: all 'if verbose' checks to logger
# TODO: all init/plim values to config file
# TODO: ability to resume from line test and ml results
# TODO: ability to resume mid run (save status at certain checkpoints?)
# TODO: ability to multiprocess mcmc runs?
# TODO: line type classes? or just a general line class?
# TODO: remove any whitespace at ends of lines
# TODO: use rng seed to be able to reproduce fits

class FitStage:
    INIT = 1
    BOOTSTRAP = 2
    MCMC = 3


def run_BADASS(inputs, **kwargs):
    # utils.options.BadassOptions.get_options_dep(kwargs)
    opts = BadassOptions.get_options(kwargs['options_file'])
    targets = BadassInput.get_inputs(inputs, opts)

    # TODO: multiprocess
    for target in targets:
        BadassRunContext(target).run()


# TODO: move logger to here instead of target
class BadassRunContext:

    def __init__(self, target):
        self.target = target
        self.options = target.options
        self.verbose = self.options.output_options.verbose
        self.fit_stage = FitStage.INIT

        # The spectral data currently being fit
        self.fit_wave = self.target.wave.copy()
        self.fit_spec = self.target.spec.copy()
        self.fit_noise = self.target.noise.copy()

        self.start_time = None
        self.force_thresh = np.inf

        self.templates = None
        self.param_dict = {}
        self.prior_params = []
        self.cur_params = {} # contains the parameter values of the current fit
        self.line_list = []
        self.combined_line_list = []
        self.soft_cons = []
        self.blob_pars = {}

        self.mc_attr_store = {}
        self.fit_results = {}
        self.model = None
        self.comp_dict = {}

        self.chain_df = None
        self.mcmc_results_dict = {}
        self.mcmc_result_chains = {}


    def run(self):
        plotting.create_input_plot(self)
        self.initialize_fit()

        # Line testing is performed first for a better line list determination and number of components
        if not self.run_tests():
            return

        self.fit_stage = FitStage.BOOTSTRAP
        if not self.max_likelihood():
            return

        self.fit_stage = FitStage.MCMC
        self.run_emcee()


    def initialize_fit(self):
        if self.options.io_options.dust_cache != None:
            IrsaDust.cache_location = str(dust_cache)

        # Check to make sure plotly is installed for HTML interactive plots:
        if (self.options.plot_options.plot_HTML) and (not importlib.util.find_spec('plotly')):
            self.options.plot_options.plot_HTML = False

        print('\n > Starting fit for %s' % self.target.infile.parent.name)
        self.target.log.log_target_info()

        sys.stdout.flush()
        # Start a timer to record the total runtime
        self.start_time = time.time()

        self.target.log.log_fit_information()
        # TODO: the templates ctx is BadassRunContext
        self.templates = initialize_templates(self.target)

        # TODO: input from past line test or user config
        # Set force_thresh to np.inf. This will get overridden if the user does the line test
        self.force_thresh = badass_test_suite.root_mean_squared_error(self.target.spec, np.full_like(self.target.spec,np.nanmedian(self.target.spec)))
        if not np.isfinite(self.force_thresh):
            self.force_thresh = np.inf

        # Initialize free parameters (all components, lines, etc.)
        self.target.log.info('\n Initializing parameters...')
        self.target.log.info('----------------------------------------------------------------------------------------------------')

        # TODO: don't need to do this before line/config testing
        self.initialize_pars()

        # Output all free parameters of fit prior to fitting (useful for diagnostics)
        if self.options.fit_options.output_pars or self.verbose:
            self.target.log.output_free_pars(self.line_list, self.param_dict, self.soft_cons)
        self.target.log.output_line_list(self.line_list, self.soft_cons)

        self.set_blob_pars()
        self.target.log.output_options()


    def initialize_pars(self, user_lines=None):
        """
        Initializes all free parameters for the fit based on user input and options.
        """

        # Initial conditions for some parameters
        max_flux = np.nanmax(self.fit_spec)*1.5
        median_flux = np.nanmedian(self.fit_spec)

        # Padding on the edges; any line(s) within this many angstroms is omitted
        # from the fit so problems do not occur with the fit
        edge_pad = 10.0
        par_input = {} # initialize an empty dictionary to store free parameter dicts
        template_args = {'median_flux':median_flux, 'max_flux':max_flux}

        for template in self.templates.values():
            template.initialize_parameters(par_input, template_args)

        # TODO: config file
        # TODO: separate classes?
        if self.options.comp_options.fit_poly:
            if (self.options.poly_options.apoly.bool) and (self.options.poly_options.apoly.order >= 0):
                if self.verbose:
                    print('\t- Fitting additive legendre polynomial component')
                for n in range(1, int(self.options.poly_options.apoly.order)+1):
                    par_input['APOLY_COEFF_%d' % n] = {'init':0.0, 'plim':(-1.0e2,1.0e2),}
            if (self.options.poly_options.mpoly.bool) and (self.options.poly_options.mpoly.order >= 0):
                if self.verbose:
                    print('\t- Fitting multiplicative legendre polynomial component')
                for n in range(1, int(self.options.poly_options.mpoly.order)+1):
                    par_input['MPOLY_COEFF_%d' % n] = {'init':0.0,'plim':(-1.0e2,1.0e2),}

        # TODO: config file
        if self.options.comp_options.fit_power:
            #### Simple Power-Law (AGN continuum)
            if self.options.power_options.type == 'simple':
                if self.verbose:
                    print('\t- Fitting Simple AGN power-law continuum')
                # AGN simple power-law amplitude
                par_input['POWER_AMP'] = {'init':(0.5*median_flux), 'plim':(0,max_flux),}
                # AGN simple power-law slope
                par_input['POWER_SLOPE'] = {'init':-1.0, 'plim':(-6.0,6.0),}
                
            #### Smoothly-Broken Power-Law (AGN continuum)
            if self.options.power_options.type == 'broken':
                if self.verbose:
                    print('\t- Fitting Smoothly-Broken AGN power-law continuum.')
                # AGN simple power-law amplitude
                par_input['POWER_AMP'] = {'init':(0.5*median_flux), 'plim':(0,max_flux),}
                # AGN simple power-law break wavelength
                par_input['POWER_BREAK'] = {'init':(np.max(self.target.wave) - (0.5*(np.max(self.target.wave)-np.min(self.target.wave)))),
                                            'plim':(np.min(self.target.wave), np.max(self.target.wave)),}
                # AGN simple power-law slope 1 (blue side)
                par_input['POWER_SLOPE_1'] = {'init':-1.0, 'plim':(-6.0,6.0),}
                # AGN simple power-law slope 2 (red side)
                par_input['POWER_SLOPE_2'] = {'init':-1.0, 'plim':(-6.0,6.0),}
                # Power-law curvature parameter (Delta)
                par_input['POWER_CURVATURE'] = {'init':0.10, 'plim':(0.01,1.0),}


        # Emission Lines
        self.line_list = user_lines if user_lines else self.options.user_lines if self.options.user_lines else optical_qso_default()
        self.add_line_comps()

        # Add the FWHM resolution and central pixel locations for each line so we don't have to find them during the fit.
        self.add_disp_res()

        # Generate line free parameters based on input line_list
        line_par_input = self.initialize_line_pars()

        param_keys = list(par_input.keys()) + list(line_par_input.keys())
        # Check hard line constraints
        self.check_hard_cons(param_keys)

        # TODO: way to not have to run this twice?
        # Re-Generate line free parameters based on revised line_list
        line_par_input = self.initialize_line_pars()

        # Append line_par_input to par_input
        self.param_dict = {**par_input, **line_par_input}

        # The default line list is automatically generated from lines with multiple components.
        # User can provide a combined line list, which can override the default.
        self.generate_comb_line_list()

        # Check soft-constraints
        # Default soft constraints
        # Soft constraints: If you want to vary a free parameter relative to another free parameter (such as
        # requiring that broad lines have larger widths than narrow lines), these are called "soft" constraints,
        # or "inequality" constraints. 
        # These are passed through a separate list of tuples which are used by the maximum likelihood constraints 
        # and prior constraints by emcee.  Soft constraints have a very specific format following
        # the scipy optimize SLSQP syntax: 
        #
        #               (parameter1 - parameter2) >= 0.0 OR (parameter1 >= parameter2)
        #
        self.check_soft_cons()


    def add_line_comps(self):
        """
        Checks each entry in the complete (narrow, broad, absorption, and user) line list
        and ensures all necessary keywords are input. It also checks every line entry against the 
        front-end component options (comp_options). The only required keyword for a line entry is 
        the "center" wavelength of the line. If "amp", "disp", "voff", "h3" and "h4" (for Gauss-Hermite)
        line profiles are missing, it assumes these are all "free" parameters in the fitting of that line. 
        If "line_type" is not defined, it is assumed to be "na" (narrow).  If "line_profile" is not defined, 
        it is assumed to be "gaussian". 

        Line list hyper-parameters:
        amp, amp_init, amp_plim
        disp, disp_init, disp_plim
        voff, voff_init, voff_plim
        shape, shape_init, shape_plim,
        h3, h4, h5, h6, h7, h8, h9, h10, _init, _plim
        line_type
        line_profile
        ncomp
        parent
        label
        """

        comp_options = self.options.comp_options
        verbose = self.options.output_options.verbose
        edge_pad = 10 # TODO: config

        # TODO: better handle types + 'user'
        line_types = {
            'na': 'narrow',
            'br': 'broad',
            'abs': 'absorp',
        }
        line_profiles = ['gaussian', 'lorentzian', 'gauss-hermite', 'voigt', 'laplace', 'uniform']
        line_attrs = ['amp', 'disp', 'voff']

        for line_name, line_dict in self.line_list.items():

            if 'line_type' not in line_dict:
                line_dict['line_type'] = 'user'

            line_type_s = line_dict['line_type'] # short name
            if (not line_type_s == 'user') and (not line_type_s in line_types):
                print('Unsupported line type: %s' % line_type_s)
                continue

            line_type = line_types[line_type_s] if line_type_s in line_types else 'user'
            is_user = line_type == 'user'
            type_options = self.options[line_type+'_options']

            # TODO: center is unit-configurable
            if ('center' not in line_dict) or (not isinstance(line_dict['center'],(int,float))):
                # TODO: just log and continue?
                raise ValueError('\n Line list entry requires at least \'center\' wavelength (in Angstroms) to be defined as an int or float type\n')

            # TODO: should use fitting region?
            # Check line in wavelength region
            if (line_dict['center'] <= self.target.wave[0]+edge_pad) or (line_dict['center'] >= self.target.wave[-1]-edge_pad):
                continue

            # Check if we are fitting lines of this type
            if not comp_options['fit_'+line_type]:
                continue

            if is_user:
                if 'line_profile' not in line_dict:
                    line_dict['line_profile'] = 'gaussian' # TODO: default in config
            else:
                line_dict['line_profile'] = type_options['line_profile']

            line_profile = line_dict['line_profile']
            if line_profile not in line_profiles:
                print('Unsupported line profile: %s' % line_profile)

            for attr in line_attrs:
                if attr not in line_dict:
                    line_dict[attr] = 'free'

            # TODO: something else for these specialized line profiles?
            if (not is_user) and (line_profile == 'gauss-hermite'):
                # Gauss-Hermite higher-order moments for each narrow, broad, and absorp
                if type_options['n_moments'] > 2:
                    for m in range(3, 3+(type_options['n_moments']-2)):
                        attr = 'h%d'%m
                        if attr not in line_dict:
                            line_dict[attr] = 'free'

                # If the line profile is Gauss-Hermite, but the number of higher-order moments is 
                # less than or equal to 2 (for which the line profile is just Gaussian), remove any 
                # unnecessary higher-order line parameters that may be in the line dictionary.
                for m in range(type_options['n_moments']+1, 11):
                    attr = 'h%d'%m
                    line_dict.pop(attr, None)
                    line_dict.pop(attr+'_init', None)
                    line_dict.pop(attr+'_plim', None)


            # Higher-order moments for laplace and uniform (h3 and h4) only for each narrow, broad, and absorp.
            if line_profile in ['laplace','uniform']:
                if 'h3' not in line_dict:
                    line_dict['h3'] = 'free'
                if 'h4' not in line_dict:
                    line_dict['h4'] = 'free'

            if line_profile not in ['gauss-hermite', 'laplace', 'uniform']:
                for m in range(3,11,1):
                    attr = 'h%d'%m
                    line_dict.pop(attr, None)
                    line_dict.pop(attr+'_init', None)
                    line_dict.pop(attr+'_plim', None)


            if line_profile == 'voigt':
                if 'shape' not in line_dict:
                    line_dict['shape'] = 'free'
            else:
                line_dict.pop('shape', None)
                line_dict.pop('shape_init', None)
                line_dict.pop('shape_plim', None)

            # line widths (narrow, broad, and absorption disp) are tied, respectively.
            if comp_options.tie_line_disp:
                for m in range(3, 3+type_options['n_moments']-2):
                    line_dict.pop('h%d'%m, None)
                line_dict.pop('shape', None)

                type_prefix = line_type_s.upper()
                if line_type == 'user': type_prefix = 'NA' # default to narrow

                line_dict['disp'] = type_prefix + '_DISP'
                if line_profile == 'gauss-hermite':
                    for m in range(3, 3+type_options['n_moments']-2):
                        line_dict['h%d'%m] = type_prefix + '_H%d'%m
                elif line_profile == 'voigt':
                    line_dict['shape'] = type_prefix + '_SHAPE'
                elif line_profile in ['laplace', 'uniform']:
                    line_dict['h3'] = type_prefix + '_H3'
                    line_dict['h4'] = type_prefix + '_H4'

            # line velocity offsets (narrow, broad, and absorption voff) are tied, respectively.
            if comp_options.tie_line_voff:
                type_prefix = line_type_s.upper()
                if line_type == 'user': type_prefix = 'NA' # default to narrow
                line_dict['voff'] = type_prefix + '_VOFF'

            if ('ncomp' not in line_dict) or (line_dict['ncomp'] <= 0):
                line_dict['ncomp'] = 1

            # Check parent keyword if exists against line list; will be used for generating combined lines
            if ('parent' in line_dict) and (line_dict['parent'] not in self.line_list):
                line_dict.pop('parent', None)


        valid_keys = ['center','center_pix','disp_res_kms','disp_res_ang','amp','disp','voff','shape','line_type',
                      'line_profile','amp_init','amp_plim','disp_init','disp_plim','voff_init','voff_plim',
                      'shape_init','shape_plim','amp_prior','disp_prior','voff_prior','shape_prior',
                      'label','ncomp','parent']

        for line_type in line_types.values():
            type_options = self.target.options[line_type+'_options']
            for suffix in ['', '_init', '_plim', '_prior']:
                valid_keys.extend(['h%d%s'%(m,suffix) for m in range(3, 3+type_options['n_moments']-2)])

        for line_dict in self.line_list.values():
            for key in line_dict.keys():
                if key not in valid_keys:
                    raise ValueError('\n %s not a valid keyword for the line list!\n'%key)


    def add_disp_res(self):
        # Perform linear interpolation on the disp_res array as a function of wavelength 
        # We will use this to determine the dispersion resolution as a function of wavelenth for each 
        # emission line so we can correct for the resolution at every iteration.
        disp_res_ftn = interp1d(self.target.wave,self.target.disp_res,kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))
        # Interpolation function that maps x (in angstroms) to pixels so we can get the exact
        # location in pixel space of the emission line.
        x_pix = np.array(range(len(self.target.wave)))
        pix_interp_ftn = interp1d(self.target.wave,x_pix,kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))

        # iterate through the line_list and add the keywords
        for line_dict in self.line_list.values():
            center = line_dict['center'] # line center in Angstroms
            line_dict['center_pix'] = float(pix_interp_ftn(center)) # line center in pixels
            disp_res_ang = float(disp_res_ftn(center)) # instrumental FWHM resolution in angstroms
            line_dict['disp_res_ang'] = disp_res_ang
            c = const.c.to('km/s').value
            line_dict['disp_res_kms'] = (disp_res_ang/center)*c # instrumental FWHM resolution in km/s


    def initialize_line_pars(self):
        """
        This function initializes the initial guess, parameter limits (lower and upper), and 
        priors if not explicily defined by the user in the line list for each line.

        Special care is taken with tring to determine the location of the particular line
        in terms of velocity.
        """

        c = const.c.to('km/s').value

        # TODO: config file
        # type_name, disp_init, disp_plim, h_init, h_plim, shape_init, shape_plim
        line_types = {
            'na': ('narrow', 250.0, (0.0,1200.0), 0.0, (-0.5,0.5), 0.0, (0.0,1.0),),
            'br': ('broad', 2500.0, (500.0,15000.0), 0.0, (-0.5,0.5), 0.0, (0.0,1.0),),
            'abs': ('absorp', 450.0, (0.1,2500.0), 0.0, (-0.5,0.5), 0.0, (0.0,1.0),),
            'out': ('outflow', 100.0, (0.0,800.0), 0.0, (-0.5,0.5), 0.0, (0.0,1.0),),
        }

        # First we remove the continuum 
        galaxy_csub = badass_tools.continuum_subtract(self.target.wave,self.target.spec,self.target.noise,sigma_clip=2.0,clip_iter=25,filter_size=[25,50,100,150,200,250,500],
                       noise_scale=1.0,opt_rchi2=True,plot=False,
                       fig_scale=8,fontsize=16,verbose=False)

        try:
            # normalize by noise
            norm_csub = galaxy_csub/self.target.noise

            peaks,_ = signal.find_peaks(norm_csub, height=2.0, width=3.0, prominence=1)
            troughs,_ = signal.find_peaks(-norm_csub, height=2.0, width=3.0, prominence=1)
            peak_wave = self.target.wave[peaks]
            trough_wave = self.target.wave[troughs]
        except:
            if self.verbose:
                print('\n Warning! Peak finding algorithm used for initial guesses of amplitude and velocity failed! Defaulting to user-defined locations...')
            peak_wave = np.array([line_dict['center'] for line_dict in self.line_list.values() if line_dict['line_type'] in ['na','br']])
            trough_wave = np.array([line_dict['center'] for line_dict in self.line_list.values() if line_dict['line_type'] in ['abs']])
            if len(peak_wave) == 0:
                peak_wave = np.array([0])
            if len(trough_wave) == 0:
                trough_wave = np.array([0])


        def amp_hyperpars(line_type, line_center, voff_init, voff_plim, amp_factor):
            """
            Assigns the user-defined or default line amplitude initial guesses and limits.
            """
            line_center = float(line_center)

            line_types = {
                'na': ('narrow', peak_wave, 1),
                'br': ('broad', peak_wave, 1),
                'abs': ('absorp', trough_wave, -1),
            }

            type_options = self.options[line_types[line_type][0]+'_options']

            if (line_type in line_types) and type_options['amp_plim']:
                min_amp, max_amp = np.abs(np.min(type_options['amp_plim'])), np.abs(np.max(type_options['amp_plim']))
            else:
                min_amp, max_amp = 0.0, 2*np.nanmax(self.target.spec)

            mf = line_types[line_type][2] # multiplicative factor (1 or -1) to handle troughs
            feature_wave = line_types[line_type][1]

            # calculate velocities of features around line center
            feature_center = feature_wave[np.argmin(np.abs(feature_wave-line_center))] # feature in angstroms
            feature_vel = (feature_center-line_center)/line_center*c # feature in velocity offset

            center = line_center
            # if velocity less than search_kms, calculate amplitude at that point
            if (feature_vel >= voff_plim[0]) and (feature_vel <= voff_plim[1]):
                center = feature_center

            init_amp = self.target.spec[ba_utils.find_nearest(self.target.wave,center)[1]]
            if (init_amp >= min_amp) and (init_amp <= max_amp):
                return mf*init_amp/amp_factor, (min(mf*min_amp,mf*max_amp), max(mf*min_amp,mf*max_amp))
            return mf*max_amp-mf*(max_amp-min_amp)/2.0/amp_factor, (min(mf*min_amp,mf*max_amp), max(mf*min_amp,mf*max_amp))


        def disp_hyperpars(line_type,line_center,line_profile): # FWHM hyperparameters
            """
            Assigns the user-defined or default line width (dispersion)
            initial guesses and limits.
            """

            # TODO: in config file
            line_types = {
                'na': ('narrow', 50.0, (0.001,300.0)),
                'br': ('broad', 500.0, (300.0,3000.0)),
                'abs': ('absorp', 50.0, (0.001,300.0))
            }

            default_init = line_types[line_type][1]
            type_options = self.options[line_types[line_type][0] + '_options']

            min_disp, max_disp = type_options['disp_plim'] if type_options['disp_plim'] else line_types[line_type][2]

            if (default_init >= min_disp) and (default_init <= max_disp):
                return default_init, (min_disp, max_disp)
            return max_disp-(max_disp-min_disp)/2.0, (min_disp, max_disp)


        def voff_hyperpars(line_type, line_center):
            """
            Assigns the user-defined or default line velocity offset (voff)
            initial guesses and limits.
            """

            voff_default_init = 0.0

            # TODO: in config file
            line_types = {
                'na': ('narrow', (-500,500), peak_wave),
                'br': ('broad', (-1000,1000), peak_wave),
                'abs': ('absorp', (-500,500), trough_wave)
            }

            if line_type not in line_types:
                min_voff, max_voff = line_types['na'][1] # default narrow
                if (min_voff <= voff_default_init) and (max_voff >= voff_default_init):
                    return voff_default_init, (min_voff, max_voff)
                return max_voff - ((max_voff-min_voff)/2.0), (min_voff, max_voff)

            type_options = self.options[line_types[line_type][0] + '_options']
            min_voff, max_voff = line_types[line_type][1]
            if type_options['voff_plim']:
                min_voff, max_voff = type_options['voff_plim']

            # calculate velocities of features around line center
            feature_wave = line_types[line_type][2]
            feature_ang = feature_wave[np.argmin(np.abs(feature_wave-line_center))] # feature in angstroms
            feature_vel = (feature_ang-line_center)/line_center*c # feature in velocity offset
            if (feature_vel >= min_voff) and (feature_vel <= max_voff):
                return feature_vel, (min_voff, max_voff)
            else:
                return voff_default_init, (min_voff, max_voff)


        def h_moment_hyperpars():
            # Higher-order moments for Gauss-Hermite line profiles
            # extends to Laplace and Uniform kernels
            # TODO: config file
            h_init = 0.0
            h_lim  = (-0.5,0.5)
            return h_init, h_lim


        def shape_hyperpars(): # shape of the Voigt profile; if line_profile="voigt"
            # TODO: config file
            shape_init = 0.0
            shape_lim = (0.0,1.0)
            return shape_init, shape_lim    


        line_par_input = {}

        # We start with standard lines and options. These are added one-by-one. Then we check specific line options and then override any lines that have
        # been already added. Params are added regardless of component options as long as the parameter is set to "free"
        for line_name, line_dict in self.line_list.items():

            line_type = line_dict['line_type']
            line_center = line_dict['center']

            # Velocity offsets determine both the intial guess in line velocity as well as amplitude, so it makes sense to perform the voff for each line first.
            if (('voff' in line_dict) and (line_dict['voff'] == 'free')):
                voff_default_init, voff_default_plim = voff_hyperpars(line_type, line_center)
                line_par_input[line_name+'_VOFF'] = {'init': line_dict.get('voff_init', voff_default_init), 
                                                     'plim':line_dict.get('voff_plim', voff_default_plim),
                                                     'prior':line_dict.get('voff_prior', {'type':'gaussian'}),
                                                    }
                if line_par_input[line_name+'_VOFF']['prior'] is None: line_par_input[line_name+'_VOFF'].pop('prior',None)

                # Check to make sure init value is within limits of plim
                if (line_par_input[line_name+'_VOFF']['init'] < line_par_input[line_name+'_VOFF']['plim'][0]) or (line_par_input[line_name+'_VOFF']['init'] > line_par_input[line_name+'_VOFF']['plim'][1]):
                    raise ValueError('\n Velocity offset (voff) initial value (voff_init) for %s outside of parameter limits (voff_plim)!\n' % (line_name))


            if (('amp' in line_dict) and (line_dict['amp'] == 'free')):
                # If amplitude parameter limits are already set in (narrow,broad,absorp)_options, then use those, otherwise, automatically generate them
                amp_factor = 1
                if 'ncomp' in line_dict:
                    # Get number of components that are in the line list for this line
                    total_ncomp = [1]
                    # If line is a parent line
                    if line_dict['ncomp'] == 1:
                        for ld in self.line_list.values():
                            if ('parent' in ld) and (ld['parent'] == line_name):
                                total_ncomp.append(ld['ncomp'])

                    # if line is a child line
                    if 'parent' in line_dict:
                        # Look in the line list for any other lines that have the same parent and append them
                        for ld in self.line_list.values():
                            if ('parent' in ld) and (ld['parent'] == line_dict['parent']):
                                total_ncomp.append(ld['ncomp'])

                    amp_factor = np.max(total_ncomp)

                # Amplitude is dependent on velocity offset from expected location, which we determined above.  If the amplitude is free but voff 
                # is tied to another line, we must extract whatever tied voff is
                # TODO: config file
                voff_init = 0.0
                voff_plim = (-500,500)
                if ('voff' in line_dict) and (line_dict['voff'] == 'free'):
                    voff_init = line_par_input[line_name+'_VOFF']['init']
                    voff_plim = line_par_input[line_name+'_VOFF']['plim']

                amp_default_init, amp_default_plim = amp_hyperpars(line_type, line_center, voff_init, voff_plim, amp_factor)
                line_par_input[line_name+'_AMP'] = {'init': line_dict.get('amp_init', amp_default_init), 
                                                    'plim': line_dict.get('amp_plim', amp_default_plim),
                                                    'prior': line_dict.get('amp_prior'),
                                                   }
                if line_par_input[line_name+'_AMP']['prior'] is None: line_par_input[line_name+'_AMP'].pop('prior',None)

                # Check to make sure init value is within limits of plim
                if (line_par_input[line_name+'_AMP']['init'] < line_par_input[line_name+'_AMP']['plim'][0]) or (line_par_input[line_name+'_AMP']['init'] > line_par_input[line_name+'_AMP']['plim'][1]):
                    raise ValueError('\n Amplitude (amp) initial value (amp_init) for %s outside of parameter limits (amp_plim)!\n' % (line_name))


            if (('disp' in line_dict) and (line_dict['disp'] == 'free')):
                disp_default_init, disp_default_plim = disp_hyperpars(line_type, line_center, line_dict['line_profile'])
                line_par_input[line_name+'_DISP'] = {'init': line_dict.get('disp_init', disp_default_init), 
                                                     'plim': line_dict.get('disp_plim', disp_default_plim),
                                                     'prior':line_dict.get('disp_prior')
                                                    }
                if line_par_input[line_name+'_DISP']['prior'] is None: line_par_input[line_name+'_DISP'].pop('prior',None)

                # Check to make sure init value is within limits of plim
                if (line_par_input[line_name+'_DISP']['init'] < line_par_input[line_name+'_DISP']['plim'][0]) or (line_par_input[line_name+'_DISP']['init'] > line_par_input[line_name+'_DISP']['plim'][1]):
                    raise ValueError('\n DISP (disp) initial value (disp_init) for %s outside of parameter limits (disp_plim)!\n' % (line_name))


            if (line_dict['line_profile'] == 'gauss-hermite'):
                type_options = target.options[line_types[line_type][0] + '_options']
                n_moments = type_options['n_moments']

                # TODO: combine with below
                h_default_init, h_default_plim = h_moment_hyperpars()
                for m in range(3,3+(n_moments-2)):
                    attr = 'h%d'%m
                    par_attr = '%s_H%d'%(line_name,m)
                    if (attr in line_dict) and (line_dict[attr] == 'free'):
                        line_par_input[par_attr] = {'init': line_dict.get(attr+'_init', h_default_init),
                                                    'plim': line_dict.get(attr+'_plim', h_default_plim),
                                                    'prior': line_dict.get(attr+'_prior', {'type':'gaussian'})
                                                   }
                    if line_par_input[par_attr]['prior'] is None: line_par_input[par_attr].pop('prior',None)

                    # Check to make sure init value is within limits of plim
                    if (line_par_input[par_attr]['init'] < line_par_input[par_attr]['plim'][0]) or (line_par_input[par_attr]["init"] > line_par_input[par_attr]['plim'][1]):
                        raise ValueError('\n Gauss-Hermite moment h%d initial value (h%d_init) for %s outside of parameter limits (h%d_plim)!\n' % (m,m,line_name,m))


            if line_dict['line_profile'] in ['laplace','uniform']:
                h_default_init, h_default_plim = h_moment_hyperpars()
                for m in range(3,5):
                    attr = 'h%d'%m
                    par_attr = '%s_H%d'%(line_name,m)
                    if (attr in line_dict) and (line_dict[attr] == 'free'):
                        line_par_input[par_attr] = {'init': line_dict.get(attr+'_init', h_default_init),
                                                    'plim': line_dict.get(attr+'_plim', h_default_plim),
                                                    'prior': line_dict.get(attr+'_prior', {'type':'halfnorm'})
                                                   }
                    if line_par_input[par_attr]['prior'] is None: line_par_input[par_attr].pop('prior',None)

                    # Check to make sure init value is within limits of plim
                    if (line_par_input[par_attr]['init'] < line_par_input[par_attr]['plim'][0]) or (line_par_input[par_attr]['init'] > line_par_input[par_attr]['plim'][1]):
                        raise ValueError('\n Laplace or Uniform moment h%d initial value (h%d_init) for %s outside of parameter limits (h%d_plim)!\n' % (m,m,line_name,m))

                # TODO: config file
                # add exceptions for h4 in each line profile; laplace h4>=0, uniform h4<0
                if line_dict['line_profile'] == 'laplace':
                    line_par_input['%s_H3'%line_name]['init'] = 0.01
                    line_par_input['%s_H3'%line_name]['plim'] = (-0.15,0.15)
                    line_par_input['%s_H4'%line_name]['init'] = 0.01
                    line_par_input['%s_H4'%line_name]['plim'] = (0,0.2)

                if line_dict['line_profile'] == 'uniform':
                    line_par_input['%s_H4'%line_name]['init'] = -0.01
                    line_par_input['%s_H4'%line_name]['plim'] = (-0.3,-1e-4)


            if ('shape' in line_dict) and (line_dict['shape'] == 'free'):
                par_attr = '%s_SHAPE'%line_name
                shape_default_init, shape_default_plim = shape_hyperpars()
                line_par_input[par_attr] = {'init': line_dict.get('shape_init', shape_default_init),
                                            'plim': line_dict.get('shape_plim', shape_default_plim),
                                            'prior': line_dict.get('shape_prior')
                                           }
                if line_par_input[par_attr]['prior'] is None: line_par_input[par_attr].pop('prior',None)

                # Check to make sure init value is within limits of plim
                if (line_par_input[par_attr]['init'] < line_par_input[par_attr]['plim'][0]) or (line_par_input[par_attr]['init'] > line_par_input[par_attr]['plim'][1]):
                    raise ValueError('\n Voigt profile shape parameter (shape) initial value (shape_init) for %s outside of parameter limits (shape_plim)!\n' % (line_name))


        # If tie_line_disp, we tie all widths (including any higher order moments) by respective line groups (Na, Br, Out, Abs)
        comp_options = self.options.comp_options
        if comp_options.tie_line_disp:
            for line_type, type_attrs in line_types.items():
                line_profile = comp_options[line_type+'_line_profile']
                if (comp_options['fit_'+type_attrs[0]]) or (line_type in [line_dict['line_type'] for line_dict in self.line_list.values()]):
                    line_par_input[line_type.upper()+'_DISP'] = {'init': type_attrs[1], 'plim': type_attrs[2]}
                if (line_profile == 'gauss-hermite') and (comp_options['n_moments'] > 2):
                    for m in range(3,3+comp_options['n_moments']-2):
                        line_par_input[line_type.upper()+'_H%d'%m] = {'init': type_attrs[3], 'plim': type_attrs[4]}
                if line_profile == 'voigt':
                    line_par_input[line_type.upper()+'_SHAPE'] = {'init': type_attrs[5], 'plim': type_attrs[6]}
                if line_profile in ['laplace', 'uniform']:
                    for m in range(3,5):
                        line_par_input[line_type.upper()+'_H%d'%m] = {'init': type_attrs[3], 'plim': type_attrs[4]}

        # If tie_line_voff, we tie all velocity offsets (including any higher order moments) by respective line groups (Na, Br, Out, Abs)   
        if comp_options.tie_line_voff:
            for line_type, type_attrs in line_types.items():
                if (comp_options['fit_'+type_attrs[0]]) or (line_type in [line_dict['line_type'] for line_dict in self.line_list.values()]):
                    # TODO: config file
                    line_par_input[line_type.upper()+'_VOFF'] = {'init': 0.0, 'plim': (-500.0,500.0), 'prior': {'type': 'gaussian'}}

        return line_par_input


    def check_hard_cons(self, param_keys, remove_lines=False):
        valid_keys = ['amp','disp','voff', 'shape'] + ['h%d'%m for m in range(3,11)]
        param_dict = {par:0 for par in param_keys}
        for line_name, line_dict in self.line_list.items():
            for hpar, value in line_dict.items():
                if (hpar not in valid_keys) or (value == 'free'):
                    continue

                if isinstance(value, (int,float)):
                    line_dict[hpar] = float(value)
                    continue

                # hpar value is an expression, make sure it's valid
                if ne.validate(value, local_dict=param_dict) is not None:
                    if remove_lines:
                        if self.verbose:
                            print('\n WARNING: Hard-constraint %s not found in parameter list or could not be parsed; removing %s line from line list.\n' % (value,line_name))
                        self.line_list.pop(line, None)
                    else:
                        if self.verbose:
                            print('Hard-constraint %s not found in parameter list or could not be parsed; converting to free parameter.\n' % value)
                        line_dict[hpar] = 'free'


    def check_soft_cons(self):
        out_cons = []
        soft_cons = self.options.user_constraints if self.options.user_constraints else []
        expr_dict = {k:v['init'] for k,v in self.param_dict.items()}

        # Check that soft cons can be parsed; if not, convert to free parameter
        for con in soft_cons:
            # validate returns None if successful
            if any([ne.validate(c,local_dict=expr_dict) for c in con]):
                print('\n - %s soft constraint removed because one or more free parameters is not available.' % str(con))
            else:
                out_cons.append(con)

        # Now check to see that initial values are obeyed; if not, throw exception and warning message
        for con in out_cons:
            val1 = ne.evaluate(con[0],local_dict=expr_dict).item()
            val2 = ne.evaluate(con[1],local_dict=expr_dict).item()
            if val1 < val2:
                raise ValueError('\n The initial value for %s is less than the initial value for %s, but the constraint %s says otherwise.  Either remove the constraint or initialize the values appropriately.\n' % (con[0],con[1],con))

        self.soft_cons = out_cons


    def generate_comb_line_list(self):
        """
        Generate a list of 'combined lines' for lines with multiple components, for which 
        velocity moments (integrated velocity and dispersion) and other quantities will 
        be calculated during the fit. This is done automatically for lines that have a valid
        "parent" explicitly defined, for which the parent line is the 1st component
        """
        if len(self.line_list) == 0:
            self.combined_line_list = {}
            return

        combined_line_list = {}

        for line_name, line_dict in self.line_list.items():
            if (line_dict['ncomp'] <= 1) or ('parent' not in line_dict) or (line_dict['parent'] not in self.line_list):
                continue

            parent = line_dict['parent']
            comb_name = '%s_COMB' % parent
            if comb_name not in combined_line_list:
                combined_line_list[comb_name] = {'lines':[parent,]}
            combined_line_list[comb_name]['lines'].append(line_name)
            for attr in ['center', 'center_pix', 'disp_res_kms', 'line_profile']:
                combined_line_list[comb_name][attr] = self.line_list[parent][attr]

        for comb_name, comb_lines in self.options.combined_lines.items():
            # Check to make sure lines are in line list; only add the lines that are valid
            valid_lines = [line for line in comb_lines if line in self.line_list.keys()]
            if len(valid_lines) < 2: # need at least two valid lines to add a combined line
                continue

            combined_line_list[comb_name] = {
                'lines': valid_lines,
                'center': self.line_list[valid_lines[0]]['center'],
                'center_pix': self.line_list[valid_lines[0]]['center_pix'],
                'disp_res_kms': self.line_list[valid_lines[0]]['disp_res_kms'],
            }

        self.combined_line_list = combined_line_list


    def set_blob_pars(self):
        """
        The blob-parameter dictionary is a dictionary for any non-free "blob" parameters for values that need 
        to be calculated during the fit. For MCMC, these equate to non-fitted parameters like fluxes, equivalent widths, 
        or continuum fluxes that aren't explicitly fit as free paramters, but need to be calculated as output /during the fitting process/
        such that full chains can be constructed out of their values (as opposed to calculated after the fitting is over).
        We mainly use blob-pars for the indices of the wavelength vector at which to calculate continuum luminosities, so we don't have to 
        interpolate during the fit, which is computationally expensive.
        This needs to be passed throughout the fit_model() algorithm so it can be used.
        """

        blob_pars = {}

        # Values of velocity scale corresponding to wavelengths; this is used to calculate
        # integrated dispersions and velocity offsets for combined lines.
        interp_ftn = interp1d(self.target.wave, np.arange(len(self.target.wave))*self.target.velscale, kind='linear', bounds_error=False)
        
        for line_name, line_dict in self.combined_line_list.items():
            blob_pars[line_name+'_LINE_VEL'] = interp_ftn(line_dict['center'])

        # Indices for continuum wavelengths
        # TODO: config file
        # TODO: unit agnostic
        for wave in [1350, 3000, 4000, 5100, 7000]:
            if (self.target.wave[0] < wave) & (self.target.wave[-1] > wave):
                blob_pars['INDEX_%d'%wave] = ba_utils.find_nearest(self.target.wave,float(wave))[1]

        self.blob_pars = blob_pars


    # TODO: move to separate testing file
    def run_tests(self):
        test_mode_dict = {
            'line': self.line_test,
            'config': self.config_test,
        }

        if not self.options.fit_options.test_lines:
            return True # continue fit

        test_mode = self.options.test_options.test_mode
        if test_mode not in test_mode_dict:
            raise Exception('Unimplemented test mode: %s'%test_mode)

        # TODO: log test results
        return test_mode_dict[test_mode]()


    def line_test(self):
        # TODO: update once new line list specifications are in place
        test_options = self.options.test_options
        if isinstance(test_options.lines[0], str): test_options.lines = [line for line in test_options.lines]

        if not self.options.user_lines:
            raise ValueError('The input user line list is None or empty. There are no lines to test. You cannot use the default line list to test for lines, as they must be explicitly defined by user lines. See examples for details...')

        if self.verbose:
            print('Performing line testing for %s' % (test_options.lines))

        # TODO: validate lines to test are in line list and within the fitting region

        all_test_fits = []
        all_test_metrics = []

        # TODO: deepcopy's needed?
        for i, test_lines in enumerate(test_options.lines):
            # TODO: make class or dict
            # test_set = [(label, [test_lines], {full_line_list}), ...]
            test_set = []
            full_line_list = {}
            max_ncomp = 0

            for label, line_dict in self.line_list.items():
                if (label not in test_lines) and (line_dict.get('parent') not in test_lines):
                    # add any line that's not a test line or a test line child to line list
                    full_line_list[label] = line_dict
                else:
                    # get the max ncomps for all test lines
                    max_ncomp = max(max_ncomp, line_dict.get('ncomp', -1))

            # first test contains none of the test lines
            test_set.append(('NULL_TEST', [], copy.deepcopy(full_line_list)))

            # add subsequent tests, adding a component to each test line as specified
            for ncomp in range(1, max_ncomp+1):
                for label, line_dict in self.line_list.items():
                    if ((label in test_lines) or (line_dict.get('parent') in test_lines)) and line_dict.get('ncomp', 1) == ncomp:
                        full_line_list[label] = line_dict
                test_set.append(('NCOMP_%d'%ncomp, test_lines, copy.deepcopy(full_line_list)))

            test_fit_results, test_metrics = self.run_test_set(test_set, test_title=str(i))
            all_test_fits.append(test_fit_results)
            all_test_metrics.append(test_metrics)

        res_dir = self.target.outdir.joinpath('line_test_results')
        res_dir.mkdir(parents=True, exist_ok=True)
        with open(res_dir.joinpath('fit_results.pkl'), 'wb') as f: pickle.dump(all_test_fits, f)
        with open(res_dir.joinpath('test_results.pkl'), 'wb') as f: pickle.dump(all_test_metrics, f)

        # recreate line list based on results
        force_thresh = np.inf
        new_line_list = {}

        all_test_lines = [line for line in test_lines for test_lines in test_options.lines]
        for label, line_dict in self.line_list.items():
            if (label not in all_test_lines) and (line_dict.get('parent') not in all_test_lines):
                new_line_list[label] = line_dict

        for test_fit, test_metrics in zip(all_test_fits,all_test_metrics):
            # look in reverse to find test with most ncomps that passed
            for label_A, label_B, metrics in test_metrics[::-1]:
                if test_fit_results[label_B]['pass']:
                    force_thresh = np.min([force_thresh, test_fit_results[label_B]['rmse']])
                    new_line_list.update(test_fit_results[label_B]['line_list'])
                    break

        self.line_list = new_line_list
        self.force_thresh = force_thresh if np.isfinite(force_thresh) else np.inf

        # TODO: fix to limit number of times initialize_pars needs to be called
        # TODO: instead of user_lines, use whatever line_list is set in the context
        self.initialize_pars(user_lines=self.line_list)

        return test_options.continue_fit


    def config_test(self):
        test_options = self.options.test_options

        if not self.options.user_lines:
            raise ValueError('The input user line list is None or empty.  There are no lines to test.  You cannot use the default line list to test for lines, as they must be explicitly defined by user lines.  See examples for details...')

        if len(test_options.lines) < 2: 
            raise ValueError('The number of configurations to test must be more than 1!')

        # Check to see that each line in each configuration is in the line list
        for config in test_options.lines:
            if not np.all([True if line in self.line_list else False for line in config]):
                raise ValueError('A line in a configuration is not defined in the input line list!')

        if self.verbose:
            print('Performing configuration testing for %d configurations...' % len(test_options.lines))
            print('----------------------------------------------------------------------------------------------------')

        # test_set = [(label, [test_lines], {full_line_list}), ...]
        test_set = []

        for i, test_config in enumerate(test_options.lines):
            # For each config, we want *only* the lines specified
            test_line_list = {line_name:line_dict for line_name,line_dict in self.line_list.items() if line_name in test_config}
            test_set.append(('CONFIG_%d'%(i+1), test_config, test_line_list))

        test_fit_results, test_metrics = self.run_test_set(test_set)

        res_dir = self.target.outdir.joinpath('config_test_results')
        res_dir.mkdir(parents=True, exist_ok=True)
        with open(res_dir.joinpath('fit_results.pkl'), 'wb') as f: pickle.dump(test_fit_results, f)
        with open(res_dir.joinpath('test_results.pkl'), 'wb') as f: pickle.dump(test_metrics, f)

        self.force_thresh = np.inf
        self.line_list = test_set[0][2] # default line_list to first config
        # look in reverse to find last test config that passed
        for label_A, label_B, metrics in test_metrics[::-1]:
            if test_fit_results[label_B]['pass']:
                self.line_list = test_fit_results[label_B]['line_list']
                self.force_thresh = test_fit_results[label_B]['rmse']
                break

        if not test_options.continue_fit:
            return

        # TODO: fix to limit number of times initialize_pars needs to be called
        self.initialize_pars(user_lines=self.line_list)

        return test_options.continue_fit


    # TODO: do something with:
    # Calculate R-Squared statistic of best fit
    # r2 = badass_test_suite.r_squared(copy.deepcopy(mccomps["DATA"][0]),copy.deepcopy(mccomps["MODEL"][0]))
    # Calculate rCHI2 statistic of best fit
    # rchi2 = badass_test_suite.r_chi_squared(copy.deepcopy(mccomps["DATA"][0]),copy.deepcopy(mccomps["MODEL"][0]),copy.deepcopy(mccomps["NOISE"][0]),len(_param_dict))
    # Calculate RMSE statistic of best fit
    # rmse = lowest_rmse#badass_test_suite.root_mean_squared_error(copy.deepcopy(mccomps["DATA"][0]),copy.deepcopy(mccomps["MODEL"][0]))
    # Calculate MAE statistic of best fit
    # mae = badass_test_suite.mean_abs_error(copy.deepcopy(mccomps["DATA"][0]),copy.deepcopy(mccomps["MODEL"][0]))

    def run_test_set(self, test_set, test_title=None):

        # test_set = [(label, [test_lines], {full_line_list}), ...]

        # TODO: all test_labels are unique
        # {label: {fit_results}), ...}
        test_fit_results = {}

        # [(label1, label2, {metrics}), ...]
        test_metrics = []

        for test_label, test_lines, full_line_list in test_set:

            self.initialize_pars(user_lines=full_line_list)
            mcpars, mccomps, mcLL, lowest_rmse = self.max_likelihood(line_test=True)

            # Calculate degrees of freedom of fit; nu = n - m (n number of observations minus m degrees of freedom (free fitted parameters))
            dof = len(self.target.wave)-len(self.param_dict)
            if dof <= 0:
                if self.verbose:
                    print('WARNING: Degrees-of-Freedom in fit is <= 0.  One should increase the test range and/or decrease the number of free parameters of the model appropriately')
                dof = 1

            if self.options.fit_options.reweighting:
                rchi2 = badass_test_suite.r_chi_squared(mccomps['DATA'][0], mccomps['MODEL'][0], mccomps['NOISE'][0], len(self.param_dict))
                aon = badass_test_suite.calculate_aon(test_lines, full_line_list, mccomps, self.target.noise*np.sqrt(rchi2))
            else:
                aon = badass_test_suite.calculate_aon(test_lines, full_line_list, mccomps, self.target.noise)

            fit_results = {'mcpars':mcpars,'mccomps':mccomps,'mcLL':mcLL,'line_list':full_line_list,'dof':dof,'npar':len(self.param_dict),'rmse':lowest_rmse, 'aon':aon}
            test_fit_results[test_label] = fit_results

            if len(test_fit_results.keys()) == 1: # first run
                prev_label, prev_results = test_label, fit_results
                continue

            # get metrics
            resid_A = prev_results['mccomps']['RESID'][0,:][self.target.fit_mask]
            resid_B = fit_results['mccomps']['RESID'][0,:][self.target.fit_mask]

            # TODO: test suite util to get all metrics
            metrics = {}

            ddof = np.abs(prev_results['dof']-fit_results['dof'])
            _,_,_,conf,_,_,_,_,_,_ = badass_test_suite.bayesian_AB_test(resid_B, resid_A, self.target.wave[self.target.fit_mask], self.target.noise[self.target.fit_mask], self.target.spec[self.target.fit_mask], np.arange(len(resid_A)), ddof, self.target.options.io_options.output_dir, plot=False)
            metrics['BADASS'] = conf

            ssr_ratio, ssr_A, ssr_B = badass_test_suite.ssr_test(resid_B, resid_A, self.target.options.io_options.output_dir)
            metrics['SSR_RATIO'] = ssr_ratio

            k_A, k_B = prev_results['npar'], fit_results['npar']
            f_stat, f_pval, f_conf = badass_test_suite.anova_test(resid_B, resid_A, k_A, k_B, self.target.options.io_options.output_dir)
            metrics['ANOVA'] = f_conf

            metrics['F_RATIO'] = badass_test_suite.f_ratio(resid_B, resid_A)

            chi2_B, chi2_A, chi2_ratio = badass_test_suite.chi2_metric(np.arange(len(resid_A)), fit_results['mccomps'], prev_results['mccomps'])
            metrics['CHI2_RATIO'] = chi2_ratio

            if self.target.options.test_options.plot_tests:
                plotting.create_test_plot(self.target, test_fit_results, prev_label, test_label, test_title=test_title)

            test_pass = badass_test_suite.thresholds_met(self.target.options.test_options, metrics, fit_results)
            fit_results['pass'] = test_pass

            test_metrics.append((prev_label, test_label, metrics))
            ptbl = PrettyTable()
            ptbl.field_names = ['TEST A', 'TEST B'] + list(metrics.keys()) + ['AON', 'PASS']
            ptbl.add_row([prev_label, test_label] + ['%f'%v for v in metrics.values()] + ['%f'%aon, test_pass])
            print(ptbl)

            print('(Test %s)' % 'Passed' if test_pass else 'Failed')
            if test_pass and self.target.options.test_options.auto_stop:
                print('metric thresholds met, stopping')
                break

            prev_label, prev_results = test_label, fit_results

        ptbl = PrettyTable()
        ptbl.field_names = ['TEST A', 'TEST B'] + list(test_metrics[0][2].keys()) + ['AON', 'PASS']
        for label_A, label_B, metrics in test_metrics:
            ptbl.add_row([label_A, label_B] + ['%f'%v for v in metrics.values()] + ['%f'%test_fit_results[label_B]['aon'], test_fit_results[label_B]['pass']])
        print('Test Results:')
        print(ptbl)

        return test_fit_results, test_metrics


    skip_comps = ['DATA','WAVE','MODEL','NOISE','RESID','POWER','HOST_GALAXY','BALMER_CONT','APOLY','MPOLY',]
    cont_comps = ['POWER', 'HOST_GALAXY', 'BALMER_CONT', 'APOLY', 'MPOLY']
    tied_target_pars = ['amp', 'disp', 'voff', 'shape', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10']

    def max_likelihood(self, line_test=False):

        print('Performing max likelihood fitting')

        self.prior_params = [key for key,val in self.param_dict.items() if ('prior' in val)]
        self.cur_params = {k:v['init'] for k,v in self.param_dict.items()}

        bounds = [val['plim'] for key,val in self.param_dict.items()]
        lb, ub = zip(*bounds)
        param_bounds = op.Bounds(lb, ub, keep_feasible=True)

        def eval_con(params, param_names, expr1, expr2):
            local_dict = dict(zip(param_names,params))
            r1 = ne.evaluate(expr1, local_dict=local_dict).item()
            r2 = ne.evaluate(expr2, local_dict=local_dict).item()
            return r1 - r2
        cons = [{'type':'ineq', 'fun':eval_con, 'args':(list(self.param_dict.keys()), con[0], con[1])} for con in self.soft_cons]

        n_basinhop = self.options.fit_options.n_basinhop
        lowest_rmse = badass_test_suite.root_mean_squared_error(self.fit_spec, np.zeros(len(self.fit_spec)))
        callback_ftn = None
        if np.isfinite(self.force_thresh):
            print('Required Maximum Likelihood RMSE threshold: %0.4f' % (self.force_thresh))
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
                rmse = badass_test_suite.root_mean_squared_error(self.comp_dict['DATA'], self.comp_dict['MODEL'])
                lowest_rmse = min(lowest_rmse, rmse)

                accept_thresh = 0.001 # Define an acceptance threshold
                if (basinhop_count > n_basinhop) and (accepted_count >=1) and ((lowest_rmse-accept_thresh > self.force_thresh) or (lowest_rmse > self.force_thresh)):
                    print('Warning: basinhopping has exceeded %d attemps to find a new global maximum. Terminating fit...'%n_basinhop)
                    return True

                terminate = False
                if (accepted_count > 1) and (basinhop_count >= force_basinhop) and (((lowest_rmse-accept_thresh) <= self.force_thresh) or (lowest_rmse <= self.force_thresh)):
                    terminate = True

                print('\tFit Status: %s\n\tForce threshold: %0.4f\n\tLowest RMSE: %0.4f\n\tCurrent RMSE: %0.4f\n\tAccepted Count: %d\n\tBasinhop Count:%d'%(terminate,self.force_thresh,lowest_rmse,rmse,accepted_count,basinhop_count))
                return terminate


        def lnprob_wrapper(fit_vals):
            self.cur_params = dict(zip(self.cur_params.keys(), fit_vals))
            return -(self.lnprob()[0]) # only care about the first returned value

        minimizer_args = {'method':'SLSQP', 'bounds':param_bounds,'constraints':cons,'options':{'disp':False,}}
        result = op.basinhopping(func=lnprob_wrapper, x0=list(self.cur_params.values()), stepsize=1.0, interval=1, niter=2500, minimizer_kwargs=minimizer_args,
                                 disp=self.verbose, niter_success=n_basinhop, callback=callback_ftn)

        best_fit = result['x']
        self.cur_params = dict(zip(self.cur_params.keys(), best_fit))
        fun_result = result['fun']
        self.fit_model()
        self.reweight()

        max_like_niter = self.options.fit_options.max_like_niter
        self.init_mc_store(max_like_niter+1)
        self.update_mc_store(0, fun_result)

        if max_like_niter:
            print('Performing Monte Carlo bootstrapping')
            orig_fit_spec = self.fit_spec.copy()

            for n in range(1, max_like_niter+1):
                # Generate a simulated galaxy spectrum with noise added at each pixel
                mcgal = np.random.normal(self.target.spec, self.fit_noise)
                # Get rid of any infs or nan if there are none; this will cause scipy.optimize to fail
                mcgal[~np.isfinite(mcgal)] = np.nanmedian(mcgal)
                self.fit_spec = mcgal

                resultmc = op.minimize(fun=lnprob_wrapper, x0=list(self.cur_params.values()), method='SLSQP', 
                                       bounds=param_bounds, constraints=cons, options={'maxiter':1000,'disp': False})

                # return original spectrum for fitting model
                self.fit_spec = orig_fit_spec
                best_fit = resultmc['x']
                self.cur_params = dict(zip(self.cur_params.keys(), best_fit))
                fun_result = resultmc['fun']
                self.fit_model()
                self.update_mc_store(n, fun_result)

        self.compile_mc_results()

        # TODO: handle differently
        if line_test:
            mccomps = {k:v for k,v in self.mc_attr_store.items() if k in self.comp_dict.keys()}
            return self.fit_results, mccomps, self.mc_attr_store['LOG_LIKE'], lowest_rmse

        self.cur_params = {k:self.fit_results[k]['med'] for k in self.cur_params.keys()}
        self.fit_model()

        self.write_max_like_results()

        return self.target.options.mcmc_options.mcmc_fit


    def init_mc_store(self, iters):
        for key in self.cur_params.keys():
            self.mc_attr_store[key] = np.zeros(iters)

        # TODO: make dict for 'name'-> calc_func
        comp_attrs = ['FLUX', 'LUM', 'EW',]
        for key, val in self.comp_dict.items():
            self.mc_attr_store[key] = np.zeros((iters, len(val)))
            if not key in self.skip_comps:
                for attr in comp_attrs:
                    self.mc_attr_store[key+'_'+attr] = np.zeros(iters)

        cll_attrs = ['DISP', 'VOFF',]
        for key in self.combined_line_list.keys():
            for attr in cll_attrs:
                self.mc_attr_store[key+'_'+attr] = np.zeros(iters)

        all_attrs = ['FWHM', 'W80', 'NPIX', 'SNR',]
        for key in (list(self.combined_line_list.keys())+list(self.line_list.keys())):
            for attr in all_attrs:
                self.mc_attr_store[key+'_'+attr] = np.zeros(iters)

        self.mc_attr_store['LOG_LIKE'] = np.zeros(iters)
        self.mc_attr_store['R_SQUARED'] = np.zeros(iters)
        self.mc_attr_store['RCHI_SQUARED'] = np.zeros(iters)

        # TODO: store elsewhere; unit agnostic
        cont_lum_attrs = {
            1350.0: ['L_CONT_AGN_1350', 'L_CONT_HOST_1350', 'L_CONT_TOT_1350'],
            3000.0: ['L_CONT_AGN_3000', 'L_CONT_HOST_3000', 'L_CONT_TOT_3000'],
            4000.0: ['HOST_FRAC_4000', 'AGN_FRAC_4000'],
            5100.0: ['L_CONT_AGN_5100', 'L_CONT_HOST_5100', 'L_CONT_TOT_5100'],
            7000.0: ['HOST_FRAC_7000', 'AGN_FRAC_7000'],
        }
        for wave, attrs in cont_lum_attrs.items():
            if (self.fit_wave[0] < wave) and (self.fit_wave[-1] > wave):
                for key in attrs:
                    self.mc_attr_store[key] = np.zeros(iters)


    def update_mc_store(self, n, fun_result):

        wave_comp = self.comp_dict['WAVE']
        flux_norm = self.target.options.fit_options.flux_norm
        fit_norm = self.target.fit_norm
        blob_pars = self.blob_pars

        for key, val in self.cur_params.items():
            self.mc_attr_store[key][n] = val

        # continuum
        cont = np.zeros(len(wave_comp))
        for key in self.cont_comps:
            if not key in self.comp_dict:
                continue
            cont += self.comp_dict[key]

        for key, val in self.comp_dict.items():
            self.mc_attr_store[key][n,:] = val
            if key in self.skip_comps:
                continue

            # TODO: utility functions for calculations
            # TODO: use rest -> obs frame util
            # TODO: better way to integrate?
            # FLUX
            # Correct for redshift (integrate over observed wavelength, not rest)
            flux = np.trapz(val, wave_comp)*(1.0+self.target.z)
            flux = np.abs(flux)*flux_norm*fit_norm
            self.mc_attr_store[key+'_FLUX'][n] = np.log10(flux)

            # LUM
            self.mc_attr_store[key+'_LUM'][n] = np.log10(self.flux_to_lum(flux))

            # EW
            ew = np.trapz(val/cont, wave_comp)*(1.0+self.target.z)
            self.mc_attr_store[key+'_EW'][n] = ew if np.isfinite(ew) else 0.0


        total_cont, agn_cont, host_cont = get_continuums(self.comp_dict, len(wave_comp))

        for wave in [1350, 3000, 5100]:
            tot_attr = 'L_CONT_TOT_%d'%wave
            if not tot_attr in self.mc_attr_store:
                continue
            index_attr = 'INDEX_%d'%wave

            # Total Luminosities
            flux = total_cont[blob_pars[index_attr]]*flux_norm*fit_norm
            self.mc_attr_store[tot_attr][n] = np.log10(self.flux_to_lum(flux)*wave)

            # AGN Luminosities
            agn_attr = 'L_CONT_AGN_%d'%wave
            flux = agn_cont[blob_pars[index_attr]]*flux_norm*fit_norm
            self.mc_attr_store[agn_attr][n] = np.log10(self.flux_to_lum(flux)*wave)

            # Host Luminosities
            host_attr = 'L_CONT_HOST_%d'%wave
            flux = host_cont[blob_pars[index_attr]]*flux_norm*fit_norm
            self.mc_attr_store[host_attr][n] = np.log10(self.flux_to_lum(flux)*wave)

        for wave in [4000, 7000]:
            host_attr = 'HOST_FRAC_%d'%wave
            if not host_attr in self.mc_attr_store:
                continue

            agn_attr = 'AGN_FRAC_%d'%wave
            index_attr = 'INDEX_%d'%wave
            self.mc_attr_store[host_attr][n] = host_cont[blob_pars[index_attr]]/total_cont[blob_pars[index_attr]]
            self.mc_attr_store[agn_attr][n] = agn_cont[blob_pars[index_attr]]/total_cont[blob_pars[index_attr]]


        for line, line_dict in {**self.line_list,**self.combined_line_list}.items():
            line_comp = self.comp_dict[line]
            self.mc_attr_store[line+'_FWHM'][n] = combined_fwhm(wave_comp, np.abs(line_comp), line_dict['disp_res_kms'], self.target.velscale)
            self.mc_attr_store[line+'_W80'][n] = calculate_w80(wave_comp, np.abs(line_comp), line_dict['disp_res_kms'], self.target.velscale, line_dict['center'])

            # compute number of pixels (NPIX) for each line in the line list;
            # this is done by determining the number of pixels of the line model
            # that are above the raw noise. 
            self.mc_attr_store[line+'_NPIX'][n] = len(np.where(np.abs(line_comp) > self.fit_noise)[0])

            # compute the signal-to-noise ratio (SNR) for each line;
            # this is done by calculating the maximum value of the line model 
            # above the MEAN value of the noise within the channels.
            self.mc_attr_store[line+'_SNR'][n] = np.nanmax(np.abs(line_comp)) / np.nanmean(self.fit_noise)

            if not line in self.combined_line_list:
                continue

            vel = np.arange(len(self.fit_wave))*self.target.velscale - blob_pars[line+'_LINE_VEL']
            full_profile = np.abs(line_comp)
            norm_profile = full_profile/np.sum(full_profile)
            voff = np.trapz(vel*norm_profile,vel)/simpson(norm_profile,vel)
            self.mc_attr_store[line+'_VOFF'][n] = voff if np.isfinite(voff) else 0.0

            disp = np.sqrt(np.trapz(vel**2*norm_profile,vel)/np.trapz(norm_profile,vel) - (voff**2))
            self.mc_attr_store[line+'_DISP'][n] = disp if np.isfinite(disp) else 0.0

        self.mc_attr_store['LOG_LIKE'][n] = fun_result
        self.mc_attr_store['R_SQUARED'][n] = badass_test_suite.r_squared(self.comp_dict['DATA'], self.comp_dict['MODEL'])
        n_free_pars = len(self.cur_params)
        self.mc_attr_store['RCHI_SQUARED'][n] = badass_test_suite.r_chi_squared(self.comp_dict['DATA'], self.comp_dict['MODEL'], self.fit_noise, n_free_pars)


    def compile_mc_results(self):
        # TODO: class or other structure for storing mc_attr_store, fit_results, etc.
        # TODO: error handling to make sure all mc_attr_store keys are included
        # TODO: custom conditions for flags and do that elsewhere
        for key, vals in self.mc_attr_store.items():
            if key in self.comp_dict.keys():
                continue
            mc_med = np.nanmedian(vals)
            if ~np.isfinite(mc_med): mc_med = 0
            mc_std = np.nanstd(vals)
            if ~np.isfinite(mc_std): mc_std = 0
            self.fit_results[key] = {'med': mc_med, 'std': mc_std, 'flag': 0}

        # TODO: mark flags for other keys if med < 0 or std == 0
        # Mark any parameter flags
        for key, val in self.param_dict.items():
            flag = 0
            bounds = val['plim']
            if mc_std == 0: flag += 1
            if mc_med-mc_std <= bounds[0]: flag += 1
            if mc_med+mc_std >= bounds[1]: flag += 1
            self.fit_results[key]['flag'] = flag

        # Add tied parameters
        med_dict = {key:key_dict['med'] for key,key_dict in self.fit_results.items()}
        for line_name, line_dict in self.line_list.items():
            for par_name, expr in line_dict.items():
                if (expr == 'free') or (not par_name in self.tied_target_pars):
                    continue

                med = ne.evaluate(expr, local_dict=med_dict).item()
                expr_stds = np.array([key_dict['std'] for key,key_dict in self.fit_results.items() if key in expr], dtype=float)
                std = np.sqrt(np.sum(expr_stds**2))
                self.fit_results[line_name+'_'+par_name.upper()] = {'med': med, 'std': std, 'flag': 0}

        # Add dispersion resolution (in km/s) for each line
        for line_name, line_dict in {**self.line_list,**self.combined_line_list}.items():
            disp_res = line_dict['disp_res_kms']
            self.fit_results[line_name+'_DISP_RES'] = {'med': disp_res, 'std': np.nan, 'flag': 0}

            disp_dict = self.fit_results[line_name+'_DISP']
            disp_corr = np.nanmax((0.0, np.sqrt(disp_dict['med']**2-disp_res**2)))
            self.fit_results[line_name+'_DISP_CORR'] = {'med': disp_corr, 'std': disp_dict['std'], 'flag': disp_dict['flag']}

            fwhm_dict = self.fit_results[line_name+'_FWHM']
            fwhm_corr = np.nanmax((0.0, np.sqrt(fwhm_dict['med']**2-(2.3548*disp_res)**2)))
            self.fit_results[line_name+'_FWHM_CORR'] = {'med': fwhm_corr, 'std': fwhm_dict['std'], 'flag': fwhm_dict['flag']}

            w80_dict = self.fit_results[line_name+'_W80']
            w80_corr = np.nanmax((0.0, np.sqrt(w80_dict['med']**2-(2.567*disp_res)**2)))
            self.fit_results[line_name+'_W80_CORR'] = {'med': w80_corr, 'std': w80_dict['std'], 'flag': w80_dict['flag']}


    def write_max_like_results(self):
        # Write maximum likelihood fit results to FITS table

        # TODO: need to copy? just let them be rescaled
        result_dict = copy.deepcopy(self.fit_results)
        comp_dict = copy.deepcopy(self.comp_dict)

        # Rescale amplitudes
        for p in result_dict:
            if p[-4:] == '_AMP':
                result_dict[p]['med'] = result_dict[p]['med']*self.target.fit_norm
                result_dict[p]['std'] = result_dict[p]['std']*self.target.fit_norm

        # Rescale components
        for key in comp_dict:
            if key not in ['WAVE']:
                comp_dict[key] *= self.target.fit_norm

        result_dict = dict(sorted(result_dict.items()))

        sigma_noise = np.nanmedian(comp_dict['NOISE'][self.target.fit_mask])
        sigma_resid = np.nanstd(comp_dict['DATA'][self.target.fit_mask]-comp_dict['MODEL'][self.target.fit_mask])
        self.target.log.log_max_like_fit(result_dict, sigma_noise, sigma_resid)

        # Write best-fit parameters
        col1 = fits.Column(name='parameter', format='30A', array=list(result_dict.keys()))
        col2 = fits.Column(name='best_fit', format='E', array=[v['med'] for v in result_dict.values()])
        col3 = fits.Column(name='sigma', format='E', array=[v['std'] for v in result_dict.values()])
        cols = fits.ColDefs([col1,col2,col3])
        table_hdu = fits.BinTableHDU.from_columns(cols)

        hdr = fits.Header()
        hdr['z_sdss'] = self.target.z
        hdr['med_noise'] = np.nanmedian(self.target.noise)
        hdr['velscale'] = self.target.velscale
        hdr['fit_norm'] = self.target.fit_norm
        hdr['flux_norm'] = self.target.options.fit_options.flux_norm

        primary = fits.PrimaryHDU(header=hdr)
        hdu = fits.HDUList([primary, table_hdu])
        hdu.writeto(self.target.outdir.joinpath('log', 'par_table.fits'), overwrite=True)

        # Write best-fit components
        cols = []
        for key, val in comp_dict.items():
            cols.append(fits.Column(name=key, format='E', array=val))

        mask = np.zeros(len(comp_dict['WAVE']), dtype=bool)
        mask[self.target.fit_mask] = True
        cols.append(fits.Column(name='MASK', format='E', array=mask))

        cols = fits.ColDefs(cols)
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.writeto(self.target.outdir.joinpath('log', 'best_model_components.fits'), overwrite=True)

        plotting.plot_ml_results(self)
        print('Done ML fitting %s! \n' % self.target.options.io_options.output_dir)


    def reweight(self):
        if not self.options.fit_options.reweighting:
            return
        print('Reweighting noise to achieve a reduced chi-squared ~ 1')
        cur_rchi2 = badass_test_suite.r_chi_squared(self.comp_dict['DATA'], self.comp_dict['MODEL'], self.fit_noise, len(self.cur_params))
        print('\tCurrent reduced chi-squared = %0.5f' % cur_rchi2)
        self.fit_noise = self.fit_noise*np.sqrt(cur_rchi2)
        new_rchi2 = badass_test_suite.r_chi_squared(self.comp_dict['DATA'], self.comp_dict['MODEL'], self.fit_noise, len(self.cur_params))
        print('\tNew reduced chi-squared = %0.5f' % new_rchi2)


    def lnprob(self):
        # Log-probability function

        ll = self.lnlike()

        if (self.fit_stage == FitStage.BOOTSTRAP) and (self.options.fit_options.fit_stat != 'ML'):
            return ll, ll

        lp = self.lnprior()
        if not np.isfinite(lp):
            return -np.inf, ll

        # return log-prob and log-like:
        # bootstrap mode will ignore the latter, mcmc will return it as a blob
        return lp + ll, ll


    def lnlike(self):
        # Log-likelihood function

        self.fit_model()
        fit_mask = self.target.fit_mask
        fit_stat = self.options.fit_options.fit_stat

        data = self.fit_spec[fit_mask]
        model = self.model[fit_mask]
        noise = self.fit_noise[fit_mask]

        if fit_stat == 'ML':
            return -0.5*np.sum(((data-model)**2/noise**2) + np.log(2*np.pi*noise**2), axis=0)

        if fit_stat == 'OLS':
            return -np.sum((data - model)**2, axis=0)


    def lnprior(self):
        # Log-prior function

        lp_arr = []
        for key, val in self.cur_params.items():
            lower, upper = self.param_dict[key]['plim']
            assert upper > lower
            lp_arr.append(0.0 if lower <= val <= upper else -np.inf)

        # Loop through soft constraints
        for expr1, expr2 in self.soft_cons:
            con_pass = ne.evaluate(expr1, local_dict=self.cur_params).item() - ne.evaluate(expr2, local_dict=self.cur_params).item() >= 0
            lp_arr.append(0.0 if con_pass else -np.inf)

        # Loop through parameters with priors on them 
        prior_map = {'gaussian': lnprior_gaussian, 'halfnorm': lnprior_halfnorm, 'jeffreys': lnprior_jeffreys, 'flat': lnprior_flat}
        p = [prior_map[self.param_dict[key]['prior']['type']](self.cur_params[key],**self.param_dict[key]) for key in self.prior_params]

        lp_arr += p
        return np.sum(lp_arr)


    # The fit_model function controls the model for both the initial and MCMC fits.
    def fit_model(self):
        # Constructs galaxy model
        host_model = np.copy(self.fit_spec)

        self.comp_dict = {}

        # Power-law Component
        # TODO: Create a template model for the power-law continuum
        if self.options.comp_options.fit_power:
            if self.options.power_options.type == 'simple':
                power = simple_power_law(self.fit_wave, self.cur_params['POWER_AMP'], self.cur_params['POWER_SLOPE'])
            elif self.options.plot_options.type == 'broken':
                power = broken_power_law(self.fit_wave, self.cur_params['POWER_AMP'], self.cur_params['POWER_BREAK'],
                                         self.cur_params['POWER_SLOPE_1'], self.cur_params['POWER_SLOPE_2'],
                                         self.cur_params['POWER_CURVATURE'])

            # Subtract off continuum from galaxy, since we only want template weights to be fit
            host_model = host_model - power
            self.comp_dict['POWER'] = power


        # Polynomial Components
        # TODO: create a template
        poly_options = self.options.poly_options
        if self.options.comp_options.fit_poly:
            if poly_options.apoly.bool:
                nw = np.linspace(-1, 1, len(self.fit_wave))
                coeff = np.empty(poly_options.apoly.order+1)
                coeff[0] = 0.0
                for n in range(1, len(coeff)):
                    coeff[n] = self.cur_params['APOLY_COEFF_%d' % n]
                apoly = np.polynomial.legendre.legval(nw, coeff)
                host_model = host_model - apoly
                self.comp_dict['APOLY'] = apoly

            if poly_options.mpoly.bool:
                nw = np.linspace(-1, 1, len(self.fit_wave))
                coeff = np.empty(poly_options.mpoly.order+1)
                for n in range(1, len(coeff)):
                    coeff[n] = self.cur_params['MPOLY_COEFF_%d' % n]
                mpoly = np.polynomial.legendre.legval(nw, coeff)
                self.comp_dict['MPOLY'] = mpoly
                host_model = host_model * mpoly


        # Template Components
        # TODO: host and losvd template components were processed after emission line components,
        #       now they will be before; does this affect anything?
        for template in self.templates.values():
            # TODO: don't need to return comp_dict
            self.comp_dict, host_model = template.add_components(self.cur_params, self.comp_dict, host_model)


        # Emission Line Components
        for line_name, line_dict in self.line_list.items():
            line_model = line_constructor(self, line_name, line_dict)
            if line_model is None:
                continue
            self.comp_dict[line_name] = line_model
            host_model = host_model - self.comp_dict[line_name]


        # The final model
        self.model = np.sum((self.comp_dict[d] for d in self.comp_dict), axis=0)

        # Add combined lines to comp_dict
        for comb_line, line_dict in self.combined_line_list.items():
            self.comp_dict[comb_line] = np.zeros(len(self.fit_wave))
            for line_name in line_dict['lines']:
                self.comp_dict[comb_line] += self.comp_dict[line_name]

        # Add last components to comp_dict for plotting purposes
        # Add galaxy, sigma, model, and residuals to comp_dict
        # TODO: does this need to be done every fit_model call?
        self.comp_dict['DATA']  = self.fit_spec
        self.comp_dict['WAVE']  = self.fit_wave
        self.comp_dict['NOISE'] = self.fit_noise
        self.comp_dict['MODEL'] = self.model
        self.comp_dict['RESID'] = self.fit_spec-self.model


    # TODO: in utils
    def flux_to_lum(self, flux):
        # TODO: calc and store elsewhere
        cosmo = FlatLambdaCDM(self.target.options.fit_options.cosmology.H0, self.target.options.fit_options.cosmology.Om0)
        d_mpc = cosmo.luminosity_distance(self.target.z).value
        # TODO: use astropy units
        d_cm = d_mpc * 3.086E+24 # 1 Mpc = 3.086e+24 cm
        return 4*np.pi*(d_cm**2)*flux


    def run_emcee(self):
        self.reweight()

        # TODO: need to re-initalize parameters here?
        self.target.log.output_free_pars(self.line_list, self.param_dict, self.soft_cons)
        self.cur_params = dict(sorted(self.cur_params.items()))

        nwalkers = self.options.mcmc_options.nwalkers
        if nwalkers < 2*len(self.cur_params):
            print('Number of walkers < 2 x (# of parameters)! Setting nwalkers = %d' % (2*len(self.cur_params)))
            nwalkers = 2*len(self.cur_params)

        pos = self.initialize_walkers(nwalkers)
        ndim = len(self.cur_params)

        # Keep original burn_in and max_iter to reset convergence if jumps out of convergence
        max_iter = self.options.mcmc_options.max_iter
        min_iter = self.options.mcmc_options.min_iter
        write_iter = self.options.mcmc_options.write_iter
        write_thresh = self.options.mcmc_options.write_thresh

        # TODO: for testing
        max_iter = 10
        min_iter = 3
        write_iter = 3
        write_thresh = 3

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
        if self.options.mcmc_options.auto_stop:

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
                    self.max_tol = self.ctx.options.mcmc_options.autocorr_tol
                    self.min_samp = self.ctx.options.mcmc_options.min_samp
                    self.ncor_times = self.ctx.options.mcmc_options.ncor_times
                    self.conv_type = self.ctx.options.mcmc_options.conv_type

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

                    self.stop_iter = self.ctx.options.mcmc_options.max_iter
                    self.burn_in = self.ctx.options.mcmc_options.burn_in


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
                        print('Converged at %d iterations\nPerforming %d iterations of sampling'%(it, self.min_samp))
                        self.burn_in = it
                        self.stop_iter = it+self.min_samp
                        self.conv_tau = tau
                        self.converged = True

                    elif (self.converged) and (not self.conv_func(sampler, tau, tol)):
                        print('Iteration: %d - Jumped out of convergence, resetting burn_in and max_iter'%it)
                        self.burn_in = self.ctx.options.mcmc_options.burn_in
                        self.stop_iter = self.options.mcmc_options.max_iter
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
        #     print('MCMC iteration: %d' % sampler.iteration)
        #     sampler.run_mcmc(pos, min(write_iter, max_iter-sampler.iteration))
        #     self.add_chain(sampler=sampler)
            # TODO: verbose -> print current parameter values


        self.add_chain()
        for result in sampler.sample(pos, iterations=max_iter):
            it = sampler.iteration
            if (it >= write_thresh) and (it % write_iter == 0):
                print('MCMC iteration: %d' % it)
                self.add_chain(sampler=sampler)
                # TODO: log current parameter values

                if not autocorr:
                    continue

                autocorr.check_convergence(sampler)


        elap_time = (time.time() - start_time)
        run_time = ba_utils.time_convert(elap_time)
        print('emcee Runtime = %s' % (run_time))

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
            print(ptbl)

        # TODO: output files
        self.collect_mcmc_results(sampler, autocorr)

        if self.options.plot_options.plot_HTML:
            plotting.plotly_best_fit(self)

        if self.options.plot_options.plot_param_hist:
            for key in self.param_dict.keys():
                plotting.posterior_plot(key, self.mcmc_results_dict[key], self.mcmc_result_chains['chains'][key], autocorr.burn_in, self.target.outdir)
            plotting.posterior_plot('LOG_LIKE', self.mcmc_results_dict['LOG_LIKE'], self.mcmc_result_chains['chains']['LOG_LIKE'], autocorr.burn_in, self.target.outdir)

        if self.options.plot_options.plot_corner:
            plotting.corner_plot(self)

        plotting.plot_best_model(self, 'best_fit_model.pdf')

        elap_time = (time.time() - start_time)
        print('Total Runtime = %s' % (ba_utils.time_convert(elap_time)))

        # TODO:
        # write_log(elap_time,'total_time',run_dir)

        print('Done MCMC fitting %s! \n' % self.target.options.io_options.output_dir)


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
            blob_dict[line_name+'_FWHM'] = combined_fwhm(wave, np.abs(line_comp), line_dict['disp_res_kms'], self.target.velscale)
            blob_dict[line_name+'_W80'] = calculate_w80(wave, np.abs(line_comp), line_dict['disp_res_kms'], self.target.velscale, line_dict['center'])
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
            if (wave < self.fit_wave[0]) or (wave > self.fit_wave[-1]):
                continue

            for cont_key, cont_val in cont_types.items():
                blob_dict['F_CONT_%s_%d'%(cont_key,wave)] = cont_val[self.blob_pars['INDEX_%d'%wave]]

        for wave in [4000, 7000]:
            if (wave < self.fit_wave[0]) or (wave > self.fit_wave[-1]):
                continue

            for cont_key in ['AGN', 'HOST']:
                blob_dict['HOST_FRAC_%d'%wave] = cont_types[cont_key][self.blob_pars['INDEX_%d'%wave]]/total_cont[self.blob_pars['INDEX_%d'%wave]]


        blob_dict['R_SQUARED'] = badass_test_suite.r_squared(self.comp_dict['DATA'], self.comp_dict['MODEL'])
        blob_dict['RCHI_SQUARED'] = badass_test_suite.r_chi_squared(self.comp_dict['DATA'], self.comp_dict['MODEL'], self.comp_dict['NOISE'], len(self.cur_params))
        return blob_dict


    def collect_mcmc_results(self, sampler, autocorr):
        nwalkers, niters, nparams = sampler.chain.shape
        burn_in = autocorr.burn_in if autocorr else self.options.mcmc_options.burn_in
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
                val = val * self.options.fit_options.flux_norm * self.target.fit_norm * (1.0+self.target.z)

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

            hist, bin_edges = np.histogram(chain, bins='doane', density=False)
            par_results['post_max'] = bin_edges[hist.argmax()]

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
                if (par_val == 'free') or (not par_name in self.tied_target_pars) or (isinstance(par_val, (int,float))):
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
        if self.options.output_options.write_chain:
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
        hdr['z_sdss'] = self.target.z
        hdr['med_noise'] = np.nanmedian(self.target.noise)
        hdr['velscale'] = self.target.velscale
        hdr['fit_norm'] = self.target.fit_norm
        hdr['flux_norm'] = self.target.options.fit_options.flux_norm
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

    npar = np.shape(sampler_chain)[2] # Number of parameters

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


def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    """
    Estimates the 1d autocorrelation function for a chain.
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
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


def lnprior_gaussian(x,**kwargs):
    """
    Log-Gaussian prior based on user-input.  If not specified, mu and sigma 
    will be derived from the init and plim, with plim occurring at 5-sigma
    for the maximum plim from the mean.
    """
    sigma_level = 5
    loc = kwargs['prior'].get('loc', kwargs['init'])
    scale = kwargs['prior'].get('scale', np.max(np.abs(kwargs['plim']))/sigma_level)
    return stats.norm.logpdf(x,loc=loc,scale=scale)


def lnprior_halfnorm(x, **kwargs):
    """
    Half Log-Normal prior based on user-input.  If not specified, mu and sigma 
    will be derived from the init and plim, with plim occurring at 5-sigma
    for the maximum plim from the mean.
    """
    sigma_level = 5
    x = np.abs(x)
    loc = kwargs['prior'].get('loc', kwargs['plim'][0])
    scale = kwargs['prior'].get('scale', np.max(np.abs(kwargs['plim']))/sigma_level)
    return stats.halfnorm.logpdf(x,loc=loc,scale=scale)


def lnprior_jeffreys(x, **kwargs):
    """
    Log-Jeffreys prior based on user-input.  If not specified, mu and sigma 
    will be derived from the init and plim, with plim occurring at 5-sigma
    for the maximum plim from the mean.
    """
    x = np.abs(x)
    if np.any(x) <= 0: x = 1.e-6
    scale = 1
    if 'loc' in kwargs['prior']:
        loc = np.abs(kwargs['prior']['loc'])
    else:
        loc = np.min(np.abs(kwargs['plim']))
    a, b = np.min(np.abs(kwargs['plim'])), np.max(np.abs(kwargs['plim']))
    if a <= 0: a = 1e-6
    return stats.loguniform.logpdf(x,a=a,b=b,loc=loc,scale=scale)


def lnprior_flat(x, **kwargs):
    if (x >= kwargs['plim'][0]) and (x <= kwargs['plim'][1]):
        return 1.0
    return -np.inf


def combined_fwhm(lam_gal, full_profile, disp_res, velscale ):
    # Calculate fwhm of combined lines directly from the model
    def lin_interp(x, y, i, half):
        return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

    def half_max_x(x, y):
        half = max(y)/2.0
        signs = np.sign(np.add(y, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        zero_crossings_i = np.where(zero_crossings)[0]
        if len(zero_crossings_i) == 2:
            return [lin_interp(x, y, zero_crossings_i[0], half),
                    lin_interp(x, y, zero_crossings_i[1], half)]
        return [0.0, 0.0]

    hmx = half_max_x(range(len(lam_gal)),full_profile)
    fwhm_pix = np.abs(hmx[1]-hmx[0])
    fwhm = fwhm_pix*velscale
    # fwhm = np.sqrt((fwhm_pix*velscale)**2 - (disp_res*2.3548)**2)
    return fwhm if np.isfinite(fwhm) else 0.0


def calculate_w80(lam_gal, full_profile, disp_res, velscale, center ):
    # Calculate W80 of the full line profile for all lines

    # TODO: use astropy consts
    c = 299792.458 # speed of light (km/s)
    # Calculate the normalized CDF of the line profile
    cdf = np.cumsum(full_profile/np.sum(full_profile))
    v = (lam_gal-center)/center*c
    w80 = np.interp(0.91,cdf,v) - np.interp(0.10,cdf,v)

    # Correct for intrinsic W80
    # The formula for a Gaussian W80 = 1.09*FWHM = 2.567*disp_res (Harrison et al. 2014; Manzano-King et al. 2019)
    # w80 = np.sqrt((w80)**2-(2.567*disp_res)**2)

    return w80 if np.isfinite(w80) else 0.0


# TODO: move to separate template
def simple_power_law(x,amp,alpha):
    """
    Simple power-low function to model
    the AGN continuum (Calderone et al. 2017).

    Parameters
    ----------
    x    : array_like
            wavelength vector (angstroms)
    amp   : float 
            continuum amplitude (flux density units)
    alpha : float
            power-law slope

    Returns
    ----------
    C    : array
            AGN continuum model the same length as x
    """
    xb = np.max(x)-(0.5*(np.max(x)-np.min(x))) # take to be half of the wavelength range
    return amp*(x/xb)**alpha # un-normalized


# TODO: move to separate template
def broken_power_law(x, amp, x_break, alpha_1, alpha_2, delta):
    """
    Smoothly-broken power law continuum model; for use 
    when there is sufficient coverage in near-UV.
    (See https://docs.astropy.org/en/stable/api/astropy.modeling.
     powerlaws.SmoothlyBrokenPowerLaw1D.html#astropy.modeling.powerlaws.
     SmoothlyBrokenPowerLaw1D)

    Parameters
    ----------
    x       : array_like
              wavelength vector (angstroms)
    amp  : float [0,max]
              continuum amplitude (flux density units)
    x_break : float [x_min,x_max]
              wavelength of the break
    alpha_1 : float [-4,2]
              power-law slope on blue side.
    alpha_2 : float [-4,2]
              power-law slope on red side.
    delta   : float [0.001,1.0]

    Returns
    ----------
    C    : array
            AGN continuum model the same length as x
    """

    return amp * (x/x_break)**(alpha_1) * (0.5*(1.0+(x/x_break)**(1.0/delta)))**((alpha_2-alpha_1)*delta)
