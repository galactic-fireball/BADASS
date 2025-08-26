#!/usr/bin/env python

"""Bayesian AGN Decomposition Analysis for SDSS Spectra (BADASS3)

BADASS is an open-source spectral analysis tool designed for detailed decomposition
of Sloan Digital Sky Survey (SDSS) spectra, and specifically designed for the 
fitting of Type 1 ("broad line") Active Galactic Nuclei (AGN) in the optical. 
The fitting process utilizes the Bayesian affine-invariant Markov-Chain Monte 
Carlo sampler emcee for robust parameter and uncertainty estimation, as well 
as autocorrelation analysis to access parameter chain convergence.
"""

import numpy as np
from numpy.polynomial import hermite
from numpy import linspace, meshgrid 
import scipy.optimize as op
import pandas as pd
import numexpr as ne
import matplotlib.pyplot as plt 
from matplotlib import cm
import matplotlib.gridspec as gridspec
from scipy import optimize, linalg, special, fftpack
from scipy.interpolate import griddata, interp1d
from scipy.stats import f, chisquare
from scipy import stats
import scipy
from scipy.integrate import simpson
from astropy.io import fits
import glob
import time
import datetime
from os import path
import os
import shutil
import sys
from astropy.stats import mad_std
from scipy.special import wofz
import emcee
from astroquery.irsa_dust import IrsaDust
import astropy.units as u
from astropy import coordinates
from astropy.cosmology import FlatLambdaCDM
import re
import natsort
import copy
import pickle
from prettytable import PrettyTable
# import StringIO
import psutil
import pathlib
import importlib
import multiprocessing as mp
# import bifrost
import spectres
import corner
import astropy.constants as const
# Import BADASS tools modules
# cwd = os.getcwd() # get current working directory
# print(cwd)
BADASS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BADASS_DIR))
sys.path.insert(1,str(BADASS_DIR.joinpath('badass_utils'))) # utility functions
sys.path.insert(1,str(BADASS_DIR.joinpath('badass_tools'))) # tool functions

import badass_check_input as badass_check_input
import badass_test_suite  as badass_test_suite
import badass_tools as badass_tools
import gh_alternative as gh_alt # Gauss-Hermite alternative line profiles
from sklearn.decomposition import PCA
from astroML.datasets import sdss_corrected_spectra # SDSS templates for PCA analysis

from prodict import Prodict

from utils.options import BadassOptions
from input.input import BadassInput
from utils.utils import time_convert, find_nearest
from templates.common import initialize_templates
import utils.plotting as plotting
from line_utils.line_lists.optical_qso import optical_qso_default
from line_utils.line_profiles import line_constructor

# plt.style.use('dark_background') # For cool tron-style dark plots
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 100000
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

__author__	   = "Remington O. Sexton (USNO), Sara M. Doan (GMU), Michael A. Reefe (GMU), William Matzko (GMU), Nicholas Darden (UCR)"
__copyright__  = "Copyright (c) 2023 Remington Oliver Sexton"
__credits__	   = ["Remington O. Sexton (GMU/USNO)", "Sara M. Doan (GMU)", "Michael A. Reefe (GMU)", "William Matzko (GMU)", "Nicholas Darden (UCR)"]
__license__	   = "MIT"
__version__	   = "10.2.0"
__maintainer__ = "Remington O. Sexton"
__email__	   = "remington.o.sexton.civ@us.navy.mil"
__status__	   = "Release"

##########################################################################################################


# TODO: create BadassContext class that contains a target, options, parameters, etc. and the following relevant
#       functions are instance functions
# TODO: all print statements to logger
# TODO: all 'if verbose' checks to logger
# TODO: all init/plim values to config file
# TODO: ability to resume from line test and ml results
# TODO: ability to resume mid run (save status at certain checkpoints?)
# TODO: ability to multiprocess mcmc runs?
# TODO: line type classes? or just a general line class?
# TODO: organize imports
# TODO: remove any whitespace at ends of lines
# TODO: use rng seed to be able to reproduce fits


def run_BADASS(inputs, **kwargs):
    # utils.options.BadassOptions.get_options_dep(kwargs)
    opts = BadassOptions.get_options(kwargs['options_file'])
    targets = BadassInput.get_inputs(inputs, opts)

    # TODO: multiprocess
    for target in targets:
        BadassRunContext(target).run()


# TODO: move logger to here instead of target
# TODO: make sure all attrs are initialized
class BadassRunContext:
    def __init__(self, target):
        self.target = target
        self.options = target.options
        self.verbose = self.options.output_options.verbose

        # The spectral data currently being fit
        self.fit_wave = self.target.wave.copy()
        self.fit_spec = self.target.spec.copy()
        self.fit_noise = self.target.noise.copy()

        self.start_time = None
        self.force_thresh = np.inf
        self.fit_type = 'init' # TODO: needed?

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


    def run(self):
        plotting.create_input_plot(self)
        self.initialize_fit()

        # Line testing is performed first for a better line list determination and number of components
        if not self.run_tests():
            return

        if not self.max_likelihood():
            return

        # self.run_emcee()


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

            peaks,_ = scipy.signal.find_peaks(norm_csub, height=2.0, width=3.0, prominence=1)
            troughs,_ = scipy.signal.find_peaks(-norm_csub, height=2.0, width=3.0, prominence=1)
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

            init_amp = self.target.spec[find_nearest(self.target.wave,center)[1]]
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
                blob_pars['INDEX_%d'%wave] = find_nearest(self.target.wave,float(wave))[1]

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
            raise Exception('Unimplemented test mode: %s'%self.test_mode)

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
                aon = badass_test_suite.calculate_aon(test_lines, full_line_list, mccomps, tself.arget.noise)

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
            return -self.lnprob()

        minimizer_args = {'method':'SLSQP', 'bounds':param_bounds,'constraints':cons,'options':{'disp':False,}}
        result = op.basinhopping(func=lnprob_wrapper, x0=list(self.cur_params.values()), stepsize=1.0, interval=1, niter=2500, minimizer_kwargs=minimizer_args,
                                 disp=self.verbose, niter_success=n_basinhop, callback=callback_ftn)

        par_best = result['x']
        self.cur_params = dict(zip(self.cur_params.keys(), par_best))
        fun_result = result['fun']
        self.fit_model()

        if self.options.fit_options.reweighting:
            print('Reweighting noise to achieve a reduced chi-squared ~ 1')
            cur_rchi2 = badass_test_suite.r_chi_squared(self.comp_dict['DATA'], self.comp_dict['MODEL'], self.fit_noise, len(par_best))
            print('\tCurrent reduced chi-squared = %0.5f' % cur_rchi2)
            self.fit_noise = self.fit_noise*np.sqrt(cur_rchi2)
            new_rchi2 = badass_test_suite.r_chi_squared(self.comp_dict['DATA'], self.comp_dict['MODEL'], self.fit_noise, len(par_best))
            print('\tNew reduced chi-squared = %0.5f' % new_rchi2)


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
                self.fit_type = 'init' # TODO: need?

                resultmc = op.minimize(fun=lnprob_wrapper, x0=list(self.cur_params.values()), method='SLSQP', 
                                       bounds=param_bounds, constraints=cons, options={'maxiter':1000,'disp': False})

                # return original spectrum for fitting model
                self.fit_spec = orig_fit_spec
                par_best = resultmc['x']
                self.cur_params = dict(zip(self.cur_params.keys(), par_best))
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

        # TODO: calc and store elsewhere
        cosmo = FlatLambdaCDM(self.target.options.fit_options.cosmology.H0, self.target.options.fit_options.cosmology.Om0)
        d_mpc = cosmo.luminosity_distance(self.target.z).value
        # TODO: use astropy units
        d_cm = d_mpc * 3.086E+24 # 1 Mpc = 3.086e+24 cm

        # TODO: in utils
        def flux_to_lum(flux):
            return 4*np.pi*(d_cm**2)*flux

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
            # FLUX
            # Correct for redshift (integrate over observed wavelength, not rest)
            flux = np.trapz(val, wave_comp)*(1.0+self.target.z)
            flux = np.abs(flux)*flux_norm*fit_norm
            self.mc_attr_store[key+'_FLUX'][n] = np.log10(flux)

            # LUM
            self.mc_attr_store[key+'_LUM'][n] = np.log10(flux_to_lum(flux))

            # EW
            ew = np.trapz(val/cont, wave_comp)*(1.0+self.target.z)
            self.mc_attr_store[key+'_EW'][n] = ew if np.isfinite(ew) else 0.0


        # TODO: store key arrays elsewhere
        total_cont = np.zeros(len(wave_comp))
        for key in ['POWER', 'HOST_GALAXY', 'BALMER_CONT', 'APOLY', 'MPOLY']:
            if not key in self.comp_dict:
                continue
            total_cont += self.comp_dict[key]
        agn_cont = np.zeros(len(wave_comp))
        for key in ['POWER', 'BALMER_CONT', 'APOLY', 'MPOLY']:
            if not key in self.comp_dict:
                continue
            agn_cont += self.comp_dict[key]
        host_cont = np.zeros(len(wave_comp))
        for key in ['HOST_GALAXY', 'APOLY', 'MPOLY']:
            if not key in self.comp_dict:
                continue
            host_cont += self.comp_dict[key]

        for wave in [1350, 3000, 5100]:
            tot_attr = 'L_CONT_TOT_%d'%wave
            if not tot_attr in self.mc_attr_store:
                continue
            index_attr = 'INDEX_%d'%wave

            # Total Luminosities
            flux = total_cont[blob_pars[index_attr]]*flux_norm*fit_norm
            self.mc_attr_store[tot_attr][n] = np.log10(flux_to_lum(flux)*wave)

            # AGN Luminosities
            agn_attr = 'L_CONT_AGN_%d'%wave
            flux = agn_cont[blob_pars[index_attr]]*flux_norm*fit_norm
            self.mc_attr_store[agn_attr][n] = np.log10(flux_to_lum(flux)*wave)

            # Host Luminosities
            host_attr = 'L_CONT_HOST_%d'%wave
            flux = host_cont[blob_pars[index_attr]]*flux_norm*fit_norm
            self.mc_attr_store[host_attr][n] = np.log10(flux_to_lum(flux)*wave)

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
        target_pars = ['amp', 'disp', 'voff', 'shape', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10']
        for line_name, line_dict in self.line_list.items():
            for par_name, expr in line_dict.items():
                if (expr == 'free') or (not par_name in target_pars):
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


    def lnprob(self):
        # Log-probability function

        res = self.lnlike()

        # MCMC fitting
        if self.fit_type == 'final':
            ll, flux_blob, eqwidth_blob, cont_flux_blob, int_vel_disp_blob = res
            lp = self.lnprior()
            if not np.isfinite(lp):
                return -np.inf, flux_blob, eqwidth_blob, cont_flux_blob, int_vel_disp_blob, ll
            return lp + ll, flux_blob, eqwidth_blob, cont_flux_blob, int_vel_disp_blob, ll

        # Maximum Likelihood, etc. fitting
        elif self.fit_type == 'init':
            ll = res
            if self.options.fit_options.fit_stat in ['ML']:
                lp = self.lnprior()
                if not np.isfinite(lp):
                    return -np.inf
                return lp + ll
            return ll


    def lnlike(self):
        # Log-likelihood function

        self.fit_model()
        fit_mask = self.target.fit_mask
        fit_stat = self.options.fit_options.fit_stat

        # TODO: make separate functions for ML, OLS, other fit stats
        if self.fit_type == 'final':
            model, flux_blob, eqwidth_blob, cont_flux_blob, int_vel_disp_blob = res
            if fit_stat == 'ML':
                # Calculate log-likelihood
                l = -0.5*(self.fit_spec[fit_mask]-model[fit_mask])**2/(self.fit_noise[fit_mask])**2 + np.log(2*np.pi*(self.fit_noise[fit_mask])**2)
                l = np.sum(l, axis=0)
            elif fit_stat == 'OLS':
                # Since emcee looks for the maximum, but Least Squares requires a minimum
                # we multiply by negative.
                l = (self.fit_spec[fit_mask]-model[fit_mask])**2
                l = -np.sum(l,axis=0)

            return l, flux_blob, eqwidth_blob, cont_flux_blob, int_vel_disp_blob

        # The maximum likelihood routine [by default] minimizes the negative likelihood
        # Thus for fit_stat="OLS", the SSR must be multiplied by -1 to minimize it. 

        if fit_stat == 'ML':
            # Calculate log-likelihood
            l = -0.5*(self.fit_spec[fit_mask]-self.model[fit_mask])**2/(self.fit_noise[fit_mask])**2 + np.log(2*np.pi*(self.fit_noise[fit_mask])**2)
            l = np.sum(l, axis=0)
        elif fit_stat == 'OLS':
            l = (self.fit_spec[fit_mask]-self.model[fit_mask])**2
            l = -np.sum(l,axis=0)
        return l 


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

        # Fluxes & Equivalent Widths
        # Equivalent widths of emission lines are stored in a dictionary and returned to emcee as metadata blob.
        # Velocity interpolation function
        # TODO: fix for mcmc
        if self.fit_type == 'final':
            fluxes, eqwidths, cont_fluxes, int_vel_disp = calc_mcmc_blob(p, fit_target.wave, self.comp_dict, bactx.options.comp_options, {**bactx.line_list, **bactx.combined_line_list}, bactx.combined_line_list, bactx.blob_pars, bactx.fit_mask, bactx.fit_stat, bactx.target.velscale)
            return gmodel, fluxes, eqwidths, cont_fluxes, int_vel_disp


def ignore_this(): 
    #### Reweighting ###################################################################

    # If True, BADASS can reweight the noise to achieve a reduced chi-squared of 1.  It does this by multiplying the noise by the 
    # square root of the resultant reduced chi-sqaured calculated from the basinhopping fit.  This is then passed to the bootstrapping 
    # fitting so the uncertainties are calculated with the re-weighted noise.

    if reweighting:
        if verbose:
            print("\n Reweighting noise to achieve a reduced chi-squared ~ 1.")
        # Calculate current rchi2
        cur_rchi2 = badass_test_suite.r_chi_squared(copy.deepcopy(comp_dict["DATA"]),copy.deepcopy(comp_dict["MODEL"]),noise,len(param_dict))
        if verbose:
            print("\tCurrent reduced chi-squared = %0.5f" % cur_rchi2)
        # Update noise
        noise = noise*np.sqrt(cur_rchi2)
        # Calculate new rchi2
        new_rchi2 = badass_test_suite.r_chi_squared(copy.deepcopy(comp_dict["DATA"]),copy.deepcopy(comp_dict["MODEL"]),noise,len(param_dict))
        if verbose:
            print("\tNew reduced chi-squared = %0.5f" % new_rchi2)

    # Initialize parameters for emcee
    if verbose:
        print('\n Initializing parameters for MCMC.')
        print('----------------------------------------------------------------------------------------------------')
    param_dict, line_list, combined_line_list, soft_cons, ncomp_dict = initialize_pars(lam_gal,galaxy,noise,fit_reg,disp_res,fit_mask,velscale,
                                                           comp_options,narrow_options,broad_options,absorp_options,
                                                           user_lines,user_constraints,combined_lines,losvd_options,host_options,power_options,poly_options,
                                                           opt_feii_options,uv_iron_options,balmer_options,
                                                           run_dir,fit_type='final',fit_stat=fit_stat,
                                                           fit_opt_feii=fit_opt_feii,fit_uv_iron=fit_uv_iron,fit_balmer=fit_balmer,
                                                           fit_losvd=fit_losvd,fit_host=fit_host,fit_power=fit_power,fit_poly=fit_poly,
                                                           fit_narrow=fit_narrow,fit_broad=fit_broad,fit_absorp=fit_absorp,
                                                           tie_line_disp=tie_line_disp,tie_line_voff=tie_line_voff,
                                                           remove_lines=False,verbose=verbose)
    #
    if verbose:
        output_free_pars(line_list,param_dict,soft_cons)
    #
    # Replace initial conditions with best fit max. likelihood parameters (the old switcharoo)
    for key in result_dict:
        if key in param_dict:
            param_dict[key]['init']=result_dict[key]['med']
    # We make an exception for FeII temperature if Kovadevic et al. (2010) templates are used because 
    # temperature is not every sensitive > 8,000 K.  This causes temperature parameter to blow up
    # during the initial max. likelihood fitting, causing it to be initialized for MCMC at an 
    # unreasonable value.  We therefroe re-initializethe FeiI temp start value to 10,000 K.
    if 'feii_temp' in param_dict:
        param_dict['feii_temp']['init']=10000.0
        




    #######################################################################################################


    # Run emcee
    if verbose:
        print('\n Performing MCMC iterations...')
        print('----------------------------------------------------------------------------------------------------')

    # Extract relevant stuff from dicts
    param_names  = [key for key in param_dict ]
    init_params  = [param_dict[key]['init'] for key in param_dict ]
    bounds		 = [param_dict[key]['plim'] for key in param_dict ]
    prior_dict   = {key:param_dict[key] for key in param_dict if ("prior" in param_dict[key])}
    # Check number of walkers
    # If number of walkers < 2*(# of params) (the minimum required), then set it to that
    if nwalkers<2*len(param_names):
        if verbose:
            print('\n Number of walkers < 2 x (# of parameters)!  Setting nwalkers = %d' % (2.0*len(param_names)))
        nwalkers = int(2.0*len(param_names))
    
    ndim, nwalkers = len(init_params), nwalkers # minimum walkers = 2*len(params)

    # initialize walker starting positions based on parameter estimation from Maximum Likelihood fitting
    pos = initialize_walkers(init_params,param_names,bounds,soft_cons,nwalkers,ndim)
    # Run emcee
    # args = arguments of lnprob (log-probability function)
    lnprob_args=(param_names,
                 prior_dict,
                 line_list,
                 combined_line_list,
                 bounds,
                 soft_cons,
                 lam_gal,
                 galaxy,
                 noise,
                 comp_options,
                 losvd_options,
                 host_options,
                 power_options,
                 poly_options,
                 opt_feii_options,
                 uv_iron_options,
                 balmer_options,
                 outflow_test_options,
                 host_template,
                 opt_feii_templates,
                 uv_iron_template,
                 balmer_template,
                 stel_templates,
                 blob_pars,
                 disp_res,
                 fit_mask,
                 velscale,
                 "final",
                 fit_stat,
                 False,
                 run_dir)
    
    emcee_data = run_emcee(pos,ndim,nwalkers,run_dir,lnprob_args,init_params,param_names,
                            auto_stop,conv_type,min_samp,ncor_times,autocorr_tol,write_iter,write_thresh,
                            burn_in,min_iter,max_iter,verbose=verbose)

    sampler_chain, burn_in, flux_blob, eqwidth_blob, cont_flux_blob, int_vel_disp_blob, log_like_blob = emcee_data
    # Add chains to each parameter in param dictionary
    for k,key in enumerate(param_names):
        if key in param_dict:
            param_dict[key]['chain']=sampler_chain[:,:,k]

    if verbose:
        print('\n > Fitting MCMC chains...')
    # These three functions produce parameter, flux, and luminosity histograms and chains from the MCMC sampling.
    # Free parameter values, uncertainties, and plots
    param_dict = param_plots(param_dict,fit_norm,burn_in,run_dir,plot_param_hist=plot_param_hist,verbose=verbose)
    # Add tied parameters
    param_dict = add_tied_parameters(param_dict, line_list)
    # Log Like Function values plots
    log_like_dict = log_like_plot(log_like_blob, burn_in, nwalkers, run_dir, plot_param_hist=plot_param_hist,verbose=verbose)
    # Flux values, uncertainties, and plots
    flux_dict = flux_plots(flux_blob, z, burn_in, nwalkers, flux_norm, fit_norm, run_dir,verbose=verbose)
    # Luminosity values, uncertainties, and plots
    lum_dict = lum_plots(flux_dict, burn_in, nwalkers, z, run_dir, H0=cosmology["H0"],Om0=cosmology["Om0"],verbose=verbose)
    # Continuum luminosity 
    cont_lum_dict = cont_lum_plots(cont_flux_blob, burn_in, nwalkers, z, flux_norm, fit_norm, run_dir, H0=cosmology["H0"],Om0=cosmology["Om0"],verbose=verbose)
    # Equivalent widths, uncertainties, and plots
    eqwidth_dict = eqwidth_plots(eqwidth_blob, z, burn_in, nwalkers, run_dir, verbose=verbose)
    # Auxiliary Line Dict (Combined FWHMs and Fluxes of MgII and CIV)
    int_vel_disp_dict = int_vel_disp_plots(int_vel_disp_blob, burn_in, nwalkers, z, run_dir, H0=cosmology["H0"],Om0=cosmology["Om0"],verbose=verbose)

    # If stellar velocity is fit, estimate the systemic velocity of the galaxy;
    # SDSS redshifts are based on average emission line redshifts.
    extra_dict = {}
    extra_dict["LOG_LIKE"] = log_like_dict

    if ('stel_vel' in param_dict):
        if verbose:
            print('\n > Estimating systemic velocity of galaxy...')
        z_dict = systemic_vel_est(z,param_dict,burn_in,run_dir,plot_param_hist=plot_param_hist)
        extra_dict = {**extra_dict, **z_dict}


    # Combine all the dictionaries
    combined_pdict = {**param_dict,**flux_dict,**lum_dict,**eqwidth_dict,**cont_lum_dict,**int_vel_disp_dict,**extra_dict}

    # Add the dispersion resolutions and corrected dispersion and widths for all lines
    all_lines = {**line_list,**combined_line_list}
    for line in all_lines:
        disp_res = all_lines[line]["disp_res_kms"]
        combined_pdict[line+"_DISP_RES"] = {'par_best'  : disp_res, # maximum of posterior distribution
                                            'ci_68_low'   : np.nan, # lower 68% confidence interval
                                            'ci_68_upp'   : np.nan, # upper 68% confidence interval
                                            'ci_95_low'   : np.nan, # lower 95% confidence interval
                                            'ci_95_upp'   : np.nan, # upper 95% confidence interval
                                            'post_max'  : np.nan,
                                            'mean'    : np.nan, # mean of posterior distribution
                                            'std_dev'    : np.nan,   # standard deviation
                                            'median'      : np.nan, # median of posterior distribution
                                            'med_abs_dev' : np.nan,   # median absolute deviation
                                            'flat_chain'  : np.nan,   # flattened samples used for histogram.
                                            'flag'    : np.nan, 
                                            }
        disp_corr = np.nanmax([0.0,np.sqrt(combined_pdict[line+"_DISP"]["par_best"]**2-(disp_res)**2)])
        fwhm_corr = np.nanmax([0.0,np.sqrt(combined_pdict[line+"_FWHM"]["par_best"]**2-(disp_res*2.3548)**2)]) 
        w80_corr  = np.nanmax([0.0,np.sqrt(combined_pdict[line+"_W80"]["par_best"]**2-(2.567*disp_res)**2)]) 
        # Add entires for these corrected lines (uncertainties are the same)
        combined_pdict[line+"_DISP_CORR"] = copy.deepcopy(combined_pdict[line+"_DISP"])
        combined_pdict[line+"_DISP_CORR"]["par_best"] = disp_corr
        combined_pdict[line+"_FWHM_CORR"] = copy.deepcopy(combined_pdict[line+"_FWHM"])
        combined_pdict[line+"_FWHM_CORR"]["par_best"] = fwhm_corr
        combined_pdict[line+"_W80_CORR"] = copy.deepcopy(combined_pdict[line+"_W80"])
        combined_pdict[line+"_W80_CORR"]["par_best"] = w80_corr

    if verbose:
        print('\n > Saving Data...')

    # Write all chains to a fits table
    if (write_chain==True):
        write_chains(combined_pdict,run_dir)

    # corner plot
    if (plot_corner==True):
        corner_plot(param_dict,combined_pdict,corner_options,run_dir)



    # Plot and save the best fit model and all sub-components
    comp_dict = plot_best_model(param_dict,
                    line_list,
                    combined_line_list,
                    lam_gal,
                    galaxy,
                    noise,
                    comp_options,
                    losvd_options,
                    host_options,
                    power_options,
                    poly_options,
                    opt_feii_options,
                    uv_iron_options,
                    balmer_options,
                    outflow_test_options,
                    host_template,
                    opt_feii_templates,
                    uv_iron_template,
                    balmer_template,
                    stel_templates,
                    blob_pars,
                    disp_res,
                    fit_mask,
                    fit_stat,
                    velscale,
                    flux_norm,
                    fit_norm,
                    run_dir)

    # Calculate some fit quality parameters which will be added to the dictionary
    # These will be appended to result_dict and need to be in the same format {"med": , "std", "flag":}
    # fit_quality_dict = fit_quality_pars(param_dict,len(param_dict),line_list,combined_line_list,comp_dict,fit_mask,fit_type="mcmc",fit_stat=fit_stat)
    # param_dict = {**param_dict,**fit_quality_dict}

    # Write best fit parameters to fits table
    # Header information
    header_dict = {}
    header_dict["Z_SDSS"]	= z
    header_dict["MED_NOISE"] = np.nanmedian(noise)
    header_dict["VELSCALE"]  = velscale
    header_dict["FLUX_NORM"]  = flux_norm
    header_dict["FIT_NORM"]  = fit_norm
    #
    # param_dict = {**param_dict,**flux_dict,**lum_dict,**eqwidth_dict,**cont_lum_dict,**int_vel_disp_dict,**extra_dict}
    write_params(combined_pdict,header_dict,bounds,run_dir,binnum,spaxelx,spaxely)

    # Make interactive HTML plot 
    if plot_HTML:
        plotly_best_fit(fits_file.parent.name,line_list,fit_mask,run_dir)
    
    # Total time
    elap_time = (time.time() - start_time)
    if verbose:
        print("\n Total Runtime = %s" % (time_convert(elap_time)))
    # Write to log
    write_log(elap_time,'total_time',run_dir)
    print(' - Done fitting %s! \n' % fits_file.stem)
    sys.stdout.flush()
    return


def initialize_walkers(init_params,param_names,bounds,soft_cons,nwalkers,ndim):
    """
    Initializes the MCMC walkers within bounds and soft constraints.
    """
    # Create refereence dictionary for numexpr
    pdict = {}
    for k in range(0,len(param_names),1):
        pdict[param_names[k]] = init_params[k]
        
    pos = init_params + 1.e-3 * np.random.randn(nwalkers,ndim)
    # First iterate through bounds
    for j in range(np.shape(pos)[1]): # iterate through parameter
        for i in range(np.shape(pos)[0]): # iterate through walker
            if (pos[i][j]<bounds[j][0]) | (pos[i][j]>bounds[j][1]):
                while (pos[i][j]<bounds[j][0]) | (pos[i][j]>bounds[j][1]):
                    pos[i][j] = init_params[j] + 1.e-3*np.random.randn(1)
    
    return pos

#### Calculate Sysetemic Velocity ################################################

def systemic_vel_est(z,param_dict,burn_in,run_dir,plot_param_hist=True):
    """
    Estimates the systemic (stellar) velocity of the galaxy and corrects 
    the SDSS redshift (which is based on emission lines).
    """

    c = 299792.458   
    # Get measured stellar velocity
    stel_vel = np.array(param_dict['stel_vel']['chain'])

    # Calculate new redshift
    z_best = (z+1)*(1+stel_vel/c)-1

    # Burned-in + Flattened (along walker axis) chain
    # If burn_in is larger than the size of the chain, then 
    # take 50% of the chain length instead.
    if (burn_in >= np.shape(z_best)[1]):
        burn_in = int(0.5*np.shape(z_best)[1])
        # print('\n Burn-in is larger than chain length! Using 50% of chain length for burn-in...\n')

    flat = z_best[:,burn_in:]
    # flat = flat.flat
    flat = flat.flatten()

    # Subsample the data into a manageable size for the kde and HDI
    if len(flat[np.isfinite(flat)]) > 0:
        # subsampled = np.random.choice(flat[np.isfinite(flat)],size=10000)

        # Histogram; 'Doane' binning produces the best results from tests.
        hist, bin_edges = np.histogram(flat, bins='doane', density=False)

        # Generate pseudo-data on the ends of the histogram; this prevents the KDE
        # from weird edge behavior.
        # n_pseudo = 3 # number of pseudo-bins 
        # bin_width=bin_edges[1]-bin_edges[0]
        # lower_pseudo_data = np.random.uniform(low=bin_edges[0]-bin_width*n_pseudo, high=bin_edges[0], size=hist[0]*n_pseudo)
        # upper_pseudo_data = np.random.uniform(low=bin_edges[-1], high=bin_edges[-1]+bin_width*n_pseudo, size=hist[-1]*n_pseudo)

        # Calculate bandwidth for KDE (Silverman method)
        # h = kde_bandwidth(flat)

        # Create a subsampled grid for the KDE based on the subsampled data; by
        # default, we subsample by a factor of 10.
        # xs = np.linspace(np.min(subsampled),np.max(subsampled),10*len(hist))

        # Calculate KDE
        # kde = gauss_kde(xs,np.concatenate([subsampled,lower_pseudo_data,upper_pseudo_data]),h)
        p68 = compute_HDI(flat,0.68)
        p95 = compute_HDI(flat,0.95)

        post_max  = bin_edges[hist.argmax()] # posterior max estimated from KDE
        post_mean = np.nanmean(flat)
        post_med  = np.nanmedian(flat)
        low_68  = post_med - p68[0]
        upp_68  = p68[1] - post_med
        low_95  = post_med - p95[0]
        upp_95  = p95[1] - post_med
        post_std  = np.nanstd(flat)
        post_mad  = stats.median_abs_deviation(flat)

        if ((post_med-(3.0*low_68))<0): 
            flag = 1
        else: flag = 0

        z_dict = {}
        z_dict["z_sys"] = {}
        z_dict["z_sys"]["par_best"] = post_med
        z_dict["z_sys"]["ci_68_low"]   = low_68
        z_dict["z_sys"]["ci_68_upp"]   = upp_68
        z_dict["z_sys"]["ci_95_low"]   = low_95
        z_dict["z_sys"]["ci_95_upp"]   = upp_95
        z_dict["z_sys"]['post_max'] = post_max 
        z_dict["z_sys"]["mean"] 	   = post_mean
        z_dict["z_sys"]["std_dev"] 	   = post_std
        z_dict["z_sys"]["median"]	   = post_med
        z_dict["z_sys"]["med_abs_dev"] = post_mad
        z_dict["z_sys"]["flat_chain"]  = flat
        z_dict["z_sys"]["flag"] 	   = flag
    else:
        z_dict = {}
        z_dict["z_sys"] = {}
        z_dict["z_sys"]["par_best"] = np.nan
        z_dict["z_sys"]["ci_68_low"]   = np.nan
        z_dict["z_sys"]["ci_68_upp"]   = np.nan
        z_dict["z_sys"]["ci_95_low"]   = np.nan
        z_dict["z_sys"]["ci_95_upp"]   = np.nan
        z_dict["z_sys"]['post_max'] = np.nan 
        z_dict["z_sys"]["mean"] 	   = np.nan
        z_dict["z_sys"]["std_dev"] 	   = np.nan
        z_dict["z_sys"]["median"]	   = np.nan
        z_dict["z_sys"]["med_abs_dev"] = np.nan
        z_dict["z_sys"]["flat_chain"]  = flat
        z_dict["z_sys"]["flag"] 	   = 1	
    
    return z_dict


def insert_nan(spec,ibad):
    """
    Inserts additional NaN values to neighboriing ibad pixels.
    """
    all_bad = np.unique(np.concatenate([ibad-1,ibad,ibad+1]))
    ibad_new = []
    for i in all_bad:
        if (i>0) & (i<len(spec)):
            ibad_new.append(i)
    ibad_new = np.array(ibad_new)
    try:
        spec[ibad_new] = np.nan
        return spec
    except:
        return spec


def isFloat(num):
    try:
        float(num)
        return True
    except (ValueError,TypeError) as e:
        return False

def add_tied_parameters(pdict,line_list):

    # for key in pdict:
    # 	print(key,pdict[key])
    # Make dictionaries for pdict
    param_names = [key for key in pdict]
    # init_dict  = {key:pdict[key]["init"]  for key in pdict}
    # plim_dict  = {key:pdict[key]["plim"]  for key in pdict}
    chain_dict	  = {key:pdict[key]["chain"] for key in pdict}
    par_best_dict   = {key:pdict[key]["par_best"] for key in pdict}

    ci_68_low_dict   = {key:pdict[key]["ci_68_low"] for key in pdict}
    ci_68_upp_dict   = {key:pdict[key]["ci_68_upp"] for key in pdict}
    ci_95_low_dict   = {key:pdict[key]["ci_95_low"] for key in pdict}
    ci_95_upp_dict   = {key:pdict[key]["ci_95_upp"] for key in pdict}

    mean_dict      = {key:pdict[key]["mean"] for key in pdict}
    std_dev_dict     = {key:pdict[key]["std_dev"] for key in pdict}
    median_dict   = {key:pdict[key]["median"] for key in pdict}
    med_abs_dev_dict = {key:pdict[key]["med_abs_dev"] for key in pdict}

    flat_samp_dict   = {key:pdict[key]["flat_chain"] for key in pdict}
    flag_dict	   = {key:pdict[key]["flag"] for key in pdict}
    # print()

    for line in line_list:
        for par in line_list[line]:
            if (line_list[line][par]!="free") & (par in ["amp","disp","voff","shape","h3","h4","h5","h6","h7","h8","h9","h10"]) & (isFloat(line_list[line][par]) is False):
                expr = line_list[line][par] # expression to evaluate
                expr_vars  = [i for i in param_names if i in expr]
                init	   = pdict[expr_vars[0]]["init"]
                plim	   = pdict[expr_vars[0]]["plim"]
                chain	   = ne.evaluate(line_list[line][par],local_dict = chain_dict)
                par_best   = ne.evaluate(line_list[line][par],local_dict = par_best_dict).item()
                flat_chain = ne.evaluate(line_list[line][par],local_dict = flat_samp_dict)

                
                ci_68_low  = np.sqrt(np.sum(np.array([ci_68_low_dict[i] for i in expr_vars],dtype=float)**2))
                ci_68_upp  = np.sqrt(np.sum(np.array([ci_68_upp_dict[i] for i in expr_vars],dtype=float)**2))
                ci_95_low  = np.sqrt(np.sum(np.array([ci_95_low_dict[i] for i in expr_vars],dtype=float)**2))
                ci_95_upp  = np.sqrt(np.sum(np.array([ci_95_upp_dict[i] for i in expr_vars],dtype=float)**2))

                mean        = np.sqrt(np.sum(np.array([mean_dict[i] for i in expr_vars],dtype=float)**2))
                std_dev  = np.sqrt(np.sum(np.array([std_dev_dict[i] for i in expr_vars],dtype=float)**2))
                median    = np.sqrt(np.sum(np.array([median_dict[i] for i in expr_vars],dtype=float)**2))
                med_abs_dev = np.sqrt(np.sum(np.array([med_abs_dev_dict[i] for i in expr_vars],dtype=float)**2))

                flag 	 = np.sum([flag_dict[i] for i in expr_vars])
                pdict[line+"_"+par.upper()] = {"init":init, "plim":plim, "chain":chain, 
                                               "par_best":par_best, "ci_68_low":ci_68_low, "ci_68_upp":ci_68_upp, 
                                               "ci_95_low":ci_95_low, "ci_95_upp":ci_95_upp, 
                                               "mean": mean, "std_dev":std_dev,
                                               "median":median, "med_abs_dev":med_abs_dev,
                                               "flag":flag,"flat_chain":flat_chain}

            # the case where the parameter was set to a constant value
            if (line_list[line][par]!="free") & (par in ["amp","disp","voff","shape","h3","h4","h5","h6","h7","h8","h9","h10"]) & (isFloat(line_list[line][par]) is True):

                continue
    # for key in pdict:
    # 	print(key,pdict[key])

    return pdict





#### Likelihood function #########################################################

##################################################################################

#### Priors ######################################################################
# These priors are the same constraints used for outflow testing and maximum likelihood
# fitting, simply formatted for use by emcee. 
# To relax a constraint, simply comment out the condition (*not recommended*).

def lnprior_gaussian(x,**kwargs):
    """
    Log-Gaussian prior based on user-input.  If not specified, mu and sigma 
    will be derived from the init and plim, with plim occurring at 5-sigma
    for the maximum plim from the mean.
    """
    sigma_level = 5
    if "loc" in kwargs["prior"]:
        loc = kwargs["prior"]["loc"]
    else:
        loc = kwargs["init"]
    #
    if "scale" in kwargs["prior"]:
        scale = kwargs["prior"]["scale"]
    else:
        scale = np.max(np.abs(kwargs["plim"]))/sigma_level
    #
    return stats.norm.logpdf(x,loc=loc,scale=scale)

def lnprior_halfnorm(x,**kwargs):
    """
    Half Log-Normal prior based on user-input.  If not specified, mu and sigma 
    will be derived from the init and plim, with plim occurring at 5-sigma
    for the maximum plim from the mean.
    """
    sigma_level = 5
    x = np.abs(x)
    if "loc" in kwargs["prior"]:
        loc = kwargs["prior"]["loc"]
    else:
        loc = kwargs["plim"][0]
    #
    if "scale" in kwargs["prior"]:
        scale = kwargs["prior"]["scale"]
    else:
        scale = np.max(np.abs(kwargs["plim"]))/sigma_level
    #
    return stats.halfnorm.logpdf(x,loc=loc,scale=scale)


def lnprior_jeffreys(x,**kwargs):
    """
    Log-Jeffreys prior based on user-input.  If not specified, mu and sigma 
    will be derived from the init and plim, with plim occurring at 5-sigma
    for the maximum plim from the mean.
    """
    x = np.abs(x)
    if np.any(x) <=0: x = 1.e-6
    scale = 1
    if "loc" in kwargs["prior"]:
        loc = np.abs(kwargs["prior"]["loc"])
    else:
        loc = np.min(np.abs(kwargs["plim"]))
    a, b = np.min(np.abs(kwargs["plim"])),np.max(np.abs(kwargs["plim"]))
    if a <= 0: a = 1e-6
    return stats.loguniform.logpdf(x,a=a,b=b,loc=loc,scale=scale)

def lnprior_flat(x,**kwargs):

    if (x>=kwargs["plim"][0]) & (x<=kwargs["plim"][1]):
        return 1.0
    else:
        return -np.inf


#### Model Function ##############################################################

def combined_fwhm(lam_gal, full_profile, disp_res, velscale ):
    """
    Calculate fwhm of combined lines directly from the model.
    """
    def lin_interp(x, y, i, half):
        return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

    def half_max_x(x, y):
        half = max(y)/2.0
        signs = np.sign(np.add(y, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        zero_crossings_i = np.where(zero_crossings)[0]
        if len(zero_crossings_i)==2:
            return [lin_interp(x, y, zero_crossings_i[0], half),
                    lin_interp(x, y, zero_crossings_i[1], half)]
        else:
            return [0.0, 0.0]

    hmx = half_max_x(range(len(lam_gal)),full_profile)
    fwhm_pix = np.abs(hmx[1]-hmx[0])
    fwhm = fwhm_pix*velscale
    # fwhm = np.sqrt((fwhm_pix*velscale)**2 - (disp_res*2.3548)**2)
    if ~np.isfinite(fwhm):
        fwhm = 0.0
    #
    return fwhm

##################################################################################

def calculate_w80(lam_gal, full_profile, disp_res, velscale, center ):
    """
    Calculate W80 of the full line profile for all lines.
    """
    c = 299792.458 # speed of light (km/s)
    # Calculate the normalized CDF of the line profile
    cdf = np.cumsum(full_profile/np.sum(full_profile))
    v   = (lam_gal-center)/center*c
    w80 = np.interp(0.91,cdf,v) - np.interp(0.10,cdf,v)
    # Correct for intrinsic W80.  
    # The formula for a Gaussian W80 = 1.09*FWHM = 2.567*disp_res (Harrison et al. 2014; Manzano-King et al. 2019)
    # w80 = np.sqrt((w80)**2-(2.567*disp_res)**2)
    if ~np.isfinite(w80):
        w80 = 0.0
    #
    return w80


# This function generates blob parameters for the MCMC routine,
# including continuum luminosities, fluxes, equivalent widths, 
# widths, and fit quality parameters (R-squared, reduced chi-squared)
def calc_mcmc_blob(p, lam_gal, comp_dict, comp_options, line_list, combined_line_list, blob_pars, fit_mask, fit_stat, velscale):

    _noise = comp_dict["NOISE"]
    noise2 = _noise**2

    # Continuum luminosities
    # Create a single continuum component based on what was fit
    total_cont = np.zeros(len(lam_gal))
    agn_cont   = np.zeros(len(lam_gal))
    host_cont  = np.zeros(len(lam_gal))
    for key in comp_dict:
        if key in ["POWER","HOST_GALAXY","BALMER_CONT", "APOLY", "MPOLY"]:
            total_cont+=comp_dict[key]
        if key in ["POWER","BALMER_CONT", "APOLY", "MPOLY"]:
            agn_cont+=comp_dict[key]
        if key in ["HOST_GALAXY", "APOLY", "MPOLY"]:
            host_cont+=comp_dict[key]


    # Get all spectral components, not including data, model, resid, and noise
    spec_comps = [i for i in comp_dict if i not in ["DATA","MODEL","WAVE","RESID","NOISE","POWER","HOST_GALAXY","BALMER_CONT", "APOLY", "MPOLY"]]
    # Get keys of any lines that were fit for which we will compute eq. widths for
    lines = [line for line in line_list] # list of all lines (individual lines and combined lines)
    # Storage dicts
    fluxes    = {}
    eqwidths      = {}
    int_vel_disp  = {}
    npix_dict    = {}
    snr_dict      = {}
    fit_quality   = {}
    #
    for key in spec_comps:
        flux = np.abs(np.trapz(comp_dict[key],lam_gal))
        # add key/value pair to dictionary
        fluxes[key+"_FLUX"] = flux
        #
        eqwidth = np.trapz(comp_dict[key]/total_cont,lam_gal)
        if ~np.isfinite(eqwidth):
            eqwidth=0.0
        # Add to eqwidth_dict
        eqwidths[key+"_EW"]  = eqwidth
        # For lines AND combined lines, calculate the model FWHM and W80 (NOTE: THIS IS NOT GAUSSIAN FWHM, i.e. 2.3548*DISP)
        if (key in lines):
            # Calculate FWHM
            comb_fwhm = combined_fwhm(lam_gal,np.abs(comp_dict[key]),line_list[key]["disp_res_kms"],velscale)
            int_vel_disp[key+"_FWHM"] = comb_fwhm
            # Calculate W80
            w80 = calculate_w80(lam_gal,np.abs(comp_dict[key]),line_list[key]["disp_res_kms"],velscale,line_list[key]["center"])
            int_vel_disp[key+"_W80"] = w80
            # Calculate NPIX and SNR for all lines
            eval_ind = np.where(np.abs(comp_dict[key])>_noise)[0]
            npix = len(eval_ind)
            npix_dict[key+"_NPIX"] = int(npix)
            # if len(eval_ind)>0:
            #    snr = np.nanmax(comp_dict[key][eval_ind])/np.nanmean(_noise[eval_ind])
            # else: 
            #    snr = 0
            snr = np.nanmax(np.abs(comp_dict[key]))/np.nanmean(_noise)
            snr_dict[key+"_SNR"] = snr

        # For combined lines ONLY, calculate integrated dispersions and velocity 
        if (key in combined_line_list):
            # Calculate velocity scale centered on line
            # vel = np.arange(len(lam_gal))*velscale - interp_ftn(line_list[key]["center"])
            vel = np.arange(len(lam_gal))*velscale - blob_pars[key+"_LINE_VEL"]
            full_profile = comp_dict[key]
            # Normalized line profile
            norm_profile = full_profile/np.sum(full_profile)
            # Calculate integrated velocity in pixels units
            v_int = np.trapz(vel*norm_profile,vel)/np.trapz(norm_profile,vel)
            # Calculate integrated dispersion and correct for instrumental dispersion
            d_int = np.sqrt(np.trapz(vel**2*norm_profile,vel)/np.trapz(norm_profile,vel) - (v_int**2))
            # d_int = np.sqrt(d_int**2 - (line_list[key]["disp_res_kms"])**2)
            if ~np.isfinite(d_int): d_int = 0.0
            if ~np.isfinite(v_int): v_int = 0.0
            int_vel_disp[key+"_DISP"] = d_int
            int_vel_disp[key+"_VOFF"] = v_int

    
    # Continuum fluxes (to obtain continuum luminosities)
    cont_fluxes = {}
    #
    if (lam_gal[0]<1350) & (lam_gal[-1]>1350):
        cont_fluxes["F_CONT_TOT_1350"]  = total_cont[blob_pars["INDEX_1350"]]
        cont_fluxes["F_CONT_AGN_1350"]  = agn_cont[blob_pars["INDEX_1350"]]
        cont_fluxes["F_CONT_HOST_1350"] = host_cont[blob_pars["INDEX_1350"]]
    if (lam_gal[0]<3000) & (lam_gal[-1]>3000):
        cont_fluxes["F_CONT_TOT_3000"]  = total_cont[blob_pars["INDEX_3000"]]
        cont_fluxes["F_CONT_AGN_3000"]  = agn_cont[blob_pars["INDEX_3000"]]
        cont_fluxes["F_CONT_HOST_3000"] = host_cont[blob_pars["INDEX_3000"]]
    if (lam_gal[0]<5100) & (lam_gal[-1]>5100):
        cont_fluxes["F_CONT_TOT_5100"]  = total_cont[blob_pars["INDEX_5100"]]
        cont_fluxes["F_CONT_AGN_5100"]  = agn_cont[blob_pars["INDEX_5100"]]
        cont_fluxes["F_CONT_HOST_5100"] = host_cont[blob_pars["INDEX_5100"]]
    if (lam_gal[0]<4000) & (lam_gal[-1]>4000):
        cont_fluxes["HOST_FRAC_4000"] = host_cont[blob_pars["INDEX_4000"]]/total_cont[blob_pars["INDEX_4000"]]
        cont_fluxes["AGN_FRAC_4000"]  = agn_cont[blob_pars["INDEX_4000"]]/total_cont[blob_pars["INDEX_4000"]]
    if (lam_gal[0]<7000) & (lam_gal[-1]>7000):
        cont_fluxes["HOST_FRAC_7000"] = host_cont[blob_pars["INDEX_7000"]]/total_cont[blob_pars["INDEX_7000"]]
        cont_fluxes["AGN_FRAC_7000"]  = agn_cont[blob_pars["INDEX_7000"]]/total_cont[blob_pars["INDEX_7000"]]
    #      

    # compute a total chi-squared and r-squared
    # fit_quality["R_SQUARED"] = 1-(np.sum((comp_dict["DATA"][fit_mask]-comp_dict["MODEL"][fit_mask])**2/np.sum(comp_dict["DATA"][fit_mask]**2)))
    fit_quality["R_SQUARED"] = badass_test_suite.r_squared(copy.deepcopy(comp_dict["DATA"]),copy.deepcopy(comp_dict["MODEL"]))

    # print(r_squared)
    #
    # nu = len(comp_dict["DATA"])-len(p)
    # fit_quality["RCHI_SQUARED"] = (np.sum(((comp_dict["DATA"][fit_mask]-comp_dict["MODEL"][fit_mask])**2)/((noise2[fit_mask])),axis=0))/nu
    fit_quality["RCHI_SQUARED"] = badass_test_suite.r_chi_squared(copy.deepcopy(comp_dict["DATA"]),copy.deepcopy(comp_dict["MODEL"]),copy.deepcopy(comp_dict["NOISE"]),len(p))


    return fluxes, eqwidths, cont_fluxes, {**int_vel_disp, **npix_dict, **snr_dict, **fit_quality}


def simple_power_law(x,amp,alpha):
    """
    Simple power-low function to model
    the AGN continuum (Calderone et al. 2017).

    Parameters
    ----------
    x	 : array_like
            wavelength vector (angstroms)
    amp   : float 
            continuum amplitude (flux density units)
    alpha : float
            power-law slope

    Returns
    ----------
    C	 : array
            AGN continuum model the same length as x
    """
    # This works
    xb = np.max(x)-(0.5*(np.max(x)-np.min(x))) # take to be half of the wavelength range
    C = amp*(x/xb)**alpha # un-normalized
    return C

##################################################################################

#### Smoothly-Broken Power-Law Template ##########################################

def broken_power_law(x, amp, x_break, alpha_1, alpha_2, delta):
    """
    Smoothly-broken power law continuum model; for use 
    when there is sufficient coverage in near-UV.
    (See https://docs.astropy.org/en/stable/api/astropy.modeling.
     powerlaws.SmoothlyBrokenPowerLaw1D.html#astropy.modeling.powerlaws.
     SmoothlyBrokenPowerLaw1D)

    Parameters
    ----------
    x		: array_like
              wavelength vector (angstroms)
    amp	 : float [0,max]
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
    C	 : array
            AGN continuum model the same length as x
    """

    C = amp * (x/x_break)**(alpha_1) * (0.5*(1.0+(x/x_break)**(1.0/delta)))**((alpha_2-alpha_1)*delta)

    return C


def run_emcee(pos,ndim,nwalkers,run_dir,lnprob_args,init_params,param_names,
              auto_stop,conv_type,min_samp,ncor_times,autocorr_tol,write_iter,write_thresh,burn_in,min_iter,max_iter,
              verbose=True):
    """
    Runs MCMC using emcee on all final parameters and checks for autocorrelation convergence 
    every write_iter iterations.
    """
    # Keep original burn_in and max_iter to reset convergence if jumps out of convergence
    orig_burn_in  = burn_in
    orig_max_iter = max_iter
    # Sorted parameter names
    param_names = np.array(param_names)
    i_sort = np.argsort(param_names) # this array gives the ordered indices of parameter names (alphabetical)
    # Create MCMC_chain.csv if it doesn't exist
    chain_file = run_dir.joinpath('log', 'MCMC_chain.csv')
    if not chain_file.exists():
        with chain_file.open(mode='w') as f:
            param_string = ', '.join(str(e) for e in param_names)
            f.write('# iter, ' + param_string) # Write initial parameters
            best_str = ', '.join(str(e) for e in init_params)
            f.write('\n 0, '+best_str)

    # initialize the sampler
    dtype = [('fluxes',dict),('eqwidths',dict),('cont_fluxes',dict),("int_vel_disp",dict),('log_like',float)] # mcmc blobs
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=lnprob_args,blobs_dtype=dtype) # blobs_dtype=dtype added for Python2 -> Python3

    start_time = time.time() # start timer

    write_log((ndim,nwalkers,auto_stop,conv_type,burn_in,write_iter,write_thresh,min_iter,max_iter),'emcee_options',run_dir)

    # Initialize stuff for autocorrelation analysis
    if (auto_stop==True):
        autocorr_times_all = [] # storage array for autocorrelation times
        autocorr_tols_all  = [] # storage array for autocorrelation tolerances
        old_tau = np.full(len(param_names),np.inf)
        min_samp	 = min_samp # minimum iterations to use past convergence
        ncor_times   = ncor_times # multiplicative tolerance; number of correlation times before which we stop sampling	
        autocorr_tol = autocorr_tol	
        stop_iter	= max_iter # stopping iteration; changes once convergence is reached
        converged  = False
        # write_log((min_samp,autocorr_tol,ncor_times,conv_type),'autocorr_options',run_dir)

    # If one provides a list of parameters for autocorrelation, it needs to be in the 
    # form of a tuple.  If one only provides one paraemeter, it needs to be converted to a tuple:
    if (auto_stop==True) and (conv_type != 'all') and (conv_type != 'mean') and (conv_type != 'median'):
        if not isinstance(conv_type, tuple):
            conv_type = (conv_type,) #

    # Check auto_stop convergence type:
    if (auto_stop==True) and (isinstance(conv_type,tuple)==True) :
        if all(elem in param_names  for elem in conv_type)==True:
            if (verbose):
                print('\n Only considering convergence of following parameters: ')
                for c in conv_type:	
                    print('		  %s' % c)
                pass
        # check to see that all param_names are in conv_type, if not, remove them 
        # from conv_type
        else:
            try:
                conv_type_list = list(conv_type)
                for c in conv_type:
                    if c not in param_names:
                        conv_type_list.remove(c)
                conv_type = tuple(conv_type_list)
                if all(elem in conv_type  for elem in param_names)==True:
                    if (verbose):
                        print('\n Only considering convergence of following parameters: ')
                        for c in conv_type:	
                            print('		  %s' % c)
                        pass
                    else:
                        if (verbose):
                            print('\n One of more parameters in conv_type is not a valid parameter. Defaulting to median convergence type../.\n')
                        conv_type='median'

            except:
                print('\n One of more parameters in conv_type is not a valid parameter. Defaulting to median convergence type../.\n')
                conv_type='median'

    if (auto_stop==True):
        write_log((min_samp,autocorr_tol,ncor_times,conv_type),'autocorr_options',run_dir)
    # Run emcee
    for k, result in enumerate(sampler.sample(pos, iterations=max_iter)):
            
        if ((k+1) % write_iter == 0) and verbose:
            print("MCMC iteration: %d" % (k+1))
        best = [] # For storing current chain positions (median of parameter values at write_iter iterations)
        if ((k+1) % write_iter == 0) and ((k+1)>=write_thresh): # Write every [write_iter] iteration
            # Chain location for each parameter
            # Median of last 100 positions for each walker.
            nwalkers = np.shape(sampler.chain)[0]
            npar = np.shape(sampler.chain)[2]
            
            sampler_chain = sampler.chain[:,:k+1,:]
            new_sampler_chain = []
            for i in range(0,np.shape(sampler_chain)[2],1):
                pflat = sampler_chain[:,:,i] # flattened along parameter
                flat  = np.concatenate(np.stack(pflat,axis=1),axis=0)
                new_sampler_chain.append(flat)
            # best = []
            for pp in range(0,npar,1):
                data = new_sampler_chain[pp][-int(nwalkers*write_iter):]
                med = np.nanmedian(data)
                best.append(med)
            # write to file
            with run_dir.joinpath('log', 'MCMC_chain.csv').open(mode='a') as f:
                best_str = ', '.join(str(e) for e in best)
                f.write('\n'+str(k+1)+', '+best_str)
        # Checking autocorrelation times for convergence
        if ((k+1) % write_iter == 0) and ((k+1)>=min_iter) and ((k+1)>=write_thresh) and (auto_stop==True):
            # Autocorrelation analysis of chain to determine convergence; the minimum autocorrelation time is 1.0, which results when a time cannot be accurately calculated.
            tau = autocorr_convergence(sampler.chain) # Calculate autocorrelation times for each parameter

            autocorr_times_all.append(tau) # append tau to storage array
            # Calculate tolerances
            tol = (np.abs(tau-old_tau)/old_tau) * 100.0
            autocorr_tols_all.append(tol) # append tol to storage array
            # If convergence for mean autocorrelation time 
            if (auto_stop==True) & (conv_type == 'mean'):
                par_conv = [] # converged parameter indices
                par_not_conv  = [] # non-converged parameter indices
                for x in range(0,len(param_names),1):
                    if (round(tau[x],1)>1.0):# & (0.0<round(tol[x],1)<autocorr_tol):
                        par_conv.append(x) # Append index of parameter for which an autocorrelation time can be calculated; we use these to calculate the mean
                    else: par_not_conv.append(x)
                # Calculate mean of parameters for which an autocorrelation time could be calculated
                par_conv = np.array(par_conv) # Explicitly convert to array
                par_not_conv = np.array(par_not_conv) # Explicitly convert to array

                if (par_conv.size == 0) and (stop_iter == orig_max_iter):
                    if verbose:
                        print('\nIteration = %d' % (k+1))
                        print('-------------------------------------------------------------------------------')
                        print('- Not enough iterations for any autocorrelation times!')
                elif ( (par_conv.size > 0) and (k+1)>(np.nanmean(tau[par_conv]) * ncor_times) and (np.nanmean(tol[par_conv])<autocorr_tol) and (stop_iter == max_iter) ):
                    if verbose:
                        print('\n ---------------------------------------------')
                        print(' | Converged at %d iterations.			  | ' % (k+1))
                        print(' | Performing %d iterations of sampling... | ' % min_samp )
                        print(' | Sampling will finish at %d iterations.  | ' % ((k+1)+min_samp) )
                        print(' ---------------------------------------------')
                    burn_in = (k+1)
                    stop_iter = (k+1)+min_samp
                    conv_tau = tau
                    converged = True
                elif ((par_conv.size == 0) or ( (k+1)<(np.nanmean(tau[par_conv]) * ncor_times)) or (np.nanmean(tol[par_conv])>autocorr_tol)) and (stop_iter < orig_max_iter):
                    if verbose:
                        print('\nIteration = %d' % (k+1))
                        print('-------------------------------------------------------------------------------')
                        print('- Jumped out of convergence! Resetting convergence criteria...')
                        # Reset convergence criteria
                        print('- Resetting burn_in = %d' % orig_burn_in)
                        print('- Resetting max_iter = %d' % orig_max_iter)
                    burn_in = orig_burn_in
                    stop_iter = orig_max_iter
                    converged = False

                if (par_conv.size>0):
                    pnames_sorted = param_names[i_sort]
                    tau_sorted	= tau[i_sort]
                    tol_sorted	= tol[i_sort]
                    best_sorted   = np.array(best)[i_sort]
                    if verbose:
                        print('{0:<30}{1:<40}{2:<30}'.format('\nIteration = %d' % (k+1),'%d x Mean Autocorr. Time = %0.2f' % (ncor_times,np.nanmean(tau[par_conv]) * ncor_times),'Mean Tolerance = %0.2f' % np.nanmean(tol[par_conv])))
                        print('--------------------------------------------------------------------------------------------------------')
                        print('{0:<30}{1:<20}{2:<20}{3:<20}{4:<20}'.format('Parameter','Current Value','Autocorr. Time','Tolerance','Converged?'))
                        print('--------------------------------------------------------------------------------------------------------')
                        for i in range(0,len(pnames_sorted),1):
                            if (((k+1)>tau_sorted[i]*ncor_times) and (tol_sorted[i]<autocorr_tol) and (tau_sorted[i]>1.0) ):
                                conv_bool = 'True'
                            else: conv_bool = 'False'
                            if (round(tau_sorted[i],1)>1.0):# & (tol[i]<autocorr_tol):
                                print('{0:<30}{1:<20.4f}{2:<20.4f}{3:<20.4f}{4:<20}'.format(pnames_sorted[i],best_sorted[i],tau_sorted[i],tol_sorted[i],conv_bool))
                            else: 
                                print('{0:<30}{1:<20.4f}{2:<20}{3:<20}{4:<20}'.format(pnames_sorted[i],best_sorted[i],' -------- ',' -------- ',' -------- '))
                        print('--------------------------------------------------------------------------------------------------------')

            # If convergence for median autocorrelation time 
            if (auto_stop==True) & (conv_type == 'median'):
                par_conv = [] # converged parameter indices
                par_not_conv  = [] # non-converged parameter indices
                for x in range(0,len(param_names),1):
                    if (round(tau[x],1)>1.0):# & (tol[x]<autocorr_tol):
                        par_conv.append(x) # Append index of parameter for which an autocorrelation time can be calculated; we use these to calculate the mean
                    else: par_not_conv.append(x)
                # Calculate mean of parameters for which an autocorrelation time could be calculated
                par_conv = np.array(par_conv) # Explicitly convert to array
                par_not_conv = np.array(par_not_conv) # Explicitly convert to array

                if (par_conv.size == 0) and (stop_iter == orig_max_iter):
                    if verbose:
                        print('\nIteration = %d' % (k+1))
                        print('-------------------------------------------------------------------------------')
                        print('- Not enough iterations for any autocorrelation times!')
                elif ( (par_conv.size > 0) and (k+1)>(np.nanmedian(tau[par_conv]) * ncor_times) and (np.nanmedian(tol[par_conv])<autocorr_tol) and (stop_iter == max_iter) ):
                    if verbose:
                        print('\n ---------------------------------------------')
                        print(' | Converged at %d iterations.			  |' % (k+1))
                        print(' | Performing %d iterations of sampling... |' % min_samp )
                        print(' | Sampling will finish at %d iterations.  |' % ((k+1)+min_samp) )
                        print(' ---------------------------------------------')
                    burn_in = (k+1)
                    stop_iter = (k+1)+min_samp
                    conv_tau = tau
                    converged = True
                elif ((par_conv.size == 0) or ( (k+1)<(np.nanmedian(tau[par_conv]) * ncor_times)) or (np.nanmedian(tol[par_conv])>autocorr_tol)) and (stop_iter < orig_max_iter):
                    if verbose:
                        print('\nIteration = %d' % (k+1))
                        print('-------------------------------------------------------------------------------')
                        print('- Jumped out of convergence! Resetting convergence criteria...')
                        # Reset convergence criteria
                        print('- Resetting burn_in = %d' % orig_burn_in)
                        print('- Resetting max_iter = %d' % orig_max_iter)
                    burn_in = orig_burn_in
                    stop_iter = orig_max_iter
                    converged = False

                if (par_conv.size>0):
                    pnames_sorted = param_names[i_sort]
                    tau_sorted	= tau[i_sort]
                    tol_sorted	= tol[i_sort]
                    best_sorted   = np.array(best)[i_sort]
                    if verbose:
                        print('{0:<30}{1:<40}{2:<30}'.format('\nIteration = %d' % (k+1),'%d x Median Autocorr. Time = %0.2f' % (ncor_times,np.nanmedian(tau[par_conv]) * ncor_times),'Med. Tolerance = %0.2f' % np.nanmedian(tol[par_conv])))
                        print('--------------------------------------------------------------------------------------------------------')
                        print('{0:<30}{1:<20}{2:<20}{3:<20}{4:<20}'.format('Parameter','Current Value','Autocorr. Time','Tolerance','Converged?'))
                        print('--------------------------------------------------------------------------------------------------------')
                        for i in range(0,len(pnames_sorted),1):
                            if (((k+1)>tau_sorted[i]*ncor_times) and (tol_sorted[i]<autocorr_tol) and (tau_sorted[i]>1.0)):
                                conv_bool = 'True'
                            else: conv_bool = 'False'
                            if (round(tau_sorted[i],1)>1.0):# & (tol[i]<autocorr_tol):	
                                print('{0:<30}{1:<20.4f}{2:<20.4f}{3:<20.4f}{4:<20}'.format(pnames_sorted[i],best_sorted[i],tau_sorted[i],tol_sorted[i],conv_bool))
                            else: 
                                print('{0:<30}{1:<20.4f}{2:<20}{3:<20}{4:<20}'.format(pnames_sorted[i],best_sorted[i],' -------- ',' -------- ',' -------- '))
                        print('--------------------------------------------------------------------------------------------------------')
                
            # If convergence for ALL autocorrelation times 
            if (auto_stop==True) & (conv_type == 'all'):
                if ( all( (x==1.0) for x in tau) ) and (stop_iter == orig_max_iter):
                    if verbose:
                        print('\nIteration = %d' % (k+1))
                        print('-------------------------------------------------------------------------------')
                        print('- Not enough iterations for any autocorrelation times!')
                elif all( ((k+1)>(x * ncor_times)) for x in tau) and all( (x>1.0) for x in tau) and all(y<autocorr_tol for y in tol) and (stop_iter == max_iter):
                    if verbose:
                        print('\n ---------------------------------------------')
                        print(' | Converged at %d iterations.			  | ' % (k+1))
                        print(' | Performing %d iterations of sampling... | ' % min_samp )
                        print(' | Sampling will finish at %d iterations.  | ' % ((k+1)+min_samp) )
                        print(' ---------------------------------------------')
                    burn_in = (k+1)
                    stop_iter = (k+1)+min_samp
                    conv_tau = tau
                    converged = True
                elif (any( ((k+1)<(x * ncor_times)) for x in tau) or any( (x==1.0) for x in tau) or any(y>autocorr_tol for y in tol)) and (stop_iter < orig_max_iter):
                    if verbose:
                        print('\n Iteration = %d' % (k+1))
                        print('-------------------------------------------------------------------------------')
                        print('- Jumped out of convergence! Resetting convergence criteria...')
                        # Reset convergence criteria
                        print('- Resetting burn_in = %d' % orig_burn_in)
                        print('- Resetting max_iter = %d' % orig_max_iter)
                    burn_in = orig_burn_in
                    stop_iter = orig_max_iter
                    converged = False
                if 1:
                    pnames_sorted = param_names[i_sort]
                    tau_sorted	= tau[i_sort]
                    tol_sorted	= tol[i_sort]
                    best_sorted   = np.array(best)[i_sort]
                    if verbose:
                        print('{0:<30}'.format('\nIteration = %d' % (k+1)))
                        print('--------------------------------------------------------------------------------------------------------------------------------------------')
                        print('{0:<30}{1:<20}{2:<20}{3:<25}{4:<20}{5:<20}'.format('Parameter','Current Value','Autocorr. Time','Target Autocorr. Time','Tolerance','Converged?'))
                        print('--------------------------------------------------------------------------------------------------------------------------------------------')
                        for i in range(0,len(pnames_sorted),1):
                            if (((k+1)>tau_sorted[i]*ncor_times) and (tol_sorted[i]<autocorr_tol) and (tau_sorted[i]>1.0) ):
                                conv_bool = 'True'
                            else: conv_bool = 'False'
                            if (round(tau_sorted[i],1)>1.0):# & (tol[i]<autocorr_tol):
                                print('{0:<30}{1:<20.4f}{2:<20.4f}{3:<25.4f}{4:<20.4f}{5:<20}'.format(pnames_sorted[i],best_sorted[i],tau_sorted[i],tau_sorted[i]*ncor_times,tol_sorted[i],str(conv_bool)))
                            else: 
                                print('{0:<30}{1:<20.4f}{2:<20}{3:<25}{4:<20}{5:<20}'.format(pnames_sorted[i],best_sorted[i],' -------- ',' -------- ',' -------- ',' -------- '))
                        print('--------------------------------------------------------------------------------------------------------------------------------------------')

            # If convergence for a specific set of parameters
            if (auto_stop==True) & (isinstance(conv_type,tuple)==True):
                # Get indices of parameters for which we want to converge; these will be the only ones we care about
                par_ind = np.array([i for i, item in enumerate(param_names) if item in set(conv_type)])
                # Get list of parameters, autocorrelation times, and tolerances for the ones we care about
                param_interest   = param_names[par_ind]
                tau_interest = tau[par_ind]
                tol_interest = tol[par_ind]
                best_interest = np.array(best)[par_ind]
                # New sort for selected parameters
                i_sort = np.argsort(param_interest) # this array gives the ordered indices of parameter names (alphabetical)
                if ( all( (x==1.0) for x in tau_interest) ) and (stop_iter == orig_max_iter):
                    if verbose:
                        print('\nIteration = %d' % (k+1))
                        print('-------------------------------------------------------------------------------')
                        print('- Not enough iterations for any autocorrelation times!')
                elif all( ((k+1)>(x * ncor_times)) for x in tau_interest) and all( (x>1.0) for x in tau_interest) and all(y<autocorr_tol for y in tol_interest) and (stop_iter == max_iter):
                    if verbose:
                        print('\n ---------------------------------------------')
                        print(' | Converged at %d iterations.			  | ' % (k+1))
                        print(' | Performing %d iterations of sampling... | ' % min_samp )
                        print(' | Sampling will finish at %d iterations.  | ' % ((k+1)+min_samp) )
                        print(' ---------------------------------------------')
                    burn_in = (k+1)
                    stop_iter = (k+1)+min_samp
                    conv_tau = tau
                    converged = True
                elif (any( ((k+1)<(x * ncor_times)) for x in tau_interest) or any( (x==1.0) for x in tau_interest) or any(y>autocorr_tol for y in tol_interest)) and (stop_iter < orig_max_iter):
                    if verbose:
                        print('\n Iteration = %d' % (k+1))
                        print('-------------------------------------------------------------------------------')
                        print('- Jumped out of convergence! Resetting convergence criteria...')
                        # Reset convergence criteria
                        print('- Resetting burn_in = %d' % orig_burn_in)
                        print('- Resetting max_iter = %d' % orig_max_iter)
                    burn_in = orig_burn_in
                    stop_iter = orig_max_iter
                    converged = False
                if 1:
                    pnames_sorted = param_interest[i_sort]
                    tau_sorted	= tau_interest[i_sort]
                    tol_sorted	= tol_interest[i_sort]
                    best_sorted   = np.array(best_interest)[i_sort]
                    if verbose:
                        print('{0:<30}'.format('\nIteration = %d' % (k+1)))
                        print('--------------------------------------------------------------------------------------------------------------------------------------------')
                        print('{0:<30}{1:<20}{2:<20}{3:<25}{4:<20}{5:<20}'.format('Parameter','Current Value','Autocorr. Time','Target Autocorr. Time','Tolerance','Converged?'))
                        print('--------------------------------------------------------------------------------------------------------------------------------------------')
                        for i in range(0,len(pnames_sorted),1):
                            if (((k+1)>tau_sorted[i]*ncor_times) and (tol_sorted[i]<autocorr_tol) and (tau_sorted[i]>1.0) ):
                                conv_bool = 'True'
                            else: conv_bool = 'False'
                            if (round(tau_sorted[i],1)>1.0):# & (tol[i]<autocorr_tol):
                                print('{0:<30}{1:<20.4f}{2:<20.4f}{3:<25.4f}{4:<20.4f}{5:<20}'.format(pnames_sorted[i],best_sorted[i],tau_sorted[i],tau_sorted[i]*ncor_times,tol_sorted[i],str(conv_bool)))
                            else: 
                                print('{0:<30}{1:<20.4f}{2:<20}{3:<25}{4:<20}{5:<20}'.format(pnames_sorted[i],best_sorted[i],' -------- ',' -------- ',' -------- ',' -------- '))
                        print('--------------------------------------------------------------------------------------------------------------------------------------------')

            # Stop
            if ((k+1) == stop_iter):
                break

            old_tau = tau	

        # If auto_stop=False, simply print out the parameters and their best values at that iteration
        if ((k+1) % write_iter == 0) and ((k+1)>=min_iter) and ((k+1)>=write_thresh) and (auto_stop==False):
            pnames_sorted = param_names[i_sort]
            best_sorted   = np.array(best)[i_sort]
            if verbose:
                print('{0:<30}'.format('\nIteration = %d' % (k+1)))
                print('------------------------------------------------')
                print('{0:<30}{1:<20}'.format('Parameter','Current Value'))
                print('------------------------------------------------')
                for i in range(0,len(pnames_sorted),1):
                        print('{0:<30}{1:<20.4f}'.format(pnames_sorted[i],best_sorted[i]))
                print('------------------------------------------------')

    elap_time = (time.time() - start_time)	   
    run_time = time_convert(elap_time)
    if verbose:
        print("\n emcee Runtime = %s. \n" % (run_time))

    # Write to log file
    if (auto_stop==True):
        # Write autocorrelation chain to log 
        # np.save(run_dir+'/log/autocorr_times_all',autocorr_times_all)
        # np.save(run_dir+'/log/autocorr_tols_all',autocorr_tols_all)
        # Create a dictionary with parameter names as keys, and contains
        # the autocorrelation times and tolerances for each parameter
        autocorr_times_all = np.stack(autocorr_times_all,axis=1)
        autocorr_tols_all  = np.stack(autocorr_tols_all,axis=1)
        autocorr_dict = {}
        for k in range(0,len(param_names),1):
            if (np.shape(autocorr_times_all)[0] > 1):
                autocorr_dict[param_names[k]] = {'tau':autocorr_times_all[k],
                                                  'tol':autocorr_tols_all[k]} 
        np.save(run_dir.joinpath('log', 'autocorr_dict.npy'),autocorr_dict)

        if (converged == True):
            write_log((burn_in,stop_iter,param_names,conv_tau,autocorr_tol,tol,ncor_times),'autocorr_results',run_dir)
        elif (converged == False):
            unconv_tol = (np.abs((old_tau) - (tau)) / (tau))
            write_log((burn_in,stop_iter,param_names,tau,autocorr_tol,unconv_tol,ncor_times),'autocorr_results',run_dir)
    write_log(run_time,'emcee_time',run_dir) 

    # Remove excess zeros from sampler chain if emcee converged on a solution
    # in fewer iterations than max_iter
    # Remove zeros from all chains
    a = [] # the zero-trimmed sampler.chain
    for p in range(0,np.shape(sampler.chain)[2],1):
        c = sampler.chain[:,:,p]
        c_trimmed = [np.delete(c[i,:],np.argwhere(c[i,:]==0)) for i in range(np.shape(c)[0])] # delete any occurence of zero 
        a.append(c_trimmed)
    a = np.swapaxes(a,1,0) 
    a = np.swapaxes(a,2,1)

    # Extract metadata blobs
    blobs		   = sampler.get_blobs()
    flux_blob	   = blobs["fluxes"]
    eqwidth_blob   = blobs["eqwidths"]
    cont_flux_blob = blobs["cont_fluxes"]
    int_vel_disp_blob  = blobs["int_vel_disp"]
    log_like_blob  = blobs["log_like"]

    return a, burn_in, flux_blob, eqwidth_blob, cont_flux_blob, int_vel_disp_blob, log_like_blob


##################################################################################

# Autocorrelation analysis 
##################################################################################

def autocorr_convergence(sampler_chain, c=5.0):
    """
    Estimates the autocorrelation times using the 
    methods outlined on the Autocorrelation page 
    on the emcee website:
    https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
    """
    
    nwalker = np.shape(sampler_chain)[0] # Number of walkers
    niter   = np.shape(sampler_chain)[1] # Number of iterations
    npar    = np.shape(sampler_chain)[2] # Number of parameters
        
    tau_est = np.empty(npar)
    # Iterate over all parameters
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
    """
    Automated windowing procedure following Sokal (1989)
    """
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

##################################################################################


# Plotting Routines
##################################################################################

def gauss_kde(xs,data,h):
    """
    Gaussian kernel density estimation.
    """
    def gauss_kernel(x):
        return (1./np.sqrt(2.*np.pi)) * np.exp(-x**2/2)

    kde = np.sum((1./h) * gauss_kernel((xs.reshape(len(xs),1)-data)/h), axis=1)
    kde = kde/np.trapz(kde,xs)# normalize
    return kde

def kde_bandwidth(data):
    """
    Silverman bandwidth estimation for kernel density estimation.
    """
    return (4./(3.*len(data)))**(1./5.) * np.nanstd(data)

def compute_HDI(trace, mass_frac) :
    """
    Returns highest probability density region given by
    a set of samples.
    
    Source: http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2015/tutorials/l06_credible_regions.html
    
    Parameters
    ----------
    trace : array
        1D array of MCMC samples for a single variable
    mass_frac : float with 0 < mass_frac <= 1
        The fraction of the probability to be included in
        the HPD.  For example, `massfrac` = 0.95 gives a
        95% HPD.
        
    Returns
    -------
    output : array, shape (2,)
        The bounds of the HPD
    """
    # Get sorted list
    d = np.sort(np.copy(trace))

    # Number of total samples taken
    n = len(trace)
    
    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)
    
    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n-n_samples]
    
    # Pick out minimal interval
    min_int = np.argmin(int_width)
    
    # Return interval
    return np.array([d[min_int], d[min_int+n_samples]])

def posterior_plots(key,flat,chain,burn_in,xs,kde,
                    low_68,upp_68,low_95,upp_95,post_mean,post_std,post_med,post_mad,
                    run_dir
                    ):
    """
    Plot posterior distributions and chains from MCMC.
    """
    # Initialize figures and axes
    # Make an updating plot of the chain
    fig = plt.figure(figsize=(10,8)) 
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
    ax1  = plt.subplot(gs[0,0])
    ax2  = plt.subplot(gs[0,1])
    ax3  = plt.subplot(gs[1,0:2])

    # Histogram; 'Doane' binning produces the best results from tests.
    n, bins, patches = ax1.hist(flat, bins='doane', histtype="bar" , density=True, facecolor="#4200a6", alpha=1,zorder=10)
    # Plot 1: Histogram plots
    ax1.axvline(post_med	,linewidth=0.5,linestyle="-",color='xkcd:bright aqua',alpha=1.00,zorder=20,label=r'$p(\theta|x)_{\rm{med}}$')
    #
    ax1.axvline(post_med-low_68,linewidth=0.5,linestyle="--" ,color='xkcd:bright aqua',alpha=1.00,zorder=20,label=r'$\textrm{68\% conf.}$')
    ax1.axvline(post_med+upp_68,linewidth=0.5,linestyle="--" ,color='xkcd:bright aqua',alpha=1.00,zorder=20)
    #
    ax1.axvline(post_med-low_95,linewidth=0.5,linestyle=":" ,color='xkcd:bright aqua',alpha=1.00,zorder=20,label=r'$\textrm{95\% conf.}$')
    ax1.axvline(post_med+upp_95,linewidth=0.5,linestyle=":" ,color='xkcd:bright aqua',alpha=1.00,zorder=20)
    #
    ax1.plot(xs,kde ,linewidth=0.5,linestyle="-" ,color="xkcd:bright pink",alpha=1.00,zorder=15,label="KDE")
    ax1.plot(xs,kde ,linewidth=3.0,linestyle="-" ,color="xkcd:bright pink",alpha=0.50,zorder=15)
    ax1.plot(xs,kde ,linewidth=6.0,linestyle="-" ,color="xkcd:bright pink",alpha=0.20,zorder=15)
    #
    ax1.grid(visible=True,which="major",axis="both",alpha=0.15,color="xkcd:bright pink",linewidth=0.5,zorder=0)
    # ax1.plot(xvec,yvec,color='white')
    ax1.set_xlabel(r'%s' % key,fontsize=12)
    ax1.set_ylabel(r'$p$(%s)' % key,fontsize=12)
    ax1.legend(loc="best",fontsize=6)
    
    # Plot 2: best fit values
    values = [post_med,low_68,upp_68,low_95,upp_95,post_mean,post_std,post_med,post_mad]
    labels = [r"$p(\theta|x)_{\rm{med}}$",
        r"$\rm{CI\;68\%\;low}$",r"$\rm{CI\;68\%\;upp}$",
        r"$\rm{CI\;95\%\;low}$",r"$\rm{CI\;95\%\;upp}$",
        r"$\rm{Mean}$",r"$\rm{Std.\;Dev.}$",
        r"$\rm{Median}$",r"$\rm{Med. Abs. Dev.}$"]
    start, step = 1, 0.12
    vspace = np.linspace(start,1-len(labels)*step,len(labels),endpoint=False)
    # Plot 2: best fit values
    for i in range(len(labels)):
        ax2.annotate('{0:>30}{1:<2}{2:<30.3f}'.format(labels[i],r"$\qquad=\qquad$",values[i]), 
                    xy=(0.5, vspace[i]),  xycoords='axes fraction',
                    xytext=(0.95, vspace[i]), textcoords='axes fraction',
                    horizontalalignment='right', verticalalignment='top', 
                    fontsize=10)
    ax2.axis('off')

    # Plot 3: Chain plot
    for w in range(0,np.shape(chain)[0],1):
        ax3.plot(range(np.shape(chain)[1]),chain[w,:],color='white',linewidth=0.5,alpha=0.5,zorder=0)
    # Calculate median and median absolute deviation of walkers at each iteration; we have depreciated
    # the average and standard deviation because they do not behave well for outlier walkers, which
    # also don't agree with histograms.
    c_med = np.nanmedian(chain,axis=0)
    c_madstd = mad_std(chain)
    ax3.plot(range(np.shape(chain)[1]),c_med,color='xkcd:bright pink',alpha=1.,linewidth=2.0,label='Median',zorder=10)
    ax3.fill_between(range(np.shape(chain)[1]),c_med+c_madstd,c_med-c_madstd,color='#4200a6',alpha=0.5,linewidth=1.5,label='Median Absolute Dev.',zorder=5)
    ax3.axvline(burn_in,linestyle='--',linewidth=0.5,color='xkcd:bright aqua',label='burn-in = %d' % burn_in,zorder=20)
    ax3.grid(visible=True,which="major",axis="both",alpha=0.15,color="xkcd:bright pink",linewidth=0.5,zorder=0)
    ax3.set_xlim(0,np.shape(chain)[1])
    ax3.set_xlabel('$N_\mathrm{iter}$',fontsize=12)
    ax3.set_ylabel(r'%s' % key,fontsize=12)
    ax3.legend(loc='upper left')
    
    # Save the figure
    histo_dir = run_dir.joinpath('histogram_plots')
    histo_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(histo_dir.joinpath('%s_MCMC.png' % (key)), bbox_inches="tight",dpi=300)

    # Close plot window
    fig.clear()
    plt.close()

    return

def param_plots(param_dict,fit_norm,burn_in,run_dir,plot_param_hist=True,verbose=True):
    """
    Generates best-fit values, uncertainties, and plots for 
    free parameters from MCMC sample chains.
    """
    #
    if verbose:
        print("\n Generating model parameter distributions...\n")

    for key in param_dict:
        #
        if verbose:
            print('		  %s' % key)
        chain = param_dict[key]['chain'] # shape = (nwalkers,niter)
        chain[~np.isfinite(chain)] = 0
        # Burned-in + Flattened (along walker axis) chain
        # If burn_in is larger than the size of the chain, then 
        # take 50% of the chain length instead.

        # Rescale amplitudes
        if key[-4:]=="_AMP":
            chain*=fit_norm

        if (burn_in >= np.shape(chain)[1]):
            burn_in = int(0.5*np.shape(chain)[1])
        # Flatten the chains
        flat = chain[:,burn_in:]
        # flat = flat.flat
        flat = flat.flatten()

        # 
        if len(flat) > 0:

            # Histogram; 'Doane' binning produces the best results from tests.
            hist, bin_edges = np.histogram(flat, bins='doane', density=False)

            # Generate pseudo-data on the ends of the histogram; this prevents the KDE
            # from weird edge behavior.
            n_pseudo = 3 # number of pseudo-bins 
            bin_width=bin_edges[1]-bin_edges[0]
            lower_pseudo_data = np.random.uniform(low=bin_edges[0]-bin_width*n_pseudo, high=bin_edges[0], size=hist[0]*n_pseudo)
            upper_pseudo_data = np.random.uniform(low=bin_edges[-1], high=bin_edges[-1]+bin_width*n_pseudo, size=hist[-1]*n_pseudo)

            # Calculate bandwidth for KDE (Silverman method)
            h = kde_bandwidth(flat)

            # Create a subsampled grid for the KDE based on the subsampled data; by
            # default, we subsample by a factor of 10.
            xs = np.linspace(np.min(flat),np.max(flat),10*len(hist))

            # Calculate KDE
            kde = gauss_kde(xs,np.concatenate([flat,lower_pseudo_data,upper_pseudo_data]),h)
            p68 = compute_HDI(flat,0.68)
            p95 = compute_HDI(flat,0.95)

            post_max  = bin_edges[hist.argmax()] # posterior max estimated from KDE
            post_mean = np.nanmean(flat)
            post_med  = np.nanmedian(flat)
            low_68  = post_med - p68[0]
            upp_68  = p68[1] - post_med
            low_95  = post_med - p95[0]
            upp_95  = p95[1] - post_med
            post_std  = np.nanstd(flat)
            post_mad  = stats.median_abs_deviation(flat)

            # Quality flags; flag any parameter that violates parameter limits by 1.5 sigma
            flag = 0 
            if key[-4:]=="_AMP":
                if ( (post_med-1.5*low_68) <= (param_dict[key]['plim'][0]*fit_norm) ):
                    flag+=1
                if ( (post_med+1.5*upp_68) >= (param_dict[key]['plim'][1]*fit_norm) ):
                    flag+=1
                if ~np.isfinite(post_med) or ~np.isfinite(low_68) or ~np.isfinite(upp_68):
                    flag+=1
            else:
                if ( (post_med-1.5*low_68) <= (param_dict[key]['plim'][0]) ):
                    flag+=1
                if ( (post_med+1.5*upp_68) >= (param_dict[key]['plim'][1]) ):
                    flag+=1
                if ~np.isfinite(post_med) or ~np.isfinite(low_68) or ~np.isfinite(upp_68):
                    flag+=1

            param_dict[key]['par_best'] = post_med # maximum of posterior distribution
            param_dict[key]['ci_68_low']   = low_68	# lower 68% confidence interval
            param_dict[key]['ci_68_upp']   = upp_68	# upper 68% confidence interval
            param_dict[key]['ci_95_low']   = low_95	# lower 95% confidence interval
            param_dict[key]['ci_95_upp']   = upp_95	# upper 95% confidence interval
            param_dict[key]['post_max'] = post_max # maximum of posterior distribution
            param_dict[key]['mean']  = post_mean # mean of posterior distribution
            param_dict[key]['std_dev']   = post_std	# standard deviation
            param_dict[key]['median']     = post_med # median of posterior distribution
            param_dict[key]['med_abs_dev'] = post_mad	# median absolute deviation
            param_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
            param_dict[key]['flag']	       = flag 

            if (plot_param_hist==True):

                posterior_plots(key,flat,chain,burn_in,xs,kde,
                                low_68,upp_68,low_95,upp_95,post_mean,post_std,post_med,post_mad,
                                run_dir
                                )

        else:
            param_dict[key]['par_best'] = np.nan # maximum of posterior distribution
            param_dict[key]['ci_68_low']   = np.nan	# lower 68% confidence interval
            param_dict[key]['ci_68_upp']   = np.nan	# upper 68% confidence interval
            param_dict[key]['ci_95_low']   = np.nan	# lower 95% confidence interval
            param_dict[key]['ci_95_upp']   = np.nan	# upper 95% confidence interval
            param_dict[key]['post_max'] = np.nan # maximum of posterior distribution
            param_dict[key]['mean']  = np.nan # mean of posterior distribution
            param_dict[key]['std_dev']   = np.nan	# standard deviation
            param_dict[key]['median']     = np.nan # median of posterior distribution
            param_dict[key]['med_abs_dev'] = np.nan	# median absolute deviation
            param_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
            param_dict[key]['flag']	       = 1 

    return param_dict


def log_like_plot(ll_blob, burn_in, nwalkers, run_dir, plot_param_hist=True,verbose=True):
    """
    Generates best-fit values, uncertainties, and plots for 
    component fluxes from MCMC sample chains.
    """
    
    ll = ll_blob.T

    # Burned-in + Flattened (along walker axis) chain
    # If burn_in is larger than the size of the chain, then 
    # take 50% of the chain length instead.
    if (burn_in >= np.shape(ll)[1]):
        burn_in = int(0.5*np.shape(ll)[1])
        # print('\n Burn-in is larger than chain length! Using 50% of chain length for burn-in...\n')

    flat = ll[:,burn_in:]
    # flat = flat.flat
    flat = flat.flatten()

    # Old confidence interval stuff; replaced by np.quantile
    # p = np.percentile(flat, [16, 50, 84])
    # pdfmax = p[1]
    # low1   = p[1]-p[0]
    # upp1   = p[2]-p[1]

    # 
    if len(flat[np.isfinite(flat)]) > 0:

        # Histogram; 'Doane' binning produces the best results from tests.
        hist, bin_edges = np.histogram(flat, bins='doane', density=False)

        # Generate pseudo-data on the ends of the histogram; this prevents the KDE
        # from weird edge behavior.
        n_pseudo = 3 # number of pseudo-bins 
        bin_width=bin_edges[1]-bin_edges[0]
        lower_pseudo_data = np.random.uniform(low=bin_edges[0]-bin_width*n_pseudo, high=bin_edges[0], size=hist[0]*n_pseudo)
        upper_pseudo_data = np.random.uniform(low=bin_edges[-1], high=bin_edges[-1]+bin_width*n_pseudo, size=hist[-1]*n_pseudo)

        # Calculate bandwidth for KDE (Silverman method)
        h = kde_bandwidth(flat)

        # Create a subsampled grid for the KDE based on the subsampled data; by
        # default, we subsample by a factor of 10.
        xs = np.linspace(np.min(flat),np.max(flat),10*len(hist))

        # Calculate KDE
        kde = gauss_kde(xs,np.concatenate([flat,lower_pseudo_data,upper_pseudo_data]),h)
        p68 = compute_HDI(flat,0.68)
        p95 = compute_HDI(flat,0.95)

        post_max  = bin_edges[hist.argmax()] # posterior max estimated from KDE
        post_mean = np.nanmean(flat)
        post_med  = np.nanmedian(flat)
        low_68  = post_med - p68[0]
        upp_68  = p68[1] - post_med
        low_95  = post_med - p95[0]
        upp_95  = p95[1] - post_med
        post_std  = np.nanstd(flat)
        post_mad  = stats.median_abs_deviation(flat)

        # Quality flags; flag any parameter that violates parameter limits by 1.5 sigma
        flag = 0
        if ~np.isfinite(post_med) or ~np.isfinite(low_68) or ~np.isfinite(upp_68):
            flag += 1

        ll_dict = {
                    'par_best'  : post_med, # maximum of posterior distribution
                    'ci_68_low'   : low_68,	# lower 68% confidence interval
                    'ci_68_upp'   : upp_68,	# upper 68% confidence interval
                    'ci_95_low'   : low_95,	# lower 95% confidence interval
                    'ci_95_upp'   : upp_95,	# upper 95% confidence interval
                    'post_max'  : post_max,
                    'mean'    : post_mean, # mean of posterior distribution
                    'std_dev'    : post_std,	# standard deviation
                    'median'      : post_med, # median of posterior distribution
                    'med_abs_dev' : post_mad,	# median absolute deviation
                    'flat_chain'  : flat,   # flattened samples used for histogram.
                    'flag'    : flag, 
        }

        if (plot_param_hist==True):
                posterior_plots("LOG_LIKE",flat,ll,burn_in,xs,kde,
                                low_68,upp_68,low_95,upp_95,post_mean,post_std,post_med,post_mad,
                                run_dir)
    else:
        ll_dict = {
                'par_best'  : np.nan, # maximum of posterior distribution
                'ci_68_low'   : np.nan,	# lower 68% confidence interval
                'ci_68_upp'   : np.nan,	# upper 68% confidence interval
                'ci_95_low'   : np.nan,	# lower 95% confidence interval
                'ci_95_upp'   : np.nan,	# upper 95% confidence interval
                'post_max'  : np.nan,
                'mean'    : np.nan, # mean of posterior distribution
                'std_dev'    : np.nan,	# standard deviation
                'median'      : np.nan, # median of posterior distribution
                'med_abs_dev' : np.nan,	# median absolute deviation
                'flat_chain'  : flat,   # flattened samples used for histogram.
                'flag'    : 1, 
        }	

    return ll_dict

def flux_plots(flux_blob, z, burn_in, nwalkers, flux_norm, fit_norm, run_dir, verbose=True):
    """
    Generates best-fit values, uncertainties, and plots for 
    component fluxes from MCMC sample chains.
    """
    if verbose:
        print("\n Generating model flux distributions...\n")

    # Create a flux dictionary
    niter	= np.shape(flux_blob)[0]
    nwalkers = np.shape(flux_blob)[1]
    flux_dict = {}
    for key in flux_blob[0][0]:
        flux_dict[key] = {'chain':np.empty([nwalkers,niter])}

    # Restructure the flux_blob for the flux_dict
    for i in range(niter):
        for j in range(nwalkers):
            for key in flux_blob[0][0]:
                flux_dict[key]['chain'][j,i] = flux_blob[i][j][key]

    for key in flux_dict:
        if verbose:
            print('		  %s' % key)
        chain = np.log10(flux_dict[key]['chain']*flux_norm*fit_norm*(1.0+z)) # shape = (nwalkers,niter)
        chain[~np.isfinite(chain)] = 0
        flux_dict[key]['chain'] = chain
        # Burned-in + Flattened (along walker axis) chain
        # If burn_in is larger than the size of the chain, then 
        # take 50% of the chain length instead.
        if (burn_in >= np.shape(chain)[1]):
            burn_in = int(0.5*np.shape(chain)[1])

        # Remove burn_in iterations and flatten for histogram
        flat = chain[:,burn_in:]
        # flat = flat.flat
        flat = flat.flatten()

        # 
        if len(flat) > 0:

            # Histogram; 'Doane' binning produces the best results from tests.
            hist, bin_edges = np.histogram(flat, bins='doane', density=False)

            # Calculate KDE
            p68 = compute_HDI(flat,0.68)
            p95 = compute_HDI(flat,0.95)

            post_max  = bin_edges[hist.argmax()] # posterior max estimated from KDE
            post_mean = np.nanmean(flat)
            post_med  = np.nanmedian(flat)
            low_68  = post_med - p68[0]
            upp_68  = p68[1] - post_med
            low_95  = post_med - p95[0]
            upp_95  = p95[1] - post_med
            post_std  = np.nanstd(flat)
            post_mad  = stats.median_abs_deviation(flat)

            # Quality flags; flag any parameter that violates parameter limits by 1.5 sigma
            flag = 0  
            if ( (post_med-1.5*low_68) <= -20 ):
                flag+=1
            if ~np.isfinite(post_med) or ~np.isfinite(low_68) or ~np.isfinite(upp_68):
                flag+=1

            flux_dict[key]['par_best']  = post_med # maximum of posterior distribution
            flux_dict[key]['ci_68_low']   = low_68	# lower 68% confidence interval
            flux_dict[key]['ci_68_upp']   = upp_68	# upper 68% confidence interval
            flux_dict[key]['ci_95_low']   = low_95	# lower 95% confidence interval
            flux_dict[key]['ci_95_upp']   = upp_95	# upper 95% confidence interval
            flux_dict[key]['post_max']  = post_max
            flux_dict[key]['mean']    = post_mean # mean of posterior distribution
            flux_dict[key]['std_dev']    = post_std	# standard deviation
            flux_dict[key]['median']      = post_med # median of posterior distribution
            flux_dict[key]['med_abs_dev'] = post_mad	# median absolute deviation
            flux_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
            flux_dict[key]['flag']	    = flag 

        else:
            flux_dict[key]['par_best']  = np.nan # maximum of posterior distribution
            flux_dict[key]['ci_68_low']   = np.nan	# lower 68% confidence interval
            flux_dict[key]['ci_68_upp']   = np.nan	# upper 68% confidence interval
            flux_dict[key]['ci_95_low']   = np.nan	# lower 95% confidence interval
            flux_dict[key]['ci_95_upp']   = np.nan	# upper 95% confidence interval
            flux_dict[key]['post_max']  = np.nan
            flux_dict[key]['mean']    = np.nan # mean of posterior distribution
            flux_dict[key]['std_dev']    = np.nan	# standard deviation
            flux_dict[key]['median']      = np.nan # median of posterior distribution
            flux_dict[key]['med_abs_dev'] = np.nan	# median absolute deviation
            flux_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
            flux_dict[key]['flag']	    = 1 

    return flux_dict

def lum_plots(flux_dict,burn_in,nwalkers,z,run_dir,H0=70.0,Om0=0.30,verbose=True):
    """
    Generates best-fit values, uncertainties, and plots for 
    component luminosities from MCMC sample chains.
    """
    if verbose:
        print("\n Generating model luminosity distributions...\n")

    # Compute luminosity distance (in cm) using FlatLambdaCDM cosmology
    cosmo = FlatLambdaCDM(H0, Om0)
    d_mpc = cosmo.luminosity_distance(z).value
    d_cm  = d_mpc * 3.086E+24 # 1 Mpc = 3.086e+24 cm

    # Create a flux dictionary
    lum_dict = {}
    for key in flux_dict:
        flux = 10**(flux_dict[key]['chain']) 
        # Convert fluxes to luminosities and take log10
        lum   = np.log10((flux * 4*np.pi * d_cm**2	)) #/ 1.0E+42
        lum[~np.isfinite(lum)] = 0
        lum_dict[key[:-4]+'LUM']= {'chain':lum}

    for key in lum_dict:
        if verbose:
            print('		  %s' % key)
        chain = lum_dict[key]['chain'] # shape = (nwalkers,niter)
        chain[~np.isfinite(chain)] = 0

        # Burned-in + Flattened (along walker axis) chain
        # If burn_in is larger than the size of the chain, then 
        # take 50% of the chain length instead.
        if (burn_in >= np.shape(chain)[1]):
            burn_in = int(0.5*np.shape(chain)[1])
            # print('\n Burn-in is larger than chain length! Using 50% of chain length for burn-in...\n')

        # Remove burn_in iterations and flatten for histogram
        flat = chain[:,burn_in:]
        # flat = flat.flat
        flat = flat.flatten()

        # 
        if len(flat) > 0:

            # Histogram; 'Doane' binning produces the best results from tests.
            hist, bin_edges = np.histogram(flat, bins='doane', density=False)

            # Calculate KDE
            # kde = gauss_kde(xs,np.concatenate([subsampled,lower_pseudo_data,upper_pseudo_data]),h)
            p68 = compute_HDI(flat,0.68)
            p95 = compute_HDI(flat,0.95)

            post_max  = bin_edges[hist.argmax()] # posterior max estimated from KDE
            post_mean = np.nanmean(flat)
            post_med  = np.nanmedian(flat)
            low_68  = post_med - p68[0]
            upp_68  = p68[1] - post_med
            low_95  = post_med - p95[0]
            upp_95  = p95[1] - post_med
            post_std  = np.nanstd(flat)
            post_mad  = stats.median_abs_deviation(flat)

            # Quality flags; flag any parameter that violates parameter limits by 1.5 sigma
            flag = 0  
            if ( (post_med-1.5*low_68) <= 30 ):
                flag+=1
            if ~np.isfinite(post_med) or ~np.isfinite(low_68) or ~np.isfinite(upp_68):
                flag+=1

            lum_dict[key]['par_best']   = post_med # maximum of posterior distribution
            lum_dict[key]['ci_68_low']   = low_68	# lower 68% confidence interval
            lum_dict[key]['ci_68_upp']   = upp_68	# upper 68% confidence interval
            lum_dict[key]['ci_95_low']   = low_95	# lower 95% confidence interval
            lum_dict[key]['ci_95_upp']   = upp_95	# upper 95% confidence interval
            lum_dict[key]['post_max']   = post_max
            lum_dict[key]['mean']      = post_mean # mean of posterior distribution
            lum_dict[key]['std_dev']     = post_std	# standard deviation
            lum_dict[key]['median']   = post_med # median of posterior distribution
            lum_dict[key]['med_abs_dev'] = post_mad	# median absolute deviation
            lum_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
            lum_dict[key]['flag']	   = flag

        else:
            lum_dict[key]['par_best']   = np.nan # maximum of posterior distribution
            lum_dict[key]['ci_68_low']   = np.nan	# lower 68% confidence interval
            lum_dict[key]['ci_68_upp']   = np.nan	# upper 68% confidence interval
            lum_dict[key]['ci_95_low']   = np.nan	# lower 95% confidence interval
            lum_dict[key]['ci_95_upp']   = np.nan	# upper 95% confidence interval
            lum_dict[key]['post_max']   = np.nan
            lum_dict[key]['mean']      = np.nan # mean of posterior distribution
            lum_dict[key]['std_dev']     = np.nan	# standard deviation
            lum_dict[key]['median']   = np.nan # median of posterior distribution
            lum_dict[key]['med_abs_dev'] = np.nan	# median absolute deviation
            lum_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
            lum_dict[key]['flag']	   = 1 

    return lum_dict

def eqwidth_plots(eqwidth_blob, z, burn_in, nwalkers, run_dir,verbose=True):
    """
    Generates best-fit values, uncertainties, and plots for 
    component fluxes from MCMC sample chains.
    """
    if verbose:
        print("\n Generating model equivalent width distributions...\n")
    # Create a flux dictionary
    niter	= np.shape(eqwidth_blob)[0]
    nwalkers = np.shape(eqwidth_blob)[1]
    eqwidth_dict = {}
    for key in eqwidth_blob[0][0]:
        eqwidth_dict[key] = {'chain':np.empty([nwalkers,niter])}

    # Restructure the flux_blob for the flux_dict
    for i in range(niter):
        for j in range(nwalkers):
            for key in eqwidth_blob[0][0]:
                eqwidth_dict[key]['chain'][j,i] = eqwidth_blob[i][j][key]

    for key in eqwidth_dict:
        if verbose:
            print('		  %s' % key)
        chain = eqwidth_dict[key]['chain'] * (1.0+z) # shape = (nwalkers,niter)
        chain[~np.isfinite(chain)] = 0

        # Burned-in + Flattened (along walker axis) chain
        # If burn_in is larger than the size of the chain, then 
        # take 50% of the chain length instead.
        if (burn_in >= np.shape(chain)[1]):
            burn_in = int(0.5*np.shape(chain)[1])

        # Remove burn_in iterations and flatten for histogram
        flat = chain[:,burn_in:]
        # flat = flat.flat
        flat = flat.flatten()

        # 
        if len(flat) > 0:

            # Histogram; 'Doane' binning produces the best results from tests.
            hist, bin_edges = np.histogram(flat, bins='doane', density=False)

            # Calculate HDI
            p68 = compute_HDI(flat,0.68)
            p95 = compute_HDI(flat,0.95)

            post_max  = bin_edges[hist.argmax()] # posterior max estimated from KDE
            post_mean = np.nanmean(flat)
            post_med  = np.nanmedian(flat)
            low_68  = post_med - p68[0]
            upp_68  = p68[1] - post_med
            low_95  = post_med - p95[0]
            upp_95  = p95[1] - post_med
            post_std  = np.nanstd(flat)
            post_mad  = stats.median_abs_deviation(flat)

            # Quality flags; flag any parameter that violates parameter limits by 1.5 sigma
            flag = 0  
            if ( (post_med-1.5*low_68) <= 0 ):
                flag+=1
            if ~np.isfinite(post_med) or ~np.isfinite(low_68) or ~np.isfinite(upp_68):
                flag+=1

            eqwidth_dict[key]['par_best']   = post_med # maximum of posterior distribution
            eqwidth_dict[key]['ci_68_low']   = low_68	# lower 68% confidence interval
            eqwidth_dict[key]['ci_68_upp']   = upp_68	# upper 68% confidence interval
            eqwidth_dict[key]['ci_95_low']   = low_95	# lower 95% confidence interval
            eqwidth_dict[key]['ci_95_upp']   = upp_95	# upper 95% confidence interval
            eqwidth_dict[key]['post_max']   = post_max
            eqwidth_dict[key]['mean']      = post_mean # mean of posterior distribution
            eqwidth_dict[key]['std_dev']     = post_std	# standard deviation
            eqwidth_dict[key]['median']   = post_med # median of posterior distribution
            eqwidth_dict[key]['med_abs_dev'] = post_mad	# median absolute deviation
            eqwidth_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
            eqwidth_dict[key]['flag']	     = flag

        else:
            eqwidth_dict[key]['par_best']   = np.nan # maximum of posterior distribution
            eqwidth_dict[key]['ci_68_low']   = np.nan	# lower 68% confidence interval
            eqwidth_dict[key]['ci_68_upp']   = np.nan	# upper 68% confidence interval
            eqwidth_dict[key]['ci_95_low']   = np.nan	# lower 95% confidence interval
            eqwidth_dict[key]['ci_95_upp']   = np.nan	# upper 95% confidence interval
            eqwidth_dict[key]['post_max']   = np.nan
            eqwidth_dict[key]['mean']      = np.nan # mean of posterior distribution
            eqwidth_dict[key]['std_dev']     = np.nan	# standard deviation
            eqwidth_dict[key]['median']   = np.nan # median of posterior distribution
            eqwidth_dict[key]['med_abs_dev'] = np.nan	# median absolute deviation
            eqwidth_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
            eqwidth_dict[key]['flag']	   = 1 

    return eqwidth_dict

def cont_lum_plots(cont_flux_blob,burn_in,nwalkers,z,flux_norm,fit_norm,run_dir,H0=70.0,Om0=0.30,verbose=True):
    """
    Generates best-fit values, uncertainties, and plots for 
    component luminosities from MCMC sample chains.
    """

    # Create a flux dictionary
    niter	= np.shape(cont_flux_blob)[0]
    nwalkers = np.shape(cont_flux_blob)[1]
    cont_flux_dict = {}
    for key in cont_flux_blob[0][0]:
        cont_flux_dict[key] = {'chain':np.empty([nwalkers,niter])}

    # Restructure the flux_blob for the flux_dict
    for i in range(niter):
        for j in range(nwalkers):
            for key in cont_flux_blob[0][0]:
                cont_flux_dict[key]['chain'][j,i] = cont_flux_blob[i][j][key]
    
    # Compute luminosity distance (in cm) using FlatLambdaCDM cosmology
    cosmo = FlatLambdaCDM(H0, Om0)
    d_mpc = cosmo.luminosity_distance(z).value
    d_cm  = d_mpc * 3.086E+24 # 1 Mpc = 3.086e+24 cm
    # Create a luminosity dictionary
    cont_lum_dict = {}
    for key in cont_flux_dict:
        # Total cont. lum.
        if (key=="F_CONT_TOT_1350"):
            flux = (cont_flux_dict[key]['chain']) * flux_norm * fit_norm
            # Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
            lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 1350.0) #/ 1.0E+42
            lum[~np.isfinite(lum)] = 0
            cont_lum_dict["L_CONT_TOT_1350"]= {'chain':lum}
        if (key=="F_CONT_TOT_3000"):
            flux = (cont_flux_dict[key]['chain']) * flux_norm * fit_norm
            # Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
            lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 3000.0) #/ 1.0E+42
            lum[~np.isfinite(lum)] = 0
            cont_lum_dict["L_CONT_TOT_3000"]= {'chain':lum}
        if (key=="F_CONT_TOT_5100"):
            flux = (cont_flux_dict[key]['chain']) * flux_norm * fit_norm
            # Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
            lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 5100.0) #/ 1.0E+42
            lum[~np.isfinite(lum)] = 0
            cont_lum_dict["L_CONT_TOT_5100"]= {'chain':lum}
        # AGN cont. lum.
        if (key=="F_CONT_AGN_1350"):
            flux = (cont_flux_dict[key]['chain']) * flux_norm * fit_norm
            # Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
            lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 1350.0) #/ 1.0E+42
            lum[~np.isfinite(lum)] = 0
            cont_lum_dict["L_CONT_AGN_1350"]= {'chain':lum}
        if (key=="F_CONT_AGN_3000"):
            flux = (cont_flux_dict[key]['chain']) * flux_norm * fit_norm
            # Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
            lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 3000.0) #/ 1.0E+42
            lum[~np.isfinite(lum)] = 0
            cont_lum_dict["L_CONT_AGN_3000"]= {'chain':lum}
        if (key=="F_CONT_AGN_5100"):
            flux = (cont_flux_dict[key]['chain']) * flux_norm * fit_norm
            # Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
            lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 5100.0) #/ 1.0E+42
            lum[~np.isfinite(lum)] = 0
            cont_lum_dict["L_CONT_AGN_5100"]= {'chain':lum}
        # Host cont. lum
        if (key=="F_CONT_HOST_1350"):
            flux = (cont_flux_dict[key]['chain']) * flux_norm * fit_norm
            # Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
            lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 1350.0) #/ 1.0E+42
            lum[~np.isfinite(lum)] = 0
            cont_lum_dict["L_CONT_HOST_1350"]= {'chain':lum}
        if (key=="F_CONT_HOST_3000"):
            flux = (cont_flux_dict[key]['chain']) * flux_norm * fit_norm
            # Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
            lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 3000.0) #/ 1.0E+42
            lum[~np.isfinite(lum)] = 0
            cont_lum_dict["L_CONT_HOST_3000"]= {'chain':lum}
        if (key=="F_CONT_HOST_5100"):
            flux = (cont_flux_dict[key]['chain']) * flux_norm * fit_norm
            # Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
            lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 5100.0) #/ 1.0E+42
            lum[~np.isfinite(lum)] = 0
            cont_lum_dict["L_CONT_HOST_5100"]= {'chain':lum}
        # AGN fractions
        if (key=="AGN_FRAC_4000"):
            cont_lum_dict["AGN_FRAC_4000"]= {'chain':cont_flux_dict[key]['chain']}
        if (key=="AGN_FRAC_7000"):
            cont_lum_dict["AGN_FRAC_7000"]= {'chain':cont_flux_dict[key]['chain']}	
        # Host fractions
        if (key=="HOST_FRAC_4000"):
            cont_lum_dict["HOST_FRAC_4000"]= {'chain':cont_flux_dict[key]['chain']}
        if (key=="HOST_FRAC_7000"):
            cont_lum_dict["HOST_FRAC_7000"]= {'chain':cont_flux_dict[key]['chain']}	


    for key in cont_lum_dict:
        if verbose:
            print('		  %s' % key)
        chain = cont_lum_dict[key]['chain'] # shape = (nwalkers,niter)
        chain[~np.isfinite(chain)] = 0

        # Burned-in + Flattened (along walker axis) chain
        # If burn_in is larger than the size of the chain, then 
        # take 50% of the chain length instead.
        if (burn_in >= np.shape(chain)[1]):
            burn_in = int(0.5*np.shape(chain)[1])
            # print('\n Burn-in is larger than chain length! Using 50% of chain length for burn-in...\n')

        # Remove burn_in iterations and flatten for histogram
        flat = chain[:,burn_in:]
        # flat = flat.flat
        flat = flat.flatten()

        # 
        if len(flat) > 0:

            # Histogram; 'Doane' binning produces the best results from tests.
            hist, bin_edges = np.histogram(flat, bins='doane', density=False)

            # Calculate HDI
            p68 = compute_HDI(flat,0.68)
            p95 = compute_HDI(flat,0.95)

            post_max  = bin_edges[hist.argmax()] # posterior max estimated from KDE
            post_mean = np.nanmean(flat)
            post_med  = np.nanmedian(flat)
            low_68  = post_med - p68[0]
            upp_68  = p68[1] - post_med
            low_95  = post_med - p95[0]
            upp_95  = p95[1] - post_med
            post_std  = np.nanstd(flat)
            post_mad  = stats.median_abs_deviation(flat)

            # Quality flags; flag any parameter that violates parameter limits by 1.5 sigma
            flag = 0  
            if ( (post_med-1.5*low_68) <= 0 ):
                flag+=1
            if ~np.isfinite(post_med) or ~np.isfinite(low_68) or ~np.isfinite(upp_68):
                flag+=1

            cont_lum_dict[key]['par_best']  = post_med # maximum of posterior distribution
            cont_lum_dict[key]['ci_68_low']   = low_68	# lower 68% confidence interval
            cont_lum_dict[key]['ci_68_upp']   = upp_68	# upper 68% confidence interval
            cont_lum_dict[key]['ci_95_low']   = low_95	# lower 95% confidence interval
            cont_lum_dict[key]['ci_95_upp']   = upp_95	# upper 95% confidence interval
            cont_lum_dict[key]['post_max']  = post_max
            cont_lum_dict[key]['mean']    = post_mean # mean of posterior distribution
            cont_lum_dict[key]['std_dev']    = post_std	# standard deviation
            cont_lum_dict[key]['median']      = post_med # median of posterior distribution
            cont_lum_dict[key]['med_abs_dev'] = post_mad	# median absolute deviation
            cont_lum_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
            cont_lum_dict[key]['flag']	   = flag 

        else:
            cont_lum_dict[key]['par_best']  = np.nan # maximum of posterior distribution
            cont_lum_dict[key]['ci_68_low']   = np.nan	# lower 68% confidence interval
            cont_lum_dict[key]['ci_68_upp']   = np.nan	# upper 68% confidence interval
            cont_lum_dict[key]['ci_95_low']   = np.nan	# lower 95% confidence interval
            cont_lum_dict[key]['ci_95_upp']   = np.nan	# upper 95% confidence interval
            cont_lum_dict[key]['post_max']  = np.nan
            cont_lum_dict[key]['mean']    = np.nan # mean of posterior distribution
            cont_lum_dict[key]['std_dev']    = np.nan	# standard deviation
            cont_lum_dict[key]['median']      = np.nan # median of posterior distribution
            cont_lum_dict[key]['med_abs_dev'] = np.nan	# median absolute deviation
            cont_lum_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
            cont_lum_dict[key]['flag']	    = 1 

    return cont_lum_dict

def int_vel_disp_plots(int_vel_disp_blob,burn_in,nwalkers,z,run_dir,H0=70.0,Om0=0.30,verbose=True):
    """
    Generates best-fit values, uncertainties, and plots for 
    component luminosities from MCMC sample chains.
    """
    if verbose:
        print("\n Generating model integrated velocity moment distributions...\n")

    # Create a flux dictionary
    niter	= np.shape(int_vel_disp_blob)[0]
    nwalkers = np.shape(int_vel_disp_blob)[1]
    int_vel_disp_dict = {}
    for key in int_vel_disp_blob[0][0]:
        int_vel_disp_dict[key] = {'chain':np.empty([nwalkers,niter])}

    # Restructure the int_vel_disp_blob for the int_vel_disp_dict
    for i in range(niter):
        for j in range(nwalkers):
            for key in int_vel_disp_blob[0][0]:
                int_vel_disp_dict[key]['chain'][j,i] = int_vel_disp_blob[i][j][key]

    for key in int_vel_disp_dict:
        if verbose:
            print('		  %s' % key)
        chain = int_vel_disp_dict[key]['chain'] # shape = (nwalkers,niter)
        chain[~np.isfinite(chain)] = 0

        # Burned-in + Flattened (along walker axis) chain
        # If burn_in is larger than the size of the chain, then 
        # take 50% of the chain length instead.
        if (burn_in >= np.shape(chain)[1]):
            burn_in = int(0.5*np.shape(chain)[1])
            # print('\n Burn-in is larger than chain length! Using 50% of chain length for burn-in...\n')

        # Remove burn_in iterations and flatten for histogram
        flat = chain[:,burn_in:]
        # flat = flat.flat
        flat = flat.flatten()

        # 
        if len(flat) > 0:

            # Histogram; 'Doane' binning produces the best results from tests.
            hist, bin_edges = np.histogram(flat, bins='doane', density=False)

            # Calculate HDI
            p68 = compute_HDI(flat,0.68)
            p95 = compute_HDI(flat,0.95)

            post_max  = bin_edges[hist.argmax()] # posterior max estimated from KDE
            post_mean = np.nanmean(flat)
            post_med  = np.nanmedian(flat)
            low_68  = post_med - p68[0]
            upp_68  = p68[1] - post_med
            low_95  = post_med - p95[0]
            upp_95  = p95[1] - post_med
            post_std  = np.nanstd(flat)
            post_mad  = stats.median_abs_deviation(flat)

            # Quality flags; flag any parameter that violates parameter limits by 1.5 sigma
            flag = 0  
            if ( (post_med-1.5*low_68) <= 0 ):
                flag+=1
            if ~np.isfinite(post_med) or ~np.isfinite(low_68) or ~np.isfinite(upp_68):
                flag+=1

            int_vel_disp_dict[key]['par_best']  = post_med # maximum of posterior distribution
            int_vel_disp_dict[key]['ci_68_low']   = low_68	# lower 68% confidence interval
            int_vel_disp_dict[key]['ci_68_upp']   = upp_68	# upper 68% confidence interval
            int_vel_disp_dict[key]['ci_95_low']   = low_95	# lower 95% confidence interval
            int_vel_disp_dict[key]['ci_95_upp']   = upp_95	# upper 95% confidence interval
            int_vel_disp_dict[key]['post_max']  = post_max
            int_vel_disp_dict[key]['mean']    = post_mean # mean of posterior distribution
            int_vel_disp_dict[key]['std_dev']    = post_std	# standard deviation
            int_vel_disp_dict[key]['median']      = post_med # median of posterior distribution
            int_vel_disp_dict[key]['med_abs_dev'] = post_mad	# median absolute deviation
            int_vel_disp_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
            int_vel_disp_dict[key]['flag']	   = flag 

        else:
            int_vel_disp_dict[key]['par_best']  = np.nan # maximum of posterior distribution
            int_vel_disp_dict[key]['ci_68_low']   = np.nan	# lower 68% confidence interval
            int_vel_disp_dict[key]['ci_68_upp']   = np.nan	# upper 68% confidence interval
            int_vel_disp_dict[key]['ci_95_low']   = np.nan	# lower 95% confidence interval
            int_vel_disp_dict[key]['ci_95_upp']   = np.nan	# upper 95% confidence interval
            int_vel_disp_dict[key]['post_max']  = np.nan
            int_vel_disp_dict[key]['mean']    = np.nan # mean of posterior distribution
            int_vel_disp_dict[key]['std_dev']    = np.nan	# standard deviation
            int_vel_disp_dict[key]['median']      = np.nan # median of posterior distribution
            int_vel_disp_dict[key]['med_abs_dev'] = np.nan	# median absolute deviation
            int_vel_disp_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
            int_vel_disp_dict[key]['flag']	    = 1 

    return int_vel_disp_dict


def write_params(param_dict,header_dict,bounds,run_dir,binnum=None,spaxelx=None,spaxely=None):
    """
    Writes all measured parameters, fluxes, luminosities, and extra stuff 
    (black hole mass, systemic redshifts) and all flags to a FITS table.
    """
    # Extract elements from dictionaries
    par_names   = []
    par_best    = []
    ci_68_low   = []
    ci_68_upp   = []
    ci_95_low   = []
    ci_95_upp   = []
    mean        = []
    std_dev  = []
    median    = []
    med_abs_dev = []
    flags 	 = []

    # Param dict
    for key in param_dict:
        par_names.append(key)
        par_best.append(param_dict[key]['par_best'])
        ci_68_low.append(param_dict[key]['ci_68_low'])
        ci_68_upp.append(param_dict[key]['ci_68_upp'])
        ci_95_low.append(param_dict[key]['ci_95_low'])
        ci_95_upp.append(param_dict[key]['ci_95_upp'])
        mean.append(param_dict[key]['mean'])
        std_dev.append(param_dict[key]['std_dev'])
        median.append(param_dict[key]['median'])
        med_abs_dev.append(param_dict[key]['med_abs_dev'])
        flags.append(param_dict[key]['flag'])

    # Sort param_names alphabetically
    i_sort	 = np.argsort(par_names)
    par_names   = np.array(par_names)[i_sort] 
    par_best    = np.array(par_best)[i_sort]  
    ci_68_low   = np.array(ci_68_low)[i_sort]   
    ci_68_upp   = np.array(ci_68_upp)[i_sort]
    ci_95_low   = np.array(ci_95_low)[i_sort]   
    ci_95_upp   = np.array(ci_95_upp)[i_sort]  
    mean        = np.array(mean)[i_sort]   
    std_dev  = np.array(std_dev)[i_sort]
    median    = np.array(median)[i_sort]   
    med_abs_dev = np.array(med_abs_dev)[i_sort] 
    flags	  = np.array(flags)[i_sort]	 

    # Write best-fit parameters to FITS table
    col1  = fits.Column(name='parameter', format='30A', array=par_names)
    col2  = fits.Column(name='best_fit', format='E', array=par_best)
    col3  = fits.Column(name='ci_68_low', format='E', array=ci_68_low)
    col4  = fits.Column(name='ci_68_upp', format='E', array=ci_68_upp)
    col5  = fits.Column(name='ci_95_low', format='E', array=ci_95_low)
    col6  = fits.Column(name='ci_95_upp', format='E', array=ci_95_upp)
    col7  = fits.Column(name='mean', format='E', array=mean)
    col8  = fits.Column(name='std_dev', format='E', array=std_dev)
    col9  = fits.Column(name='median', format='E', array=median)
    col10 = fits.Column(name='med_abs_dev', format='E', array=med_abs_dev)
    col11 = fits.Column(name='flag', format='E', array=flags)
    cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11])
    table_hdu  = fits.BinTableHDU.from_columns(cols)

    if binnum is not None:
        header_dict['binnum'] = binnum
    # Header information
    hdr = fits.Header()
    for key in header_dict:
        hdr[key] = header_dict[key]
    empty_primary = fits.PrimaryHDU(header=hdr)

    hdu = fits.HDUList([empty_primary,table_hdu])
    if spaxelx is not None and spaxely is not None:
        hdu2 = fits.BinTableHDU.from_columns(fits.ColDefs([
            fits.Column(name='spaxelx', array=spaxelx, format='E'),
            fits.Column(name='spaxely', array=spaxely, format='E')
        ]))
        hdu.append(hdu2)

    hdu.writeto(run_dir.joinpath('log', 'par_table.fits'), overwrite=True)

    del hdu
    # Write full param dict to log file
    write_log((par_names,par_best,ci_68_low,ci_68_upp,ci_95_low,ci_95_upp,mean,std_dev,median,med_abs_dev,flags),'emcee_results',run_dir)
    return 

def write_chains(param_dict,run_dir):
    """
    Writes all MCMC chains to a FITS Image HDU.  Each FITS 
    extension corresponds to 
    """

    # for key in param_dict:
    # 	print(key,np.shape(param_dict[key]["chain"]))

    cols = []
    # Construct a column for each parameter and chain
    for key in param_dict:
        # cols.append(fits.Column(name=key, format='D',array=param_dict[key]['chain']))
        values = param_dict[key]['chain']
        cols.append(fits.Column(name=key, format="%dD" % (values.shape[0]*values.shape[1]), dim="(%d,%d)" % (values.shape[1],values.shape[0]), array=[values]))
    # Write to fits
    cols = fits.ColDefs(cols)
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(run_dir.joinpath('log', 'MCMC_chains.fits'), overwrite=True)

    return 

def corner_plot(free_dict,param_dict,corner_options,run_dir):
    """
    Calls the corner.py package to create a corner plot of all or selected parameters.
    """

    # with open("free_dict.pickle","wb") as handle:
    #    pickle.dump(free_dict,handle)
    # with open("param_dict.pickle","wb") as handle:
    #    pickle.dump(param_dict,handle)
    # with open("corner_options.pickle","wb") as handle:
    #    pickle.dump(corner_options,handle)

    # Extract the flattened chained from the dicts
    free_dict  = {i:free_dict[i]["flat_chain"] for i in free_dict}
    param_dict = {i:param_dict[i]["flat_chain"] for i in param_dict}

    # Extract parameters that are actually in the param_dict
    valid_dict = {i:param_dict[i] for i in corner_options["pars"] if i in param_dict}
    
    if len(valid_dict)>=2:
        # Stack the flat samples in order 
        flat_samples = np.vstack([valid_dict[i] for i in valid_dict]).T
        # labels if not provided
        if len(corner_options["labels"])==len(valid_dict):
            labels = corner_options["labels"]
        else:
            labels = [key for key in valid_dict]
        with plt.style.context('default'):
            fig = corner.corner(flat_samples,labels=labels)
            plt.savefig(run_dir.joinpath('corner.pdf'))
        fig.clear()
        plt.close()
    elif len(valid_dict)<2:
        print("\n WARNING: More than two valid parameters are required to generate corner plot! Defaulting to only free parameters... \n")
        flat_samples = np.vstack([free_dict[i] for i in free_dict]).T
        labels = [key for key in free_dict]
        with plt.style.context('default'):
            fig = corner.corner(flat_samples,labels=labels)
            plt.savefig(run_dir.joinpath('corner.pdf'))
        fig.clear()
        plt.close()

    

    return


    
def plot_best_model(param_dict,
                    line_list,
                    combined_line_list,
                    lam_gal,
                    galaxy,
                    noise,
                    comp_options,
                    losvd_options,
                    host_options,
                    power_options,
                    poly_options,
                    opt_feii_options,
                    uv_iron_options,
                    balmer_options,
                    outflow_test_options,
                    host_template,
                    opt_feii_templates,
                    uv_iron_template,
                    balmer_template,
                    stel_templates,
                    blob_pars,
                    disp_res,
                    fit_mask,
                    fit_stat,
                    velscale,
                    flux_norm,
                    fit_norm,
                    run_dir):
    """
    Plots the best fig model and outputs the components to a FITS file for reproduction.
    """

    param_names  = [key for key in param_dict ]
    par_best     = [param_dict[key]['par_best'] for key in param_dict ]

    # We already multiplied the amplitudes by fit_norm in param_plots(), 
    # now we need to use the original amplitudes to generate the best fit model
    for i in range(len(param_names)):
        if param_names[i][-4:]=="_AMP":
            par_best[i]/=fit_norm


    def poly_label(kind):
        if kind=="apoly":
            order = len([p for p in param_names if p.startswith("APOLY_")])-1
        if kind=="mpoly":
            order = len([p for p in param_names if p.startswith("MPOLY_")])-1
        #
        ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
        return ordinal(order)

    def calc_new_center(center,voff):
        """
        Calculated new center shifted 
        by some velocity offset.
        """
        c = 299792.458 # speed of light (km/s)
        new_center = (voff*center)/c + center
        return new_center

    output_model = True
    fit_type	 = 'final'
    comp_dict = fit_model(par_best,
                          param_names,
                          line_list,
                          combined_line_list,
                          lam_gal,
                          galaxy,
                          noise,
                          comp_options,
                          losvd_options,
                          host_options,
                          power_options,
                          poly_options,
                          opt_feii_options,
                          uv_iron_options,
                          balmer_options,
                          outflow_test_options,
                          host_template,
                          opt_feii_templates,
                          uv_iron_template,
                          balmer_template,
                          stel_templates,
                          blob_pars,
                          disp_res,
                          fit_mask,
                          velscale,
                          run_dir,
                          fit_type,
                          fit_stat,
                          output_model)

    # Rescale all components by fit_norm
    for key in comp_dict:
        if key not in ["WAVE"]:
            comp_dict[key] *= fit_norm

    # Put params in dictionary
    p = dict(zip(param_names,par_best))

    # Maximum Likelihood plot
    fig = plt.figure(figsize=(14,6)) 
    gs = gridspec.GridSpec(4, 1)
    gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
    ax1  = plt.subplot(gs[0:3,0])
    ax2  = plt.subplot(gs[3,0])

    for key in comp_dict:
        if (key=='DATA'):
            ax1.plot(comp_dict['WAVE'],comp_dict['DATA'],linewidth=0.5,color='white',label='Data',zorder=0)
        elif (key=='MODEL'):
            ax1.plot(lam_gal,comp_dict[key], color='xkcd:bright red', linewidth=1.0, label='Model', zorder=15)
        elif (key=='HOST_GALAXY'):
            ax1.plot(comp_dict['WAVE'], comp_dict['HOST_GALAXY'], color='xkcd:bright green', linewidth=0.5, linestyle='-', label='Host/Stellar')

        elif (key=='POWER'):
            ax1.plot(comp_dict['WAVE'], comp_dict['POWER'], color='xkcd:red' , linewidth=0.5, linestyle='--', label='AGN Cont.')

        elif (key=='APOLY'):
            ax1.plot(comp_dict['WAVE'], comp_dict['APOLY'], color='xkcd:bright purple' , linewidth=0.5, linestyle='-', label='%s-order Add. Poly.' % (poly_label("apoly")))
        elif (key=='MPOLY'):
            ax1.plot(comp_dict['WAVE'], comp_dict['MPOLY'], color='xkcd:lavender' , linewidth=0.5, linestyle='-', label='%s-order Mult. Poly.' % (poly_label("mpoly")))

        elif (key in ['NA_OPT_FEII_TEMPLATE','BR_OPT_FEII_TEMPLATE']):
            ax1.plot(comp_dict['WAVE'], comp_dict['NA_OPT_FEII_TEMPLATE'], color='xkcd:yellow', linewidth=0.5, linestyle='-' , label='Narrow FeII')
            ax1.plot(comp_dict['WAVE'], comp_dict['BR_OPT_FEII_TEMPLATE'], color='xkcd:orange', linewidth=0.5, linestyle='-' , label='Broad FeII')

        elif (key in ['F_OPT_FEII_TEMPLATE','S_OPT_FEII_TEMPLATE','G_OPT_FEII_TEMPLATE','Z_OPT_FEII_TEMPLATE']):
            if key=='F_OPT_FEII_TEMPLATE':
                ax1.plot(comp_dict['WAVE'], comp_dict['F_OPT_FEII_TEMPLATE'], color='xkcd:yellow', linewidth=0.5, linestyle='-' , label='F-transition FeII')
            elif key=='S_OPT_FEII_TEMPLATE':
                ax1.plot(comp_dict['WAVE'], comp_dict['S_OPT_FEII_TEMPLATE'], color='xkcd:mustard', linewidth=0.5, linestyle='-' , label='S-transition FeII')
            elif key=='G_OPT_FEII_TEMPLATE':
                ax1.plot(comp_dict['WAVE'], comp_dict['G_OPT_FEII_TEMPLATE'], color='xkcd:orange', linewidth=0.5, linestyle='-' , label='G-transition FeII')
            elif key=='Z_OPT_FEII_TEMPLATE':
                ax1.plot(comp_dict['WAVE'], comp_dict['Z_OPT_FEII_TEMPLATE'], color='xkcd:rust', linewidth=0.5, linestyle='-' , label='Z-transition FeII')
        elif (key=='UV_IRON_TEMPLATE'):
            ax1.plot(comp_dict['WAVE'], comp_dict['UV_IRON_TEMPLATE'], color='xkcd:bright purple', linewidth=0.5, linestyle='-' , label='UV Iron'	 )
        elif (key=='BALMER_CONT'):
            ax1.plot(comp_dict['WAVE'], comp_dict['BALMER_CONT'], color='xkcd:bright green', linewidth=0.5, linestyle='--' , label='Balmer Continuum'	 )
        # Plot emission lines by cross-referencing comp_dict with line_list
        if (key in line_list):
            if (line_list[key]["line_type"]=="na"):
                ax1.plot(comp_dict['WAVE'], comp_dict[key], color='xkcd:cerulean', linewidth=0.5, linestyle='-', label='Narrow/Core Comp.')
            if (line_list[key]["line_type"]=="br"):
                ax1.plot(comp_dict['WAVE'], comp_dict[key], color='xkcd:bright teal', linewidth=0.5, linestyle='-', label='Broad Comp.')
            if (line_list[key]["line_type"]=="out"):
                ax1.plot(comp_dict['WAVE'], comp_dict[key], color='xkcd:bright pink', linewidth=0.5, linestyle='-', label='Outflow Comp.')
            if (line_list[key]["line_type"]=="abs"):
                ax1.plot(comp_dict['WAVE'], comp_dict[key], color='xkcd:pastel red', linewidth=0.5, linestyle='-', label='Absorption Comp.')
            if (line_list[key]["line_type"]=="user"):
                ax1.plot(comp_dict['WAVE'], comp_dict[key], color='xkcd:electric lime', linewidth=0.5, linestyle='-', label='Other')

    # Plot bad pixels
    ibad = [i for i in range(len(lam_gal)) if i not in fit_mask]
    if (len(ibad)>0):# and (len(ibad[0])>1):
        bad_wave = [(lam_gal[m],lam_gal[m+1]) for m in ibad if ((m+1)<len(lam_gal))]
        ax1.axvspan(bad_wave[0][0],bad_wave[0][0],alpha=0.25,color='xkcd:lime green',label="bad pixels")
        for i in bad_wave[1:]:
            ax1.axvspan(i[0],i[0],alpha=0.25,color='xkcd:lime green')

    ax1.set_xticklabels([])
    ax1.set_xlim(np.min(lam_gal)-10,np.max(lam_gal)+10)
    # ax1.set_ylim(-0.5*np.nanmedian(comp_dict['MODEL']),np.max([comp_dict['DATA'],comp_dict['MODEL']]))
    ax1.set_ylabel(r'$f_\lambda$ ($10^{%d}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)' % (np.log10(flux_norm)),fontsize=10)
    # Residuals
    sigma_resid = np.nanstd(comp_dict['DATA'][fit_mask]-comp_dict['MODEL'][fit_mask])
    sigma_noise = np.nanmedian(comp_dict['NOISE'][fit_mask])
    ax2.plot(lam_gal,(comp_dict['NOISE']*3.0),linewidth=0.5,color="xkcd:bright orange",label='$\sigma_{\mathrm{noise}}=%0.4f$' % (sigma_noise))
    ax2.plot(lam_gal,(comp_dict['RESID']*3.0),linewidth=0.5,color="white",label='$\sigma_{\mathrm{resid}}=%0.4f$' % (sigma_resid))
    ax1.axhline(0.0,linewidth=1.0,color='white',linestyle='--')
    ax2.axhline(0.0,linewidth=1.0,color='white',linestyle='--')
    # Axes limits 
    ax_low = np.min([ax1.get_ylim()[0],ax2.get_ylim()[0]])
    ax_upp = np.nanmax(comp_dict['DATA'][fit_mask])+(3.0 * np.nanmedian(comp_dict['NOISE'][fit_mask])) # np.max([ax1.get_ylim()[1], ax2.get_ylim()[1]])
    # if np.isfinite(sigma_resid):
        # ax_upp += 3.0 * sigma_resid

    minimum = [np.nanmin(comp_dict[comp][np.where(np.isfinite(comp_dict[comp]))[0]]) for comp in comp_dict
               if comp_dict[comp][np.isfinite(comp_dict[comp])[0]].size > 0]
    if len(minimum) > 0:
        minimum = np.nanmin(minimum)
    else:
        minimum = 0.0
    ax1.set_ylim(np.nanmin([0.0, minimum]), ax_upp)
    ax1.set_xlim(np.min(lam_gal),np.max(lam_gal))
    ax2.set_ylim(ax_low,ax_upp)
    ax2.set_xlim(np.min(lam_gal),np.max(lam_gal))
    # Axes labels
    ax2.set_yticklabels(np.round(np.array(ax2.get_yticks()/3.0)))
    ax2.set_ylabel(r'$\Delta f_\lambda$',fontsize=12)
    ax2.set_xlabel(r'Wavelength, $\lambda\;(\mathrm{\AA})$',fontsize=12)
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(),loc='upper right',fontsize=8)
    ax2.legend(loc='upper right',fontsize=8)

    # Emission line annotations
    # Gather up emission line center wavelengths and labels (if available, removing any duplicates)
    line_labels = []
    for line in line_list:
        if "label" in line_list[line]:
            line_labels.append([line,line_list[line]["label"]])
    line_labels = set(map(tuple, line_labels))   
    for label in line_labels:
        center = line_list[label[0]]["center"]
        if (line_list[label[0]]["voff"]=="free"):
            voff = p[label[0]+"_VOFF"]
        elif (line_list[label[0]]["voff"]!="free"):
            voff   =  ne.evaluate(line_list[label[0]]["voff"],local_dict = p).item()
        xloc = calc_new_center(center,voff)
        yloc = np.max([comp_dict["DATA"][find_nearest(lam_gal,xloc)[1]],comp_dict["MODEL"][find_nearest(lam_gal,xloc)[1]]])
        ax1.annotate(label[1], xy=(xloc, yloc),  xycoords='data',
        xytext=(xloc, yloc), textcoords='data',
        horizontalalignment='center', verticalalignment='bottom',
        color='xkcd:white',fontsize=6,
        )

    # Save figure
    plt.savefig(run_dir.joinpath('best_fit_model.pdf'))
    # Close plot
    fig.clear()
    plt.close()
    


    # Store best-fit components in a FITS file
    # Construct a column for each parameter and chain
    cols = []
    for key in comp_dict:
        cols.append(fits.Column(name=key, format='E', array=comp_dict[key]))

    # Add fit mask to cols
    cols.append(fits.Column(name="MASK", format='E', array=fit_mask))

    # Write to fits
    cols = fits.ColDefs(cols)
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(run_dir.joinpath('log', 'best_model_components.fits'), overwrite=True)
    
    return comp_dict


def write_log(output_val,output_type,run_dir):
    """
    This function writes values to a log file as the code runs.
    """

    log_file_path = run_dir.joinpath('log', 'log_file.txt')
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_file_path.is_file():
        with log_file_path.open(mode='w') as logfile:
            logfile.write(f'\n############################### BADASS {__version__} LOGFILE ####################################\n')

    if (output_type=='line_test'):
        ptbl = output_val
        with log_file_path.open(mode='a') as logfile:
            logfile.write('\n')
            # logfile.write('-----------------------------------------------------------------------------------------------------')
            logfile.write('\n Line Test Results:\n')
            logfile.write(ptbl.get_string())
            logfile.write("\n")

        return None

    # run_emcee
    if (output_type=='emcee_options'): # write user input emcee options
        ndim,nwalkers,auto_stop,conv_type,burn_in,write_iter,write_thresh,min_iter,max_iter = output_val
        # write_log((ndim,nwalkers,auto_stop,burn_in,write_iter,write_thresh,min_iter,max_iter),40)
        a = str(datetime.datetime.now())
        with log_file_path.open(mode='a') as logfile:
            logfile.write('\n')
            logfile.write('\n### Emcee Options ###')
            logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
            logfile.write('\n{0:<30}{1:<30}'.format('ndim'		, ndim ))
            logfile.write('\n{0:<30}{1:<30}'.format('nwalkers'	, nwalkers ))
            logfile.write('\n{0:<30}{1:<30}'.format('auto_stop'   , str(auto_stop) ))
            logfile.write('\n{0:<30}{1:<30}'.format('user burn_in', burn_in ))
            logfile.write('\n{0:<30}{1:<30}'.format('write_iter'  , write_iter ))
            logfile.write('\n{0:<30}{1:<30}'.format('write_thresh', write_thresh ))
            logfile.write('\n{0:<30}{1:<30}'.format('min_iter'	, min_iter ))
            logfile.write('\n{0:<30}{1:<30}'.format('max_iter'	, max_iter ))
            logfile.write('\n{0:<30}{1:<30}'.format('start_time'  , a ))
            logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
        return None

    if (output_type=='autocorr_options'): # write user input auto_stop options
        min_samp,autocorr_tol,ncor_times,conv_type = output_val
        with log_file_path.open(mode='a') as logfile:
            # write_log((min_samp,tol,ntol,atol,ncor_times,conv_type),41,run_dir)
            logfile.write('\n')
            logfile.write('\n### Autocorrelation Options ###')
            logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
            logfile.write('\n{0:<30}{1:<30}'.format('min_samp'  , min_samp	 ))
            logfile.write('\n{0:<30}{1:<30}'.format('tolerance%', autocorr_tol ))
            logfile.write('\n{0:<30}{1:<30}'.format('ncor_times', ncor_times   ))
            logfile.write('\n{0:<30}{1:<30}'.format('conv_type' , str(conv_type)	))
            logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
        return None

    if (output_type=='autocorr_results'): # write autocorrelation results to log
        # write_log((k+1,burn_in,stop_iter,param_names,tau),42,run_dir)
        burn_in,stop_iter,param_names,tau,autocorr_tol,tol,ncor_times = output_val
        with log_file_path.open(mode='a') as logfile:
            # write_log((min_samp,tol,ntol,atol,ncor_times,conv_type),41,run_dir)
            i_sort = np.argsort(param_names)
            param_names = np.array(param_names)[i_sort]
            tau = np.array(tau)[i_sort]
            tol = np.array(tol)[i_sort]
            logfile.write('\n')
            logfile.write('\n### Autocorrelation Results ###')
            logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
            logfile.write('\n{0:<30}{1:<30}'.format('conv iteration', burn_in   ))
            logfile.write('\n{0:<30}{1:<30}'.format('stop iteration', stop_iter ))
            logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
            logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}{4:<30}'.format('Parameter','Autocorr. Time','Target Autocorr. Time','Tolerance','Converged?'))
            logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
            for i in range(0,len(param_names),1):
                if (burn_in > (tau[i]*ncor_times)) and (0 < tol[i] < autocorr_tol):
                    c = 'True'
                elif (burn_in < (tau[i]*ncor_times)) or (tol[i]>= 0.0):
                    c = 'False'
                else: 
                    c = 'False'
                logfile.write('\n{0:<30}{1:<30.5f}{2:<30.5f}{3:<30.5f}{4:<30}'.format(param_names[i],tau[i],(tau[i]*ncor_times),tol[i],c))
            logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
        return None

    if (output_type=='emcee_time'): # write autocorrelation results to log
        # write_log(run_time,43,run_dir)
        run_time = output_val
        a = str(datetime.datetime.now())
        with log_file_path.open(mode='a') as logfile:
            # write_log((min_samp,tol,ntol,atol,ncor_times,conv_type),41,run_dir)
            logfile.write('\n{0:<30}{1:<30}'.format('end_time',  a ))
            logfile.write('\n{0:<30}{1:<30}'.format('emcee_runtime',run_time ))
            logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
        return None

    if (output_type=='emcee_results'): # write best fit parameters results to log
        par_names,par_best,ci_68_low,ci_68_upp,ci_95_low,ci_95_upp,mean,std_dev,median,med_abs_dev,flags = output_val 
        # write_log((par_names,par_best,sig_low,sig_upp),50,run_dir)
        with log_file_path.open(mode='a') as logfile:
            logfile.write('\n')
            logfile.write('\n### Best-fit Parameters & Uncertainties ###')
            logfile.write('\n-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
            logfile.write('\n{0:<30}{1:<16}{2:<16}{3:<16}{4:<16}{5:<16}{6:<16}{7:<16}{8:<16}{9:<16}{10:<16}'.format('Parameter','Best-fit Value','68% CI low','68% CI upp','95% CI low','95% CI upp','Mean','Std. Dev.','Median','Med. Abs. Dev.','Flag'))
            logfile.write('\n-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
            for par in range(0,len(par_names),1):
                logfile.write('\n{0:<30}{1:<16.5f}{2:<16.5f}{3:<16.5f}{4:<16.5f}{5:<16.5f}{6:<16.5f}{7:<16.5f}{8:<16.5f}{9:<16.5f}{10:<16.5f}'.format(par_names[par],par_best[par],ci_68_low[par],ci_68_upp[par],ci_95_low[par],ci_95_upp[par],mean[par],std_dev[par],median[par],med_abs_dev[par],flags[par]))
            logfile.write('\n-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        return None

    # Total runtime
    if (output_type=='total_time'): # write total time to log
        # write_log(run_time,43,run_dir)
        tot_time = output_val
        a = str(datetime.datetime.now())
        with log_file_path.open(mode='a') as logfile:
            # write_log((min_samp,tol,ntol,atol,ncor_times,conv_type),41,run_dir)
            logfile.write('\n{0:<30}{1:<30}'.format('total_runtime',time_convert(tot_time) ))
            logfile.write('\n{0:<30}{1:<30}'.format('end_time',a ))
            logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
        return None

    return None

