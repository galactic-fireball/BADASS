import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import badass.utils.constants as consts
from badass.components.templates.common import BadassTemplate, convolve_gauss_hermite, gaussian_filter1d, template_rfft
from badass.utils.utils import log_rebin


class OpticalFeIITemplate(BadassTemplate):
    OPTION_NAME = 'optfeii'
    PARAM_PREFIX = 'OPT_FEII_'

    TEMP_LAM_RANGE = [0.0, -1.0]

    @classmethod
    def initialize_template(cls, ctx):
        if not ctx.cfg.comp.fit_feii:
            return None

        temp_type = ctx.cfg.optfeii.template
        class_name = '%s_OpticalFeIITemplate' % (temp_type)
        if not class_name in globals():
            ctx.log.error('Optical FeII template unsupported: %s' % temp_type)
            return None

        temp_class = globals()[class_name]

        if (temp_class.TEMP_LAM_RANGE[1] > 0.0 and ctx.fit_wave[0] > temp_class.TEMP_LAM_RANGE[1]) or (ctx.fit_wave[-1] < temp_class.TEMP_LAM_RANGE[0]):
            ctx.log.warn('Optical FeII template disabled because template is outside of fitting region')
            ctx.log.update_opt_feii()
            ctx.cfg.comp.fit_feii = False
            return None

        return temp_class.initialize_template(ctx)


    def convolve(self, fft, feii_voff, feii_disp, npad=None):
        if npad is None: npad = self.npad
        return convolve_gauss_hermite(fft, npad, float(self.ctx.target.velscale),
                                     [feii_voff, feii_disp/2.3548], self.ctx.fit_wave.shape[0],
                                     velscale_ratio=1, sigma_diff=0, vsyst=self.vsyst)


class VC04_OpticalFeIITemplate(OpticalFeIITemplate):
    """
    'VC04' : Veron-Cetty et al. (2004) template, which utilizes a single broad
             and single narrow line template with fixed relative intensities.
             One can choose to fix FWHM and VOFF for each, and only vary
             amplitudes (2 free parameters), or vary amplitude, FWHM, and VOFF
             for each template (6 free parameters)
    """

    TEMPLATE_PARAMS = ['%s_%s'%(lt,attr) for lt in ['na','br'] for attr in ['amp','disp','voff']]
    TEMP_LAM_RANGE = [3400.0, 7200.0] # Angstrom

    vc04_data_dir = consts.BADASS_DATA_DIR.joinpath('feii_templates', 'veron-cetty_2004')
    br_path = vc04_data_dir.joinpath('VC04_br_feii_template.csv')
    na_path = vc04_data_dir.joinpath('VC04_na_feii_template.csv')


    @classmethod
    def initialize_template(cls, ctx):
        if (not VC04_OpticalFeIITemplate.br_path.exists()) or (not VC04_OpticalFeIITemplate.na_path.exists()):
            ctx.log.error('VC04 data directory not found: %s' % str(VC04_OpticalFeIITemplate.vc04_data_dir))
            return None

        return cls(ctx)


    def __init__(self, ctx):
        super().__init__(ctx)

        df_br = pd.read_csv(self.br_path)
        df_na = pd.read_csv(self.na_path)

        # Generate a new grid with the original resolution, but the size of the fitting region
        dlam_feii = df_br['angstrom'].to_numpy()[1]-df_br['angstrom'].to_numpy()[0] # angstroms
        npad = 100 # anstroms
        lam_feii = np.arange(np.min(self.ctx.fit_wave)-npad, np.max(self.ctx.fit_wave)+npad, dlam_feii) # angstroms

        # Interpolate the original template onto the new grid
        interp_ftn_br = interp1d(df_br['angstrom'].to_numpy(),df_br['flux'].to_numpy(),kind='linear',bounds_error=False,fill_value=(0.0,0.0))
        interp_ftn_na = interp1d(df_na['angstrom'].to_numpy(),df_na['flux'].to_numpy(),kind='linear',bounds_error=False,fill_value=(0.0,0.0))
        spec_feii_br = interp_ftn_br(lam_feii)
        spec_feii_na = interp_ftn_na(lam_feii)

        # Convolve templates to the native resolution of SDSS
        fwhm_feii = 1.0 # templates were created with 1.0 FWHM resolution
        disp_feii = fwhm_feii/2.3548
        disp_res_interp = np.interp(lam_feii, self.ctx.fit_wave, self.ctx.target.disp_res)
        disp_diff = np.sqrt((disp_res_interp**2 - disp_feii**2).clip(0))
        sigma = disp_diff/dlam_feii # Sigma difference in pixels
        spec_feii_br = gaussian_filter1d(spec_feii_br, sigma)
        spec_feii_na = gaussian_filter1d(spec_feii_na, sigma)

        # log-rebin the spectrum to same velocity scale as the input galaxy
        lamRange_feii = [np.min(lam_feii), np.max(lam_feii)]
        spec_feii_br_new, loglam_feii, velscale_feii = log_rebin(lamRange_feii, spec_feii_br, velscale=self.ctx.target.velscale)
        spec_feii_na_new, loglam_feii, velscale_feii = log_rebin(lamRange_feii, spec_feii_na, velscale=self.ctx.target.velscale)

        # Pre-compute FFT of templates, since they do not change (only the LOSVD and convolution changes)
        self.br_opt_feii_fft, self.npad = template_rfft(spec_feii_br_new)
        self.na_opt_feii_fft, self.npad = template_rfft(spec_feii_na_new)

        # The FeII templates are offset from the input galaxy spectrum by 100 A, so we
        # shift the spectrum to match that of the input galaxy.
        self.vsyst = np.log(lam_feii[0]/self.ctx.fit_wave[0]) * consts.c

        # if all params are constant, we can pre_convolve before the fit
        self.pre_convolve = all([param in self.const_params for param in ['%s_%s'%(lt,attr) for lt in ['na','br'] for attr in ['disp','voff']]])
        if self.pre_convolve:
            self.br_conv_temp = self.convolve(self.br_opt_feii_fft, self.const_params['br_voff'], self.const_params['br_disp'])
            self.na_conv_temp = self.convolve(self.na_opt_feii_fft, self.const_params['na_voff'], self.const_params['na_disp'])


    def add_components(self, params, comp_dict, host_model):
        if not self.pre_convolve:
            self.br_conv_temp = self.convolve(self.br_opt_feii_fft, self.get_param('br_voff',params), self.get_param('br_disp',params))
            self.na_conv_temp = self.convolve(self.na_opt_feii_fft, self.get_param('na_voff',params), self.get_param('na_disp',params))

        br_opt_feii_template = (self.get_param('br_amp',params) * self.br_conv_temp).reshape(-1)
        na_opt_feii_template = (self.get_param('na_amp',params) * self.na_conv_temp).reshape(-1)

        # Set fitting region outside of template to zero to prevent convolution loops
        br_opt_feii_template[(self.ctx.fit_wave < self.TEMP_LAM_RANGE[0]) & (self.ctx.fit_wave > self.TEMP_LAM_RANGE[1])] = 0
        na_opt_feii_template[(self.ctx.fit_wave < self.TEMP_LAM_RANGE[0]) & (self.ctx.fit_wave > self.TEMP_LAM_RANGE[1])] = 0

        # Update the component dict with the templates
        comp_dict['BR_OPT_FEII_TEMPLATE'] = br_opt_feii_template
        comp_dict['NA_OPT_FEII_TEMPLATE'] = na_opt_feii_template

        # Subtract the br and na templates from the host model and return
        host_model -= na_opt_feii_template
        host_model -= br_opt_feii_template
        return host_model


class K10_OpticalFeIITemplate(OpticalFeIITemplate):
    """
    'K10'  : Kovacevic et al. (2010) template, which treats the F, S, and G line
             groups as independent templates (each amplitude is a free parameter)
             and whose relative intensities are temperature dependent (1 free
             parameter).  There are additonal lines from IZe1 that only vary in
             amplitude.  All 4 line groups share the same FWHM and VOFF, for a
             total of 7 free parameters.  This template is only recommended
             for objects with very strong FeII emission, for which the LOSVD
             cannot be determined at all.
    """

    TEMPLATE_PARAMS = ['f_amp', 'g_amp', 's_amp', 'z_amp', 'disp', 'voff', 'temp']
    TEMP_LAM_RANGE = [4400.0, 5500.0]
    k10_data_dir = consts.BADASS_DATA_DIR.joinpath('feii_templates', 'kovacevic_2010')

    class Transition:

        # Values from Kovacevic et al. 2010
        TRANSITION_DICT = {
            'F': {
                    'range_min': 4472,
                    'range_max': 5147,
                    'lam2': 4549.474,
                    'gf2': 1.10e-02,
                    'e1': 8.896255e-19,
                 },
            'S': {
                    'range_min': 4731,
                    'range_max': 5285,
                    'lam2': 5018.440,
                    'gf2': 3.98e-02,
                    'e1': 8.589111e-19,
                 },
            'G': {
                    'range_min': 4472,
                    'range_max': 5147,
                    'lam2': 5316.615,
                    'gf2': 1.17e-02,
                    'e1': 8.786549e-19,
                 },
            'Z': {
                    'range_min': 4418,
                    'range_max': 5428,
                 },
        }

        def __init__(self, name):
            self.name = name
            self.__dict__.update(self.TRANSITION_DICT[self.name])

            self.data_path = None
            self.df = None
            self.fft = None
            self.npad = None
            self.conv_temp = None

            self.wavelength = None
            self.gf = None
            self.e2 = None
            self.rel_int = None

            self.feii_amp = None
            self.templates = None


        def read_data(self, data_path):
            self.data_path = data_path
            self.df = pd.read_csv(self.data_path)

            self.wavelength = self.df['wavelength'].to_numpy()

            if self.name == 'Z':
                self.rel_int = self.df['rel_int'].to_numpy()
            else:
                self.gf = self.df['gf'].to_numpy()
                self.e2 = self.df['E2_J'].to_numpy()


        def calc_rel_int(self, temp):
            """
            Calculate relative intensities for the S, F, and G FeII line groups
            from Kovacevic et al. 2010 template as a function a temperature.
            """
            self.rel_int = self.feii_amp*(self.lam2/self.wavelength)**3 * (self.gf/self.gf2) \
                            * np.exp(-1.0/(consts.k*temp) * (self.e2 - self.e1))


    @classmethod
    def initialize_template(cls, ctx):
        if not K10_OpticalFeIITemplate.k10_data_dir.exists():
            ctx.log.error('K10 data directory not found: %s' % str(K10_OpticalFeIITemplate.k10_data_dir))
            return None

        return cls(ctx)


    def __init__(self, ctx):
        super().__init__(ctx)

        # The procedure for the K10 templates is slightly difference since their relative intensities
        # are temperature dependent.  We must create a Gaussian emission line for each individual line,
        # and store them as an array, for each of the F, S, G, and Z transitions.  We treat each transition
        # as a group of templates, which will be convolved together, but relative intensities will be calculated
        # for separately.

        def gaussian_angstroms(x, center, amp, disp, voff):
            x = x.reshape((len(x),1))
            g = amp*np.exp(-0.5*(x-(center))**2/(disp)**2) # construct gaussian
            g = np.sum(g,axis=1)
            # Replace the ends with the same value
            g[0]  = g[1]
            g[-1] = g[-2]
            return g

        self.transitions = {name:self.Transition(name) for name in self.Transition.TRANSITION_DICT.keys()}
        for trans in self.transitions.values():
            trans.read_data(K10_OpticalFeIITemplate.k10_data_dir.joinpath('K10_%s_transitions.csv' % trans.name))

        # Generate a high-resolution wavelength scale that is universal to all transitions
        fwhm = 1.0 # Angstroms
        disp = fwhm/2.3548
        dlam_feii = 0.1 # linear spacing in Angstroms
        npad = 100
        lam_feii = np.arange(np.min(self.ctx.fit_wave)-npad, np.max(self.ctx.fit_wave)+npad, dlam_feii)
        lamRange_feii = [np.min(lam_feii), np.max(lam_feii)]
        # Get size of output log-rebinned spectrum
        ga = gaussian_angstroms(lam_feii, self.transitions['F'].wavelength[0], 1.0, disp, 0.0)
        new_size, loglam_feii, velscale_feii = log_rebin(lamRange_feii, ga, velscale=self.ctx.target.velscale)

        for trans in self.transitions.values():
            # Create storage arrays for each emission line of each transition
            trans.templates = np.empty((len(new_size), len(trans.wavelength)))

            # Generate templates with an amplitude of 1.0
            for i in range(np.shape(trans.templates)[1]):
                ga = gaussian_angstroms(lam_feii, trans.wavelength[i], 1.0, disp, 0.0)
                new_temp = log_rebin(lamRange_feii, ga, velscale=self.ctx.target.velscale)[0]
                trans.templates[:,i] = new_temp/np.max(new_temp)

            # Pre-compute the FFT for each transition
            trans.fft, trans.npad = template_rfft(trans.templates)

        self.npad = self.transitions['F'].npad
        self.vsyst = np.log(lam_feii[0]/self.ctx.fit_wave[0]) * consts.c

        # only if disp and voff are constant, can we pre_convolve before the fit
        self.pre_convolve = ('disp' in self.const_params) and ('voff' in self.const_params)

        if self.pre_convolve:
            for trans in self.transitions.values():
                trans.conv_temp = convolve(trans.fft, self.const_params['voff'], self.const_params['disp'], npad=trans.npad)


    def add_components(self, params, comp_dict, host_model):
        for trans in self.transitions.values():
            trans.feii_amp = self.get_param('%s_amp'%trans.name,params)

            if not self.pre_convolve:
                # Perform the convolution
                # TODO: set npad for each transition?
                trans.conv_temp = self.convolve(trans.fft, self.get_param('voff',params), self.get_param('disp',params))

            # TODO: if we do pre-convolve do we need to do this here? Or can we do this once in init?
            # Normalize amplitudes to 1
            norm = np.array([np.max(trans.conv_temp[:,i]) for i in range(np.shape(trans.conv_temp)[1])])
            norm[norm<1.e-6] = 1.0
            trans.conv_temp = trans.conv_temp/norm

            # Calculate temperature dependent relative intensities
            # TODO: if temp is constant, do this in init?
            if trans.name != 'Z': # relative intensity set for Z in initialization
                trans.calc_rel_int(self.get_param('temp',params))

            # Multiply by relative intensities
            trans.conv_temp *= trans.rel_int

            # Sum templates along rows
            trans.templates = np.sum(trans.conv_temp, axis=1)

            if trans.name == 'Z':
                trans.templates * self.get_param('z_amp',params)

            trans.templates[(self.ctx.fit_wave < trans.range_min) | (self.ctx.fit_wave > trans.range_max)] = 0

            comp_dict[trans.name+'_OPT_FEII_TEMPLATE'] = trans.templates
            host_model -= trans.templates

        return host_model
