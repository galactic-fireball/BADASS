import astropy.constants as const
from dataclasses import dataclass, field
import numpy as np
from prodict import Prodict
from scipy import signal
from scipy.interpolate import interp1d
from typing import Dict, Optional

from badass.badass_tools import badass_tools
import badass.utils.constants as consts
import badass.utils.utils as ba_utils
from badass.components.spectral_lines.default_hyperpars import type_default_hyperpars, profile_default_hyperpars

EDGE_PAD = 10
DEFAULT_TYPE = 'NARROW'

short_type_to_type = {
    'NA': 'NARROW',
    'BR': 'BROAD',
    'ABS': 'ABSORP',
}

def prefix(t):
    for pre, name in short_type_to_type.items():
        if name == t: return pre
    return ''


def capitalize(data):
    res = []
    for data_dict in data:
        res_dict = {}
        for key, val in data_dict.items():
            res_dict[key.upper()] = str(val).upper() if isinstance(val, str) else val
            children = data_dict.get('CHILDREN', data_dict.get('children', []))
            if children:
                res_dict['CHILDREN'] = capitalize(children)
        res.append(res_dict)
    return res


primary_pars = ['AMP', 'DISP', 'VOFF']
hyperpars = ['INIT', 'PLIM', 'PRIOR']

# TODO: move into components.py and use in templates
@dataclass
class FitParameter:
    name: str = ''
    expr: [str,float,int] = 'FREE'
    is_free: bool = False
    expr: [str,float,int] = 0.0
    init: Optional[float] = None
    plim: Optional[tuple] = None
    prior: Optional[Dict] = None
    value: Optional[float] = None

    def __post_init__(self):
        self.is_free = self.expr == 'FREE'


class SpectralLine:

    ctx = None
    line_list = []
    common_params = {}
    all_type_options = {}

    @staticmethod
    def initialize_spectral_lines(_ctx, _line_list):
        SpectralLine.ctx = _ctx
        SpectralLine.all_type_options = {t:capitalize([SpectralLine.ctx.options.get('%s_options'%t.lower(),{})])[0] for t in short_type_to_type.values()}
        SpectralLine.line_list = [SpectralLine.from_dict(line_dict, None) for line_dict in capitalize(_line_list)]
        # TODO: just call initialize_parameters here?


    @staticmethod
    def dump_lines():
        for line in SpectralLine.line_list:
            print(line) # TODO: logger


    @ staticmethod
    def add_line(line_dict):
        pass # TODO


    @staticmethod
    def remove_line(line_name):
        pass # TODO


    @staticmethod
    def get_free_parameters(line_list):
        pass # TODO


    @staticmethod
    def dump():
        pass # TODO


    @staticmethod
    def pretty_print():
        pass # TODO


    @staticmethod
    def get_hyperpar_val(par, hparam, line_type='', line_profile=''):
        profile_default = profile_default_hyperpars.get(line_profile, {}).get(par, {}).get(hparam, None)
        if not profile_default is None:
            return profile_default

        hparam_name = par + '_' + hparam
        type_options = SpectralLine.all_type_options.get(line_type, {})
        # check in order: type_options, default type options, common options
        type_default = type_options.get(hparam_name, type_default_hyperpars.get(line_type, {}).get(par, {}).get(hparam, type_default_hyperpars['COMMON'].get(par, {}).get(hparam, None)))
        return type_default


    @staticmethod
    def add_tied_param(line_type, par):
        # TODO: instead of add to params, return class parameters
        pre = prefix(line_type)
        param_name = pre + '_' + par
        if param_name in SpectralLine.common_params:
            return

        type_options = SpectralLine.all_type_options[line_type]
        fp = FitParameter(name=param_name, expr=type_options.get(par, 'FREE'))
        SpectralLine.common_params[param_name] = fp
        if not fp.is_free:
            return

        for hparam in hyperpars:
            # check each in order: type_options, default hyperpar dict
            hparam_val = SpectralLine.get_hyperpar_val(line_type, par, hparam)

            if (hparam != 'PRIOR') and (hparam_val is None):
                raise Exception('Could not find voff hyperpar [%s] for line type [%s]'%(hparam_name,line_type))

            setattr(fp, hparam.lower(), hparam_val)


    @classmethod
    def from_dict(cls, line_dict, parent):
        center = line_dict.get('CENTER', parent.center if parent else None)
        if center is None:
            raise Exception('Line center needed for: %s'%line_dict['NAME'])

        if (center <= SpectralLine.ctx.target.wave[0]+EDGE_PAD) or (center >= SpectralLine.ctx.target.wave[-1]-EDGE_PAD):
            SpectralLine.ctx.log.warning('Not fitting %s line (out of fit region): %s'%(line_type,line_dict['NAME']))
            return None

        if not 'NAME' in line_dict:
            line_dict['NAME'] = 'FEAT_.5%f'%center

        if not 'TYPE' in line_dict:
            line_dict['TYPE'] = DEFAULT_TYPE
        line_type = line_dict['TYPE'].upper()
        if len(line_type) < 4: line_type = short_type_to_type[line_type]
        line_dict['TYPE'] = line_type

        if (line_type != 'COMBINED') and (not SpectralLine.ctx.options.comp_options.get('fit_%s'%line_type.lower(), True)):
            SpectralLine.ctx.log.warning('Not fitting %s line (unfit type): %s'%(line_type,line_dict['NAME']))
            return None

        return cls(line_dict, parent)


    def __init__(self, line_dict, parent):
        self.line_dict = line_dict
        self.parent = parent
        self.name = self.line_dict['NAME']
        self.line_type = self.line_dict.get('TYPE', DEFAULT_TYPE)
        self.is_combined = self.line_type == 'COMBINED'
        self.prefix = prefix(self.line_type)
        self.center = self.line_dict.get('CENTER', parent.center if parent else None) # TODO: center is unit-configurable

        if self.is_combined:
            self.type_options = {}
            self.line_profile = None
        else:
            self.type_options = SpectralLine.all_type_options.get(self.line_type, {})
            profile = self.line_dict.get('PROFILE', self.type_options.get('profile', None))
            if profile is None:
                raise Exception('Line profile required for non-combined line: %s'%self.name)
            from badass.components.spectral_lines.line_profiles import LineProfile # need to avoid circular imports
            self.line_profile = LineProfile.get_line_profile(profile)
            if self.line_profile is None:
                raise Exception('Invalid line profile (%s) for line: %s'%(profile,self.name))

        self.children = [SpectralLine.from_dict(child_dict, self) for child_dict in self.line_dict.get('CHILDREN', [])]

        # Fit parameters, either free, constant, or expression
        # Parameters for combined lines will be calculated post-fit
        self.parameters = {}

        SpectralLine.ctx.log.debug('Added line: %s'%str(self))


    def initialize_parameters(self, params, args):
        if self.is_combined:
            for child in self.children:
                child.initialize_parameters(params, args)
            return

        for par in primary_pars:
            self.set_hyperpars(par, args)
        self.line_profile.initialize_parameters(self, args) # add profile-unique parameters
        self.validate_hyperpars()

        # TODO: pass a dict of {name: FitParameter}
        for param in self.parameters.values():
            if not param.is_free:
                continue

            params[param.name] = {
                'init': param.init,
                'plim': param.plim
            }
            if param.prior:
                params[param.name]['prior'] = param.prior

        for child in self.children:
            child.initialize_parameters(params, args)


    def set_hyperpars(self, par, args):
        par_name = self.name + '_' + par

        if ((par == 'VOFF') and (SpectralLine.ctx.options.comp_options.tie_line_voff)) \
            or ((par == 'DISP') and (SpectralLine.ctx.options.comp_options.tie_line_disp)):
            self.parameters[par_name] = FitParameter(name=par_name, expr=self.prefix + '_' + par)
            SpectralLine.add_tied_param(self.line_type, par)
            return

        # If not in line_dict or type_options => default to a free parameter
        fp = FitParameter(name=par_name, expr=self.line_dict.get(par, self.type_options.get(par, 'FREE')))
        self.parameters[par_name] = fp

        if not fp.is_free:
            return

        for hparam in hyperpars:
            hparam_name = par + '_' + hparam
            # check each in order: line_dict, type_options, default hyperpar dict
            hparam_val = self.line_dict.get(hparam_name, SpectralLine.get_hyperpar_val(par, hparam, line_type=self.line_type, line_profile=self.line_profile.name))

            # Special case if we found the amp_init value in a general source (ie. not the specific line dict), apply a factor based on the number of components
            if (par == 'AMP') and (hparam == 'INIT') and (not hasattr(self.line_dict, hparam_name)) and (not hparam_val is None):
                amp_factor = len(self.parent.children) if self.parent else 1 # number of siblings (including self)
                hparam_val /= amp_factor

            # par and hparam unique methods for finding hparam_val
            if hparam_val is None:
                if (par == 'VOFF') and (hparam == 'INIT'):
                    # derive the voff_init value based on the actual peak (or trough) wavelengths vs the provided line center
                    feat_waves = args.get(type_to_feat_type(self.line_type), None)
                    if not feat_waves is None:
                        closest_feat = feat_waves[np.argmin(np.abs(feat_waves-self.center))]
                        hparam_val = (closest_feat-self.center)/self.center*consts.c # to km/s

                if (par == 'AMP') and (hparam == 'INIT'):
                    # derive the amp_init value based on the flux close to the line center
                    hparam_val = float(self.ctx.fit_spec[ba_utils.find_nearest(self.ctx.fit_wave,self.center)[1]])
                    # apply a factor based on the number of components
                    amp_factor = len(self.parent.children) if self.parent else 1 # number of siblings (including self)
                    hparam_val /= amp_factor

                if (par == 'AMP') and (hparam == 'PLIM'):
                    hparam_val = (0.0, float(2*np.nanmax(self.ctx.fit_spec)))

            if (hparam != 'PRIOR') and (hparam_val is None):
                raise Exception('Could not find voff hyperpar [%s] for line [%s]'%(hparam_name,self.name))

            # negate the amp for absorption lines
            if (self.line_type == 'ABSORP') and (par == 'AMP') and (hparam != 'PRIOR'):
                hparam_val *= -1

            setattr(fp, hparam.lower(), hparam_val)


    def validate_hyperpars(self):
        for pname, fp in self.parameters.items():
            if not fp.is_free:
                continue

            if (fp.plim[0] < fp.init) or (fp.init > fp.plim[1]):
                new_init = fp.plim[1] - (fp.plim[1]-fp.plim[0])
                self.ctx.log.warn('init value for %s [%f] outside limits (%f,%f), resetting to %f'%(pname,fp.init,fp.plim[0],fp.plim[1],new_init))
                fp.init = new_init


    def __str__(self):
        s = '%s (%s%s) @ %.04f'%(self.name, self.line_type, ' %s'%self.line_profile.name if self.line_profile else '', self.center)
        if len(self.children):
            s += '\n\tChildren:'
            for c in self.children:
                s += '\n\t'
                s += str(c).replace('\t', '\t\t')
            s += '\n'
        if self.parameters:
            s += '\n\tParameters:'
            for key, val in self.parameters.items():
                s += '\n\t\t%s = %s' % (key, str(val))
        return s


    @staticmethod
    def add_disp_res():
        c = const.c.to('km/s').value
        # Perform linear interpolation on the disp_res array as a function of wavelength
        # We will use this to determine the dispersion resolution as a function of wavelength for each
        # emission line so we can correct for the resolution at every iteration.
        disp_res_ftn = interp1d(SpectralLine.ctx.target.wave,SpectralLine.ctx.target.disp_res,kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))
        # Interpolation function that maps x (in angstroms) to pixels so we can get the exact
        # location in pixel space of the emission line.
        x_pix = np.array(range(len(SpectralLine.ctx.target.wave)))
        pix_interp_ftn = interp1d(SpectralLine.ctx.target.wave,x_pix,kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))

        def add_disp_values(line):
            line.center_pix = float(pix_interp_ftn(line.center)) # line center in pixels
            line.disp_res_ang = float(disp_res_ftn(center)) # instrumental FWHM resolution in angstroms
            line.disp_res_kms = (line.disp_res_ang/line.center)*c # instrumental FWHM resolution in km/s

            for child_line in line.children:
                add_disp_values(child_line)

        for line in SpectralLine.ctx.line_list:
            add_disp_values(line)


    @staticmethod
    def initialize_line_parameters(params, args):
        ctx = SpectralLine.ctx
        peaks, troughs = get_spec_features(ctx.fit_wave,ctx.fit_spec,ctx.fit_noise,line_list=SpectralLine.line_list)
        args['peaks'] = peaks
        args['troughs'] = troughs

        for line in SpectralLine.line_list:
            print('Initializing: %s' % line)
            line.initialize_parameters(params, args)

        # TODO: just add the common_params to total param list and return
        for param in SpectralLine.common_params.values():
            if not param.is_free:
                continue

            params[param.name] = {
                'init': param.init,
                'plim': param.plim
            }
            if param.prior:
                params[param.name]['prior'] = param.prior


def get_spec_features(wave, spec, noise, line_list=None):
    galaxy_csub = badass_tools.continuum_subtract(wave,spec,noise,sigma_clip=2.0,plot=False,verbose=False)

    try:
        # normalize by noise
        norm_csub = galaxy_csub/noise

        peaks,_ = signal.find_peaks(norm_csub, height=2.0, width=3.0, prominence=1)
        troughs,_ = signal.find_peaks(-norm_csub, height=2.0, width=3.0, prominence=1)
        peak_wave = wave[peaks]
        trough_wave = wave[troughs]
    except:
        if line_list:
            SpectralLine.ctx.log.warn('Warning! Peak finding algorithm used for initial guesses of amplitude and velocity failed! Defaulting to user-defined locations...')
            peak_wave = np.array([line.center for line in line_list if line.line_type in ['NARROW','BROAD']])
            trough_wave = np.array([line.center for line in line_list if line.line_type in ['ABSORP']])

    if len(peak_wave) == 0:
        peak_wave = np.array([0])
    if len(trough_wave) == 0:
        trough_wave = np.array([0])

    return peak_wave, trough_wave
