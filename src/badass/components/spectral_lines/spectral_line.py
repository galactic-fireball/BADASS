import astropy.constants as const
from dataclasses import dataclass, field
import numpy as np
from prodict import Prodict
from scipy import signal
from scipy.interpolate import interp1d
from typing import Dict, Optional

import badass.utils.constants as consts
import badass.utils.utils as ba_utils
from badass.components.blobs import ComponentBlob, LineVelBlob
from badass.components.components import BadassComponent
from badass.components.params import ParameterRegistry
from badass.components.spectral_lines.default_hyperpars import type_default_hyperpars, profile_default_hyperpars

EDGE_PAD = 10

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


class SpectralLine(BadassComponent):

    ctx = None
    line_list = []
    common_params = {}
    param_reg = None

    @staticmethod
    def initialize_spectral_lines(_ctx, _line_list):
        SpectralLine.ctx = _ctx
        SpectralLine.param_reg = SpectralLine.ctx.param_reg
        return [SpectralLine.from_dict(line_dict, None) for line_dict in capitalize(_line_list)]


    @staticmethod
    def dump_lines():
        for line in SpectralLine.line_list:
            print(line) # TODO: logger


    @classmethod
    def from_dict(cls, line_dict, parent):
        if (not 'CENTER' in line_dict) or (line_dict['CENTER'] is None):
            if parent is None:
                raise Exception('Line center needed for: %s'%line_dict['NAME'])
            line_dict['CENTER'] = parent.center

        if (line_dict['CENTER'] <= SpectralLine.ctx.target.wave[0]+EDGE_PAD) or (line_dict['CENTER'] >= SpectralLine.ctx.target.wave[-1]-EDGE_PAD):
            SpectralLine.ctx.log.warning('Not fitting %s line (out of fit region): %s'%(line_type,line_dict['NAME']))
            return None

        if (not 'NAME' in line_dict) or (line_dict['NAME'] is None) or (line_dict['NAME'] == ''):
            line_dict['NAME'] = 'FEAT_.5%f'%line_dict['CENTER']

        line_type = line_dict['TYPE'].upper()
        if len(line_type) < 4: line_type = short_type_to_type[line_type]
        line_dict['TYPE'] = line_type

        if (line_type != 'COMBINED') and (not SpectralLine.ctx.cfg.comp.fit(line_type.lower())):
            SpectralLine.ctx.log.warning('Not fitting %s line (unfit type): %s'%(line_type,line_dict['NAME']))
            return None

        return cls(line_dict, parent)


    def __init__(self, line_dict, parent):
        self.line_dict = line_dict
        self.parent = parent
        self.name = self.line_dict['NAME']
        self.line_type = self.line_dict['TYPE']
        self.is_combined = self.line_type == 'COMBINED'
        self.prefix = prefix(self.line_type)
        self.center = self.line_dict['CENTER'] # TODO: center is unit-configurable

        if self.is_combined:
            self.type_options = {}
            self.line_profile = None
        else:
            profile = self.line_dict['PROFILE']
            from badass.components.spectral_lines.line_profiles import LineProfile # need to avoid circular imports
            self.line_profile = LineProfile.get_line_profile(profile)
            if self.line_profile is None:
                raise Exception('Invalid line profile (%s) for line: %s'%(profile,self.name))

        self.children = [SpectralLine.from_dict(child_dict, self) for child_dict in self.line_dict.get('CHILDREN', [])]
        self.add_disp_res()

        SpectralLine.ctx.log.debug('Added line: %s'%str(self))
        super().__init__(SpectralLine.ctx)


    def initialize_parameters(self):
        if self.is_combined:
            return

        for param in primary_pars:
            val = self.line_dict.get(param.upper())

            # TODO
            # if (param == 'amp') and (self.line_dict['amp_adjust']) and (isinstance(val,dict)):
            #     val = self.get_amp_hyperpar()

            # if (param == 'voff') and (self.line_dict['voff_adjust']) and (isinstance(val,dict)):
            #     val = self.get_voff_hyperpar()

            param_name = self.name + '_' + param.upper()
            self.pr.add_param(name=param_name, expr=val, source=self.name)
            self.comp_params.append(param_name)

        # add profile-unique parameters
        # self.line_profile.initialize_parameters()


    def register_blobs(self):
        if self.is_combined:
            self.br.register_blob(LineVelBlob(name=self.name.upper(), center=self.center, ctx=SpectralLine.ctx))
        self.br.register_blob(ComponentBlob(component=self.name))


    def get_param(self, param_name):
        full_name = self.name + '_' + param_name.upper()
        return self.pr.get_param_val(full_name)


    def add_components(self, comp_dict, host_model):
        if self.is_combined:
            for line in self.children:
                host_model = line.add_components(comp_dict, host_model)

            comp_dict[self.name] = np.sum([comp_dict[line.name] for line in self.children], axis=0)
            return host_model

        line_comp = self.line_profile.construct_line(self)
        comp_dict[self.name] = line_comp
        host_model -= line_comp

        return host_model


    def add_disp_res(self):
        c = const.c.to('km/s').value
        # Perform linear interpolation on the disp_res array as a function of wavelength
        # We will use this to determine the dispersion resolution as a function of wavelength for each
        # emission line so we can correct for the resolution at every iteration.
        # TODO: make this common
        disp_res_ftn = interp1d(SpectralLine.ctx.target.wave,SpectralLine.ctx.target.disp_res,kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))
        # Interpolation function that maps x (in angstroms) to pixels so we can get the exact
        # location in pixel space of the emission line.
        x_pix = np.array(range(len(SpectralLine.ctx.target.wave)))
        pix_interp_ftn = interp1d(SpectralLine.ctx.target.wave,x_pix,kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))

        self.center_pix = float(pix_interp_ftn(self.center)) # line center in pixels
        self.disp_res_ang = float(disp_res_ftn(self.center)) # instrumental FWHM resolution in angstroms
        self.disp_res_kms = (self.disp_res_ang/self.center)*c # instrumental FWHM resolution in km/s







    @staticmethod
    def get_hyperpar_val(par, hparam, line_type='', line_profile=''):
        profile_default = profile_default_hyperpars.get(line_profile, {}).get(par, {}).get(hparam, None)
        if not profile_default is None:
            return profile_default

        hparam_name = par + '_' + hparam
        type_cfg = SpectralLine.ctx.cfg[line_type.lower()]
        # check in order: type_cfg, default type options, common options
        type_default = type_cfg.get(hparam_name, type_default_hyperpars.get(line_type, {}).get(par, {}).get(hparam, type_default_hyperpars['COMMON'].get(par, {}).get(hparam, None)))
        return type_default


    @staticmethod
    def add_tied_param(line_type, par):
        # TODO: instead of add to params, return class parameters
        pre = prefix(line_type)
        param_name = pre + '_' + par
        if param_name in SpectralLine.common_params:
            return

        type_cfg = SpectralLine.ctx.cfg[line_type]
        fp = SpectralLine.param_reg.new_param(name=param_name, expr=type_cfg.get(par, 'FREE'))
        SpectralLine.common_params[param_name] = fp
        if not fp.is_free:
            return

        for hparam in hyperpars:
            # check each in order: type_cfg, default hyperpar dict
            hparam_val = SpectralLine.get_hyperpar_val(line_type, par, hparam)

            if (hparam != 'PRIOR') and (hparam_val is None):
                raise Exception('Could not find voff hyperpar [%s] for line type [%s]'%(hparam_name,line_type))

            setattr(fp, hparam.lower(), hparam_val)


    def set_hyperpars(self, par, args):
        par_name = self.name + '_' + par

        if ((par == 'VOFF') and (SpectralLine.ctx.cfg.comp.tie('voff'))) \
            or ((par == 'DISP') and (SpectralLine.ctx.cfg.comp.tie('disp'))):
            SpectralLine.add_tied_param(self.line_type, par)
            return

        # If not in line_dict or type_options => default to a free parameter
        # TODO: get the expr from the line_dict (should already be filled with needed defaults by config validator)
        fp = SpectralLine.param_reg.new_param(name=par_name, expr=self.line_dict.get(par, self.type_options.get(par, 'FREE')))
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

            if (fp.plim[0] > fp.init) or (fp.init > fp.plim[1]):
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
        # if self.parameters:
        #     s += '\n\tParameters:'
        #     for key, val in self.parameters.items():
        #         s += '\n\t\t%s = %s' % (key, str(val))
        return s


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
    galaxy_csub = ba_utils.continuum_subtract(wave,spec,noise,sigma_clip=2.0,plot=False,verbose=False)

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
