import astropy.constants as const
from dataclasses import dataclass, field
import numpy as np
from prodict import Prodict
from scipy import signal
from scipy.interpolate import interp1d
from typing import Dict, Optional

import badass.utils.constants as consts
import badass.utils.utils as ba_utils
from badass.components.blobs import CombinedLineComponentBlob, LineComponentBlob, LineVelBlob
from badass.components.components import BadassComponent
from badass.components.params import ParameterRegistry

EDGE_PAD = 10

short_type_to_type = {
    'NA': 'NARROW',
    'BR': 'BROAD',
    'ABS': 'ABSORP',
}

type_to_feat_type = {
    'NARROW': 'peaks',
    'BROAD': 'peaks',
    'ABSORP': 'troughs',
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
    spec_features = None

    @staticmethod
    def initialize_spectral_lines(_ctx, _line_list):
        SpectralLine.ctx = _ctx
        SpectralLine.param_reg = SpectralLine.ctx.param_reg
        return [line for line in [SpectralLine.from_dict(line_dict, None) for line_dict in capitalize(_line_list)] if not line is None]


    @staticmethod
    def dump_lines():
        if SpectralLine.ctx is None:
            return

        for line in SpectralLine.ctx.line_list:
            SpectralLine.ctx.log.info(line)


    @classmethod
    def from_dict(cls, line_dict, parent):
        # 'CENTER' is a required argument
        if (not 'CENTER' in line_dict) or (line_dict['CENTER'] is None):
            if parent is None:
                # TODO: remove from line list instead?
                raise Exception('Line center needed for: %s'%line_dict['NAME'])
            line_dict['CENTER'] = parent.center

        # make sure line is in the fitting region
        if (line_dict['CENTER'] <= SpectralLine.ctx.target.wave[0]+EDGE_PAD) or (line_dict['CENTER'] >= SpectralLine.ctx.target.wave[-1]-EDGE_PAD):
            SpectralLine.ctx.log.warn('Not fitting line %s (out of fit region)'%(line_dict['NAME']))
            return None

        # if you do not have a name, one will be provided for you
        if (not 'NAME' in line_dict) or (line_dict['NAME'] is None) or (line_dict['NAME'] == ''):
            line_dict['NAME'] = 'FEAT_.5%f'%line_dict['CENTER']

        # make sure types are all consistent to long names
        line_type = line_dict['TYPE'].upper()
        if len(line_type) < 4: line_type = short_type_to_type[line_type]
        line_dict['TYPE'] = line_type

        # check that user wants to fit this line type
        if (line_type != 'COMBINED') and (not SpectralLine.ctx.cfg.comp.fit(line_type.lower())):
            SpectralLine.ctx.log.warn('Not fitting %s line (unfit type): %s'%(line_type,line_dict['NAME']))
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
        # initialize children array, will update later
        self.children = self.line_dict.get('CHILDREN', [])

        if self.is_combined:
            self.type_options = {}
            self.line_profile = None
        else:
            profile = self.line_dict['PROFILE']
            from badass.components.spectral_lines.line_profiles import LineProfile # need to avoid circular imports
            self.line_profile = LineProfile.get_line_profile(profile)
            if self.line_profile is None:
                raise Exception('Invalid line profile (%s) for line: %s'%(profile,self.name))

        self.add_disp_res()

        super().__init__(SpectralLine.ctx)
        SpectralLine.ctx.log.debug('Added line: %s'%str(self.name))

        # setup children once the parent line is ready
        self.children = [SpectralLine.from_dict(child_dict, self) for child_dict in self.line_dict.get('CHILDREN', [])]


    def initialize_parameters(self):
        if self.is_combined:
            return

        param_vals = {}


        # AMP
        amp_val = self.line_dict.get('AMP')
        # if the user wants BADASS to determine a good amp init guess, and amp is a free parameter
        if (self.line_dict['AMP_ADJUST']) and (isinstance(amp_val,dict)):
            init = self.get_amp_init()
            if (amp_val['plim'][0] <= init) and (init <= amp_val['plim'][1]):
                self.ctx.log.info('Adjusting %s amp to %0.04f'%(self.name,init))
                amp_val['init'] = init

        param_vals['AMP'] = amp_val


        # VOFF
        voff_val = self.line_dict.get('VOFF')
        if SpectralLine.ctx.cfg.comp.tie('voff'):
            # add a parameter for the voff of this line type and set the expr for the line voff to that parameter
            voff_val = self.prefix + '_VOFF'
            tied_voff_val = SpectralLine.ctx.cfg[self.line_type.lower()].voff
            self.pr.add_param(name=voff_val, expr=tied_voff_val, source=self.name)

        # if the user wants BADASS to determine a good voff init guess, and voff is a free parameter
        elif (self.line_dict['VOFF_ADJUST']) and (isinstance(voff_val,dict)):
            init = self.get_voff_init()
            if (voff_val['plim'][0] <= init) and (init <= voff_val['plim'][1]):
                self.ctx.log.info('Adjusting %s voff to %0.04f'%(self.name,init))
                voff_val['init'] = init

        param_vals['VOFF'] = voff_val


        # DISP
        disp_val = self.line_dict.get('DISP')
        if SpectralLine.ctx.cfg.comp.tie('disp'):
            # add a parameter for the disp of this line type and set the expr for the line disp to that parameter
            disp_val = self.prefix + '_DISP'
            tied_disp_val = SpectralLine.ctx.cfg[self.line_type.lower()].disp
            self.pr.add_param(name=disp_val, expr=tied_disp_val, source=self.name)

        param_vals['DISP'] = disp_val

        # register primary parameters
        for param, param_val in param_vals.items():
            param_name = self.name + '_' + param.upper()
            self.pr.add_param(name=param_name, expr=param_val, source=self.name)
            self.comp_params.append(param_name)

        # add profile-unique parameters
        self.line_profile.initialize_parameters(self)


    def get_voff_init(self):
        # derive the voff_init value based on the actual peak (or trough) wavelengths vs the provided line center
        feat_waves = SpectralLine.get_spec_features()[type_to_feat_type[self.line_type]]

        if feat_waves is None:
            return 0.0

        closest_feat = feat_waves[np.argmin(np.abs(feat_waves-self.center))]
        return (closest_feat-self.center)/self.center*consts.c # to km/s


    def get_amp_init(self):
        # derive the amp_init value based on the flux close to the line center
        init = float(self.ctx.fit_spec[ba_utils.find_nearest(self.ctx.fit_wave,self.center)[1]])

        # apply a factor based on the number of components
        amp_factor = len(self.parent.children) if self.parent else 1 # number of siblings (including self)
        init /= amp_factor

        return init


    def register_blobs(self):
        if self.is_combined:
            self.br.register_blob(LineVelBlob(name=self.name.upper(), center=self.center, ctx=SpectralLine.ctx))
            self.br.register_blob(CombinedLineComponentBlob(name=self.name, line=self))
            return

        self.br.register_blob(LineComponentBlob(name=self.name, line=self))


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
        line_comp.resize(len(host_model), refcheck=False) # pad with zeros if needed
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


    @classmethod
    def get_spec_features(cls):
        if not cls.spec_features is None:
            return cls.spec_features

        galaxy_csub = ba_utils.continuum_subtract(cls.ctx.fit_wave,cls.ctx.fit_spec,cls.ctx.fit_noise,sigma_clip=2.0,plot=False,verbose=False)

        try:
            # normalize by noise
            norm_csub = galaxy_csub/cls.ctx.fit_noise

            peaks,_ = signal.find_peaks(norm_csub, height=2.0, width=3.0, prominence=1)
            troughs,_ = signal.find_peaks(-norm_csub, height=2.0, width=3.0, prominence=1)
            peak_wave = cls.ctx.fit_wave[peaks]
            trough_wave = cls.ctx.fit_wave[troughs]
        except:
            peak_wave = np.array()
            trough_wave = np.array()

        if len(peak_wave) == 0:
            peak_wave = np.array()
        if len(trough_wave) == 0:
            trough_wave = np.array()

        cls.spec_features = {
            'peaks': peak_wave,
            'trough': trough_wave,
        }

        return cls.spec_features


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

