import astropy.constants as const
from dataclasses import dataclass
from prodict import Prodict
from scipy import signal
from scipy.interpolate import interp1d

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
            res_dict['CHILDREN'] = capitalize(children)
        res.append(res_dict)
    return res


primary_pars = ['AMP', 'DISP', 'VOFF']

@dataclass
class FitParameter:
    name: str = ''
    value: str = 'FREE'
    is_free: bool = False
    init: float = 0.0
    plim: tuple = (0.0,0.0)
    prior: str = ''

    def __post_init__(self):
        self.is_free = self.value == 'FREE'


class SpectralLine:

    ctx = None
    line_list = []

    @staticmethod
    def initialize_spectral_lines(_ctx, _line_list):
        SpectralLine.ctx = _ctx
        SpectralLine.line_list = [SpectralLine.from_dict(line_dict, None) for line_dict in capitalize(_line_list)]


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


    @classmethod
    def from_dict(cls, line_dict, parent):
        if not 'NAME' in line_dict: # TODO: assign name based on center?
            raise Exception('Line with missing name')

        if not 'TYPE' in line_dict: # TODO: default to narrow?
            raise Exception('Line type needed for: %s'%line_dict['NAME'])
        line_type = line_dict['TYPE'].upper()
        if len(line_type) < 4: line_type = short_type_to_type[line_type]
        line_dict['TYPE'] = line_type

        if (line_type != 'COMBINED') and (not SpectralLine.ctx.options.comp_options['fit_%s'%line_type.lower()]):
            SpectralLine.ctx.log.warning('Not fitting %s line (unfit type): %s'%(line_type,line_dict['NAME']))
            return None

        center = line_dict.get('CENTER', parent.center if parent else None)
        if center is None:
            raise Exception('Line center needed for: %s'%line_dict['NAME'])

        if (center <= SpectralLine.ctx.target.wave[0]+EDGE_PAD) or (center >= SpectralLine.ctx.target.wave[-1]-EDGE_PAD):
            SpectralLine.ctx.log.warning('Not fitting %s line (out of fit region): %s'%(line_type,line_dict['NAME']))
            return None

        return cls(line_dict, parent)


    def __init__(self, line_dict, parent):
        self.line_dict = line_dict
        self.parent = parent
        self.name = self.line_dict['NAME']
        self.line_type = self.line_dict['TYPE']
        self.is_combined = self.line_type == 'COMBINED'
        self.prefix = prefix(self.line_type)
        self.center = self.line_dict.get('CENTER', parent.center if parent else None) # TODO: center is unit-configurable

        if self.is_combined:
            self.type_options = {}
            self.line_profile = None
        else:
            self.type_options = SpectralLine.ctx.options['%s_options'%self.line_type.lower()]
            profile = self.line_dict.get('PROFILE', self.type_options.get('profile', None))
            if profile is None:
                raise Exception('Line profile required for non-combined line: %s'%self.name)
            from components.spectral_lines.line_profiles import LineProfile # need to avoid circular imports
            self.line_profile = LineProfile.get_line_profile(profile)
            if self.line_profile is None:
                raise Exception('Invalid line profile (%s) for line: %s'%(profile,self.name))

        self.children = [SpectralLine.from_dict(child_dict, self) for child_dict in self.line_dict.get('CHILDREN', [])]

        # Fit parameters, either free, constant, or expression
        # Parameters for combined lines will be calculated post-fit
        self.parameters = {}
        if self.is_combined:
            SpectralLine.ctx.log.debug('Added line: %s'%str(self))
            return

        for par in primary_pars:
            par_name = self.name + '_' + par
            self.parameters[par_name] = FitParameter(name=par_name, value=self.line_dict.get(par, 'FREE'))

        if SpectralLine.ctx.options.comp_options.tie_line_disp:
            par_name = self.name + '_DISP'
            self.parameters[par_name] = FitParameter(name=par_name, value=self.prefix + '_DISP')

        if SpectralLine.ctx.options.comp_options.tie_line_voff:
            par_name = self.name + '_VOFF'
            self.parameters[par_name] = FitParameter(name=par_name, value=self.prefix + '_VOFF')

        self.line_profile.add_parameters(self) # add profile-unique parameters
        SpectralLine.ctx.log.debug('Added line: %s'%str(self))


    def __str__(self):
        s = '%s (%s%s) @ %.04f'%(self.name, self.line_type, ' %s'%self.line_profile.name if self.line_profile else '', self.center)
        if len(self.children):
            s += '\n\tChildren: %s' % ', '.join([child.name for child in self.children])
        if self.parameters:
            s += '\n\tParameters:'
            for key, val in self.parameters.items():
                s += '\n\t\t%s = %s' % (key, str(val))
        return s


    def initialize_parameters(self, params, args):
        if self.is_combined:
            for child in self.children:
                child.initialize_parameters(params, args)
            return


        self.line_profile.init


        for param in self.parameters.value():
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
    def initialize_parameters(params, args):
        ctx = SpectralLine.ctx
        peaks, troughs = get_spec_features(ctx.fit_wave,ctx.fit_spec,ctx.fit_noise,line_list=SpectralLine.line_list)
        args['peaks'] = peaks
        args['troughs'] = troughs

        for line in ctx.line_list:
            line.initialize_parameters(params, args)






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

















