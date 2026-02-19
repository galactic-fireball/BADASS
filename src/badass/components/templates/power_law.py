import numpy as np

from badass.components.templates.common import BadassTemplate

def simple_power_law(x, amp, alpha):
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


class PowerLawTemplate(BadassTemplate):

    OPTION_NAME = 'power'
    PARAM_PREFIX = 'POWER_'

    @classmethod
    def initialize_component(cls, ctx):
        if not ctx.cfg.comp.fit_power:
            return None

        temp_type = ctx.cfg.power.type
        class_name = '%sPowerLawTemplate'%temp_type.capitalize()
        if not class_name in globals():
            ctx.log.error('Power Law template unsupported: %s' % temp_type)
            return None

        temp_class = globals()[class_name]
        return temp_class(ctx)


# Simple Power-Law (AGN continuum)
class SimplePowerLawTemplate(PowerLawTemplate):

    TEMPLATE_PARAMS = ['amp', 'slope']

    # TODO: have a 'description' variable to log instead of overriding each time
    def __init__(self, ctx):
        super().__init__(ctx)
        self.ctx.log.info('- Fitting Simple AGN power-law continuum')


    def add_components(self, comp_dict, host_model):
        amp = self.get_param('amp')
        slope = self.get_param('slope')

        power = simple_power_law(self.ctx.fit_wave, amp, slope)
        comp_dict['POWER'] = power
        return host_model - power


# Smoothly-Broken Power-Law (AGN continuum)
class BrokenPowerLawTemplate(PowerLawTemplate):

    TEMPLATE_PARAMS = ['amp', 'break_', 'slope_1', 'slope_2', 'curvature']

    def __init__(self, ctx):
        super().__init__(ctx)
        self.ctx.log.info('- Fitting Smoothly-Broken AGN power-law continuum')


    def add_components(self, comp_dict, host_model):
        power = broken_power_law(self.ctx.fit_wave, self.get_param('amp'), self.get_param('break_'),
                                         self.get_param('slope_1'), self.get_param('slope_2'),
                                         self.get_param('curvature'))
        comp_dict['POWER'] = power
        return host_model - power
