import numpy as np

from templates.common import BadassTemplate

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

    @classmethod
    def initialize_template(cls, ctx):
        if not ctx.options.comp_options.fit_power:
            return None

        temp_type = ctx.options.power_options.type
        class_name = '%sPowerLawTemplate'%temp_type.capitalize()
        if not class_name in globals():
            ctx.log.error('Power Law template unsupported: %s' % temp_type)
            return None

        temp_class = globals()[class_name]
        return temp_class(ctx)


    def __init__(self, ctx):
        self.ctx = ctx


# Simple Power-Law (AGN continuum)
class SimplePowerLawTemplate(PowerLawTemplate):

    def __init__(self, ctx):
        super().__init__(ctx)
        self.ctx.log.info('- Fitting Simple AGN power-law continuum')


    def initialize_parameters(self, params, args):
        # AGN simple power-law amplitude
        params['POWER_AMP'] = {
                                'init':(0.5*args['median_flux']),
                                'plim':(0,args['max_flux']),
                              }

        # AGN simple power-law slope
        params['POWER_SLOPE'] = {
                                    'init':-1.0,
                                    'plim':(-6.0,6.0),
                                }


    def add_components(self, params, comp_dict, host_model):
        power = simple_power_law(self.ctx.fit_wave, params['POWER_AMP'], params['POWER_SLOPE'])
        comp_dict['POWER'] = power
        return host_model - power


# Smoothly-Broken Power-Law (AGN continuum)
class BrokenPowerLawTemplate(PowerLawTemplate):

    def __init__(self, ctx):
        super().__init__(ctx)
        self.ctx.log.info('- Fitting Smoothly-Broken AGN power-law continuum')


    def initialize_parameters(self, params, args):
        # AGN simple power-law amplitude
        params['POWER_AMP'] = {
                                'init':(0.5*args['median_flux']),
                                'plim':(0,args['max_flux']),
                              }

        # AGN simple power-law break wavelength
        params['POWER_BREAK'] = {
                                    'init':(np.max(self.ctx.fit_wave) - (0.5*(np.max(self.ctx.fit_wave)-np.min(self.ctx.fit_wave)))),
                                    'plim':(np.min(self.ctx.fit_wave), np.max(self.ctx.fit_wave)),
                                }

        # AGN simple power-law slope 1 (blue side)
        params['POWER_SLOPE_1'] = {
                                    'init':-1.0,
                                    'plim':(-6.0,6.0),
                                  }

        # AGN simple power-law slope 2 (red side)
        params['POWER_SLOPE_2'] = {
                                    'init':-1.0,
                                    'plim':(-6.0,6.0),
                                  }

        # Power-law curvature parameter (Delta)
        params['POWER_CURVATURE'] = {
                                        'init':0.10,
                                        'plim':(0.01,1.0),
                                    }


    def add_components(self, params, comp_dict, host_model):
        power = broken_power_law(self.ctx.fit_wave, params['POWER_AMP'], params['POWER_BREAK'],
                                         params['POWER_SLOPE_1'], cur_params['POWER_SLOPE_2'],
                                         params['POWER_CURVATURE'])
        comp_dict['POWER'] = power
        return host_model - power
