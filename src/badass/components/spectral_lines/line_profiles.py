from badass.badass_utils import gh_alternative as gh_alt
from numbers import Number
import numexpr as ne
import numpy as np
from numpy.polynomial import hermite
from scipy import special

from badass.components.spectral_lines.spectral_line import SpectralLine, hyperpars

# Valid line profiles: will be populated as each profile class is defined
line_profiles = {}

class LineProfile:

    @staticmethod
    def get_line_profile(profile_name):
        return line_profiles.get(profile_name, None)


    @staticmethod
    def initialize_parameters(line):
        pass


    @staticmethod
    def construct_line(line):
        return None


class GaussianProfile(LineProfile):
    name = 'GAUSSIAN'

    @staticmethod
    def construct_line(line):

        # dispersion in pixels (velscale = km/s/pixel)
        sigma_pix = line.get_param('disp') / SpectralLine.ctx.target.velscale
        if sigma_pix <= 0.01: sigma_pix = 0.01

        # velocity offset in pixels
        voff_pix = line.get_param('voff') / SpectralLine.ctx.target.velscale
        # shift the line center by voff in pixels
        center_pix = line.center_pix + voff_pix

        # pixels vector
        x_pix = np.array(range(len(SpectralLine.ctx.fit_wave)), dtype=float)
        # reshape into row
        x_pix = x_pix.reshape((len(x_pix), 1))

        # construct Gaussian
        g = line.get_param('amp') * np.exp(-0.5*(x_pix-center_pix)**2/sigma_pix**2)
        g = np.sum(g, axis=1)

        # Make sure edges of gaussian are zero to avoid wierd things
        g[(g > -1e-6) & (g < 1e-6)] = 0.0
        g[np.isnan(g)] = 0.0
        g[0] = g[1]
        g[-1] = g[-2]

        if np.all([np.isnan(g)]):
            breakpoint()

        return g

line_profiles[GaussianProfile.name] = GaussianProfile



class LorentzianProfile(LineProfile):
    name = 'LORENTZIAN'

    @staticmethod
    def construct_line(line):
        # Produces a lorentzian vector the length of x with the specified parameters
        # (See: https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Lorentz1D.html)

        fwhm = line.get_param('disp')*2.3548
        # fwhm in pixels (velscale = km/s/pixel)
        fwhm_pix = fwhm / SpectralLine.ctx.target.velscale
        if fwhm_pix <= 0.01: fwhm_pix = 0.01

        # velocity offset in pixels
        voff_pix = line.get_param('voff') / SpectralLine.ctx.target.velscale
        # shift the line center by voff in pixels
        center_pix = line.center_pix + voff_pix

        # pixels vector
        x_pix = np.array(range(len(SpectralLine.ctx.fit_wave)), dtype=float)
        # reshape into row
        x_pix = x_pix.reshape((len(x_pix), 1))

        gamma = 0.5*fwhm_pix
        # construct lorenzian
        l = amp*((gamma**2) / (gamma**2+(x_pix-center_pix)**2))
        l = np.sum(l, axis=1)

        # Make sure edges of lorenzian are zero to avoid wierd things
        l[(l > -1e-6) & (l < 1e-6)] = 0.0
        l[0] = l[1]
        l[-1] = l[-2]
        return l

line_profiles[LorentzianProfile.name] = LorentzianProfile



class VoigtProfile(LineProfile):
    name = 'VOIGT'

    @staticmethod
    def initialize_parameters(line):
        shape_val = line.line_dict.get('SHAPE')

        if SpectralLine.ctx.cfg.comp.tie('disp'):
            # add a parameter for the shape of this line type and set the expr for the line shape to that parameter
            shape_val = line.prefix + '_SHAPE'
            tied_shape_val = SpectralLine.ctx.cfg[line.line_type.lower()].shape
            line.pr.add_param(name=shape_val, expr=tied_shape_val, source=line.name)


        param_name = line.name + '_SHAPE'
        line.pr.add_param(name=param_name, expr=shape_val, source=line.name)
        line.comp_params.append(param_name)


    @staticmethod
    def construct_line(line):
        # Pseudo-Voigt profile implementation from:
        # https://docs.mantidproject.org/nightly/fitting/fitfunctions/PseudoVoigt.html

        # fwhm in pixels (velscale = km/s/pixel)
        fwhm_pix = (line.get_param('disp')*2.3548) / SpectralLine.ctx.target.velscale
        if fwhm_pix <= 0.01: fwhm_pix = 0.01

        sigma_pix = fwhm_pix/2.3548
        if sigma_pix <= 0.01: sigma_pix = 0.01

        # velocity offset in pixels
        voff_pix = line.get_param('voff') / SpectralLine.ctx.target.velscale
        # shift the line center by voff in pixels
        center_pix = line.center_pix + voff_pix

        # pixels vector
        x_pix = np.array(range(len(SpectralLine.ctx.fit_wave)), dtype=float)
        # reshape into row
        x_pix = x_pix.reshape((len(x_pix), 1))

        # Gaussian contribution
        a_G = 1.0/(sigma_pix * np.sqrt(2.0*np.pi))
        g = a_G * np.exp(-0.5*(x_pix-(center_pix))**2/(sigma_pix)**2)
        g = np.sum(g, axis=1)

        # Lorentzian contribution
        l = (1.0/np.pi) * (fwhm_pix/2.0)/((x_pix-center_pix)**2 + (fwhm_pix/2.0)**2)
        l = np.sum(l,axis=1)

        # Voigt profile
        shape = line.get_param('shape')
        pv = (float(shape) * g) + ((1.0-float(shape))*l)

        # Normalize and multiply by amplitude
        pv = pv/np.max(pv) * line.get_param('amp')

        # Replace the ends with the same value
        pv[(pv > -1e-6) & (pv < 1e-6)] = 0.0
        pv[0] = pv[1]
        pv[-1] = pv[-2]
        return pv


line_profiles[VoigtProfile.name] = VoigtProfile



class GaussHermiteProfile(LineProfile):
    name = 'GAUSS-HERMITE'

    @staticmethod
    def initialize_parameters(line):
        n_moments = line.line_dict.get('N_MOMENTS')
        if n_moments < 3:
            return

        h_val = line.line_dict.get('H')
        for par in ['H%d'%m for m in range(3, 3+n_moments-2)]:
            par_val = line.line_dict.get(par, h_val)
            if SpectralLine.ctx.cfg.comp.tie('disp'):
                # add a parameter for the moments of this line type and set the expr for the line moments to that parameter
                par_val = line.prefix + '_' + par
                tied_h_val = SpectralLine.ctx.cfg[line.line_type.lower()].h
                line.pr.add_param(name=par_val, expr=tied_h_val, source=line.name)

            param_name = line.name + '_' + par
            line.pr.add_param(name=param_name, expr=par_val, source=line.name)
            line.comp_params.append(param_name)


    @staticmethod
    def construct_line(line):
        # Produces a Gauss-Hermite vector the length of x with the specified parameters
        h_moments = None
        n_moments = line.line_dict.get('N_MOMENTS')
        if n_moments > 0:
            h_moments = np.empty(n_moments)
            for i, m in enumerate(range(3,3+n_moments)):
                h_moments[i] = line.get_param('H%d'%m)

        # dispersion in pixels (velscale = km/s/pixel)
        sigma_pix = line.get_param('disp') / SpectralLine.ctx.target.velscale
        if sigma_pix <= 0.01: sigma_pix = 0.01

        # velocity offset in pixels
        voff_pix = line.get_param('voff') / SpectralLine.ctx.target.velscale
        # shift the line center by voff in pixels
        center_pix = line.center_pix + voff_pix

        # pixels vector
        x_pix = np.array(range(len(SpectralLine.ctx.fit_wave)), dtype=float)
        x_pix = x_pix.reshape((len(x_pix), 1))

        # Taken from Riffel 2010 - profit: a new alternative for emission-line profile fitting
        w = (x_pix-center_pix) / sigma_pix
        alpha = 1.0/np.sqrt(2.0)*np.exp(-w**2/2.0)

        amp = line.get_param('amp')
        if h_moments is None:
            coeff = np.array([1, 0, 0])
        else:
            mom = len(h_moments)+2
            n = np.arange(3, mom + 1)
            nrm = np.sqrt(special.factorial(n)*2**n) # Normalization
            coeff = np.append([1, 0, 0], h_moments/nrm)

        h = hermite.hermval(w,coeff)
        g = (amp*alpha)/sigma_pix*h
        g = np.sum(g, axis=1)

        # we ensure any values of the line profile that are negative are zeroed out (See Van der Marel 1993)
        g[g < 0] = 0.0
        g = g/np.max(g) # normalize to 1
        g = amp*g # apply amplitude

        # replace the ends with the same value
        g[(g > -1e-6) & (g < 1e-6)] = 0.0
        g[0] = g[1]
        g[-1] = g[-2]
        return g


line_profiles[GaussHermiteProfile.name] = GaussHermiteProfile



class LaplaceProfile(LineProfile):
    name = 'LAPLACE'

    @staticmethod
    def initialize_parameters(line):
        h_val = line.line_dict.get('H')
        for par in ['H3','H4']:
            par_val = line.line_dict.get(par, h_val)
            if SpectralLine.ctx.cfg.comp.tie('disp'):
                # add a parameter for the moments of this line type and set the expr for the line moments to that parameter
                par_val = line.prefix + '_' + par
                tied_h_val = SpectralLine.ctx.cfg[line.line_type.lower()].h
                line.pr.add_param(name=par_val, expr=tied_h_val, source=line.name)

            param_name = line.name + '_' + par
            line.pr.add_param(name=param_name, expr=par_val, source=line.name)
            line.comp_params.append(param_name)


    @staticmethod
    def construct_line(line):
        # Produces a Laplace kernel vector the length of x with the specified parameters
        # Laplace kernel from Sanders & Evans (2020): https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5806S/abstract

        # dispersion in pixels (velscale = km/s/pixel)
        sigma_pix = line.get_param('disp') / SpectralLine.ctx.target.velscale
        if sigma_pix <= 0.01: sigma_pix = 0.01

        # velocity offset in pixels
        voff_pix = line.get_param('voff') / SpectralLine.ctx.target.velscale
        # shift the line center by voff in pixels
        center_pix = line.center_pix + voff_pix

        # Note that the pixel vector must be a float type otherwise
        # the GH alternative functions return NaN.
        x_pix = np.array(range(len(SpectralLine.ctx.fit_wave)), dtype=float) # pixels vector
        g = gh_alt.laplace_kernel_pdf(x_pix, 0.0, center_pix, sigma_pix, line.get_param('h3'), line.get_param('h4'))

        # We ensure any values of the line profile that are negative
        g[g < 0] = 0.0
        g = g/np.nanmax(g) # Normalize to 1
        g = line.get_param('amp')*g # Apply amplitude

        # Replace the ends with the same value
        g[(g > -1e-6) & (g < 1e-6)] = 0.0
        g[0] = g[1]
        g[-1] = g[-2]
        return g

line_profiles[LaplaceProfile.name] = LaplaceProfile



class UniformProfile(LineProfile):
    name = 'UNIFORM'

    @staticmethod
    def initialize_parameters(line):
        LaplaceProfile.initialize_parameters(line)


    @staticmethod
    def construct_line(line):
        # Produces a Uniform kernel vector the length of x with the specified parameters
        # Uniform kernel from Sanders & Evans (2020): https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5806S/abstract

        # dispersion in pixels (velscale = km/s/pixel)
        sigma_pix = line.get_param('disp') / SpectralLine.ctx.target.velscale
        if sigma_pix <= 0.01: sigma_pix = 0.01
        
        # velocity offset in pixels
        voff_pix = line.get_param('voff') / SpectralLine.ctx.target.velscale
        # shift the line center by voff in pixels
        center_pix = line.center_pix + voff_pix

        # Note that the pixel vector must be a float type otherwise
        # the GH alternative functions return NaN.
        x_pix = np.array(range(len(SpectralLine.ctx.fit_wave)), dtype=float) # pixels vector
        g = gh_alt.uniform_kernel_pdf(x_pix, 0.0, center_pix, sigma_pix, line.get_param('h3'), line.get_param('h4'))

        # We ensure any values of the line profile that are negative
        g[g < 0] = 0.0
        g = g/np.nanmax(g) # Normalize to 1
        g = line.get_param('amp')*g # Apply amplitude

        # Replace the ends with the same value
        g[(g > -1e-6) & (g < 1e-6)] = 0.0
        g[0] = g[1]
        g[-1] = g[-2]
        return g

line_profiles[UniformProfile.name] = UniformProfile

