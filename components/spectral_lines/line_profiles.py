import gh_alternative as gh_alt
import numexpr as ne
import numpy as np
from numpy.polynomial import hermite
from scipy import special

def line_constructor(ctx, line_name, line_dict):

    line_profile = line_dict['line_profile']

    def get_attr(attr):
        attr_val = line_dict.get(attr, 'free')
        if (isinstance(attr_val, str)) and (attr_val != 'free'):
            return ne.evaluate(attr_val, local_dict=ctx.cur_params).item()
        return ctx.cur_params[line_name+'_'+attr.upper()]

    amp = get_attr('amp')
    amp = amp if np.isfinite(amp) else 0.0
    disp = get_attr('disp')
    disp = disp if np.isfinite(disp) else 100.0
    voff = get_attr('voff')
    voff = voff if np.isfinite(voff) else 0.0

    if line_profile == 'gaussian':
        line_model = gaussian_line_profile(ctx.fit_wave, amp, disp, voff, line_dict['center_pix'], ctx.target.velscale)

    elif line_profile == 'lorentzian':
        line_model = lorentzian_line_profile(ctx.fit_wave, amp, disp, voff, line_dict['center_pix'], ctx.target.velscale)

    elif line_profile == 'gauss-hermite':
        n_moments = len([k for k in line_dict.keys() if k in ['h3','h4','h5','h6','h7','h8','h9','h10']])
        hmoments = None
        if n_moments > 0:
            hmoments = np.empty(n_moments)
            for i,m in enumerate(range(3,3+n_moments)):
                hattr = 'h'+str(m)
                hmoments[i] = get_attr(hattr)
        line_model = gauss_hermite_line_profile(ctx.fit_wave, amp, disp, voff, hmoments, line_dict['center_pix'], ctx.target.velscale)

    elif line_profile == 'laplace':
        hmoments = np.empty(2)
        for i,m in enumerate(range(3,5)):
            hattr = 'h'+str(m)
            hmoments[i] = get_attr(hattr)
        line_model = laplace_line_profile(ctx.fit_wave, amp, disp, voff, hmoments, line_dict['center_pix'], ctx.target.velscale)

    elif line_profile == 'uniform':
        hmoments = np.empty(2)
        for i,m in enumerate(range(3,5)):
            hattr = 'h'+str(m)
            hmoments[i] = get_attr(hattr)
        line_model = uniform_line_profile(ctx.fit_wave, amp, disp, voff, hmoments, line_dict['center_pix'], ctx.target.velscale)

    elif line_profile == 'voigt':
        shape = get_attr('shape')
        line_model = voigt_line_profile(ctx.fit_wave, amp, disp, voff, shape, line_dict['center_pix'], ctx.target.velscale)

    else:
        return None

    line_model[~np.isfinite(line_model)] = 0.0
    return line_model


def gaussian_line_profile(wave, amp, disp, voff, center_pix, velscale):
    # Produces a gaussian vector the length of x with the specified parameters

    sigma = disp # Gaussian dispersion in km/s
    sigma_pix = sigma / velscale # dispersion in pixels (velscale = km/s/pixel)
    if sigma_pix <= 0.01: sigma_pix = 0.01
    voff_pix = voff / velscale # velocity offset in pixels
    center_pix = center_pix + voff_pix # shift the line center by voff in pixels

    x_pix = np.array(range(len(wave)), dtype=float) # pixels vector
    x_pix = x_pix.reshape((len(x_pix), 1)) # reshape into row
    g = amp*np.exp(-0.5*(x_pix-center_pix)**2/sigma_pix**2) # construct gaussian
    g = np.sum(g, axis=1)

    # Make sure edges of gaussian are zero to avoid wierd things
    g[(g > -1e-6) & (g < 1e-6)] = 0.0
    g[0] = g[1]
    g[-1] = g[-2]
    return g


def lorentzian_line_profile(wave, amp, disp, voff, center_pix, velscale):
    # Produces a lorentzian vector the length of x with the specified parameters
    # (See: https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Lorentz1D.html)

    fwhm = disp*2.3548
    fwhm_pix = fwhm / velscale # fwhm in pixels (velscale = km/s/pixel)
    if fwhm_pix <= 0.01: fwhm_pix = 0.01
    voff_pix = voff / velscale # velocity offset in pixels
    center_pix = center_pix + voff_pix # shift the line center by voff in pixels

    x_pix = np.array(range(len(wave)), dtype=float) # pixels vector
    x_pix = x_pix.reshape((len(x_pix), 1)) # reshape into row
    gamma = 0.5*fwhm_pix
    l = amp*((gamma**2) / (gamma**2+(x_pix-center_pix)**2)) # construct lorenzian
    l= np.sum(l, axis=1)

    # Make sure edges of lorenzian are zero to avoid wierd things
    l[(l > -1e-6) & (l < 1e-6)] = 0.0
    l[0] = l[1]
    l[-1] = l[-2]
    return l


def gauss_hermite_line_profile(wave, amp, disp, voff, hmoments, center_pix, velscale):
    # Produces a Gauss-Hermite vector the length of x with the specified parameters

    sigma_pix = disp / velscale # dispersion in pixels (velscale = km/s/pixel)
    if sigma_pix <= 0.01: sigma_pix = 0.01
    voff_pix = voff / velscale # velocity offset in pixels
    center_pix = center_pix + voff_pix # shift the line center by voff in pixels

    x_pix = np.array(range(len(wave)), dtype=float) # pixels vector
    x_pix = x_pix.reshape((len(x_pix), 1))

    # Taken from Riffel 2010 - profit: a new alternative for emission-line profile fitting
    w = (x_pix-center_pix) / sigma_pix
    alpha = 1.0/np.sqrt(2.0)*np.exp(-w**2/2.0)

    if hmoments is None:
        coeff = np.array([1, 0, 0])
        h = hermite.hermval(w, coeff)
        g = (amp*alpha) / sigma_pix*h
    else:
        mom = len(hmoments)+2
        n = np.arange(3, mom + 1)
        nrm = np.sqrt(special.factorial(n)*2**n) # Normalization
        coeff = np.append([1, 0, 0], hmoments/nrm)
        h = hermite.hermval(w,coeff)
        g = (amp*alpha)/sigma_pix*h

    g = np.sum(g, axis=1)
    # We ensure any values of the line profile that are negative are zeroed out (See Van der Marel 1993)
    g[g < 0] = 0.0
    g = g/np.max(g) # Normalize to 1
    g = amp*g # Apply amplitude

    # Replace the ends with the same value
    g[(g > -1e-6) & (g < 1e-6)] = 0.0
    g[0] = g[1]
    g[-1] = g[-2]
    return g


def laplace_line_profile(wave, amp, disp, voff, hmoments, center_pix, velscale):
    # Produces a Laplace kernel vector the length of x with the specified parameters
    # Laplace kernel from Sanders & Evans (2020): https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5806S/abstract

    sigma_pix = disp / velscale # dispersion in pixels (velscale = km/s/pixel)
    if sigma_pix <= 0.01: sigma_pix = 0.01
    voff_pix = voff / velscale # velocity offset in pixels
    center_pix = center_pix + voff_pix # shift the line center by voff in pixels

    # Note that the pixel vector must be a float type otherwise
    # the GH alternative functions return NaN.
    x_pix = np.array(range(len(wave)), dtype=float) # pixels vector
    g = gh_alt.laplace_kernel_pdf(x_pix, 0.0, center_pix, sigma_pix, hmoments[0], hmoments[1])

    # We ensure any values of the line profile that are negative
    g[g < 0] = 0.0
    g = g/np.nanmax(g) # Normalize to 1
    g = amp*g # Apply amplitude

    # Replace the ends with the same value
    g[(g > -1e-6) & (g < 1e-6)] = 0.0
    g[0] = g[1]
    g[-1] = g[-2]
    return g


def uniform_line_profile(wave, amp, disp, voff, hmoments, center_pix, velscale):
    # Produces a Uniform kernel vector the length of x with the specified parameters
    # Uniform kernel from Sanders & Evans (2020): https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5806S/abstract

    sigma_pix = disp / velscale # dispersion in pixels (velscale = km/s/pixel)
    if sigma_pix <= 0.01: sigma_pix = 0.01
    voff_pix = voff / velscale # velocity offset in pixels
    center_pix = center_pix + voff_pix # shift the line center by voff in pixels

    # Note that the pixel vector must be a float type otherwise
    # the GH alternative functions return NaN.
    x_pix = np.array(range(len(wave)), dtype=float) # pixels vector
    g = gh_alt.uniform_kernel_pdf(x_pix, 0.0, center_pix, sigma_pix, hmoments[0], hmoments[1])

    # We ensure any values of the line profile that are negative
    g[g < 0] = 0.0
    g = g/np.nanmax(g) # Normalize to 1
    g = amp*g # Apply amplitude

    # Replace the ends with the same value
    g[(g > -1e-6) & (g < 1e-6)] = 0.0
    g[0] = g[1]
    g[-1] = g[-2]
    return g


def voigt_line_profile(wave, amp, disp, voff, shape, center_pix, velscale):
    # Pseudo-Voigt profile implementation from:
    # https://docs.mantidproject.org/nightly/fitting/fitfunctions/PseudoVoigt.html

    fwhm_pix = (disp*2.3548) / velscale # fwhm in pixels (velscale = km/s/pixel)
    if fwhm_pix <= 0.01: fwhm_pix = 0.01
    sigma_pix = fwhm_pix/2.3548
    if sigma_pix <= 0.01: sigma_pix = 0.01
    voff_pix = voff / velscale # velocity offset in pixels
    center_pix = center_pix + voff_pix # shift the line center by voff in pixels

    x_pix = np.array(range(len(wave)), dtype=float) # pixels vector
    x_pix = x_pix.reshape((len(x_pix), 1)) # reshape into row

    # Gaussian contribution
    a_G = 1.0/(sigma_pix * np.sqrt(2.0*np.pi))
    g = a_G * np.exp(-0.5*(x_pix-(center_pix))**2/(sigma_pix)**2)
    g = np.sum(g, axis=1)

    # Lorentzian contribution
    l = (1.0/np.pi) * (fwhm_pix/2.0)/((x_pix-center_pix)**2 + (fwhm_pix/2.0)**2)
    l = np.sum(l,axis=1)

    # Voigt profile
    pv = (float(shape) * g) + ((1.0-float(shape))*l)

    # Normalize and multiply by amplitude
    pv = pv/np.max(pv)*amp

    # Replace the ends with the same value
    pv[(pv > -1e-6) & (pv < 1e-6)] = 0.0
    pv[0] = pv[1]
    pv[-1] = pv[-2]
    return pv
