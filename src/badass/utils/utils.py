from astropy import coordinates
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
from scipy.ndimage import generic_filter
from scipy.integrate import simpson
import scipy.optimize as op

from badass.utils.constants import *


def dered(wave, z=0.0):
    return wave / (1 + z)


def redden(wave, z=0.0):
    return wave * (1 + z)


def flux_to_lum(flux, cosmology, z):
    # TODO: calc and store elsewhere
    cosmo = FlatLambdaCDM(cosmology.H0, cosmology.Om0)
    d_mpc = cosmo.luminosity_distance(z).value
    # TODO: use astropy units
    d_cm = d_mpc * 3.086E+24 # 1 Mpc = 3.086e+24 cm
    return 4*np.pi*(d_cm**2)*flux


def valid_expression(expr, local_dict):
    try:
        val = ne.evaluate(expr, local_dict=local_dict).item()
        return True
    except KeyError:
        return False


def find_nearest(array, value):
    """
    This function finds the nearest value in an array and returns the 
    closest value and the corresponding index.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def time_convert(seconds): 
    """
    Converts runtimes in seconds to hours:minutes:seconds format.
    """
    seconds = seconds % (24. * 3600.) 
    hour = seconds // 3600.
    seconds %= 3600.
    minutes = seconds // 60.
    seconds %= 60.
      
    return "%d:%02d:%02d" % (hour, minutes, seconds)


# TODO: implement generating numbered output diretories
def get_default_outdir(infile):
    return infile.parent.joinpath(DEFAULT_OUTDIR)


def nan_helper(y):
    """
    Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def get_ebv(ra, dec):
    if (ra is None) or (dec is None):
        return GALACTIC_EBV

    co = coordinates.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='fk5')
    try:
        table = IrsaDust.get_query_table(co, section='ebv')
        ebv = table['ext SandF mean'][0]
    except:
        return GALACTIC_EBV

    # If E(B-V) is large, it can significantly affect normalization of the
    # spectrum, in addition to changing its shape.  Re-normalizing the spectrum
    # throws off the maximum likelihood fitting, so instead of re-normalizing,
    # we set an upper limit on the allowed ebv value for Galactic de-reddening.
    if ebv >= 1.0:
        return GALACTIC_EBV
    return ebv


# Galactic Extinction Correction
def ccm_unred(wave, flux, ebv, r_v=3.1):
    """ccm_unred(wave, flux, ebv, r_v="")
    Deredden a flux vector using the CCM 1989 parameterization 
    Returns an array of the unreddened flux
    
    INPUTS:
    wave - array of wavelengths (in Angstroms)
    dec - calibrated flux array, same number of elements as wave
    ebv - colour excess E(B-V) float. If a negative ebv is supplied
          fluxes will be reddened rather than dereddened     
    
    OPTIONAL INPUT:
    r_v - float specifying the ratio of total selective
          extinction R(V) = A(V)/E(B-V). If not specified,
          then r_v = 3.1
            
    OUTPUTS:
    funred - unreddened calibrated flux array, same number of 
             elements as wave
             
    NOTES:
    1. This function was converted from the IDL Astrolib procedure
       last updated in April 1998. All notes from that function
       (provided below) are relevant to this function 
       
    2. (From IDL:) The CCM curve shows good agreement with the Savage & Mathis (1979)
       ultraviolet curve shortward of 1400 A, but is probably
       preferable between 1200 and 1400 A.
    3. (From IDL:) Many sightlines with peculiar ultraviolet interstellar extinction 
       can be represented with a CCM curve, if the proper value of 
       R(V) is supplied.
    4. (From IDL:) Curve is extrapolated between 912 and 1000 A as suggested by
       Longo et al. (1989, ApJ, 339,474)
    5. (From IDL:) Use the 4 parameter calling sequence if you wish to save the 
       original flux vector.
    6. (From IDL:) Valencic et al. (2004, ApJ, 616, 912) revise the ultraviolet CCM
       curve (3.3 -- 8.0 um-1). But since their revised curve does
       not connect smoothly with longer and shorter wavelengths, it is
       not included here.
    
    7. For the optical/NIR transformation, the coefficients from 
       O'Donnell (1994) are used
    
    >>> ccm_unred([1000, 2000, 3000], [1, 1, 1], 2 ) 
    array([9.7976e+012, 1.12064e+07, 32287.1])
    """
    wave = np.array(wave, float)
    flux = np.array(flux, float)
    
    if wave.size != flux.size: raise TypeError( 'ERROR - wave and flux vectors must be the same size')

    x = 10000.0/wave
    # Correction invalid for x>11:
    if np.any(x>11):
        return flux 

    npts = wave.size
    a = np.zeros(npts, float)
    b = np.zeros(npts, float)
    
    ###############################
    #Infrared
    
    good = np.where( (x > 0.3) & (x < 1.1) )
    a[good] = 0.574 * x[good]**(1.61)
    b[good] = -0.527 * x[good]**(1.61)
    
    ###############################
    # Optical & Near IR

    good = np.where( (x  >= 1.1) & (x < 3.3) )
    y = x[good] - 1.82
    
    c1 = np.array([ 1.0 , 0.104,   -0.609,  0.701,  1.137, \
                  -1.718,   -0.827, 1.647, -0.505 ])
    c2 = np.array([ 0.0,  1.952,    2.908,   -3.989, -7.985, \
                  11.102,   5.491,  -10.805,  3.347 ] )

    a[good] = np.polyval(c1[::-1], y)
    b[good] = np.polyval(c2[::-1], y)

    ###############################
    # Mid-UV
    
    good = np.where( (x >= 3.3) & (x < 8) )   
    y = x[good]
    F_a = np.zeros(np.size(good),float)
    F_b = np.zeros(np.size(good),float)
    good1 = np.where( y > 5.9 ) 
    
    if np.size(good1) > 0:
        y1 = y[good1] - 5.9
        F_a[ good1] = -0.04473 * y1**2 - 0.009779 * y1**3
        F_b[ good1] =   0.2130 * y1**2  +  0.1207 * y1**3

    a[good] =  1.752 - 0.316*y - (0.104 / ( (y-4.67)**2 + 0.341 )) + F_a
    b[good] = -3.090 + 1.825*y + (1.206 / ( (y-4.62)**2 + 0.263 )) + F_b
    
    ###############################
    # Far-UV
    
    good = np.where( (x >= 8) & (x <= 11) )   
    y = x[good] - 8.0
    c1 = [ -1.073, -0.628,  0.137, -0.070 ]
    c2 = [ 13.670,  4.257, -0.420,  0.374 ]
    a[good] = np.polyval(c1[::-1], y)
    b[good] = np.polyval(c2[::-1], y)

    # Applying Extinction Correction
    
    a_v = r_v * ebv
    a_lambda = a_v * (a + b/r_v)
    
    funred = flux * 10.0**(0.4*a_lambda)   

    return funred


def window_filter(spec,size):
    """
    Estimates the median value of the spectrum 
    within a pixel window.
    """
    med_spec = np.empty(len(spec))
    pix = np.arange(0,len(spec),1)
    for i,p in enumerate(pix):
        # Get n-nearest pixels
        # Calculate distance from i to each pixel
        i_sort =np.argsort(np.abs(i-pix))
        idx = pix[i_sort][:size] # indices we estimate from
        med = np.median(spec[idx])
        med_spec[i] = med
    #
    return med_spec


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


def interpolate_metal(spec,noise):
    """
    Interpolates over metal absorption lines for 
    high-redshift spectra using a moving median
    filter.
    """
    sig_clip = 3.0
    nclip = 10
    bandwidth= 15
    med_spec = window_filter(spec,bandwidth)
    count = 0 
    new_spec = np.copy(spec)
    while (count<=nclip) and ((np.std(new_spec-med_spec)*sig_clip)>np.median(noise)):
        count+=1
        # Get locations of nan or -inf pixels
        nan_spec = np.where((np.abs(new_spec-med_spec)>(np.std(new_spec-med_spec)*sig_clip)) & (new_spec < (med_spec-sig_clip*noise)) )[0]
        if len(nan_spec)>0:
            inan = np.unique(np.concatenate([nan_spec]))
            buffer = 0
            inan_buffer_upp = np.array([(i+buffer) for i in inan if (i+buffer) < len(spec)],dtype=int)
            inan_buffer_low = np.array([(i-buffer) for i in inan if (i-buffer) > 0],dtype=int)
            inan = np.concatenate([inan,inan_buffer_low, inan_buffer_upp])
            # Interpolate over nans and infs if in spec
            new_spec[inan] = np.nan
            new_spec = insert_nan(new_spec,inan)
            nans, x= nan_helper(new_spec)
            new_spec[nans]= np.interp(x(nans), x(~nans), new_spec[~nans])
        else:
            break

    return new_spec


def log_rebin(lam, spec, velscale=None, oversample=1, flux=False):
    """Logarithmically rebin a spectrum or an array of spectra.

    This function logarithmically rebins a spectrum, or the first dimension of
    an array of spectra, while rigorously conserving flux. The photons in the
    spectrum are redistributed to a new grid of pixels with logarithmic
    sampling in the spectral direction.

    The function can operate in two modes based on the `flux` parameter.
    When `flux=True`, it performs an exact integration of the original spectrum,
    assuming it is a step function constant within each pixel, onto the new
    logarithmically-spaced pixels. This preserves the total flux.
    When `flux=False` (default), the integrated flux is divided by the width
    of each new pixel, preserving the flux density (e.g., in units of
    erg/(s cm^2 A)). This mode is generally recommended as it preserves the
    spectral shape.

    Parameters
    ----------
    lam : array_like
        Wavelength values. This can be either a 2-element array specifying the
        minimum and maximum wavelengths `[lam_min, lam_max]` for a regularly
        sampled spectrum, or a 1-D array with the central wavelength of each
        pixel for an irregularly sampled spectrum.
        - If `lam` has two elements, it defines the central wavelengths of the
          first and last pixels. The wavelength scale is assumed to be linear.
          This method is faster for regular sampling.
        - If `lam` is a 1-D array, it provides the central wavelength for each
          spectral pixel, allowing for arbitrary irregular sampling. The pixel
          edges are assumed to be the midpoints between adjacent wavelengths.

        Example for uniform wavelength sampling from FITS keywords::

            lam = CRVAL1 + CDELT1 * np.arange(NAXIS1)

    spec : array_like
        The input spectrum or an array of spectra to be rebinned. This can be a
        1-D array `spec[npixels]` or a 2-D array `spec[npixels, nspec]`.
    velscale : float, optional
        The desired velocity scale in km/s per pixel for the output spectrum.
        If not provided, it is computed to produce the same number of output
        pixels as the input. If specified, it determines the number of pixels
        and the wavelength scale of the output.
    oversample : int, default=1
        Oversampling factor. A value greater than 1 increases the number of
        output pixels, which can help prevent degradation of spectral
        resolution, especially over extended wavelength ranges, and avoid
        aliasing. An `oversample` of 1 results in approximately the same
        number of output pixels as input pixels.
    flux : bool, default=False
        Determines whether to preserve total flux or flux density.
        - If `True`, the total flux is conserved. The flux in each new pixel
          is proportional to its wavelength width (`dlam`), which can alter
          the visual shape of the spectrum.
        - If `False`, the flux density is conserved. The rebinned spectrum will
          closely overlap the original spectrum when plotted.

        Example of plotting the output::

            # With flux=True, the spectral shape changes
            plt.plot(np.exp(ln_lam), specNew)
            plt.plot(np.linspace(lam[0], lam[1], spec.size), spec)

            # With flux=False, the shapes are nearly identical
            plt.plot(np.exp(ln_lam), specNew)
            plt.plot(np.linspace(lam[0], lam[1], spec.size), spec)

    Returns
    -------
    spec_new : ndarray
        The logarithmically-rebinned spectrum or array of spectra.
    ln_lam : ndarray
        The natural logarithm of the wavelength for the new pixel grid. This
        represents the geometric mean of the wavelength at the borders of
        each pixel.
    velscale : float
        The velocity scale per pixel in km/s.

    """
    lam, spec = np.asarray(lam, dtype=float), np.asarray(spec, dtype=float)
    assert np.all(np.diff(lam) > 0), '`lam` must be monotonically increasing'
    n = len(spec)
    assert lam.size in [2, n], '`lam` must be either a 2-elements range or a vector with the length of `spec`'

    if lam.size == 2:
        dlam = np.diff(lam)/(n - 1) # Assume constant dlam
        lim = lam + [-0.5, 0.5]*dlam
        borders = np.linspace(*lim, n + 1)
    else:
        lim = 1.5*lam[[0, -1]] - 0.5*lam[[1, -2]]
        borders = np.hstack([lim[0], (lam[1:] + lam[:-1])/2, lim[1]])
        dlam = np.diff(borders)
    ln_lim = np.log(lim)

    if velscale is None:
        m = int(n*oversample) # Number of output elements
        velscale = c*np.diff(ln_lim).item()/m # Only for output (eq. 8 of Cappellari 2017, MNRAS)
    else:
        ln_scale = velscale/c
        m = int(round(np.diff(ln_lim).item()/ln_scale)) # Number of output pixels

    new_borders = np.exp(ln_lim[0] + velscale/c*np.arange(m + 1))

    if lam.size == 2:
        k = ((new_borders - lim[0])/dlam).clip(0, n-1).astype(int)
    else:
        k = (np.searchsorted(borders, new_borders) - 1).clip(0, n-1)

    spec_new = np.add.reduceat((spec.T*dlam).T, k)[:-1] # Do analytic integral of step function
    spec_new.T[...] *= np.diff(k) > 0 # fix for design flaw of reduceat()
    spec_new.T[...] += np.diff(((new_borders - borders[k]))*spec[k].T) # Add to 1st dimension

    if not flux:
        spec_new.T[...] /= np.diff(new_borders) # Divide 1st dimension

    # Output np.log(wavelength): natural log of geometric mean
    ln_lam = 0.5*np.log(new_borders[1:]*new_borders[:-1])

    return spec_new, ln_lam, velscale


def rebin(x, factor):
    """
    Rebin a vector, or the first dimension of an array,
    by averaging within groups of "factor" adjacent values.

    """
    if factor == 1:
        xx = x
    else:
        xx = x.reshape(len(x)//factor, factor, -1).mean(1).squeeze()

    return xx


def gauss_kde(xs,data,h):
    # Gaussian kernel density estimation.
    def gauss_kernel(x):
        return (1./np.sqrt(2.*np.pi)) * np.exp(-x**2/2)

    kde = np.sum((1./h) * gauss_kernel((xs.reshape(len(xs),1)-data)/h), axis=1)
    kde = kde/simpson(kde,xs) # normalize
    return kde


def kde_bandwidth(data):
    # Silverman bandwidth estimation for kernel density estimation.
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



# TODO: move to badass_tools
def emline_masker(wave,spec,noise):
    """
    Runs a multiple moving window median  
    to determine location of emission lines
    to generate an emission line mask for 
    continuum fitting.
    """
    # First we remove the continuum 
    galaxy_csub = continuum_subtract(wave,spec,noise,sigma_clip=2.0,clip_iter=25,filter_size=[25,50,100,150,200,250,500],
                   noise_scale=1.0,opt_rchi2=True,plot=False,
                   fig_scale=8,fontsize=16,verbose=False)
    #
    signif = 3.0
    pad = 3 # pixels on each side 
    mask_bad = np.unique(np.where(((galaxy_csub)>(signif*(noise))) | ((galaxy_csub)<(-signif*(noise)))))
    # Pad masked bad by pad pixels on each side
    padded_mask_bad = np.array([])
    for b in mask_bad:
        # backwards pix
        # forwards pix
        pix = np.unique(np.abs(np.arange(b-pad,b+pad+1,1)))
        padded_mask_bad = np.concatenate([padded_mask_bad,pix],axis=0)


    mask_bad = np.array(np.unique(np.ravel(padded_mask_bad)),dtype=int)
    #
    edge_ignore = 25 # ignore this many pixels on the edges of the spectrum
    mask_bad  = [m for m in mask_bad if m not in np.concatenate([np.arange(0,edge_ignore),np.arange(len(wave)-edge_ignore,len(wave))])]
    #
    return mask_bad


def metal_masker(wave,spec,noise):
    """
    Performs masking on metal absorption features.
    """
    # First we remove the continuum 
    galaxy_csub = continuum_subtract(wave,spec,noise,sigma_clip=2.0,clip_iter=25,filter_size=[3,5,8],#[25,50,100,150,200,250,500],
                   noise_scale=1.0,opt_rchi2=True,plot=False,
                   fig_scale=8,fontsize=16,verbose=False)
    #
    signif = 3.0
    pad = 3 # pixels on each side 
    mask_bad = np.unique(np.where(((galaxy_csub)>(signif*np.nanmean(noise))) | ((galaxy_csub)<(-signif*np.nanmean(noise)))))
    # Pad masked bad by pad pixels on each side
    padded_mask_bad = np.array([])
    for b in mask_bad:
        # backwards pix
        # forwards pix
        pix = np.unique(np.abs(np.arange(b-pad,b+pad+1,1)))
        padded_mask_bad = np.concatenate([padded_mask_bad,pix],axis=0)


    mask_bad = np.array(np.unique(np.ravel(padded_mask_bad)),dtype=int)
    #
    edge_ignore = 25 # ignore this many pixels on the edges of the spectrum
    mask_bad  = [m for m in mask_bad if m not in np.concatenate([np.arange(0,edge_ignore),np.arange(len(wave)-edge_ignore,len(wave))])]
    #
    return mask_bad


def continuum_subtract(wave,flux,noise,sigma_clip=3.0,clip_iter=25,filter_size=[25,50,100,150,200,250,500],
                       noise_scale=1.0,opt_rchi2=True,plot=True,fig_scale=8,fontsize=16,verbose=True):
        """
        This function performs a first-order continuum subtraction of the spectrum using 
        a series of median filters ranging from narrow to broad bandwidths, while also 
        using sigma clipping for a default threshold of 3.0. It works well with both 
        small and large fitting regions, and with a number of types of objects. It 
        does poorly when large fraction of the fitting region is occupied with strong
        metal absorption features.
        """

        def rchi2_optimize(flux,model,noise,verbose=False):
            # Performs optimization to achieve a reduced chi-squared of 1.0.
            # Optimization function for reduced chi-squared
            def f(n,flux,model,noise,nu):
                rchi2 = np.nansum((flux-model)**2/(n*noise)**2)/nu
                return np.abs(rchi2-1)
            nu = len(flux) # deg of freedom
            init = np.nansum((flux-model)**2/noise**2)/nu
            noise_scale = op.fmin(f,1.0,args=(flux,model,noise,nu,),disp=verbose)
            return noise_scale[0]


        def plot_cont_sub():
            # Determine 
            masked_vals = np.where(mask/mask!=1)[0]
            x_ = np.arange(len(flux))
                    
            # Plot
            fig = plt.figure(figsize=(fig_scale*2,fig_scale*1))
            ax1 = fig.add_subplot(2,1,1)
            ax2 = fig.add_subplot(2,1,2)
            #
            ax1.step(wave,flux,linestyle="-",linewidth=0.5,label=r"$\textrm{Data}$")
            ax1.step(wave,masked_flux,linestyle="-",linewidth=0.5,label=r"$\textrm{Masked}$")
            ax1.step(wave,noise,linestyle="-",linewidth=0.5,label=r"$\textrm{Noise}$")
            ax1.step(wave,smoothed,linestyle="-",linewidth=1,color="xkcd:bright orange",label=r"$\textrm{Median Filter}$")
            ax2.step(wave,resid,linestyle="-",linewidth=0.5,color="xkcd:white",label=r"$\textrm{Residuals}$")
            # Noise intervals
    #         ax2.fill_between(wave,-sigma_clip*np.nanmedian(noise)*noise_scale,sigma_clip*np.nanmedian(noise)*noise_scale,color="xkcd:bright red",alpha=0.5)
            # Masked pixels
            for m in masked_vals:
                try:
                    lower, upper = wave[m], wave[m+1]
                except:
                    lower, upper = wave[m], wave[-1]+1
                ax1.axvspan(lower,upper,color="xkcd:bright green",alpha=0.15)
            #
            ax1.axhline(0.0,color="xkcd:white",linestyle="--",linewidth=1)
            ax2.axhline(0.0,color="xkcd:white",linestyle="--",linewidth=1)
            ax1.set_xlim(wave.min(),wave.max())
            ax2.set_xlim(wave.min(),wave.max())
            plt.suptitle(r"$\textrm{sigma clip iteration %d}$" % (i),fontsize=fontsize+4)
            ax1.set_xlabel(r"$\lambda_\textrm{rest}~(\textrm{\AA})$",fontsize=fontsize)
            ax1.set_ylabel(r"$f_\lambda~(10^{-17}~\textrm{erg}~\textrm{cm}^{-2}~\textrm{s}^{-1}~\textrm{\AA}^{-1})$",fontsize=fontsize)
            ax1.tick_params(axis='both', labelsize=fontsize-4)
            ax1.legend(loc="best",fontsize=fontsize-4)
            ax2.set_xlabel(r"$\lambda_\textrm{rest}~(\textrm{\AA})$",fontsize=fontsize)
            ax2.set_ylabel(r"$f_\lambda~(10^{-17}~\textrm{erg}~\textrm{cm}^{-2}~\textrm{s}^{-1}~\textrm{\AA}^{-1})$",fontsize=fontsize)
            ax2.tick_params(axis='both', labelsize=fontsize-4)
            ax2.legend(loc="best",fontsize=fontsize-4)
            plt.tight_layout()
            return
            
        ############################################################################################################################################

        mask = np.ones(len(flux))
        clip_sum = None
        # sigma clipping iterations
        for i in range(clip_iter):
            # Apply mask
            masked_flux = flux*mask
            # Perform median smoothing
            # scipy's median filter doesn't respect masked arrays so we use a generic filter and pass it numpy's nanmedian()
            if isinstance(filter_size,(int,float)):
                smoothed = generic_filter(masked_flux,function=np.nanmedian,size=filter_size,mode="mirror")
                # Interpolate over nans
                nans, x= nan_helper(smoothed)
                smoothed[nans]= np.interp(x(nans), x(~nans), smoothed[~nans])
            if isinstance(filter_size,(list,tuple)):
                # Storage array for all 
                smoothed_arr = np.empty((len(filter_size),len(flux)))
                for j,f in enumerate(filter_size):
        #             print(j,f)
                    smoothedf = generic_filter(masked_flux,function=np.nanmedian,size=f,mode="mirror")
                    # Interpolate over nans
                    nans, x= nan_helper(smoothedf)
                    smoothedf[nans]= np.interp(x(nans), x(~nans), smoothedf[~nans])
                    smoothed_arr[j,:] = smoothedf
        #         smoothed = np.nanmin(smoothed_arr,axis=0)
        #         smoothed = np.nanmean(smoothed_arr,axis=0)
                smoothed = np.nanmedian(smoothed_arr,axis=0)
            
        
        
            # Calculate residuals
            resid    = flux-smoothed 
        
            # Perform optimization on the noise scaling factor to acheive
            # reduced chi-squared of 1.0 on 
            if opt_rchi2:
                noise_scale = rchi2_optimize(flux,smoothed,noise*sigma_clip,verbose=False)
            else: 
                noise_scale =1
        
            # mask to be iteratively updated
            mask = np.ones(len(flux))
        #     print(noise_scale,np.nanmedian(noise),np.nanmedian(noise)*noise_scale)
            mask[np.where((resid <= -sigma_clip*np.nanmedian(noise)*noise_scale) | (resid >= sigma_clip*np.nanmedian(noise)*noise_scale))[0]] = np.nan
        #     mask[np.where((resid <= -sigma_clip*noise) | (resid >= sigma_clip*noise))[0]] = np.nan
            
            # Check to see if any new 
            if len(np.where(mask/mask!=1)[0])==clip_sum:
                if verbose:
                    print("\t sigma clipping successfully stopped at %d iterations" % (i))
                if plot:
                    plot_cont_sub()
                break    
            if i+1==clip_iter:
                if plot:
                    plot_cont_sub()
        
            # Update clip_sum
            clip_sum = len(np.where(mask/mask!=1)[0])
            if verbose:
                print(" sigma clip iteration %d out of %d (%s clipped pixels)" % (i+1, clip_iter, clip_sum))
            #
        return resid

