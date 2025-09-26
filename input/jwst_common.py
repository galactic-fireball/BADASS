from astropy.io import fits
import astropy.units as u
import numpy as np

TARGET_WAVE_UNIT = u.AA
TARGET_FLUX_UNIT_UM = u.erg / u.s / (u.cm**2) / u.um
TARGET_FLUX_UNIT_AA = u.erg / u.s / (u.cm**2) / u.AA

def get_jwst_cube(fits_file):
    hdu = fits.open(fits_file)
    header = hdu['SCI'].header
    cunit = header['CUNIT3']
    bunit = header['BUNIT']

    nwave = hdu['SCI'].data.shape[0]
    wave0 = header['CRVAL3'] - (header['CRPIX3'] - 1) * header['CDELT3']
    wave = (wave0 + np.arange(nwave)*header['CDELT3']) * u.Unit(cunit)

    spec = hdu['SCI'].data.T * u.Unit(bunit)
    err = hdu['ERR'].data.T * u.Unit(bunit)
    if '/sr' in bunit:
        pxar = header['PIXAR_SR'] * u.sr
        spec *= pxar
        err *= pxar

    spec = spec.to(TARGET_FLUX_UNIT_UM, equivalencies=u.spectral_density(wave))
    err = err.to(TARGET_FLUX_UNIT_UM, equivalencies=u.spectral_density(wave))
    err[np.isnan(err)] = np.nanmedian(err)

    ra, dec = hdu[0].header['TARG_RA'], hdu[0].header['TARG_DEC']
    hdu.close()

    return {
        'ra': ra,
        'dec': dec,
        'wave': wave,
        'spec': spec,
        'err': err,
    }
