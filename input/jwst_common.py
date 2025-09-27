from astropy.io import fits
import astropy.units as u
import numpy as np
import pathlib

from input.input import BadassInput
from utils.utils import dered, log_rebin

TARGET_WAVE_UNIT = u.AA
TARGET_FLUX_UNIT_UM = u.erg / u.s / (u.cm**2) / u.um
TARGET_FLUX_UNIT_AA = u.erg / u.s / (u.cm**2) / u.AA

class JWSTReader(BadassInput):

    def __init__(self, input_data, options):
        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading JWST spectra from data currently unsupported') # TODO

        self.infile = input_data
        if not self.infile.exists():
            raise Exception('Not found: %s'%str(self.infile))

        if not 'redshift' in options.io_options:
            raise Exception('Redshift for NIRSpec cube must be provided')

        if not 'spaxel' in options.io_options:
            raise Exception('Spaxel for NIRSpec cube must be provided')

        self.z = options.io_options.redshift
        self.spaxel = options.io_options.spaxel

        hdu = fits.open(self.infile)
        header = hdu['SCI'].header
        cunit = header['CUNIT3']
        bunit = header['BUNIT']

        nwave = hdu['SCI'].data.shape[0]
        wave0 = header['CRVAL3'] - (header['CRPIX3'] - 1) * header['CDELT3']
        obs_wave = (wave0 + np.arange(nwave)*header['CDELT3']) * u.Unit(cunit)

        cube_spec = hdu['SCI'].data.T * u.Unit(bunit)
        cube_err = hdu['ERR'].data.T * u.Unit(bunit)
        if '/sr' in bunit:
            pxar = header['PIXAR_SR'] * u.sr
            cube_spec *= pxar
            cube_err *= pxar

        cube_spec = cube_spec.to(TARGET_FLUX_UNIT_UM, equivalencies=u.spectral_density(obs_wave))
        cube_err = cube_err.to(TARGET_FLUX_UNIT_UM, equivalencies=u.spectral_density(obs_wave))
        cube_err[np.isnan(cube_err)] = np.nanmedian(cube_err)
        self.ra, self.dec = hdu[0].header['TARG_RA'], hdu[0].header['TARG_DEC']
        hdu.close()

        self.set_dispersion(options, obs_wave.value)

        self.spec = cube_spec[self.spaxel[0],self.spaxel[1],:]
        self.noise = cube_err[self.spaxel[0],self.spaxel[1],:]

        self.wave = dered(obs_wave, self.z)
        self.disp_res = dered(self.disp_res, self.z)

        # TODO: after fit, return wave to original units
        # TODO: unit agnostic
        self.wave = self.wave.to(TARGET_WAVE_UNIT).value
        self.spec = self.spec.to(TARGET_FLUX_UNIT_AA).value
        self.noise = self.noise.to(TARGET_FLUX_UNIT_AA).value

        div = int(np.floor(np.log10(np.abs(np.nanmedian(self.spec)))))
        self.spec = self.spec / (10**div)
        self.noise = self.noise / (10**div)
        self.flux_norm = 10**div

        lam_range = (np.min(self.wave),np.max(self.wave))
        self.spec, log_lam, velscale = log_rebin(lam_range, self.spec, velscale=None, flux=False)
        self.noise, _, _ = log_rebin(lam_range, self.noise, velscale=velscale, flux=False)
        self.wave = np.exp(log_lam)
        self.velscale = velscale[0]


    def set_dispersion(self, options, obs_wave):
        # Instrument child classes will override
        self.disp_res = None
