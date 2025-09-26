import astropy.constants as consts
from astropy.io import fits
import astropy.units as u
import numpy as np
import pathlib

from input.input import BadassInput
from utils.utils import dered

MUSE_FLUX_NORM = 1e-20

class MUSEReader(BadassInput):

    def __init__(self, input_data, options):
        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading MUSE spectra from data currently unsupported') # TODO

        if not 'redshift' in options.io_options:
            raise Exception('Redshift for MUSE cube must be provided')

        if not 'spaxel' in options.io_options:
            raise Exception('Spaxel for MUSE cube must be provided')

        self.z = options.io_options.redshift
        self.spaxel = options.io_options.spaxel

        self.infile = input_data
        with fits.open(self.infile) as hdu:
            primary = hdu['PRIMARY'].header
            header = hdu['DATA'].header

            self.nx, self.ny, self.nz = header['NAXIS1'], header['NAXIS2'], header['NAXIS3']
            self.ra, self.dec = primary['RA'], primary['DEC']

            obs_wave = np.array(header['CRVAL3'] + header['CD3_3']*np.arange(header['NAXIS3']))

            cube_spec = hdu['DATA'].data
            self.spec = cube_spec[:,self.spaxel[0],self.spaxel[1]]
            cube_noise = np.sqrt(hdu['STAT'].data)
            self.noise = cube_noise[:,self.spaxel[0],self.spaxel[1]]
            self.flux_norm = MUSE_FLUX_NORM

            # Default behavior for MUSE data cubes using https://www.aanda.org/articles/aa/pdf/2017/12/aa30833-17.pdf equation 7
            fwhm_res = 5.835e-8 * obs_wave**2 - 9.080e-4 * obs_wave + 5.983
            R = obs_wave / fwhm_res

            # scale by center wavelength
            c = consts.c.to(u.km/u.s).value
            R_cent = primary['SPEC_RES']
            cwave = np.nanmedian(obs_wave)
            c_dlambda = 5.835e-8 * cwave**2 - 9.080e-4 * cwave + 5.983
            scale = 1 + (R_cent - cwave/c_dlambda) / R_cent
            R *= scale

            fwhm_res = obs_wave / R
            self.disp_res = fwhm_res / 2.3548
            self.velscale = np.log(obs_wave[1] / obs_wave[0]) * c  # Constant velocity scale in km/s per pixel

            self.wave = dered(obs_wave, self.z)
            self.disp_res = dered(self.disp_res, self.z)


Reader = MUSEReader
