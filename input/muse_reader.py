import astropy.constants as consts
from astropy.io import fits
import astropy.units as u
import numpy as np
import pathlib

from input.cube_reader import CubeReader
from utils.utils import dered

MUSE_FLUX_NORM = 1e-20

class MUSEReader(CubeReader):

    @classmethod
    def get_cube_data(cls, input_data, options):
        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading MUSE spectra from data currently unsupported') # TODO

        if not 'redshift' in options.io_options:
            raise Exception('Redshift for MUSE cube must be provided')

        cube_data = {}
        cube_data['z'] = options.io_options.redshift

        cube_data['infile'] = input_data
        with fits.open(input_data) as hdu:
            primary = hdu['PRIMARY'].header
            header = hdu['DATA'].header

            cube_data['nx'], cube_data['ny'], cube_data['nz'] = header['NAXIS1'], header['NAXIS2'], header['NAXIS3']
            cube_data['ra'], cube_data['dec'] = primary['RA'], primary['DEC']

            obs_wave = np.array(header['CRVAL3'] + header['CD3_3']*np.arange(header['NAXIS3']))

            cube_spec = hdu['DATA'].data.T
            cube_noise = np.sqrt(hdu['STAT'].data.T)

            # flux_norm = MUSE_FLUX_NORM
            div = int(np.floor(np.log10(np.abs(np.nanmedian(cube_spec)))))
            cube_data['spec'] = cube_spec / (10**div)
            cube_data['noise'] = cube_noise / (10**div)
            cube_data['flux_norm'] = 10**div

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
            disp_res = fwhm_res / 2.3548
            cube_data['velscale'] = np.log(obs_wave[1] / obs_wave[0]) * c  # Constant velocity scale in km/s per pixel

            cube_data['wave'] = dered(obs_wave, cube_data['z'])
            cube_data['disp_res'] = dered(disp_res, cube_data['z'])

            cube_data['splitable'] = ['spec','noise']

            return cube_data


Reader = MUSEReader
