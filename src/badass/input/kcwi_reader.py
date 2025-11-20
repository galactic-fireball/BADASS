from astropy import constants as const
from astropy.io import fits
import astropy.units as u
import numpy as np
import pathlib

from badass.input.cube_reader import CubeReader
from badass.utils.utils import dered

KCWI_FLUX_NORM = 1e-16

GRATING_TO_R = {
    'BH3-L': 4500, 'BH2-L': 4500, 'BH1-L': 4500,
    'BH3-M': 9000, 'BH2-M': 9000, 'BH1-M': 9000,
    'BH3-S': 18000, 'BH2-S': 18000, 'BH1-S': 18000,
    'BM-L': 2000, 'BM-M': 4000, 'BM-S': 8000,
    'BL-L': 900, 'BL-M': 1800, 'BL-S': 3600,
    ### Red arm measurements are as of 04/26/2025 00:19
    'RH4-L': 3250, 'RH3-L': 3250, 'RH2-L': 3250, 'RH1-L': 3250,
    'RH4-M': 6500, 'RH3-M': 6500, 'RH2-M': 6500, 'RH1-M': 6500,
    'RH4-S': 13000, 'RH3-S': 13000, 'RH2-S': 13000, 'RH1-S': 13000,
    'RM2-L': 1400, 'RM1-L': 1400,
    'RM2-M': 2800, 'RM1-M': 2800,
    'RM2-S': 5600, 'RM1-S': 5600,
    'RL-L': 500, # > 500
    'RL-M': 1000, # > 1000
    'RL-S': 2000
}

def get_R(grating):
    for grating_name, R in GRATING_TO_R.items():
        if grating_name in grating:
            return R
    print('No matching grating found in STATENAM')
    return None


class KCWIReader(CubeReader):

    @classmethod
    def get_cube_data(cls, input_data, options):
        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading KCWI spectra from data currently unsupported') # TODO

        if not 'redshift' in options.fit_options:
            raise Exception('Redshift for KCWI cube must be provided')

        cube_data = {}
        cube_data['z'] = options.fit_options.redshift
        cube_data['infile'] = input_data

        with fits.open(input_data) as hdu:
            header = hdu[0].header

            if 'RA' in header:
                cube_data['ra'] = header['RA']
                cube_data['dec'] = header['DEC']
            elif 'TARGRA' in header:
                cube_data['ra'] = header['TARGRA']
                cube_data['dec'] = header['TARGDEC']
            else:
                cube_data['ra'] = None
                cube_data['dec'] = None

            cube_data['spec'] = hdu[0].data.T
            try:
                cube_data['bad_pix'] = hdu[1].data.T
            except:
                cube_data['bad_pix'] = np.zeros_like(cube_data['spec'], dtype=bool)
            cube_data['noise'] = np.sqrt(hdu[2].data.T)

            lam_pix_prop = 'CDELT3' if 'CDELT3' in header else 'CD3_3'
            obs_wave = np.array(header['CRVAL3'] + header[lam_pix_prop]*np.arange(header['NAXIS3']))

            cube_data['flux_norm'] = KCWI_FLUX_NORM

            ### KCWI gratings: https://www2.keck.hawaii.edu/inst/kcwi/configurations.html
            # 'STATENAM' may not be the correct parameter to use for all data. 
            # TODO: Maybe change to 'BGRATNAM'/'RGRATNAM' + 'IFUNAM'
            R = options.io_options.get('R', get_R(header['STATENAM']))
            if R is None:
                raise Exception('No grating information found... please add \'STATENAM\' to the header')

            fwhm_res = header['WAVMID'] / R # Resolution FWHM of every pixel, in angstroms
            disp_res = fwhm_res / 2.3548
            cube_data['velscale'] = const.c.to(u.km/u.s).value / R / 2.3548 # Instrumental velocity broadening (km/s)

            cube_data['wave'] = dered(obs_wave, cube_data['z'])
            cube_data['disp_res'] = dered(disp_res, cube_data['z'])

            cube_data['splitable'] = ['spec', 'noise']

            return cube_data


Reader = KCWIReader
