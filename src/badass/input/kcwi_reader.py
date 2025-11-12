from astropy import constants as const
from astropy.io import fits
import astropy.units as u
import numpy as np
import pathlib

from badass.input.input import BadassInput
from badass.utils.utils import dered

KCWI_FLUX_NORM = 1e-16

class KCWIReader(BadassInput):

    def __init__(self, input_data, options):
        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading KCWI spectra from data currently unsupported') # TODO
        self.infile = input_data
        with fits.open(self.infile) as hdu:
            self.z = options.io_options.redshift

            if 'RA' in hdu[0].header:
                self.ra = hdu[0].header['RA']
                self.dec = hdu[0].header['DEC']
            elif 'TARGRA' in hdu[0].header:
                self.ra = hdu[0].header['TARGRA']
                self.dec = hdu[0].header['TARGDEC']
            else:
                self.ra = None
                self.dec = None

            self.spec = hdu[0].data
            obs_wave = np.array(hdu[0].header['CRVAL3'] + hdu[0].header['CDELT3']*np.arange(hdu[0].header['NAXIS3']))
            self.noise = hdu[2].data
            self.bad_pix = hdu[1].data # np.where(t['and_mask'] != 0)[0] # TODO: need?
            self.flux_norm = KCWI_FLUX_NORM

            # KCWI gratings: https://www2.keck.hawaii.edu/inst/kcwi/configurations.html
            try:
                print('You are using grating ', hdu[0].header['STATENAM'])
                if 'BH3-L' or 'BH2-L' or 'BH1-L' in hdu[0].header['STATENAM']:
                    R = 4500
                elif 'BH3-M' or 'BH2-M' or 'BH1-M' in hdu[0].header['STATENAM']:
                    R = 9000
                elif 'BH3-S' or 'BH2-S' or 'BH1-S' in hdu[0].header['STATENAM']:
                    R = 18000
                elif 'BM-L' in hdu[0].header['STATENAM']:
                    R = 2000
                elif 'BM-M' in hdu[0].header['STATENAM']:
                    R = 4000
                elif 'BM-S' in hdu[0].header['STATENAM']:
                    R = 8000
                elif 'BL-L' in hdu[0].header['STATENAM']:
                    R = 900
                elif 'BL-M' in hdu[0].header['STATENAM']:
                    R = 1800
                elif 'BL-S' in hdu[0].header['STATENAM']:
                    R = 3600
                ### Red arm measurements are as of 04/26/2025 00:19
                elif 'RH4-L' or 'RH3-L' or 'RH2-L' or 'RH1-L' in hdu[0].header['STATENAM']:
                    R = 3250 # > 3250
                elif 'RH4-M' or 'RH3-M' or 'RH2-M' or 'RH1-M' in hdu[0].header['STATENAM']:
                    R = 6500 # > 6500
                elif 'RH4-S' or 'RH3-S' or 'RH2-S' or 'RH1-S' in hdu[0].header['STATENAM']:
                    R = 13000 # > 13000
                elif 'RM2-L' or 'RM1-L' in hdu[0].header['STATENAM']:
                    R = 1400 # > 1400
                elif 'RM2-M' or 'RM1-M' in hdu[0].header['STATENAM']:
                    R = 2800 # > 2800
                elif 'RM2-S' or 'RM1-S' in hdu[0].header['STATENAM']:
                    R = 5600 # > 5600
                elif 'RL-L' in hdu[0].header['STATENAM']:
                    R = 500 # > 500
                elif 'RL-M' in hdu[0].header['STATENAM']:
                    R = 1000 # > 1000
                elif 'RL-S' in hdu[0].header['STATENAM']:
                    R = 2000 # > 2000
                else:
                    print('No matching grating found in STATENAM')
                    R = None
            except:
                R = None
                print('No grating information found... please add \'STATENAM\' to the header')

            self.disp_res = hdu[0].header['WAVMID'] / R # Resolution FWHM of every pixel, in angstroms
            self.velscale = const.c.to(u.km/u.s).value / R / 2.3548 # Instrumental velocity broadening (km/s)

            self.wave = dered(obs_wave, self.z)
            self.disp_res = dered(self.disp_res, self.z)

        super().__init__(input_data, options)

Reader = KCWIReader
