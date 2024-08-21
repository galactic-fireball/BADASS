from astropy import constants as const
from astropy.io import fits
import astropy.units as u
import numpy as np
import pathlib

from input.input import BadassInput
from utils.utils import dered

class SDSSReader(BadassInput):

    def __init__(self, input_data, options):
        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading SDSS spectra from data currently unsupported')

        self.infile = input_data
        with fits.open(self.infile) as hdu:
            specobj = hdu[2].data
            self.z = specobj['z'][0]

            if 'RA' in hdu[0].header:
                self.ra = hdu[0].header['RA']
                self.dec = hdu[0].header['DEC']
            elif 'PLUG_RA' in hdu[0].header:
                self.ra = specobj['PLUG_RA'][0]
                self.dec = specobj['PLUG_DEC'][0]
            else:
                self.ra = None
                self.dec = None

            t = hdu[1].data

            # Unpack the spectra
            self.spec = t['flux']
            obs_wave = np.power(10, t['loglam'])
            self.noise = np.sqrt(1 / t['ivar'])
            self.bad_pix = np.where(t['and_mask'] != 0)[0] # TODO: need?

            frac = obs_wave[1]/obs_wave[0] # Constant lambda fraction per pixel
            dlam_gal = (frac - 1)*obs_wave # Size of every pixel in Angstrom
            wdisp = t['wdisp'] # Intrinsic dispersion of every pixel, in pixels units
            self.disp_res = wdisp*dlam_gal # Resolution FWHM of every pixel, in angstroms
            self.velscale = np.log(frac) * const.c.to(u.km/u.s).value

            self.wave = dered(obs_wave, self.z)
            self.disp_res = dered(self.disp_res, self.z)

Reader = SDSSReader
