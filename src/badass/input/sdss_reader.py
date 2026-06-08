from astropy import constants as const
from astropy.io import fits
import astropy.units as u
from dataclasses import dataclass
import numpy as np
import pathlib

from spark.io.readers.sdss import SDSSSpectrum
from spark.utils import redden, deredden

from badass.input.input import BadassSpec

SDSS_FLUX_NORM = 1e-17

@dataclass
class SDSSReader(BadassSpec, SDSSSpectrum):

    def __post_init__(self):
        super().__post_init__()

        self.flux_norm = SDSS_FLUX_NORM

        if self.wave_is_rest:
            self.obs_wave = redden(self.wave, z=self.target.z)
        else:
            self.obs_wave = self.wave.copy()
            self.wave = deredden(self.obs_wave, z=self.target.z)


        # TODO: implement bad_pix masking
        # self.bad_pix = np.where(t['and_mask'] != 0)[0]

        frac = self.obs_wave[1]/self.obs_wave[0] # Constant lambda fraction per pixel
        dlam_gal = (frac - 1)*self.obs_wave # Size of every pixel in Angstrom
        self.disp_res = self.disp*dlam_gal # Resolution FWHM of every pixel, in angstroms
        self.disp_res = deredden(self.disp_res, self.target.z)
        self.velscale = np.log(frac) * const.c.to(u.km/u.s).value

Reader = SDSSReader
