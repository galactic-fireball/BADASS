import astropy.constants as consts
from astropy.io import fits
import astropy.units as u
from dataclasses import dataclass
import numpy as np
import pathlib

from spark.io.readers.muse import MUSECube
from spark.utils import deredden

from badass.input.input import BadassCube

MUSE_FLUX_NORM = 1e-20

@dataclass
class MUSEReader(BadassCube, MUSECube):

    def __post_init__(self):
        super().__post_init__()

        self.flux_norm = MUSE_FLUX_NORM
        self.flux / u.Unit(self.flux_norm)
        self.err / u.Unit(self.flux_norm)
        self.flux_unit / u.Unit(self.flux_norm)

        # Default behavior for MUSE data cubes using https://www.aanda.org/articles/aa/pdf/2017/12/aa30833-17.pdf equation 7
        fwhm_res = 5.835e-8 * self.obs_wave**2 - 9.080e-4 * self.obs_wave + 5.983
        R = self.obs_wave / fwhm_res
        # scale by center wavelength
        c = consts.c.to(u.km/u.s).value
        R_cent = self.spec_res
        cwave = np.nanmedian(self.obs_wave)
        c_dlambda = 5.835e-8 * cwave**2 - 9.080e-4 * cwave + 5.983
        scale = 1 + (R_cent - cwave/c_dlambda) / R_cent
        R *= scale
        fwhm_res = self.obs_wave / R
        self.disp_res = fwhm_res / 2.3548
        self.disp_res = deredden(self.disp_res, z=self.target.z)

        self.velscale = np.log(self.obs_wave[1] / self.obs_wave[0]) * c  # Constant velocity scale in km/s per pixel

Reader = MUSEReader
