from astropy.io import fits
import astropy.units as u
from dataclasses import dataclass
import numpy as np
import pathlib

from spark.io.readers.jwst import JWSTCube
from spark.utils import deredden

from badass.input.input import BadassCube, LogRebinSpaxel, LogRebinCircularAperture, LogRebinRectangularAperture


# TODO: after fit, return wave to original units
# TODO: unit agnostic
@dataclass
class JWSTReader(BadassCube, JWSTCube):
    spaxel_class = LogRebinSpaxel
    ap_shapes = {
        'circular': LogRebinCircularAperture,
        'rectangular': LogRebinRectangularAperture,
    }

    def __post_init__(self):
        super().__post_init__()

        self.set_dispersion()
        self.disp_res = deredden(self.disp_res, self.target.z)
        self.flux_norm = 1
        self.velscale = np.nan


    @classmethod
    def set_dispersion(cls, cube_data, options, obs_wave):
        # Instrument child classes will override
        pass
