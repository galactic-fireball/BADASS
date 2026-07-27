from astropy.io import fits
import astropy.units as u
from dataclasses import dataclass
import numpy as np
import pathlib
from scipy import interpolate

from spark.io.readers.jwst import JWSTCube
from spark.utils import deredden

from badass.input.input import BadassCube, LogRebinMixin, LogRebinSpaxel, LogRebinCircularAperture, LogRebinEllipticalAperture, LogRebinRectangularAperture


# TODO: after fit, return wave to original units
# TODO: unit agnostic
@dataclass
class JWSTReader(BadassCube, JWSTCube, LogRebinMixin):
    spaxel_class = LogRebinSpaxel
    ap_shapes = {
        'circular': LogRebinCircularAperture,
        'elliptical': LogRebinEllipticalAperture,
        'rectangular': LogRebinRectangularAperture,
    }

    def __post_init__(self):
        super().__post_init__()

        self.set_dispersion()
        self.disp_res = deredden(self.disp_res, self.target.z)
        self.flux_norm = 1.0
        self.flux = self.flux / self.flux_norm
        self.err = self.err / self.flux_norm
        self.velscale = np.nan

        self.log_rebin()


    def set_dispersion(self, cube_data, options, obs_wave):
        # Instrument child classes will override
        pass


    @staticmethod
    def interp_dispersion(data_file, wave_array, wave_unit=u.um):
        hdu = fits.open(data_file)
        wave_um = (wave_array*wave_unit).to(u.um)

        interp_func = interpolate.interp1d(hdu[1].data['WAVELENGTH']*u.um, hdu[1].data['R'], bounds_error=False, fill_value='extrapolate')
        hdu.close()

        disp = (wave_um / interp_func(wave_um)).to(wave_unit).value / 2.355
        return disp

