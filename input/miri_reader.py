from astropy.io import fits
import pathlib
from scipy import interpolate

from input.jwst_common import JWSTReader

inst_data_file = pathlib.Path(__file__).resolve().parent.joinpath('instrument_data', 'miri', 'MIRI_INST_DATA_FULL.fits')

class MIRIReader(JWSTReader):

    def set_dispersion(self, options, obs_wave):
        hdu = fits.open(inst_data_file)
        interp_func = interpolate.interp1d(hdu[1].data['WAVELENGTH'], hdu[1].data['R'], bounds_error=False, fill_value='extrapolate')
        self.disp_res = obs_wave / interp_func(obs_wave)

Reader = MIRIReader
