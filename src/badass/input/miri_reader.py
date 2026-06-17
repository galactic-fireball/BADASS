from astropy.io import fits
import astropy.units as u
import pathlib
from scipy import interpolate

from badass.input.jwst_common import JWSTReader

inst_data_file = pathlib.Path(__file__).resolve().parent.joinpath('instrument_data', 'miri', 'MIRI_INST_DATA_FULL.fits')

class MIRIReader(JWSTReader):

    def set_dispersion(self):
        hdu = fits.open(inst_data_file)
        wave_um = (self.obs_wave*self.wave_unit).to(u.um)
        interp_func = interpolate.interp1d(hdu[1].data['WAVELENGTH']*u.um, hdu[1].data['R'], bounds_error=False, fill_value='extrapolate')
        self.disp_res = (wave_um / interp_func(wave_um)).to(self.wave_unit).value
        hdu.close()

Reader = MIRIReader
