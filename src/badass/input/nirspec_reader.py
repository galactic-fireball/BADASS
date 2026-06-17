from astropy.io import fits
import astropy.units as u
import pathlib
from scipy import interpolate

from badass.input.jwst_common import JWSTReader

inst_data_dir = pathlib.Path(__file__).resolve().parent.joinpath('instrument_data', 'nirspec')

FILT_GRAT = {
    '100': '140',
    '170': '235',
    '290': '395',
}

class NIRSpecReader(JWSTReader):

    def set_dispersion(self):
        grating = FILT_GRAT[self.cfg.io.filter]
        disperser = self.cfg.io.disperser.lower()

        inst_data_file = inst_data_dir.joinpath('jwst_nirspec_g%s%s_disp.fits'%(grating,disperser))
        hdu = fits.open(inst_data_file)

        wave_um = (self.obs_wave*self.wave_unit).to(u.um)
        interp_func = interpolate.interp1d(hdu[1].data['WAVELENGTH']*u.um, hdu[1].data['R'], bounds_error=False, fill_value='extrapolate')
        self.disp_res = (wave_um / interp_func(wave_um)).to(self.wave_unit).value
        hdu.close()


Reader = NIRSpecReader
