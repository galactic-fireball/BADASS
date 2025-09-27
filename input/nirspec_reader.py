from astropy.io import fits
import pathlib
from scipy import interpolate

from input.jwst_common import JWSTReader

inst_data_dir = pathlib.Path(__file__).resolve().parent.joinpath('instrument_data', 'nirspec')

FILT_GRAT = {
    '100': '140',
    '170': '235',
    '290': '395',
}

class NIRSpecReader(JWSTReader):

    def set_dispersion(self, options, obs_wave):
        if not 'filter' in options.io_options:
            raise Exception('Filter for NIRSpec cube must be provided')

        if not 'disperser' in options.io_options:
            raise Exception('Disperser for NIRSpec cube must be provided')

        self.filter = options.io_options.filter
        self.grating = FILT_GRAT[self.filter]
        self.disperser = options.io_options.disperser.lower()

        inst_data_file = inst_data_dir.joinpath('jwst_nirspec_g%s%s_disp.fits'%(self.grating,self.disperser))
        hdu = fits.open(inst_data_file)

        interp_func = interpolate.interp1d(hdu[1].data['WAVELENGTH'], hdu[1].data['R'], bounds_error=False, fill_value='extrapolate')
        self.disp_res = obs_wave / interp_func(obs_wave)


Reader = NIRSpecReader
