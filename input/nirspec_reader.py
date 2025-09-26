from astropy.io import fits
import numpy as np
import pathlib
from scipy import interpolate

from input.input import BadassInput
from utils.utils import dered, log_rebin
import input.jwst_common as jwc

inst_data_dir = pathlib.Path(__file__).resolve().parent.joinpath('instrument_data', 'nirspec')

FILT_GRAT = {
    '100': '140',
    '170': '235',
    '290': '395',
}

def get_dispersion(filt, disperser, wave):
    grating = FILT_GRAT[filt]
    inst_data_file = inst_data_dir.joinpath('jwst_nirspec_g%s%s_disp.fits'%(grating,disperser))
    hdu = fits.open(inst_data_file)

    interp_func = interpolate.interp1d(hdu[1].data['WAVELENGTH'], hdu[1].data['R'], bounds_error=False)
    return wave / interp_func(wave)


class NIRSpecReader(BadassInput):

    def __init__(self, input_data, options):
        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading NIRSpec spectra from data currently unsupported') # TODO

        self.infile = input_data
        if not self.infile.exists():
            raise Exception('Not found: %s'%str(self.infile))

        if not 'redshift' in options.io_options:
            raise Exception('Redshift for NIRSpec cube must be provided')

        if not 'filter' in options.io_options:
            raise Exception('Filter for NIRSpec cube must be provided')

        if not 'disperser' in options.io_options:
            raise Exception('Disperser for NIRSpec cube must be provided')

        if not 'spaxel' in options.io_options:
            raise Exception('Spaxel for NIRSpec cube must be provided')

        self.z = options.io_options.redshift
        self.spaxel = options.io_options.spaxel
        self.filter = options.io_options.filter
        self.disperser = options.io_options.disperser.lower()

        cube = jwc.get_jwst_cube(self.infile)

        self.ra = cube['ra']
        self.dec = cube['dec']
        obs_wave = cube['wave']
        self.spec = cube['spec'][self.spaxel[0],self.spaxel[1],:]
        self.noise = cube['err'][self.spaxel[0],self.spaxel[1],:]

        self.disp_res = get_dispersion(self.filter, self.disperser, obs_wave.value)
        self.wave = dered(obs_wave, self.z)
        self.disp_res = dered(self.disp_res, self.z)

        # TODO: after fit, return wave to original units
        # TODO: unit agnostic
        self.wave = self.wave.to(jwc.TARGET_WAVE_UNIT).value
        self.spec = self.spec.to(jwc.TARGET_FLUX_UNIT_AA).value
        self.noise = self.noise.to(jwc.TARGET_FLUX_UNIT_AA).value

        div = int(np.floor(np.log10(np.abs(np.nanmedian(self.spec)))))
        self.spec = self.spec / (10**div)
        self.noise = self.noise / (10**div)
        self.flux_norm = 10**div

        lam_range = (np.min(self.wave),np.max(self.wave))
        self.spec, log_lam, velscale = log_rebin(lam_range, self.spec, velscale=None, flux=False)
        self.noise, _, _ = log_rebin(lam_range, self.noise, velscale=velscale, flux=False)
        self.wave = np.exp(log_lam)
        self.velscale = velscale[0]


Reader = NIRSpecReader
