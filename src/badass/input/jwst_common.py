from astropy.io import fits
import astropy.units as u
import numpy as np
import pathlib

from badass.input.cube_reader import CubeReader
from badass.utils.utils import dered, log_rebin

TARGET_WAVE_UNIT = u.AA
TARGET_FLUX_UNIT_UM = u.erg / u.s / (u.cm**2) / u.um
TARGET_FLUX_UNIT_AA = u.erg / u.s / (u.cm**2) / u.AA

class JWSTReader(CubeReader):

    @classmethod
    def get_cube_data(cls, input_data, options):
        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading JWST spectra from data currently unsupported') # TODO

        if not input_data.exists():
            raise Exception('Not found: %s'%str(input_data))

        if not 'redshift' in options.fit_options:
            raise Exception('Redshift for JWST cube must be provided')

        cube_data = {}
        cube_data['z'] = options.fit_options.redshift
        cube_data['infile'] = input_data

        hdu = fits.open(input_data)
        header = hdu['SCI'].header
        cunit = header['CUNIT3']
        bunit = header['BUNIT']

        nwave = hdu['SCI'].data.shape[0]
        wave0 = header['CRVAL3'] - (header['CRPIX3'] - 1) * header['CDELT3']
        obs_wave = (wave0 + np.arange(nwave)*header['CDELT3']) * u.Unit(cunit)

        cube_spec = hdu['SCI'].data.T * u.Unit(bunit)
        cube_err = hdu['ERR'].data.T * u.Unit(bunit)
        if '/sr' in bunit:
            pxar = header['PIXAR_SR'] * u.sr
            cube_spec *= pxar
            cube_err *= pxar

        cube_spec = cube_spec.to(TARGET_FLUX_UNIT_UM, equivalencies=u.spectral_density(obs_wave))
        cube_err = cube_err.to(TARGET_FLUX_UNIT_UM, equivalencies=u.spectral_density(obs_wave))
        cube_err[np.isnan(cube_err)] = np.nanmedian(cube_err)
        cube_data['ra'], cube_data['dec'] = hdu[0].header['TARG_RA'], hdu[0].header['TARG_DEC']
        hdu.close()

        cls.set_dispersion(cube_data, options, obs_wave.value)

        wave = dered(obs_wave, cube_data['z'])
        cube_data['disp_res'] = dered(cube_data['disp_res'], cube_data['z'])

        # TODO: after fit, return wave to original units
        # TODO: unit agnostic
        wave = wave.to(TARGET_WAVE_UNIT).value
        cube_spec = cube_spec.to(TARGET_FLUX_UNIT_AA).value
        cube_err = cube_err.to(TARGET_FLUX_UNIT_AA).value

        div = int(np.floor(np.log10(np.abs(np.nanmedian(cube_spec)))))
        flux_norm = 10**div
        cube_spec = cube_spec / flux_norm
        cube_err = cube_err / flux_norm
        cube_data['flux_norm'] = flux_norm

        cube_data['wave'] = wave
        cube_data['velscale'] = np.nan # will be set when the class is initialized
        cube_data['spec'] = cube_spec
        cube_data['noise'] = cube_err
        cube_data['splitable'] = ['spec', 'noise']
        return cube_data


    def postinit(self):
        # TODO: LogRebinMixin
        lam_range = (np.min(self.wave),np.max(self.wave))
        self.spec, log_lam, velscale = log_rebin(lam_range, self.spec, velscale=None, flux=False)
        self.noise, _, _ = log_rebin(lam_range, self.noise, velscale=velscale, flux=False)
        self.wave = np.exp(log_lam)
        self.velscale = velscale[0]
        return super().postinit()


    @classmethod
    def set_dispersion(cls, cube_data, options, obs_wave):
        # Instrument child classes will override
        pass
