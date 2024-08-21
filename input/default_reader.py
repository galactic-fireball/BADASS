from astropy import constants as const
import numpy as np
import spectres

from input.input import BadassInput
from utils.utils import dered, log_rebin

class DefaultReader(BadassInput):
    def __init__(self, input_data, options):
        if not isinstance(input_data, dict):
            raise Exception('Default user data input should be dict')

        self.__dict__.update(input_data)

        expected_vals = ['wave', 'spec', 'noise', 'z', 'fwhm_res']
        for attr in expected_vals:
            if not hasattr(self, attr) or getattr(self, attr, None) is None:
                raise Exception('BADASS user input missing expected value: {attr}'.format(attr=attr))

        # First, we must log-rebin the linearly-binned input spectrum
        # If the spectrum is NOT linearly binned, we need to do that before we try to log-rebin
        wave = self.wave
        if not np.isclose(wave[1]-wave[0], wave[-1]-wave[-2]):
            # TODO: how to handle before logger setup?
            # if verbose:
            #     print("\n Input spectrum is not linearly binned. BADASS will linearly rebin and conserve flux...")
            new_wave = np.linspace(wave[0], wave[-1], len(wave))
            spec, err = spectres.spectres(new_wavs=new_wave, spec_wavs=wave, spec_fluxes=self.spec,
                                          spec_errs=self.noise, fill=None, verbose=False)
            # Fill in any NaN
            mask = np.isnan(spec)
            spec[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), spec[~mask])
            mask = np.isnan(err)
            err[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), err[~mask])
            wave = new_wave

        lam_range = (np.min(wave),np.max(wave))
        spec, log_lam, velscale = log_rebin(lam_range, spec, velscale=None, flux=False)
        noise, _, _ = log_rebin(lam_range, err, velscale=velscale, flux=False)
        lam_gal = np.exp(log_lam)

        self.wave = lam_gal
        self.spec = spec
        self.noise = noise
        self.velscale = velscale[0]

        # if noise vector is zero, set it to 10%
        if np.nansum(self.noise) == 0:
            self.noise = np.full_like(self.spec, 0.05)

        frac = self.wave[1]/self.wave[0] # Constant lambda fraction per pixel
        dlam_gal = (frac - 1)*self.wave # Size of every pixel in Angstrom
        if type(self.fwhm_res) in (list, np.ndarray):
            self.disp_res = self.fwhm_res/2.3548
        else:
            self.disp_res = np.full(self.wave.shape, fill_value=self.fwhm_res/2.3548)

        self.wave = dered(self.wave, self.z)
        self.disp_res = dered(self.disp_res, self.z)

        # TODO: add?
        # Mask pixels exactly equal to zero (but not negative pixels)
        # mask_zeros = True 
        # edge_mask_pix = 5 
        # zero_pix = np.where(galaxy==0)[0]
        # if mask_zeros:
        #     for i in zero_pix:
        #         m = np.arange(i-edge_mask_pix,i+edge_mask_pix,1)
        #         for b in m:
        #             fit_mask_bad.append(b)


Reader = DefaultReader
