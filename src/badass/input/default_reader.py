from astropy import constants as const
import numpy as np
import spectres

from badass.input.input import BadassInput
from badass.utils.utils import dered, log_rebin

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
        if not np.isclose(self.wave[1]-self.wave[0], self.wave[-1]-self.wave[-2]):
            # TODO: how to handle before logger setup?
            # if verbose:
            #     print("\n Input spectrum is not linearly binned. BADASS will linearly rebin and conserve flux...")
            new_wave = np.linspace(self.wave[0], self.wave[-1], len(self.wave))
            self.spec, self.noise = spectres.spectres(new_wavs=new_wave, spec_wavs=self.wave, spec_fluxes=self.spec,
                                          spec_errs=self.noise, fill=None, verbose=False)
            self.wave = new_wave

            # Fill in any NaN
            mask = np.isnan(self.spec)
            self.spec[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), self.spec[~mask])
            mask = np.isnan(self.noise)
            self.noise[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), self.noise[~mask])

        lam_range = (np.min(self.wave),np.max(self.wave))
        self.spec, log_lam, velscale = log_rebin(lam_range, self.spec, velscale=None, flux=False)
        self.noise, _, _ = log_rebin(lam_range, self.noise, velscale=velscale, flux=False)
        self.wave = np.exp(log_lam)
        self.velscale = velscale[0]

        # if noise vector is zero, set it to 10%
        if np.nansum(self.noise) == 0:
            self.noise = np.full_like(self.spec, 0.1*self.spec)

        frac = self.wave[1]/self.wave[0] # Constant lambda fraction per pixel
        dlam_gal = (frac - 1)*self.wave # Size of every pixel in Angstrom
        if isinstance(self.fwhm_res, (list, np.ndarray)):
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

        super().__init__(input_data, options)

Reader = DefaultReader
