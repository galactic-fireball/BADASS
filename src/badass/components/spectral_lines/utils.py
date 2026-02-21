import numpy as np

import badass.utils.constants as consts

def calculate_fwhm(wave, line_profile, velscale):
    # Calculate fwhm of combined lines directly from the model
    line_profile = np.abs(line_profile)

    def lin_interp(x, y, i, half):
        return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

    def half_max_x(x, y):
        half = max(y)/2.0
        signs = np.sign(np.add(y, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        zero_crossings_i = np.where(zero_crossings)[0]
        if len(zero_crossings_i) == 2:
            return [lin_interp(x, y, zero_crossings_i[0], half),
                    lin_interp(x, y, zero_crossings_i[1], half)]
        return [0.0, 0.0]

    hmx = half_max_x(range(len(wave)),line_profile)
    fwhm_pix = np.abs(hmx[1]-hmx[0])
    fwhm = fwhm_pix*velscale
    return fwhm if np.isfinite(fwhm) else 0.0


def calculate_w80(wave, line_profile, line_center):
    # Calculate W80 of the full line profile
    line_profile = np.abs(line_profile)

    # Calculate the normalized CDF of the line profile
    cdf = np.cumsum(line_profile/np.sum(line_profile))
    v = (wave-line_center)/line_center*consts.c
    w80 = np.interp(0.91,cdf,v) - np.interp(0.10,cdf,v)

    # Correct for intrinsic W80
    # The formula for a Gaussian W80 = 1.09*FWHM = 2.567*disp_res (Harrison et al. 2014; Manzano-King et al. 2019)
    # w80 = np.sqrt((w80)**2-(2.567*disp_res)**2)

    return w80 if np.isfinite(w80) else 0.0
