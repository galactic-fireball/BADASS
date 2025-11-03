
**`fit_reg`**: (*tuple/list of length (2,)*; *Default: (4400,5500)*)<br/>
the minimum and maximum desired fitting wavelength in angstroms

<!-- **`good_thresh`**: (*float [0.0,1.0]*; *Default: 0.0*)<br/>
the cutoff for minimum fraction of "good" pixels (determined by SDSS) within the fitting range to allow for fitting of a given spectrum.  If the spectrum has fewer good pixels than this value, BADASS skips over it and moves onto the next spectrum. -->

**`mask_bad_pix`**: (*bool*; *Default: False*)<br/>
Mask pixels which the specified instrument has flagged as bad due to sky line subtraction or cosmic rays. 

**`mask_emline`**: (*bool*; *Default: False*)<br/>
Mask any significant absorption and emission features relative to the continuum.  This uses an automated iterative moving median filter of various sizes to detect significant flux differences between window sizes.  Good for continuum fitting but tends to over mask lots of features near the edges of the spectrum.

**`mask_metal`**: (*bool*; *Default: False*)<br/>
Performs the same moving median filter algorithm as `mask_emline` but only to absorption features.  Works well for metal absorption features seen typically in high-redshift spectra.

**`fit_stat`**: (*str*; *Default: : "ML"*)<br/>
The fit statistic used for the likelihood. Options:

* "ML" for standard maximum likelihood (pixels weighted by noise with no noise scaling)
* "OLS" for ordinary least-squares fitting (all pixels weighted by same amount).

**`n_basinhop`**: (*int*; *Default: 25*)<br/>
Number of successive `niter_success` times the basinhopping algorithm needs to achieve a solution. The fit becomes much better with more success times, however this can increase the time to a solution significantly. Recommended 5-10.

**`reweighting`**: (*bool*; *Default: True*)<br/>
If true, BADASS will reweight the noise vector to achieve a reduced chi-squared ~ 1. This is done after the initial basinhopping fit, and applied to any bootstrapped uncertainties and MCMC fitting performed afterward. This does not affect the chi-squared ratio metric used in line and configuration testing, but does effect the amplitude-over-noise and SNR calculations in BADASS.

**`test_lines`**: (`bool`:*bool*; *Default: False*)<br/>
Performs tests for lines. Options are specified in `test_options`.

**`max_like_niter`**: (*int*; *Default: 10*)<br/>
Number of bootstrapping iterations to perform after the initial basinhopping fit. This is a means to obtain uncertainties on parameters without performing MCMC fitting, however, do not produce as robust uncertainties as MCMC.

**`output_pars`**: (*bool*; *Default: False*)<br/>
Convenience feature that prints out all free parameters.

**`cosmology`**: (*Default*: `{"H0":70.0, "Om0": 0.30}`)<br/>
The flat Lambda-CDM cosmology assumed for calculating luminosities from fluxes.