## `fit_reg`
*Type:* `list | str`<br/>
*Default:* `auto`<br/>
*Description:* The minimum and maximum desired fitting wavelength in angstroms.
*Examples:* `(4400, 5500)`, `'auto'`

## `n_basinhop`
*Type:* `int`<br/>
*Default:* `25`<br/>
*Description:* Number of successive `niter_success` times the basinhopping algorithm needs to achieve a solution. The fit becomes much better with more success times, however this can increase the time to a solution significantly.

## `max_like_niter`
*Type:* `int`<br/>
*Default:* `10`<br/>
*Description:* Number of bootstrapping iterations to perform after the initial basinhopping fit. This is a means to obtain uncertainties on parameters without performing MCMC fitting, however, do not produce as robust uncertainties as MCMC.

## `fit_area`
*Type:* `dict`<br/>
*Default:* `{}`<br/>
*Description:* Defines the area to be fit for data cubes. See `examples/muse_examples.py` for usage.

## `reweighting`
*Type:* `bool`<br/>
*Default:* `True`<br/>
*Description:* If `True`, BADASS will reweight the noise vector to achieve a reduced chi-squared ~ 1. This is done after the initial basinhopping fit, and applied to any bootstrapped uncertainties and MCMC fitting performed afterward. This does not affect the chi-squared ratio metric used in line and configuration testing, but does effect the amplitude-over-noise and SNR calculations in BADASS.

## `fit_stat`
*Type:* `str`<br/>
*Default:* `ML`<br/>
*Description:* The fit statistic used for the likelihood. Options:

* `'ML'` for standard maximum likelihood (pixels weighted by noise with no noise scaling).
* `'OLS'` for ordinary least-squares fitting (all pixels weighted by same amount).

## `cosmology`
*Type:* `dict | Prodict`<br/>
*Default:* `{'H0': 70.0, 'Om0': 0.3}`<br/>
*Description:* The flat Lambda-CDM cosmology assumed for calculating luminosities from fluxes.

## `test_lines`
*Type:* `bool`<br/>
*Default:* `False`<br/>
*Description:* Performs tests for lines. Options are specified in `test_options`.

## `mask_emline`
*Type:* `bool`<br/>
*Default:* `False`<br/>
*Description:* Mask any significant absorption and emission features relative to the continuum. This uses an automated iterative moving median filter of various sizes to detect significant flux differences between window sizes. Good for continuum fitting but tends to over mask lots of features near the edges of the spectrum.

## `mask_bad_pix`
*Type:* `bool`<br/>
*Default:* `False`<br/>
*Description:* Mask pixels which the specified instrument has flagged as bad due to sky line subtraction or cosmic rays.

## `mask_metal`
*Type:* `bool`<br/>
*Default:* `False`<br/>
*Description:* Performs the same moving median filter algorithm as `mask_emline` but only to absorption features. Works well for metal absorption features seen typically in high-redshift spectra.

## `feature_edge_pad`
*Type:* `int | float`<br/>
*Default:* `10`<br/>

