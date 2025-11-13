## `fit_feii`
*Type:* `bool`<br/>
*Default:* `True`<br/>
*Description:* Broad and narrow optical FeII templates are taken from [Véron-Cetty et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004A%26A...417..515V/abstract) with each line modeled using a Gaussian. One can also optionally using the template from [Kovačević et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010MSAIS..15..176K/abstract), however with limited coverage (4400 Å - 5500 Å). FeII emission can be very strong in some Type 1 (broad line) AGN, but is almost undetectable in Type 2 (narrow line) AGN.

## `fit_uv_iron`
*Type:* `bool`<br/>
*Default:* `False`<br/>
*Description:* Fits the empirical UV iron template from [Vestergaard and Wilkes (2001)](https://ui.adsabs.harvard.edu/abs/2001ApJS..134....1V/abstract), for high-redshift spectra with coverage < 3500 Å.

## `fit_balmer`
*Type:* `bool`<br/>
*Default:* `False`<br/>
*Description:* Fits a series of higher-order Balmer lines and Balmer pseudo-continuum for high-redshift spectra with coverage < 3500 Å.

## `fit_losvd`
*Type:* `bool`<br/>
*Default:* `True`<br/>
*Description:* Stellar line-of-sight velocity distribution (LOSVD) using Penalized Pixel-Fitting ([pPXF](https://www-astro.physics.ox.ac.uk/~mxc/software/#ppxf), [Cappellari et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C/abstract)) using templates from the [Indo-U.S. Library of Coudé Feed Stellar Spectra](https://www.noao.edu/cflib/) ([Valdes et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004ApJS..152..251V/abstract)) in the optical region 3460 Å - 9464 Å. This is used to obtain stellar kinematics in spectra with resolvable absorption features, such as stellar velocity and dispersion.

## `fit_host`
*Type:* `bool`<br/>
*Default:* `False`<br/>
*Description:* Fits a host galaxy template using single-stellar population templates from the EMILES library. Note that this method does not estimate stellar LOSVD, but can shift in velocity and convolve to match the data as best as it can.

## `fit_power`
*Type:* `bool`<br/>
*Default:* `True`<br/>
*Description:* This fits a power-law component to simulate the effect of the AGN "blue-bump" continuum.

## `fit_poly`
*Type:* `bool`<br/>
*Default:* `False`<br/>
*Description:* Fit a polynomial continuum component of a specified order. Polynomial options are specified by `poly_options` dictionary. Options are additive Legendre polynomial or multiplicative Legendre polynomial. The order must be within the range 0 <= order <= 99. Note: caution should be used when using polynomial components, as these can be degenerate with other continuum components, and higher-order polynomials can lead to overfitting.

## `fit_narrow`
*Type:* `bool`<br/>
*Default:* `True`<br/>
*Description:* Fit lines of the `line_type`:`na` in the line list. Narrow forbidden emission lines are seen in both Type 1 and Type 2 AGNs, as well as starforming galaxies.

## `fit_broad`
*Type:* `bool`<br/>
*Default:* `True`<br/>
*Description:* Fit lines of the `line_type`:`br` in the line list. Broad permitted emission lines are commonly seen in Type 1 AGN.

## `fit_absorp`
*Type:* `bool`<br/>
*Default:* `False`<br/>
*Description:* Fit lines of the `line_type`:`abs` in the line list. Occasionally one might need to fit a strong absorption feature that isn't described by stellar processes, such as a broad absorption line in a quasar.

## `tie_line_disp`
*Type:* `bool`<br/>
*Default:* `False`<br/>
*Description:* Ties the widths of all respective line types (all narrow lines are tied, all broad lines are tied, etc.). This can be done to significantly reduce the number of free parameters in the fit if fitting many lines, however it is not recommended.

## `tie_line_voff`
*Type:* `bool`<br/>
*Default:* `False`<br/>
*Description:* Ties the velocity offsets of all respective line types (all narrow lines are tied, all broad lines are tied, etc.). This can be done to significantly reduce the number of free parameters in the fit if fitting many lines, however it is not recommended.
