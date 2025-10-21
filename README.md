
![BADASS logo](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS2_logo.gif)

Ridiculous acronyms are a [long-running joke in astronomy](https://lweb.cfa.harvard.edu/~gpetitpas/Links/Astroacro.html), but here, spectral fitting ain't no joke!

[BADASS](https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.2871S/abstract) is an open-source spectral analysis tool designed for detailed decomposition of Sloan Digital Sky Survey (SDSS) spectra, and specifically designed for the fitting of Type 1 ("broad line") Active Galactic Nuclei (AGN) in the optical.  The fitting process utilizes the Bayesian affine-invariant Markov-Chain Monte Carlo sampler [emcee](https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F/abstract) for robust parameter and uncertainty estimation, as well as autocorrelation analysis to access parameter chain convergence.  BADASS can fit the following spectral features:
- Stellar line-of-sight velocity distribution (LOSVD) using Penalized Pixel-Fitting ([pPXF](https://www-astro.physics.ox.ac.uk/~mxc/software/#ppxf), [Cappellari et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C/abstract)) using templates from the [Indo-U.S. Library of Coudé Feed Stellar Spectra](https://www.noao.edu/cflib/) ([Valdes et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004ApJS..152..251V/abstract)) in the optical region 3460 Å - 9464 Å.
- Broad and Narrow FeII emission features using the FeII templates from [Véron-Cetty et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004A%26A...417..515V/abstract) or [Kovačević et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010MSAIS..15..176K/abstract).
- UV iron template from [Vestergaard and Wilkes (2001)](https://ui.adsabs.harvard.edu/abs/2001ApJS..134....1V/abstract)
- Individual narrow, broad, and/or absorption line features.
- AGN power-law continuum and Balmer pseudo-continuum.
- "Blue-wing" outflow emission components found in narrow-line emission. 

A more-detailed summary of BADASS, as well as a case-study of ionized gas outflows, is given in [Sexton et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.2871S/abstract). 

All spectral components can be turned off and on via the [Jupyter Notebook](https://jupyter.org/) interface, from which all fitting options can be easily changed to fit non-AGN-host galaxies (or even stars!).  BADASS uses multiprocessing to fit multiple spectra simultaneously depending on your hardware configuration.  The code was originally written in Python 2.7 to fit Keck Low-Resolution Imaging Spectrometer (LRIS) data ([Sexton et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...878..101S/abstract)), but because BADASS is open-source and *not* written in an expensive proprietary language, one can easily contribute to or modify the code to fit data from other instruments. Out of the box, BADASS fits SDSS spectra, MANGA IFU cube data, and examples are provided for fitting user-input spectra of any instrument.

Before getting started, check out the [Documentation](https://galactic-fireball.github.io/BADASS)


# Known Issues

We've done our best to find any bugs or issues. BADASS is under constant development for this reason. Please let us know if there are any features you'd like us to implement, or bugs that need to be fixed.

# Credits

- [Dr. Remington Oliver Sexton (USNO/GMU)](https://r-magnitude.com/) 
- [Sara M. Doan (GMU)](https://mason.gmu.edu/~sdoan2/)
- [Michael Reefe (GMU)](https://github.com/Michael-Reefe)
- William Matzko (GMU)

# License

MIT License

Copyright (c) 2022 Remington Oliver Sexton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
