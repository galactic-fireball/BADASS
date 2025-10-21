
## New in Version 11.0.0
* Code has been reorganized and made more modular.
* An options schema has been developed: see `utils/schema.py`.
* Supported input types have been moved to individual `BadassInput` classes: see `input/`.
* Templates have been moved to individual `BadassTemplate` classes: see `components/templates/`.
* The main fitting execution (including line/config testing, max likelihood bootstrapping, and MCMC) has been moved to a `BadassRunContext` class.
* Inherent support for JWST NIRSpec and MIRI instruments.
* Plotting (`utils/plotting.py`) and logging (`utils/logger.py`) functionality moved to separate files.


## New in Version 10.2.0
* New line and configuration testing framework. See [Line Testing and Options](#line-testing-and-options).
* Line component options.  See [Line Component Options](#line-component-options).
* Improvements in autocorrelation calculations.
* W80 now a standard output parameter for all lines.
* Outputs line widths that are both corrected and uncorrected for input resolution
* BADASS now normalizes the spectrum internally for fitting purposes.
* Various improvements to global optimizer used in initial fit.
* **Note**: The algorithm used for scaling the noise to achieve a $\chi_\nu^2=1$ was found to be numerically unstable and users should use `fit_stat='ML'` instead.
