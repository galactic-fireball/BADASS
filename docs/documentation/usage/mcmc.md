
**`mcmc_fit`**: (*bool*; *Default: False*)<br/>
Perform fit with MCMC using the initial maximum likelihood fit as initial parameters for the fit. It is *highly recommended* that one use MCMC to perform the fit, although sampling will require a significant amount of time compared to a maximum likelihood fit using `scipy.optimize.minimize()`.

**`nwalkers`**: (*int*; *Default: 100*)<br/>
Number of "walkers" per parameter used by emcee to explore each parameter space. The minimum number of walkers is 2 x ( # of free parameters), set by emcee.

**`auto_stop`**: (*bool*; *Default: True*)<br/>
If `True`, autocorrelation is used to automatically stop the fitting process when a convergence criteria (`conv_type`) is achieved. 

**`conv_type`**: (*str*; *Default: "median"*; *options: "all", "median", "mean", list of parameters*)<br/>
Mode of convergence.  Convergence of 'all' ensures all fitting parameters have achieved the desired `ncor_times` and `autocorr_tol` criteria, while "median" and "mean" only ensure that `ncor_times` and `autocorr_tol` criteria have been met for the median or mean of all parameters, respectively. A list of valid parameters is also acceptable to ensure specific parameters have achieved convergence even if others have not. In general "median" requires the fewest number of iterations and is not sensitive to poorly-constrained parameters, and "all" and "mean" require the most number of iterations and are much more sensitive to fluctuations in calculated autocorrelation times and tolerances. A list of parameters is suitable in cases where one is only interested in certain spectral features.

**`min_samp`**: (*int*; *Default: 1000*)<br/>
If `auto_stop=True`, then the `burn_in` is the iteration at which convergence is achieved, and `min_samp` is the number of iterations *past convergence* used for posterior sampling (the samples used for histograms and estimating best-fit parameters and uncertainties). If for some reason the parameters "jump out" of convergence, the `burn_in` will reset and BADASS will continue to sample until convergence is met again. If emcee completes `min_samp` iterations after convergence is achieved without jumping out of convergence, this concludes the MCMC sampling.

**`ncor_times`**: (*int* or *float*; *Default=10*)<br/>
The number of integrated autocorrelation times (iterations) needed for convergence. We recommend a minimum of `ncor_times=2.0`. In general, it will require more than 2.0 autocorrelation times to calculate the autocorrelation time for a parameter chain. Increasing `ncor_times` ensures that the parameter chain has stopped exploring the parameter space and is ready to begin sampling for the posterior distribution.

**`autocorr_tol`**: (*int* or *float*; *Default=10*)<br/>
The percent change in the current integrated autocorrelation time and the previously calculated integrated autocorrelation time. The `write_iter` determines how often BADASS checks a parameter's integrated autocorrelation time. In general, we find that `autocorr_tol=5` (a 5% change) is acceptable for a converged parameter chain. A parameter chain that diverges more than 10% in 100 iterations could still be exploring the parameter space for a stable solution. A `autocorr_tol=1` (a 1% change) typically requires many more iterations than necessary for convergence.

**`write_iter`**: (*int*; *Default=100*)<br/>
The frequency at which BADASS writes the current parameter values (median walker positions). If `auto_stop=True`, then BADASS checks for convergence every `write_iter` iteration for convergence.

**`write_thresh`**: (*int*; *Default=100*)<br/>
The iteration at which writing (and checking for convergence if `auto_stop=True`) begins. BADASS does not check for convergence before this value.

**`burn_in`**: (*int*; *Default=1500*)<br/>
If `auto_stop=False` then this serves as the burn-in for a maximum number of iterations. If `auto_stop=True`, this value is ignored.

**`min_iter`**: (*int*; *Default=100*)<br/>
The minimum number of iterations BADASS performs before it is allowed to stop. This is true regardless of the value of `auto_stop`.

**`max_iter`**: (*int*; *Default=2500*)<br/>
The maximum number of iterations BADASS performs before stopping. This value is adhered to regardless of the value of `auto_stop` to set a limit on the number of iterations before BADASS should "give up."
