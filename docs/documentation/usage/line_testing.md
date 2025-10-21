**`test_mode`**: (str; *Default="line"*)<br/>
Future releases of BADASS will have different testing methods; the only option available now is `line`.

**`lines`**: (list; *Default=[]*)<br/>
The lines (or group of lines) to test.  BADASS will perform these tests in the order the are defined in the list.

**`metrics`**: (list; *Default=["BADASS", "CHI2_RATIO", "AON"]*)<br/>
The testing metrics used to determine when the appropriate number of components is reached. Options are "BADASS", "ANOVA", "CHI2_RATIO", "SSR_RATIO", "F_RATIO", "AON".  We describe these below:
* *BADASS*: this test generates a confidence between 0 and 1 of the Monte-Carlo resampled maximum likelihood values of the simple and complex models.  For example, if the confidence is 0.95, you can say that the difference between residuals is significant, and an additional component is justified with 95% confidence.
* *ANOVA*: this test is anlogous to an analysis-of-variance (ANOVA) test.  It haves similarly to the BADASS test, but tends to prefer a more complex model, while the BADASS metric tends to prefer a simpler one.
* *CHI2_RATIO*: the ratio of the reduced chi-squared values for each fitted model.  This tends to be a robust method with thresholds between 0.1 and 0.25.  Interpreted as the fraction of the difference in the residuals weighted by the noise.
* *SSR_RATIO*: the ratio of the sum-of-squares of the residuals (SSR).  Values close to 1 indicate residuals that are very similar.  
* *F_RATIO*: the ratio of variances betweent he two fitted models, behaving similarly to the SSR ratio.
* *AON*: amplitude-over-noise (AON) of the final line.  This is not a test between models, but a final check to determine if the lines that comprise the best model have an amplitude-over-noise (otherwise known as signal-to-noise) above a certain threshold.  This metric is good to have to ensure you aren't fitting noise.

**`thresholds`**: (list; *Default=[0.95, 0.10, 3.0]*)<br/>
Thresholds for the above chosen metrics.  When a calculated metric goes *below* a threshold for a given metric, "convergence" is achieved.

**`conv_mode`**: (str; *Default="any"*; options are "any" or "all")<br/>
Whether "all" or "any" of the metrics must achieve convergence to determine the appropriate number of line components.  If "all" is chosen, then all metrics must go below the given threshold for convergence.  If "any" is chosen, any single metric must achive convergence.  

**`auto_stop`**: (bool; *Default=True*)<br/>
Automatically stop testing for a line when convergence is met.  For example, if we define five components for a line, but convergence is reached at three components, BADASS will not test that line further.

**`full_verbose`**: (bool; *Default=False*)<br/>
Print out the results of fitting each test (basinhopping callbacks, etc.) to screen.  This is `False` by default because it can be excessive (no exagerration, it will print *everything*).  This is useful to monitor the fit of each test as it is performed, but not recommended for the uninitiated.

**`plot_tests`**: (bool; *Default=True*)<br/>
Plot the results of each test and save them for visual comparison.

**`force_best`**: (bool; *Default=True*)<br/>
Forces the complex model to achieve a root-mean-squared-error (RMSE) comparable to or less than the simpler model.  This is highly recommended because of the caveats we give below.

**`continue_fit`**: (bool; *Default=True*)<br/>
Continue the fit (to max likelihood and/or MCMC) after the test is completed.  If False, BADASS terminates when line testing is completed.

We also test for different lines in the same fit.  Here we specify we want to fit the three narrow lines from above, but now we also want to test the broad H-beta line (`BR_H_BETA`) and some unknown line at 5100 Å (`NA_UNK_1`) in a separate test region:

```python
test_options = {
"test_mode":"line",
"lines": [["NA_OIII_5007","NA_OIII_4960","NA_H_BETA"],"BR_H_BETA","NA_UNK_1"], # The lines to test
"metrics": ["BADASS", "ANOVA", "CHI2_RATIO","AON"],# Fitting metrics to use when determining the 
"thresholds": [0.95, 0.95, 0.10, 3.0],
"auto_stop":True, # automatically stop testing once threshold is reached; False test all no matter 
"plot_tests":True,
"force_best":True, # this forces the more-complex model to have a fit better than the previous.
"continue_fit":True, # continue the fit with the best chosen model
}
```

**A few caveats to line testing the user must be aware of!**

The process of determining the "correct" number of components for a line is a naturally degenerate problem, especially if the individual components are not resolved.  A number of underlying gaussian processes can define the observed emission line shape, and the "best fit" to those processes can be achived multiple ways depending on the allowed widths, velocities, and number of components you include in the model.  Because of this, a local minimization technique usually fails (such as linear least squares or Levenberg-Marquardt) unless the user supplies very accurate apriori guesses for each component.  Determining these apriori values defeats the purpose of automated fitting.  
Instead, BADASS implements a stochastic global minimizer (basinhopping) to ensure that the fit can achive a global minimum and not get stuck in local minima.  Basinhopping tends to be just as accurate as simulated or dual-annealing, but much faster than brute-force methods.  However, because it *is* stochastic, this can lead to inconsistent results between line components.  To this I say: it doesn't matter; if individual line components are not resolved, then only the total fit to the line profile shape matters, and is the greatest amount of information you can actually recover from your data.  
There are cases in which decomposing partially resolved narrow "core" components from another more-broad "outflow" component can be done (i.e., a two-component fit), and one can make reasonable assumptions about the physical nature of those components, but this typically ends at two components.  With more than 2 components, the fit can become highly degenerate *between* components, that is, you can achieve the same fit multiple ways depending on the allowed withs, velocities, and even initial guesses.  Discerning the physical nature of individual components for `ncomp`>2 for a given line should be treated with many ceveats unless those underlying physical processes are understood apriori. 

Now, if you're getting unexpected behavior from your multiple component tests, there are a number of things you can do:
* plot out each test to visually confirm that the fit is doing what you expected (set `plot_tests` to `True` in `test_options`.
* set higher `n_basinhop` threshold (15-25); this is the number of sucessive basinhopping thresholds before a solution is achieved. A low number means basinhopping gives up sooner (less time), and a higher threshold gives basinhopping a greater chance of finding the better fit (more time).  Often times, if `n_basinhop` is too low, the best fit is usually not achieved for a given test, which means that even if `force_best` is used, successive tests might not achieve their best fit either, because they just have to be better than the previous fit.
* check parameter limits; if a paramter such as line dispersion or velocity hits its limit, it might mean that a better fit could not be achieved because a line was not allowed to go outside of its defined parameter space to achieve a better fit.  This will happen if `voff` or `disp` parameter limits are too restrictive.  You can control global line parameter limits using `narrow_options`, `broad_options`, or `absorp_options`, or set them individually for each line using `disp_plim` (for dispersion) or `voff_plim` (for velocity offset). 
* check parameter constraints; if you have soft constraints (for example `["OIII_5007_2_DISP","OIII_5007_DISP"]`, which forces the second component dispersion to be greater than the first component dispersion), you may be over-constraining the model.  This can lead to `force_best` never reaching a good solution (or just taking a very long time).  The best fits are usually acheived when you don't use any soft constraints, especially when the number of components exceeds 2.
* simplify the continuum.  Lots of continuum flexibility can lead to strange behavior.  For this reason, we already restrict the polynomial continuum in the test regions to be 2. Removing some continuum components (e.g., FeII) might help. 
* check behavior of fit; if you really want to know what the testing is doing under the hood, set `"full_verbose":True` and BADASS will print every step of the testing process to screen.  Warning: its a lot of output, but its the only way to monitor how the fit is performing for each test.