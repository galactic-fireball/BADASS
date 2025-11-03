## Prior Distributions

The newest version of BADASS provides an easy way to implement some non-uniform priors, which can give the user more control over some parameters that may be poorly-constrained by a uniform prior or soft-constraints.

By default, all parameters are subject to an (improper, i.e. unnormalized) uniform prior with values [0,1], or [-inf,0] in log.  The value of this prior is zero if its respective parameter violates bounds and/or any constraints, and 1 if it does not.  This log-prior value is added to the log-likelihood.  

BADASS currently allows for three different non-uniform priors on any free parameter, all of which follow the `scipy.stats` standard:

**Normal (`gaussian`) Prior**: `stats.norm.logpdf(x,loc,scale)`; `loc`  and `scale` are the mean and standard deviation of the distribution, respectively.  If `loc` and/or `scale` are not provided, they will be calculated from the `init` (initial value) and `plim` (parameter limits) keywords that define each free parameter.  If not provided, the assumed `loc` is set equal to `init`, and `scale` is determined such that the difference between `init` and the absolute maximum `plim` is at 5 standard deviations, i.e., `scale = (|plim|-init)/5`. 

**Half-Norm (`halfnorm`) Prior**: `stats.halfnorm.logpdf(x,loc,scale)`; `loc`  and `scale` are the mean and standard deviation of the  positively bounded [0,+inf] distribution, respectively.  If `loc` and/or `scale` are not provided, they will be calculated from the `init` (initial value) and `plim` (parameter limits) keywords that define each free parameter.  If not provided, the assumed `loc` is set equal to `init`, and `scale` is determined such that the difference between `init` and the absolute maximum `plim` is at 5 standard deviations, i.e., `scale = (|plim|-init)/5`. 

**Log-Uniform (`jeffreys`) Prior**: `stats.loguniform.logpdf(x,a,b,loc)`; `loc` is the mean of the distribution, and `a` and `b` are shape parameters, where `0<a<b`.  The parameters `a` and `b` are determined from `plim` automatically, whereas `loc` can be specified by the user. 

By default, all model continuum priors (i.e., stellar kinematics, power law continuum, FeII emission) have uniform priors.  Changes to these priors can be made in the `initialize_pars()` function.  For line parameters, the only prior implemented across all lines is the velocity (`voff`) parameter, which is chosen to be `gaussian`, with a `loc=0` and `scale=100`.  This is done so that lines are biased to not stray too far way from where they are expected to be.   Gaussian priors are also put on any higher-order moments (`h3`, `h4`, ...) of the gauss-hermite, laplace, or uniform line profiles. 

To specify a prior on a parameter, one must provide the `kind`  (`gaussian`, `halfnorm`, or `jeffreys`).  Other parameters such as `loc` or `scale` are optional because they can be inferred from the `init` and `plim` keywords for each free parameter.

Here is an example of a 4-moment gauss-hermite line with a strict gaussian prior on `h3` and `h4`.  By doing this, it biases the moments to be very close to zero, and thus the gauss-hermite profile to be more gaussian, i.e., `h3~0` and `h4~0`.

```python
"BR_H_BETA"   :{"center":4862.691, 
                    "amp":"free", 
                    "disp":"free", 
                    "voff":"free",
                    "h3":"free",
                    "h3_prior":{"type":"gaussian","loc":0.0,"scale":0.01},
                    "h4":"free",
                    "h4_prior":{"type":"gaussian","loc":0.0,"scale":0.01},
                    "disp_plim":(500,5000),
                    "disp_init":1000.0,
                    "line_profile":"GH",
                    "line_type":"br"
                   },
```

## Line Lists

The default line list built into BADASS references most of the [standard SDSS lines](http://classic.sdss.org/dr6/algorithms/linestable.html).  The actual reference list is inside the `line_list_default()` function in the `badass.py` script.   BADASS expects a line entry to be in the following form

```python
"NA_OIII_5007" :{"center"   :5008.240, # rest-frame wavelength of line
         "amp"      :"free", # "free" parameter or tied to another valid parameter
         "amp_init" : float, # initial guess value 
         "amp_plim" : tuple, # tuple of (lower,upper) bounds of parameter
         
         "disp"     :"free", # "free" parameter or tied to another valid parameter
         "disp_init": float, # initial guess value 
         "disp_plim": tuple, # tuple of (lower,upper) bounds of parameter
         
         "voff"     :"free",  # "free" parameter or tied to another valid parameter
         "voff_init": float, # initial guess value 
         "voff_plim": tuple, # tuple of (lower,upper) bounds of parameter
         
         "line_type": "na", # line type ["na","br","out","abs", or "user]
         "line_profile": "gaussian" # gaussian, lorentzian, voigt, gauss-hermite, laplace, or uniform
         "label"    : string, # a name for the line for plotting purposes
         },
```

If `_init` or `_plim` keys are not explicitly assigned, BADASS will assume some reasonable values based on the `line_type`.  There are additional keys available when `line_profile` is Voigt (a `shape` key) or Gauss-Hermite (higher orders `h3`, `h4`, etc.).  Keep in mind that BADASS will enforce line profiles defined in the `fit_options` for `line_types` `na`, `br`, `out`, and `abs`; if one wants to define a custom line that isn't the same line profile shape as those defined in `fit_options`, one should use the `user` `line_type`.

For example:

```python
"NA_OIII_5007" :{"center"   : 5008.240, 
         "amp"      : "free",
         "disp"     : "free",
         "voff"     : "free",
         "line_type": "na",
         "label"    : r"[O III]"
         },
```

or in a more general case

```python
"RANDOM_USER_LINE" :{"center"      : 3094.394, # some random wavelength 
             "amp"         : "free",
             "disp"        : "free",
             "voff"        : "free",
             "h3"          : "free",
             "h4"          : "free",
             "line_type"   : "user",
             "line_profile": "gauss-hermite"
             "label"       : r"User Line"
         },
```

## Hard Constraints 

BADASS uses the `numexpr` module to allow for hard constraints on line parameters (i.e., "tying" one line's parameter's to another line's parameters).  To do this, the only requirement is that the constraint be a valid free parameter.  The most common case is tying the [OIII] doublet widths and velocity offsets:

```python
    "NA_OIII_4960" :{"center":4960.295,
             "amp":"(NA_OIII_5007_AMP/2.98)", 
             "disp":"NA_OIII_5007_DISP", 
             "voff":"NA_OIII_5007_VOFF", 
             "line_type":"na" ,
             "label":r"[O III]"
             },
             
    "NA_OIII_5007" :{"center":5008.240, 
             "amp":"free", 
             "disp":"free", 
             "voff":"free", 
             "line_type":"na" ,
             "label":r"[O III]"
             },

```
This works because when we define `NA_OIII_5007`, free parameters are created for the amplitude (`NA_OIII_5007_AMP`), dispersion (`NA_OIII_5007_DISP`) and velocity offset (`NA_OIII_5007_VOFF`), because we specified that they are *free* parameters.  These free parameters are the actual parameters that are solved for.  We can then reference those free valid parameters for `NA_OIII_4960`.  The power of the `numexpr` module is that we can also perform mathematical operations on those parameters *during* the fit, for example we can fix the amplitude of [OIII]4960 to be the [OIII]5007 amplitude divided by 2.93.  This makes implementing hard constraints very easy and is a very powerful feature.  With that said, you can do some pretty wild and unrealistic stuff, so use it responsibly.

## Soft Constraints

BADASS also uses the `numexpr` module to implement soft constrains on free parameters.  A soft constraint is defined here as a limit of a free parameter with respect to another free parameter, i.e., soft constraints are inequality constraints.  For example, if we want BADASS to enforce the requirement that broad H-beta has a greater dispersion than narrow [OIII]5007, we would say 

<img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{110}\bg{black}(\rm{broad~H}\beta\rm{~DISP})&space;>=&space;(\rm{narrow~[OIII]5007~DISP})" title="https://latex.codecogs.com/png.image?\inline \large \dpi{110}\bg{black}(\rm{broad~H}\beta\rm{~DISP}) >= (\rm{narrow~[OIII]5007~DISP})" />

or in the way the `scipy.optimize()` module requires it

<img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{110}\bg{black}(\rm{broad~H}\beta\rm{~DISP}&space;)-&space;&space;(\rm{narrow~[OIII]5007~DISP})&space;>=&space;0" title="https://latex.codecogs.com/png.image?\inline \large \dpi{110}\bg{black}(\rm{broad~H}\beta\rm{~DISP} )- (\rm{narrow~[OIII]5007~DISP}) >= 0" />

In BADASS, this soft constraint would be implemented as a tuple of length 2: 
```python
("BR_H_BETA_DISP","NA_OIII_5007_DISP")
```
By default BADASS includes the following list of soft constraints in the `initialize_pars()` function:
```python
soft_cons = [
      ("BR_H_BETA_DISP","NA_OIII_5007_DISP"), # broad H-beta width > narrow [OIII] width
      ("BR_H_BETA_DISP","OUT_OIII_5007_DISP"), # broad H-beta width > outflow width
      ("OUT_OIII_5007_DISP","NA_OIII_5007_DISP"), # outflow width > narrow [OIII] width
      ]
```