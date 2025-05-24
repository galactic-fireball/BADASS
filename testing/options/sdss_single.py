################################# IO Options ################################
io_options={
    "infmt" : "sdss",
    "output_dir" : 'sdss_test', # same directory as input file
    "dust_cache" : None,
    "overwrite" : False,
    "log_level" : "debug",
    "err_level": "warning",
    "multiprocess": False,
}

################################## Fit Options #################################
# Fitting Parameters
fit_options={
"fit_reg"    : (4400,5500),# Fitting region; Note: Indo-US Library=(3460,9464)
"good_thresh": 0.0, # percentage of "good" pixels required in fig_reg for fit.
"mask_bad_pix": False, # mask pixels SDSS flagged as 'bad' (careful!)
"mask_emline" : False, # automatically mask lines for continuum fitting.
"mask_metal": False, # interpolate over metal absorption lines for high-z spectra
"fit_stat": "ML", # fit statistic; ML = Max. Like. , OLS = Ordinary Least Squares
"n_basinhop": 25, # Number of consecutive basinhopping thresholds before solution achieved
"reweighting":True, # re-weight the noise after initial fit to achieve RCHI2 = 1
"test_lines": False, # Perform line/configuration testing for multiple components
"max_like_niter": 25, # number of maximum likelihood iterations
"output_pars": False, # only output free parameters of fit and stop code (diagnostic)
"cosmology": {"H0":70.0, "Om0": 0.30}, # Flat Lam-CDM Cosmology
}
################################################################################

########################### MCMC algorithm parameters ##########################
mcmc_options={
"mcmc_fit"    : False, # Perform robust fitting using emcee
"nwalkers"    : 100,  # Number of emcee walkers; min = 2 x N_parameters
"auto_stop"   : False, # Automatic stop using autocorrelation analysis
"conv_type"   : "all", # "median", "mean", "all", or (tuple) of parameters
"min_samp"    : 1000,  # min number of iterations for sampling post-convergence
"ncor_times"  : 10.0,  # number of autocorrelation times for convergence
"autocorr_tol": 10.0,  # percent tolerance between checking autocorr. times
"write_iter"  : 100,   # write/check autocorrelation times interval
"write_thresh": 100,   # iteration to start writing/checking parameters
"burn_in"     : 1500, # burn-in if max_iter is reached
"min_iter"    : 1500, # min number of iterations before stopping
"max_iter"    : 2500, # max number of MCMC iterations
}
################################################################################

############################ Fit component op dtions #############################
comp_options={
"fit_opt_feii"     : True, # optical FeII
"fit_uv_iron"      : False, # UV Iron 
"fit_balmer"       : False, # Balmer continuum (<4000 A)
"fit_losvd"        : False, # stellar LOSVD
"fit_host"         : True, # host template
"fit_power"        : True, # AGN power-law
"fit_poly"         : True, # Add polynomial continuum component
"fit_narrow"       : True, # narrow lines
"fit_broad"        : True, # broad lines
"fit_absorp"       : False, # absorption lines
"tie_line_disp"    : False, # tie line widths (dispersions)
"tie_line_voff"    : False, # tie line velocity offsets
}

# Line options for each narrow, broad, and absorption.
narrow_options = {
#     "amp_plim": (0,1), # line amplitude parameter limits; default (0,)
    "disp_plim": (0,500), # line dispersion parameter limits; default (0,)
    "voff_plim": (-500,500), # line velocity offset parameter limits; default (0,)
    "line_profile": "gaussian", # line profile shape*
    "n_moments": 4, # number of higher order Gauss-Hermite moments (if line profile is gauss-hermite, laplace, or uniform)
}

broad_options ={
#     "amp_plim": (0,40), # line amplitude parameter limits; default (0,)
    "disp_plim": (500,3000), # line dispersion parameter limits; default (0,)
    "voff_plim": (-1000,1000), # line velocity offset parameter limits; default (0,)
    "line_profile": "gaussian", # line profile shape*
    "n_moments": 4, # number of higher order Gauss-Hermite moments (if line profile is gauss-hermite, laplace, or uniform)
}

absorp_options = {
#     "amp_plim": (-1,0), # line amplitude parameter limits; default (0,)
#     "disp_plim": (0,10), # line dispersion parameter limits; default (0,)
#     "voff_plim": (-2500,2500), # line velocity offset parameter limits; default (0,)
    "line_profile": "gaussian", # line profile shape*
    "n_moments": 4, # number of higher order Gauss-Hermite moments (if line profile is gauss-hermite, laplace, or uniform)        
}

# Choices for line profile shape include 'gaussian', 'lorentzian', 'voigt',
# 'gauss-hermite', 'laplace', and 'uniform'
################################################################################

########################### Emission Lines & Options ###########################
# If not specified, defaults to SDSS-QSO Emission Lines (http://classic.sdss.org/dr6/algorithms/linestable.html)
################################################################################
# User lines overrides the default line list with a user-input line list!
user_lines = {
    "NA_H_BETA"      :{"center":4862.691,"amp":"free","disp":"NA_OIII_5007_DISP","voff":"free","line_type":"na","label":r"H$\beta$","ncomp":1,},
    "NA_H_BETA_2"    :{"center":4862.691,"amp":"NA_H_BETA_AMP*NA_OIII_5007_2_AMP/NA_OIII_5007_AMP","disp":"NA_OIII_5007_2_DISP","voff":"NA_OIII_5007_2_VOFF","line_type":"na","ncomp":2,"parent":"NA_H_BETA"},

    "NA_OIII_4960"   :{"center":4960.295,"amp":"(NA_OIII_5007_AMP/2.98)","disp":"NA_OIII_5007_DISP","voff":"NA_OIII_5007_VOFF","line_type":"na","label":r"[O III]","ncomp":1,},
    "NA_OIII_4960_2" :{"center":4960.295,"amp":"(NA_OIII_5007_2_AMP/2.98)","disp":"NA_OIII_5007_2_DISP","voff":"NA_OIII_5007_2_VOFF","line_type":"na","ncomp":2,"parent":"NA_OIII_4960"},

    "NA_OIII_5007"   :{"center":5008.240,"amp":"free","disp":"free","voff":"free","line_type":"na","label":r"[O III]","ncomp":1,},
    "NA_OIII_5007_2" :{"center":5008.240,"amp":"free","disp":"free","voff":"free","line_type":"na","ncomp":2,"parent":"NA_OIII_5007"},

    "BR_H_BETA"      :{"center":4862.691,"amp":"free","disp":"free","voff":"free","line_type":"br","ncomp":1,},
}
user_constraints = [
    ("NA_OIII_5007_AMP","NA_OIII_5007_2_AMP"),
    ("NA_OIII_5007_2_DISP","NA_OIII_5007_DISP"),
]
# User defined masked regions (list of tuples)
user_mask = [
#     (4840,5015),
]

# Combined lines; define a composite line and calculate
# its combined parameters.  These are automatically
# generated for lines with multiple components (parent+child lines)
combined_lines = {
    "H_BETA_COMP"   :["NA_H_BETA","BR_H_BETA"],
}
########################## LOSVD Fitting & Options #############################
# For direct fitting of the stellar kinematics (stellar LOSVD), one can 
# specify a stellar template library (Indo-US or Vazdekis 2010).
# One can also hold velocity or dispersion constant to avoid template
# convolution during the fitting process.
################################################################################

losvd_options = {
"library"   : "IndoUS", # Options: IndoUS, Vazdekis2010
"vel_const" :  {"bool":False, "val":0.0},
"disp_const":  {"bool":False, "val":250.0},
}

########################## SSP Host Galaxy Template & Options ##################
# The default is zero velocity, 100 km/s dispersion 10 Gyr template from 
# the eMILES stellar library. 
################################################################################

host_options = {
"age"       : [1.0,5.0,10.0], # Gyr; [0.09 Gyr - 14 Gyr] 
"vel_const" : {"bool":False, "val":0.0},
"disp_const": {"bool":False, "val":150.0}
}

########################### AGN power-law continuum & Options ##################
# The default is a simple power law.
################################################################################

power_options = {
"type" : "simple" # alternatively, "broken" for smoothly-broken power-law
}

########################### Polynomial Continuum Options #######################
# Options for an additive legendre polynomial or multiplicative polynomial to be 
# included in the fit.  NOTE: these polynomials do not include the zeroth-order
# (constant) term to avoid degeneracies with other continuum components.
################################################################################

poly_options = {
"apoly" : {"bool": True , "order": 3}, # Legendre additive polynomial 
"mpoly" : {"bool": False, "order": 3}, # Legendre multiplicative polynomial 
}

############################### Optical FeII options ###############################
# Below are options for fitting optical FeII.  For most objects, you don't need to 
# perform detailed fitting on FeII (only fit for amplitudes) use the 
# Veron-Cetty 2004 template ('VC04') (2-6 free parameters)
# However in NLS1 objects, FeII is much stronger, and sometimes more detailed 
# fitting is necessary, use the Kovacevic 2010 template 
# ('K10'; 7 free parameters).

# The options are:
# template   : VC04 (Veron-Cetty 2004) or K10 (Kovacevic 2010)
# amp_const  : constant amplitude (default False)
# disp_const : constant dispersion (default True)
# voff_const : constant velocity offset (default True)
# temp_const : constant temp ('K10' only)

opt_feii_options={
"opt_template"  :{"type":"VC04"}, 
"opt_amp_const" :{"bool":False, "br_opt_feii_val":1.0   , "na_opt_feii_val":1.0},
"opt_disp_const":{"bool":False, "br_opt_feii_val":3000.0, "na_opt_feii_val":500.0},
"opt_voff_const":{"bool":False, "br_opt_feii_val":0.0   , "na_opt_feii_val":0.0},
}
# or
# opt_feii_options={
# "opt_template"  :{"type":"K10"},
# "opt_amp_const" :{"bool":False,"f_feii_val":1.0,"s_feii_val":1.0,"g_feii_val":1.0,"z_feii_val":1.0},
# "opt_disp_const":{"bool":False,"opt_feii_val":1500.0},
# "opt_voff_const":{"bool":False,"opt_feii_val":0.0},
# "opt_temp_const":{"bool":True,"opt_feii_val":10000.0},
# }
################################################################################

############################### UV Iron options ################################
uv_iron_options={
"uv_amp_const"  :{"bool":False, "uv_iron_val":1.0},
"uv_disp_const" :{"bool":False, "uv_iron_val":3000.0},
"uv_voff_const" :{"bool":True,  "uv_iron_val":0.0},
}
################################################################################

########################### Balmer Continuum options ###########################
# For most purposes, only the ratio R, and the overall amplitude are free paramters
# but if you want to go crazy, you can fit everything.
balmer_options = {
"R_const"          :{"bool":True,  "R_val":1.0}, # ratio between balmer continuum and higher-order balmer lines
"balmer_amp_const" :{"bool":False, "balmer_amp_val":1.0}, # amplitude of overall balmer model (continuum + higher-order lines)
"balmer_disp_const":{"bool":True,  "balmer_disp_val":5000.0}, # broadening of higher-order Balmer lines
"balmer_voff_const":{"bool":True,  "balmer_voff_val":0.0}, # velocity offset of higher-order Balmer lines
"Teff_const"       :{"bool":True,  "Teff_val":15000.0}, # effective temperature
"tau_const"        :{"bool":True,  "tau_val":1.0}, # optical depth
}

################################################################################

############################### Plotting options ###############################
plot_options={
"plot_param_hist"    : False,# Plot MCMC histograms and chains for each parameter
"plot_HTML"          : True,# make interactive plotly HTML best-fit plot
}
################################################################################

################################ Output options ################################
output_options={
"res_correct"  : True,  # Correct final emission line widths for the intrinsic 
                        # resolution of the spectrum; 
                        # WARNING: make sure your resolution is well-determined!
"write_chain"  : False, # Write MCMC chains for all paramters, fluxes, and
                         # luminosities to a FITS table We set this to false 
                         # because MCMC_chains.FITS file can become very large, 
                         # especially  if you are running multiple objects.  
                         # You only need this if you want to reconstruct full chains 
                         # and histograms. 
"write_options": False,  # output restart file
"verbose"      : True,   # print out all steps of fitting process
}
################################################################################
