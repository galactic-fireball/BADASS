io_options = {
    'infmt' : 'muse',
    'output_dir' : '/Users/sara/Dropbox/research/bgc/projects/spectra_fitting/badass_mini_class/BADASS/example_spectra/MUSE/muse_test', # same directory as input file
    'product_name': 'NGC1068',
    'overwrite' : True,
    'log_level' : 'debug',
}
fit_options = {
    'fit_reg': (4800,5200),
    'redshift': 0.00348372,
    'fit_area': {
        # 'bins': {'side_length':3},
        # 'bins': {'side_length':3, 'x': (10,40), 'y':(10,35), 'method': 'mean'},
        # 'spaxels': {'x': (20,30), 'y': (15,30)},
        'spaxels': [(20,30),(24,33),(30,39)],
        # 'spaxels': 'all',
        # 'plot_input': True,
    },
    'n_basinhop': 15, # Number of consecutive basinhopping thresholds before solution achieved
}

mcmc_options = {
'mcmc_fit'    : False, # Perform robust fitting using emcee
'nwalkers'    : 100,  # Number of emcee walkers; min = 2 x N_parameters
'auto_stop'   : True, # Automatic stop using autocorrelation analysis
'conv_type'   : ('NA_OIII_5007_AMP','NA_OIII_5007_DISP'), # 'median', 'mean', 'all', or (tuple) of parameters
'min_samp'    : 1000,  # min number of iterations for sampling post-convergence
'ncor_times'  : 10,  # number of autocorrelation times for convergence
'autocorr_tol': 10.0,  # percent tolerance between checking autocorr. times
'burn_in'     : 250, # burn-in if max_iter is reached
# 'write_iter'  : 100,   # write/check autocorrelation times interval
# 'write_thresh': 100,   # iteration to start writing/checking parameters
# 'min_iter'    : 1500, # min number of iterations before stopping
# 'max_iter'    : 5000, # max number of MCMC iterations
'write_iter'  : 3,   # write/check autocorrelation times interval
'write_thresh': 3,   # iteration to start writing/checking parameters
'min_iter'    : 3, # min number of iterations before stopping
'max_iter'    : 10, # max number of MCMC iterations
}


from badass.components.spectral_lines.line_lists import type1agn_default1
user_lines = type1agn_default1.user_lines
