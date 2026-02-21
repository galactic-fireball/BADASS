io = {
    'infmt': 'sdss',
    'output_dir': 'sdss_test', # same directory as input file
    'overwrite': True,
    'log_level': 'debug',
}

fit = {
    'fit_reg': (4400,5500),
    'n_basinhop': 15,
}

mcmc = {
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

comp = {
    # 'fit_losvd': False,
    # 'fit_host': True,
    # 'fit_poly': True,
    'tie_line_voff': True,
    'tie_line_disp': True,
}


from badass.components.spectral_lines.line_lists import type1agn_default1
user_lines = type1agn_default1.user_lines

# from badass.components.spectral_lines.line_lists import common_lines

# hbeta = common_lines.NA_H_BETA.copy()
# hbeta.pop('disp')

# user_lines = [hbeta, common_lines.BR_H_BETA]
