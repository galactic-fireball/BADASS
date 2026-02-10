io = {
    'infmt': 'sdss',
    'output_dir': 'line_test',
    'overwrite': True,
    'log_level': 'debug',
}

fit = {
    'fit_reg': (4400,5500),
    'test_lines': True,
}

comp = {
    'fit_losvd': False,
    'fit_host': True,
    'fit_poly': True,
}

narrow = {
    'disp_plim': (0,500),
}

user_lines = {
    'H_BETA': {'center':4862.691, 'disp':'OIII_5007_DISP','line_type':'na','label':r'H$\beta$','ncomp':1,},
    'H_BETA_2': {'center':4862.691,'amp':'H_BETA_AMP*OIII_5007_2_AMP/OIII_5007_AMP','disp':'OIII_5007_2_DISP','voff':'OIII_5007_2_VOFF','line_type':'na','ncomp':2,'parent':'H_BETA'},

    'OIII_4960': {'center':4960.295,'amp':'(OIII_5007_AMP/2.98)','disp':'OIII_5007_DISP','voff':'OIII_5007_VOFF','line_type':'na','label':r'[O III]','ncomp':1,},
    'OIII_4960_2': {'center':4960.295,'amp':'(OIII_5007_2_AMP/2.98)','disp':'OIII_5007_2_DISP','voff':'OIII_5007_2_VOFF','line_type':'na','ncomp':2,'parent':'OIII_4960'},

    'OIII_5007': {'center':5008.240,'line_type':'na','label':r'[O III]','ncomp':1,},
    'OIII_5007_2': {'center':5008.240,'line_type':'na','ncomp':2,'parent':'OIII_5007'},

    'BR_H_BETA': {'center':4862.691,'line_type':'br','ncomp':1,},
}

test_options = {
    'test_mode':'line',
    'lines': [['OIII_5007','OIII_4960','H_BETA']], # The lines to test
    'metrics': {'BADASS':0.95, 'ANOVA':0.95, 'CHI2_RATIO':0.10, 'AON':3.0}, # Fitting metrics to use when determining the best model
    'conv_mode': 'all', # 'any' single threshold satisfies the solution, or 'all' must satisfy thresholds
    'auto_stop':False, # automatically stop testing once threshold is reached; False test all no matter what
    'full_verbose':True, # prints out all test fitting to screen
    'plot_tests':True, # plot the fit of each model comparison
    'force_best':True, # this forces the more-complex model to have a fit better than the previous.
    'continue_fit':True, # continue the fit with the best chosen model
}

user_constraints = [
    ('OIII_5007_AMP','OIII_5007_2_AMP'),
    ('OIII_5007_2_DISP','OIII_5007_DISP'),
]

combined_lines = {
    'H_BETA_COMP': ['NA_H_BETA','BR_H_BETA'],
}

poly = {
    'apoly_order': 3, # Legendre additive polynomial 
}

plot = {
    'param_hist': False,
}
