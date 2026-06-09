io = {
    'infmt' : 'muse',
    'output_dir' : 'muse_test',
    'product_name': 'NGC1068',
    'overwrite' : True,
    'log_level' : 'debug',
}

fit = {
    'fit_reg': (4800,5200),
    'redshift': 0.00348372,
    'fit_area': {
        'type': 'spaxels',
        'spaxels': [(20,30),(24,33),(30,39)],
        # 'bins': {'side_length':3},
        # 'bins': {'side_length':3, 'x': (10,40), 'y':(10,35), 'method': 'mean'},
        # 'spaxels': {'x': (20,30), 'y': (15,30)},
        # 'spaxels': [(20,30),(24,33),(30,39)],
        # 'spaxels': 'all',
        # 'plot_input': True,
    },
    'n_basinhop': 15, # Number of consecutive basinhopping thresholds before solution achieved
}


from badass.components.spectral_lines.line_lists import type1agn_default1
user_lines = type1agn_default1.user_lines
