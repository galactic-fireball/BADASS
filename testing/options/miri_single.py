io = {
    'infmt': 'miri',
    'output_dir': 'miri_test', # same directory as input file
    'overwrite': True,
    'plots': {
        'style': 'light',
    }
}

fit = {
    'fit_reg': (50000,57000),
    'redshift': 0.002336,
    'fit_area': {'type':'aperture','apertures': [
        {'shape':'circular', 'center':(30,27), 'radius':8},
        # {'shape':'rectangular', 'center':(25,25), 'width':5}
    ],
    'plot_input':False,},
}

mcmc = {
    'mcmc_fit': False,
    'burn_in': 2,
    'min_iter': 3,
    'max_iter': 5,
    'write_iter': 1,
    'write_thresh': 1,
    'auto_stop': True,
}


comp = {
    'fit_losvd': False,
    # 'fit_host': True,
    'fit_feii': False,
    'fit_power': True,
}

user_lines = {
    "FEAT1": {"center":54466.91,"line_type":"user","line_profile":"gaussian","ncomp":1,"line_type":"na"},
    "FEAT2": {"center":55029.82,"line_type":"user","line_profile":"gaussian","ncomp":1,"line_type":"na"},
    "FEAT3": {"center":56090.69,"line_type":"user","line_profile":"gaussian","ncomp":1,"line_type":"na"},
}

