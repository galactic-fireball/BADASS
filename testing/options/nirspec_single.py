io = {
    'infmt': 'nirspec',
    'output_dir': 'nirspec_test', # same directory as input file
    'overwrite': True,
    'log_level': 'info',
    'filter': '290',
    'disperser': 'h',
}

fit = {
    'fit_reg': (36400,40000),
    'redshift': 0.002336,
    'fit_area': {'type':'spaxels','spaxel': [(30,27),],},
}


comp = {
    'fit_losvd': False,
    # 'fit_host': True,
    'fit_feii': False,
}

# user_lines = {
#     'FEAT1': {'center':36584.56,'line_type':'user','line_profile':'gaussian','ncomp':1,'line_type':'na'},
#     'FEAT2': {'center':37400.61,'line_type':'user','line_profile':'gaussian','ncomp':1,'line_type':'na'},
#     'FEAT3': {'center':38077.33,'line_type':'user','line_profile':'gaussian','ncomp':1,'line_type':'na'},
#     'FEAT4': {'center':38462.13,'line_type':'user','line_profile':'gaussian','ncomp':1,'line_type':'na'},
#     'FEAT5': {'center':39337.88,'line_type':'user','line_profile':'gaussian','ncomp':1,'line_type':'na'},
# }


user_lines = [
    {'FEAT5': 'FEAT1', 'center': 39337.88}
]
