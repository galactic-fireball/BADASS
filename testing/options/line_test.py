io = {
    'infmt': 'sdss',
    'output_dir': 'line_test',
    'overwrite': True,
    'log_level': 'debug',
}

fit = {
    'fit_reg': (4400,5500),
    'test_models': True,
}


from badass.components.spectral_lines.line_lists.common_lines import OIII_5007, NA_OIII_5007, NA_OIII_5007_BLUE, BR_OIII_5007, H_BETA

user_lines = [H_BETA]


test_sets = []

oiii_5007 = OIII_5007.copy()
oiii_5007['children'] = [NA_OIII_5007,]
test_sets.append([oiii_5007,])

# oiii_5007 = OIII_5007.copy()
# oiii_5007['children'] = [NA_OIII_5007, NA_OIII_5007_BLUE]
# test_sets.append([oiii_5007,])

# oiii_5007 = OIII_5007.copy()
# oiii_5007['children'] = [NA_OIII_5007, NA_OIII_5007_BLUE, BR_OIII_5007]
# test_sets.append([oiii_5007,])


test = {
    'mode': 'line',
    'test_sets': test_sets,
    'continue_fit': False,
}


user_constraints = [
    ('NA_OIII_5007_AMP','NA_OIII_5007_BLUE_AMP'),
    ('NA_OIII_5007_BLUE_DISP','NA_OIII_5007_DISP'),
    ('NA_OIII_5007_VOFF','NA_OIII_5007_BLUE_VOFF'),
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
