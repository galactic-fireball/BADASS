io = {
    'infmt': 'sdss',
    'output_dir': 'sdss_test',
    'overwrite': True,
    'log_level': 'debug',
    'nprocesses': 1,
}

fit = {
    'fit_reg': (4400,5500),
    'max_like_niter': 1,
    'n_basinhop': 1,
}

mcmc = {
    'mcmc_fit': True,
    'burn_in': 0,
    'min_iter': 5,
    'max_iter': 20,
    'write_iter': 5,
    'write_thresh': 5,
}

comp = {
    'fit_losvd': False,
    # 'fit_host': True,
    'fit_feii': False,
    # 'fit_poly': True,
    # 'tie_line_voff': True,
    # 'tie_line_disp': True,
}


# user_mask = [
#     (4981,5056),
# ]


from badass.components.spectral_lines.line_lists import type1agn_default1
user_lines = type1agn_default1.user_lines

from badass.components.spectral_lines.line_lists import common_lines, coronal_lines
from badass.common_lines import *
from badass.coronal_lines import *



# hbeta = BR_H_BETA
# hbeta['name'] = 'H_BETA'
H_BETA_LAM = 4862.691
hbeta = {'name': 'H_BETA', 'center': H_BETA_LAM, 'type': 'combined', 'children': [NA_H_BETA, BR_H_BETA,],}
user_lines = [hbeta,]#, OIII_5007]


# br_hbeta = common_lines.BR_H_BETA.copy()
# br_hbeta['profile'] = 'gauss-hermite'
# br_hbeta['profile'] = 'uniform'
# br_hbeta['n_moments'] = 5

# na_hbeta = common_lines.NA_H_BETA.copy()
# na_hbeta.pop('disp')

# user_lines = [na_hbeta, br_hbeta,]
# user_lines = [
#     BR_CII_2326,
#     BR_FeIII_UV47,
#     MgV_2782,
#     MgII_2799,
#     MgV_2928,
#     BR_FeII_3100,
#     NA_OIII_3133,
#     BR_FeII_3200,
#     NA_HeII_3203,
#     NeV_3426,
#     NeV_3346,
# ]
