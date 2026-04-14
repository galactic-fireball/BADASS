# TODO: make SpectralLine

# Region 8 (< 2000 Å)

BR_OVI_1034 = {'name': 'BR_OVI_1034', 'center': 1033.820, 'type': 'broad',}

LY_ALPHA_LAM = 1215.240
NA_LY_ALPHA = {'name': 'NA_LY_ALPHA', 'center': LY_ALPHA_LAM,}
BR_LY_ALPHA = {'name': 'BR_LY_ALPHA', 'center': LY_ALPHA_LAM, 'type': 'broad',}
LY_ALPHA = {'name': 'LY_ALPHA', 'center': LY_ALPHA_LAM, 'type': 'combined', 'children': [NA_LY_ALPHA, BR_LY_ALPHA,],}

BR_NV_1241 = {'name': 'BR_NV_1241', 'center': 1240.810, 'type': 'broad',}
BR_OI_1305 = {'name': 'BR_OI_1305', 'center': 1305.530, 'type': 'broad',}
BR_CII_1335 = {'name': 'BR_CII_1335', 'center': 1335.310, 'type': 'broad',}
BR_SiIV_1398 = {'name': 'BR_SiIV_1398', 'center': 1397.610, 'type': 'broad',}
BR_SiIV_OIV = {'name': 'BR_SiIV+OIV', 'center': 1399.800, 'type': 'broad',}

CIV_1549_LAM = 1549.480
NA_CIV_1549 = {'name': 'NA_CIV_1549', 'center': CIV_1549_LAM,}
BR_CIV_1549 = {'name': 'BR_CIV_1549', 'center': CIV_1549_LAM, 'type': 'broad',}
CIV_1549 = {'name': 'CIV_1549', 'center': CIV_1549_LAM, 'type': 'combined', 'children': [NA_CIV_1549, BR_CIV_1549,],}

BR_HeII_1640 = {'name': 'BR_HeII_1640', 'center': 1640.400, 'type': 'broad',}

CIII_1908_LAM = 1908.734
NA_CIII_1908 = {'name': 'NA_CIII_1908', 'center': CIII_1908_LAM,}
BR_CIII_1908 = {'name': 'BR_CIII_1908', 'center': CIII_1908_LAM, 'type': 'broad',}
CIII_1908 = {'name': 'CIII_1908', 'center': CIII_1908_LAM, 'type': 'combined', 'children': [NA_CIII_1908, BR_CIII_1908,],}


# Region 7 (2000 Å - 3500 Å)

BR_CII_2326 = {'name': 'BR_CII_2326', 'center': 2326.000, 'type': 'broad',}
BR_FeIII_UV47 = {'name': 'BR_FeIII_UV47', 'center': 2418.000, 'type': 'broad',}

MgII_2799_LAM = 2799.117
NA_MgII_2799 = {'name': 'NA_MgII_2799', 'center': MgII_2799_LAM,}
BR_MgII_2799 = {'name': 'BR_MgII_2799', 'center': MgII_2799_LAM, 'type': 'broad',}
MgII_2799 = {'name': 'MgII_2799', 'center': MgII_2799_LAM, 'type': 'combined', 'children': [NA_MgII_2799, BR_MgII_2799,],}

BR_FeII_3100 = {'name': 'BR_FeII_3100', 'center': 3100.000, 'type': 'broad',}
NA_OIII_3133 = {'name': 'NA_OIII_3133', 'center': 3132.794,}
BR_FeII_3200 = {'name': 'BR_FeII_3200', 'center': 3200.000, 'type': 'broad',}
NA_HeII_3203 = {'name': 'NA_HeII_3203', 'center': 3203.100,}


# Region 6 (3500 Å - 4400 Å)

NA_OII_3727 = {'name': 'NA_OII_3727', 'center': 3727.092, 'disp': 'NA_OII_3729_DISP', 'voff': 'NA_OII_3729_VOFF',}
NA_OII_3729 = {'name': 'NA_OII_3729', 'center': 3729.875,}

NeIII_3869_LAM = 3869.857
NA_NeIII_3869 = {'name': 'NA_NeIII_3869', 'center': NeIII_3869_LAM,}
NA_NeIII_3869_2 = {'name': 'NA_NeIII_3869_2', 'center': NeIII_3869_LAM}
NeIII_3869 = {'name': 'NeIII_3869', 'center': NeIII_3869_LAM, 'children': [NA_NeIII_3869, NA_NeIII_3869_2,],}

NA_HeI_3889 = {'name': 'NA_HeI_3889', 'center': 3888.647}

NeIII_3968_LAM = 3968.593
NA_NeIII_3968 = {'name': 'NA_NeIII_3968', 'center': NeIII_3968_LAM, 'disp': 'NA_NeIII_3869_DISP', 'voff': 'NA_NeIII_3869_VOFF',}
NA_NeIII_3968_2 = {'name': 'NA_NeIII_3968_2', 'center': NeIII_3968_LAM, 'amp': 'NA_NeIII_3869_2_AMP/NA_NeIII_3869_AMP*NA_NeIII_3968_AMP',}
NeIII_3968 = {'name': 'NeIII_3968', 'center': NeIII_3968_LAM, 'children': [NA_NeIII_3968, NA_NeIII_3968_2,],}

H_DELTA_LAM = 4102.900
NA_H_DELTA = {'name': 'NA_H_DELTA', 'center': H_DELTA_LAM, 'disp': 'NA_H_GAMMA_DISP', 'voff': 'NA_H_GAMMA_VOFF',}
BR_H_DELTA = {'name': 'BR_H_DELTA', 'center': H_DELTA_LAM, 'type': 'broad',}
H_DELTA = {'name': 'H_DELTA', 'center': H_DELTA_LAM, 'type': 'combined', 'children': [NA_H_DELTA, BR_H_DELTA,],}

H_GAMMA_LAM = 4341.691
NA_H_GAMMA = {'name': 'NA_H_GAMMA', 'center': H_GAMMA_LAM,}
BR_H_GAMMA = {'name': 'BR_H_GAMMA', 'center': H_GAMMA_LAM, 'type': 'broad',}
H_GAMMA = {'name': 'H_GAMMA', 'center': H_GAMMA_LAM, 'type': 'combined', 'children': [NA_H_GAMMA, BR_H_GAMMA,],}

NA_OIII_4364 = {'name': 'NA_OIII_4364', 'center': 4364.436, 'disp': 'NA_H_GAMMA_DISP', 'voff': 'NA_H_GAMMA_VOFF',}


# Region 5 (4400 Å - 5500 Å)

NA_HeI_4471 = {'name': 'NA_HeI_4471', 'center': 4471.479,}
NA_HeII_4687 = {'name': 'NA_HeII_4687', 'center': 4687.021,}

H_BETA_LAM = 4862.691
NA_H_BETA = {'name': 'NA_H_BETA', 'center': H_BETA_LAM, 'disp': 'NA_OIII_5007_DISP',}
NA_H_BETA_2 = {'name': 'NA_H_BETA_2', 'center': H_BETA_LAM, 'amp': 'NA_H_BETA_AMP*NA_OIII_5007_2_AMP/NA_OIII_5007_AMP', 'disp': 'NA_OIII_5007_2_DISP', 'voff': 'NA_OIII_5007_2_VOFF',}
BR_H_BETA = {'name': 'BR_H_BETA', 'center': H_BETA_LAM, 'type': 'broad',}
H_BETA = {'name': 'H_BETA', 'center': H_BETA_LAM, 'type': 'combined', 'children': [NA_H_BETA, NA_H_BETA_2, BR_H_BETA,],}

OIII_4960_LAM = 4960.295
NA_OIII_4960 = {'name': 'NA_OIII_4960', 'center': OIII_4960_LAM, 'amp': '(NA_OIII_5007_AMP/2.98)', 'disp': 'NA_OIII_5007_DISP', 'voff': 'NA_OIII_5007_VOFF', 'h3': 'NA_OIII_5007_H3', 'h4': 'NA_OIII_5007_H4',}
NA_OIII_4960_2 = {'name': 'NA_OIII_4960_2', 'center': OIII_4960_LAM, 'amp': 'NA_OIII_5007_2_amp/NA_OIII_5007_amp*NA_OIII_5007_amp/2.98',}
OIII_4960 = {'name': 'OIII_4960', 'center': OIII_4960_LAM, 'type': 'combined', 'children': [NA_OIII_4960, NA_OIII_4960_2,],}

OIII_5007_LAM = 5008.240
NA_OIII_5007 = {'name': 'NA_OIII_5007', 'center': OIII_5007_LAM,}
NA_OIII_5007_2 = {'name': 'NA_OIII_5007_2', 'center': OIII_5007_LAM,}
BR_OIII_5007 = {'name': 'BR_OIII_5007', 'center': OIII_5007_LAM, 'type': 'broad',}
OIII_5007 = {'name': 'OIII_5007', 'center': OIII_5007_LAM, 'type': 'combined', 'children': [NA_OIII_5007, NA_OIII_5007_2, BR_OIII_5007,],}


# Region 4 (5500 Å - 6200 Å)

NA_HeI_5876 = {'name': 'NA_HeI_5876', 'center': 5875.624,}
ABS_NaI_5897 = {'name': 'ABS_NaI_5897', 'center': 5897.558, 'type': 'absorp',}


# Region 3 (6200 Å - 6800 Å)

NA_OI_6302 = {'name': 'NA_OI_6302', 'center': 6302.046, 'disp': 'NA_NII_6585_DISP', 'voff': 'NA_NII_6585_VOFF',}
NA_SIII_6312 = {'name': 'NA_SIII_6312', 'center': 6312.060, 'disp': 'NA_NII_6585_DISP',}
NA_OI_6365 = {'name': 'NA_OI_6365', 'center': 6365.535, 'amp': 'NA_OI_6302_AMP/3.0', 'disp': 'NA_NII_6585_DISP', 'voff': 'NA_NII_6585_VOFF',}

NII_6549_LAM = 6549.859
NA_NII_6549 = {'name': 'NA_NII_6549', 'center': NII_6549_LAM, 'amp': 'NA_NII_6585_AMP/2.93', 'disp': 'NA_NII_6585_DISP', 'voff': 'NA_NII_6585_VOFF',}
NA_NII_6549_2 = {'name': 'NA_NII_6549_2', 'center': NII_6549_LAM, 'amp': 'NA_NII_6585_2_AMP/NA_NII_6585_AMP*NA_NII_6585_AMP/2.93',}
NII_6549 = {'name': 'NII_6549', 'center': NII_6549_LAM, 'type': 'combined', 'children': [NA_NII_6549, NA_NII_6549_2,],}

H_ALPHA_LAM = 6564.632
NA_H_ALPHA = {'name': 'NA_H_ALPHA', 'center': H_ALPHA_LAM, 'disp': 'NA_NII_6585_DISP', 'voff': 'NA_NII_6585_VOFF',}
BR_H_ALPHA = {'name': 'BR_H_ALPHA', 'center': H_ALPHA_LAM, 'type': 'broad',}
H_ALPHA = {'name': 'H_ALPHA', 'center': H_ALPHA_LAM, 'type': 'combined', 'children': [NA_H_ALPHA, BR_H_ALPHA,],}

NII_6585_LAM = 6585.278
NA_NII_6585 = {'name': 'NA_NII_6585', 'center': NII_6585_LAM,}
NA_NII_6585_2 = {'name': 'NA_NII_6585_2', 'center': NII_6585_LAM,}
NII_6585 = {'name': 'NII_6585', 'center': NII_6585_LAM, 'type': 'combined', 'children': [NA_NII_6585, NA_NII_6585_2,],}

SII_6718_LAM = 6718.294
NA_SII_6718 = {'name': 'NA_SII_6718', 'center': SII_6718_LAM, 'disp': 'NA_NII_6585_DISP', 'voff': 'NA_NII_6585_VOFF',}
NA_SII_6718_2 = {'name': 'NA_SII_6718_2', 'center': SII_6718_LAM, 'amp': 'NA_NII_6585_2_AMP/NA_NII_6585_AMP*NA_NII_6585_AMP',}
SII_6718 = {'name': 'SII_6718', 'center': SII_6718_LAM, 'type': 'combined', 'children': [NA_SII_6718, NA_SII_6718_2,],}

SII_6732_LAM = 6732.668
NA_SII_6732 = {'name': 'NA_SII_6732', 'center': SII_6732_LAM, 'disp': 'NA_NII_6585_DISP', 'voff': 'NA_NII_6585_VOFF',}
NA_SII_6732_2 = {'name': 'NA_SII_6732_2', 'center':SII_6732_LAM, 'amp': 'NA_NII_6585_2_AMP/NA_NII_6585_AMP*NA_NII_6585_AMP',}
SII_6732 = {'name': 'SII_6732', 'center': SII_6732_LAM, 'type': 'combined', 'children': [NA_SII_6732, NA_SII_6732_2,],}


# Region 2 (6800 Å - 8000 Å)

NA_HeI_7062 = {'name': 'NA_HeI_7062', 'center': 7065.196,}
NA_ArIII_7135 = {'name': 'NA_ArIII_7135', 'center': 7135.790,}
NA_OII_7319 = {'name': 'NA_OII_7319', 'center': 7319.990, 'disp': 'NA_OII_7331_DISP', 'voff': 'NA_OII_7331_VOFF',}
NA_OII_7331 = {'name': 'NA_OII_7331', 'center': 7330.730,}
NA_NiIII_7890 = {'name': 'NA_NiIII_7890', 'center': 7889.900,}


# Region 1 (8000 Å - 9000 Å)

NA_HeII_8236 = {'name': 'NA_HeII_8236', 'center': 8236.790,}
NA_OI_8446 = {'name': 'NA_OI_8446', 'center': 8446.359,}
NA_FeII_8616 = {'name': 'NA_FeII_8616', 'center': 8616.950, 'disp': 'NA_FeII_8891_DISP', 'voff': 'NA_FeII_8891_VOFF',}
NA_FeII_8891 = {'name': 'NA_FeII_8891', 'center': 8891.910,}
NA_SIII_9069 = {'name': 'NA_SIII_9069', 'center': 9068.600,}

__all__ = [key for key, val in globals().items() if not key.startswith('_') and isinstance(val,dict)]
