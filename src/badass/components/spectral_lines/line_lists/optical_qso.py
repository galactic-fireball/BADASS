def optical_qso_default():
    narrow_lines = {
        ### Region 8 (< 2000 Å)
        'NA_LY_ALPHA': {'center': 1215.24, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na'},
        'NA_CIV_1549': {'center': 1549.48, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na'},
        'NA_CIII_1908': {'center': 1908.734, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na'},

        ### Region 7 (2000 Å - 3500 Å)
        'NA_MGII_2799': {'center': 2799.117, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': 'Mg II'},
        'NA_HEII_3203': {'center': 3203.1, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': 'He II'},
        'NA_NEV_3346': {'center': 3346.783, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': '[Ne V]'},
        'NA_NEV_3426': {'center': 3426.863, 'amp': 'free', 'disp': 'NA_NEV_3346_DISP', 'voff': 'NA_NEV_3346_VOFF', 'line_type': 'na', 'label': '[Ne V]'},

        ### Region 6 (3500 Å - 4400 Å)
        'NA_OII_3727': {'center': 3727.092, 'amp': 'free', 'disp': 'NA_OII_3729_DISP', 'voff': 'NA_OII_3729_VOFF', 'line_type': 'na', 'label': '[O II]'},
        'NA_OII_3729': {'center': 3729.875, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na'},
        'NA_NEIII_3869': {'center': 3869.857, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': '[Ne III]'},
        'NA_HEI_3889': {'center': 3888.647, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': 'He I'},
        'NA_NEIII_3968': {'center': 3968.593, 'amp': 'free', 'disp': 'NA_NEIII_3869_DISP', 'voff': 'NA_NEIII_3869_VOFF', 'line_type': 'na', 'label': '[Ne III]'},
        'NA_H_DELTA': {'center': 4102.9, 'amp': 'free', 'disp': 'NA_H_GAMMA_DISP', 'voff': 'NA_H_GAMMA_VOFF', 'line_type': 'na', 'label': 'H$\\delta$'},
        'NA_H_GAMMA': {'center': 4341.691, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': 'H$\\gamma$'},
        'NA_OIII_4364': {'center': 4364.436, 'amp': 'free', 'disp': 'NA_H_GAMMA_DISP', 'voff': 'NA_H_GAMMA_VOFF', 'line_type': 'na', 'label': '[O III]'},

        ### Region 5 (4400 Å - 5500 Å)
        'NA_HEII_4687': {'center': 4687.021, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': 'He II'},
        'NA_H_BETA': {'center': 4862.691, 'amp': 'free', 'disp': 'NA_OIII_5007_DISP', 'voff': 'free', 'h3': 'NA_OIII_5007_H3', 'h4': 'NA_OIII_5007_H4', 'line_type': 'na', 'label': 'H$\\beta$'},
        'NA_OIII_4960': {'center': 4960.295, 'amp': '(NA_OIII_5007_AMP/2.98)', 'disp': 'NA_OIII_5007_DISP', 'voff': 'NA_OIII_5007_VOFF', 'h3': 'NA_OIII_5007_H3', 'h4': 'NA_OIII_5007_H4', 'line_type': 'na', 'label': '[O III]'},
        'NA_OIII_5007': {'center': 5008.24, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'h3': 'free', 'h4': 'free', 'line_type': 'na', 'label': '[O III]'},

        ### Region 4 (5500 Å - 6200 Å)
        'NA_FEVI_5638': {'center': 5637.6, 'amp': 'free', 'disp': 'NA_FEVI_5677_DISP', 'voff': 'NA_FEVI_5677_VOFF', 'line_type': 'na', 'label': '[Fe VI]'},
        'NA_FEVI_5677': {'center': 5677.0, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': '[Fe VI]'},
        'NA_FEVII_5720': {'center': 5720.7, 'amp': 'free', 'disp': 'NA_FEVII_6087_DISP', 'voff': 'NA_FEVII_6087_VOFF', 'line_type': 'na', 'label': '[Fe VII]'},
        'NA_HEI_5876': {'center': 5875.624, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': 'He I'},
        'NA_FEVII_6087': {'center': 6087.0, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': '[Fe VII]'},

        ### Region 3 (6200 Å - 6800 Å)
        'NA_OI_6302': {'center': 6302.046, 'amp': 'free', 'disp': 'NA_NII_6585_DISP', 'voff': 'NA_NII_6585_VOFF', 'line_type': 'na', 'label': '[O I]'},
        'NA_SIII_6312': {'center': 6312.06, 'amp': 'free', 'disp': 'NA_NII_6585_DISP', 'voff': 'free', 'line_type': 'na', 'label': '[S III]'},
        'NA_OI_6365': {'center': 6365.535, 'amp': 'NA_OI_6302_AMP/3.0', 'disp': 'NA_NII_6585_DISP', 'voff': 'NA_NII_6585_VOFF', 'line_type': 'na', 'label': '[O I]'},
        'NA_FEX_6374': {'center': 6374.51, 'amp': 'free', 'disp': 'NA_NII_6585_DISP', 'voff': 'free', 'line_type': 'na', 'label': '[Fe X]'},
        'NA_NII_6549': {'center': 6549.859, 'amp': 'NA_NII_6585_AMP/2.93', 'disp': 'NA_NII_6585_DISP', 'voff': 'NA_NII_6585_VOFF', 'line_type': 'na', 'label': '[N II]'},
        'NA_H_ALPHA': {'center': 6564.632, 'amp': 'free', 'disp': 'NA_NII_6585_DISP', 'voff': 'NA_NII_6585_VOFF', 'line_type': 'na', 'label': 'H$\\alpha$'},
        'NA_NII_6585': {'center': 6585.278, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': '[N II]'},
        'NA_SII_6718': {'center': 6718.294, 'amp': 'free', 'disp': 'NA_NII_6585_DISP', 'voff': 'NA_NII_6585_VOFF', 'line_type': 'na', 'label': '[S II]'},
        'NA_SII_6732': {'center': 6732.668, 'amp': 'free', 'disp': 'NA_NII_6585_DISP', 'voff': 'NA_NII_6585_VOFF', 'line_type': 'na', 'label': '[S II]'},

        ### Region 2 (6800 Å - 8000 Å)
        'NA_HEI_7062': {'center': 7065.196, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': 'He I'},
        'NA_ARIII_7135': {'center': 7135.79, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': '[Ar III]'},
        'NA_OII_7319': {'center': 7319.99, 'amp': 'free', 'disp': 'NA_OII_7331_DISP', 'voff': 'NA_OII_7331_VOFF', 'line_type': 'na', 'label': '[O II]'},
        'NA_OII_7331': {'center': 7330.73, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': '[O II]'},
        'NA_NIIII_7890': {'center': 7889.9, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': '[Ni III]'},
        'NA_FEXI_7892': {'center': 7891.8, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': '[Fe XI]'},

        ### Region 1 (8000 Å - 9000 Å)
        'NA_HEII_8236': {'center': 8236.79, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': 'He II'},
        'NA_OI_8446': {'center': 8446.359, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': 'O I'},
        'NA_FEII_8616': {'center': 8616.95, 'amp': 'free', 'disp': 'NA_FEII_8891_DISP', 'voff': 'NA_FEII_8891_VOFF', 'line_type': 'na', 'label': '[Fe II]'},
        'NA_FEII_8891': {'center': 8891.91, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': '[Fe II]'},
    }

    broad_lines = {
        ### Region 8 (< 2000 Å)
        'BR_OVI_1034': {'center': 1033.82, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'O VI'},
        'BR_LY_ALPHA': {'center': 1215.24, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'Ly$\\alpha$'},
        'BR_NV_1241': {'center': 1240.81, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'N V'},
        'BR_OI_1305': {'center': 1305.53, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'O I'},
        'BR_CII_1335': {'center': 1335.31, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'C II'},
        'BR_SIIV_1398': {'center': 1397.61, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'Si IV + O IV'},
        'BR_SIIV+OIV': {'center': 1399.8, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'Si IV + O IV'},
        'BR_CIV_1549': {'center': 1549.48, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'C IV'},
        'BR_HEII_1640': {'center': 1640.4, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'He II'},
        'BR_CIII_1908': {'center': 1908.734, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'C III]'},

        ### Region 7 (2000 Å - 3500 Å)
        'BR_CII_2326': {'center': 2326.0, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_profile': 'gaussian', 'line_type': 'br', 'label': 'C II]'},
        'BR_FEIII_UV47': {'center': 2418.0, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_profile': 'gaussian', 'line_type': 'br', 'label': 'Fe III'},
        'BR_MGII_2799': {'center': 2799.117, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'Mg II'},

        ### Region 6 (3500 Å - 4400 Å):
        'BR_H_DELTA': {'center': 4102.9, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br'},
        'BR_H_GAMMA': {'center': 4341.691, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br'},

        ### Region 5 (4400 Å - 5500 Å)
        'BR_H_BETA': {'center': 4862.691, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br'},

        ### Region 3 (6200 Å - 6800 Å)
        'BR_H_ALPHA': {'center': 6564.632, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br'},
    }

    absorp_lines = {
        'ABS_NAI_5897': {'center': 5897.558, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'abs', 'label': 'Na D'}
    }


    line_list = {**narrow_lines, **broad_lines, **absorp_lines}
    return line_list
