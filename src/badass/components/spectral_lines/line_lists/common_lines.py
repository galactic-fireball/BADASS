H_BETA_LAM = 4862.691
NA_H_BETA = {"name": "NA_H_BETA", "center": H_BETA_LAM, "disp": "NA_OIII_5007_DISP",}
NA_H_BETA_2 = {"name": "NA_H_BETA_2", "center": H_BETA_LAM, "amp": "NA_H_BETA_AMP*NA_OIII_5007_2_AMP/NA_OIII_5007_AMP", "disp": "NA_OIII_5007_2_DISP", "voff": "NA_OIII_5007_2_VOFF",}
BR_H_BETA = {"name": "BR_H_BETA", "center": H_BETA_LAM, "type": "broad",}
H_BETA = {"name": "H_BETA", "center": H_BETA_LAM, "type": "combined", "children": [NA_H_BETA, NA_H_BETA_2, BR_H_BETA],}

OIII_5007_LAM = 5008.240
NA_OIII_5007 = {"name": "NA_OIII_5007", "center": OIII_5007_LAM,}
NA_OIII_5007_2 = {"name": "NA_OIII_5007_2", "center": OIII_5007_LAM,}
BR_OIII_5007 = {"name": "BR_OIII_5007", "center": OIII_5007_LAM, "type": "broad",}
OIII_5007 = {"name": "OIII_5007", "center": OIII_5007_LAM, "type": "combined", "children": [NA_OIII_5007, NA_OIII_5007_2, BR_OIII_5007],}

