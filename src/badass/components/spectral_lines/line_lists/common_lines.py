NA_H_BETA = {
    "name": "NA_H_BETA",
    "center": 4862.691,
    "type": "narrow",
    "profile": "gaussian",
    "disp": "NA_OIII_5007_DISP",
}

NA_H_BETA_BLUE = {
    "name": "NA_H_BETA_BLUE",
    "center": 4862.691,
    "type": "narrow",
    "profile": "gaussian",
    "amp": "NA_H_BETA_AMP*NA_OIII_5007_BLUE_AMP/NA_OIII_5007_AMP",
    "disp": "NA_OIII_5007_BLUE_DISP",
    "voff": "NA_OIII_5007_BLUE_VOFF",
}

BR_H_BETA = {
    "name": "BR_H_BETA",
    "center": 4862.691,
    "type": "broad",
    "profile": "gaussian",
}

H_BETA = {
    "name": "H_BETA",
    "center": 4862.691,
    "type": "combined",
    "children": [NA_H_BETA, NA_H_BETA_BLUE, BR_H_BETA]
}

NA_OIII_5007 = {
    "name": "NA_OIII_5007",
    "center": 5008.240,
    "type": "narrow",
    "profile": "gaussian",
}

NA_OIII_5007_BLUE = {
    "name": "NA_OIII_5007_BLUE",
    "center": 5008.240,
    "type": "narrow",
    "profile": "gaussian",
}

BR_OIII_5007 = {
    "name": "BR_OIII_5007",
    "center": 5008.240,
    "type": "broad",
    "profile": "gaussian",
}

OIII_5007 = {
    "name": "OIII_5007",
    "center": 5008.240,
    "type": "combined",
    "children": [NA_OIII_5007, NA_OIII_5007_BLUE, BR_OIII_5007]
}
