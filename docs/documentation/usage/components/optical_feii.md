There are two FeII templates built into BADASS.  The default is the broad and narrow templates from [Véron-Cetty et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004A%26A...417..515V/abstract) (`VC04`).  This model allows the user to have amplitude, dispersion, and velocity offset as free-parameters, with options to constrain them to constant values during the fit.  BADASS can also use the temperature-dependent template from [Kovačević et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010MSAIS..15..176K/abstract) (`K10`), which allows for the fitting of individual F, S, G, and I Zw 1 atomic transitions, as well as temperature.  The K10 template is best suited for modeling FeII in NLS1 objects with strong FeII emission.

```python
############################### FeII Fit options ###############################
# Below are options for fitting FeII.  For most objects, you don't need to 
# perform detailed fitting on FeII (only fit for amplitudes) use the 
# Veron-Cetty 2004 template ('VC04') (2-6 free parameters)
# However in NLS1 objects, FeII is much stronger, and sometimes more detailed 
# fitting is necessary, use the Kovacevic 2010 template 
# ('K10'; 7 free parameters).

# The options are:
# template   : VC04 (Veron-Cetty 2004) or K10 (Kovacevic 2010)
# amp_const  : constant amplitude (default False)
# disp_const : constant disp (default True)
# voff_const : constant velocity offset (default True)
# temp_const : constant temp ('K10' only)

feii_options={
  'template'  :{'type':'VC04'}, 
  'amp_const' :{'bool':False,'br_feii_val':1.0,'na_feii_val':1.0},
  'disp_const':{'bool':True,'br_feii_val':3000.0,'na_feii_val':500.0},
  'voff_const':{'bool':True,'br_feii_val':0.0,'na_feii_val':0.0},
}
# or
# feii_options={
# 'template'  :{'type':'K10'},
# 'amp_const' :{'bool':False,'f_feii_val':1.0,'s_feii_val':1.0,'g_feii_val':1.0,'z_feii_val':1.0},
# 'disp_const':{'bool':False,'val':1500.0},
# 'voff_const':{'bool':False,'val':0.0},
# 'temp_const':{'bool':True,'val':10000.0} 
# }
################################################################################
```