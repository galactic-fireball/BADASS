```python
balmer_options = {
  "R_const" :{"bool":True,  "R_val":1.0}, # ratio between balmer continuum and higher-order lines
  "balmer_amp_const" :{"bool":False, "balmer_amp_val":1.0}, # hold amplitude constant?
  "balmer_disp_const" :{"bool":True,  "balmer_disp_val":5000.0}, # hold dispersion constant?
  "balmer_voff_const" :{"bool":True,  "balmer_voff_val":0.0}, # hold velocity constant?
  "Teff_const" :{"bool":True,  "Teff_val":15000.0}, # effective temperature
  "tau_const" :{"bool":True,  "tau_val":1.0}, # optical depth
}
```