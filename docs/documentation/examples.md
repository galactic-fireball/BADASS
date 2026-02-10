## Under Construction

## Quick Start

```python
import badass

spec_file # a string path or a pathlib.Path
options # the path (string or pathlib) to a .py or .json file; or a python dict

badass.run_BADASS(spec_file, options=options)
```

## SDSS Sample

```python
BA_OPTIONS = {
	'io_options': {
		'infmt': 'sdss',
		'output_dir': cwd.joinpath('sample_test_out'),
		'log_level': 'debug',
	},
	'fit_options': {
		'fit_reg': (4400, 5500),
		'n_basinhop': 15,
		'max_line_niter': 10,
	},
	'comp_options': {
		'fit_opt_feii': True,
		'fit_uv_iron': True,
		'fit_balmer': False,
		'fit_losvd': True,
		'fit_host': True,
		'fit_power': True,
		'fit_poly': True,
	},
	'user_lines': {
		'NA_H_BETA': {'center':4862.691, 'amp':'free', 'disp':'NA_OIII_5007_DISP', 'voff':'free', 'line_type':'na', 'label':r'H$\beta$', 'ncomp':1,},
		'NA_H_BETA_2': {'center':4862.691,'amp':'NA_H_BETA_AMP*NA_OIII_5007_2_AMP/NA_OIII_5007_AMP','disp':'NA_OIII_5007_2_DISP','voff':'NA_OIII_5007_2_VOFF','line_type':'na','ncomp':2,'parent':'NA_H_BETA'},

		'NA_OIII_4960': {'center':4960.295,'amp':'(NA_OIII_5007_AMP/2.98)','disp':'NA_OIII_5007_DISP','voff':'NA_OIII_5007_VOFF','line_type':'na','label':r'[O III]','ncomp':1,},
		'NA_OIII_4960_2': {'center':4960.295,'amp':'(NA_OIII_5007_2_AMP/2.98)','disp':'NA_OIII_5007_2_DISP','voff':'NA_OIII_5007_2_VOFF','line_type':'na','ncomp':2,'parent':'NA_OIII_4960'},

		'NA_OIII_5007': {'center':5008.240,'amp':'free','disp':'free','voff':'free','line_type':'na','label':r'[O III]','ncomp':1,},
		'NA_OIII_5007_2': {'center':5008.240,'amp':'free','disp':'free','voff':'free','line_type':'na','ncomp':2,'parent':'NA_OIII_5007'},

		'BR_H_BETA': {'center':4862.691,'amp':'free','disp':'free','voff':'free','line_type':'br','ncomp':1,},
		'BR_OIII_5007': {'center':5008.240,'amp':'free','disp':'free','voff':'free','line_type':'br','ncomp':1,},
	},
	'combined_lines': {
		'H_BETA_COMB': ['NA_H_BETA', 'BR_H_BETA'],
		'OIII_COMB': ['NA_OIII_5007', 'BR_OIII_5007'],
	},
	'opt_feii_options': {
		'opt_template': {'type': 'VC04'},
		'opt_amp_const': {'bool': False},
		'opt_disp_const': {'bool': False},
		'opt_voff_const': {'bool': False},
	},
	'uv_iron_options': {
		'uv_amp_const': {'bool': False},
		'uv_disp_const': {'bool': False},
		'uv_voff_const': {'bool': False},
	},
	'losvd_options': {
		'library': 'IndoUS',
		'vel_const': {'bool': False},
		'disp_const': {'bool': False},
	},
	'host_options': {
		'age': [1.0,5.0,10.0],
		'vel_const': {'bool': False},
		'disp_const': {'bool': False},
	},
	'power_options': {
		'type': 'simple',
	},
	'poly_options': {
		'apoly': {'bool': True, 'order': 3},
	},
	'plot_options': {
		'plot_HTML': True,
	},
}

def run_sample():
	sample = ['spec-0948-52428-0454.fits', 'spec-1180-52995-0420.fits', 'spec-10237-58154-0550.fits', 'spec-4216-55477-0310.fits', 'spec-8862-57461-0145.fits']
	sample = [spectra_dir.joinpath(spec_file) for spec_file in sample]

	badass.run_BADASS(sample, options=BA_OPTIONS, nprocesses=2)
```


## NIRSpec

```python
options_file = OPTIONS_DIR.joinpath('nirspec_single.py')
test_file = EX_SPEC_DIR.joinpath('JWST_NIRSpec', 'NGC4051_nirspec_290_s3d.fits')
badass.run_BADASS(test_file, options_file=options_file)
```

where in `options_file` we have:

```python
io_options = {
	'infmt': 'nirspec',
    'output_dir': 'nirspec_out',
	'filter': '290',
	'disperser': 'h',
	'redshift': 0.002336,
	'spaxel': (25,25),
}

# Other fit options...
```
