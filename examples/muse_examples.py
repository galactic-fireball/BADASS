from astropy.io import fits
import matplotlib.pyplot as plt
import pathlib

cwd = pathlib.Path(__file__).resolve().parent
EXAMPLE_FILE = cwd.joinpath('example_spectra', 'MUSE', 'NGC1068_subcube.fits')

import badass

NGC1068_REDSHIFT = 0.00348372


# NOTE:
# options['fit_options']['fit_area']['plot_input'] == True => BADASS will pop up a median cube plot with the fit area designated


def run_single_spaxel():
	options = MUSE_OPTIONS
	options['fit_options']['fit_area'] = {'spaxels':(25,22), 'plot_input':True}
	out_dir = options['io_options']['output_dir'].joinpath('single_spaxel')
	options['io_options']['output_dir'] = out_dir
	badass.run_BADASS(EXAMPLE_FILE, options=options)


def run_spaxel_range():
	options = MUSE_OPTIONS
	options['fit_options']['fit_area'] = {'spaxels':{'x':(20,30), 'y':(15,30)}, 'plot_input':True}
	out_dir = options['io_options']['output_dir'].joinpath('spaxel_range')
	options['io_options']['output_dir'] = out_dir
	badass.run_BADASS(EXAMPLE_FILE, options=options, nprocesses=2)

	maps_file = out_dir.joinpath('maps.fits')
	if not maps_file.exists():
		print('RUN FAILED!')
		return

	map_line = 'OIII_5007_COMB'
	flux_attr = map_line + '_FLUX'
	hdu = fits.open(maps_file)
	if not flux_attr in hdu:
		print('Could not find %s in maps data!'%flux_attr)
		return

	flux_data = 10**(hdu[flux_attr].data[20:,15:])
	plt.figure()
	plt.imshow(flux_data.T, origin='lower')
	plt.show()


def run_bins():
	options = MUSE_OPTIONS
	options['fit_options']['fit_area'] = {'bins':{'side_length':3, 'x':(20,30), 'y':(15,30)}, 'plot_input':True}
	out_dir = options['io_options']['output_dir'].joinpath('bins')
	options['io_options']['output_dir'] = out_dir
	badass.run_BADASS(EXAMPLE_FILE, options=options, nprocesses=2)

	maps_file = out_dir.joinpath('maps.fits')
	if not maps_file.exists():
		print('RUN FAILED!')
		return

	map_line = 'OIII_5007_COMB'
	flux_attr = map_line + '_FLUX'
	hdu = fits.open(maps_file)
	if not flux_attr in hdu:
		print('Could not find %s in maps data!'%flux_attr)
		return

	flux_data = 10**(hdu[flux_attr].data)
	plt.figure()
	plt.imshow(flux_data.T, origin='lower')
	plt.show()


def run_aperture():
	options = MUSE_OPTIONS
	options['fit_options']['fit_area'] = {'aperture': {'type':'circular', 'center':(25,22), 'radius':5}, 'plot_input':True}
	out_dir = options['io_options']['output_dir'].joinpath('aperture')
	options['io_options']['output_dir'] = out_dir
	badass.run_BADASS(EXAMPLE_FILE, options=options)


def run_line_test():
	options = MUSE_OPTIONS
	options['fit_options']['fit_area'] = {'aperture': {'type':'circular', 'center':(25,22), 'radius':5},}
	out_dir = options['io_options']['output_dir'].joinpath('line_test')
	options['io_options']['output_dir'] = out_dir
	options['fit_options']['test_lines'] = True
	badass.run_BADASS(EXAMPLE_FILE, options=options)


# Uncomment target function
def main():
	# run_single_spaxel()
	# run_spaxel_range()
	# run_bins()
	# run_aperture()
	run_line_test()
	pass


USER_LINES = {
    'NA_H_BETA': {'center':4862.691, 'line_type':'na', 'ncomp':1, 'disp':'NA_OIII_5007_DISP', 'label':r'H$\beta$',},
    'NA_H_BETA_2': {'center':4862.691, 'line_type':'na', 'ncomp':2, 'parent':'NA_H_BETA', 'amp':'NA_H_BETA_AMP*NA_OIII_5007_2_AMP/NA_OIII_5007_AMP', 'disp':'NA_OIII_5007_2_DISP', 'voff':'NA_OIII_5007_2_VOFF',},

    'NA_OIII_4960': {'center':4960.295, 'line_type':'na', 'ncomp':1, 'amp':'(NA_OIII_5007_AMP/2.98)', 'disp':'NA_OIII_5007_DISP', 'voff':'NA_OIII_5007_VOFF','label':r'[O III]',},

    'NA_OIII_5007': {'center':5008.240, 'line_type':'na', 'ncomp':1, 'label':r'[O III]',},
    'NA_OIII_5007_2': {'center':5008.240, 'line_type':'na', 'ncomp':2, 'parent':'NA_OIII_5007'},

    'BR_H_BETA': {'center':4862.691, 'line_type':'br','ncomp':1,},
    'BR_OIII_5007': {'center':5008.240, 'line_type':'br','ncomp':1,},
}

CONSTRAINTS = [
    ('NA_OIII_5007_AMP', 'NA_OIII_5007_2_AMP'),
    ('NA_OIII_5007_2_DISP', 'NA_OIII_5007_DISP'),
]

COMBINED_LINES = {
    'H_BETA_COMB':['NA_H_BETA', 'NA_H_BETA_2', 'BR_H_BETA'],
    'OIII_5007_COMB':['NA_OIII_5007', 'NA_OIII_5007_2', 'BR_OIII_5007'],
}


MUSE_OPTIONS = {
	'io_options': {
		'infmt': 'muse',
		'output_dir': EXAMPLE_FILE.parent.joinpath('example_runs'),
		'product_name': 'NGC1068',
		'log_level': 'info',
		'overwrite': True,
	},
	'fit_options': {
		'redshift': NGC1068_REDSHIFT,
		'fit_reg': (4800,5200),
		'fit_area': {}, # to be set by each example
		'n_basinhop': 15,
		'test_lines': False,
	},
	'comp_options': {
		'fit_opt_feii': True,
		'fit_uv_iron': False,
		'fit_balmer': False,
		'fit_losvd': False,
		'fit_host': True,
		'fit_power': True,
		'fit_poly': True,
		'fit_narrow': True,
		'fit_broad': True,
		'tie_line_disp': False,
		'tie_line_voff': False,
	},
	'user_lines': USER_LINES,
	'user_constraints': CONSTRAINTS,
	'combined_lines': COMBINED_LINES,
	'narrow_options': {
		'disp_plim': (0,500),
		'voff_plim': (-500,500),
		'line_profile': 'gaussian',
	},
	'broad_options': {
		'disp_plim': (500,3000),
		'voff_plim': (-1000,1000),
		'line_profile': 'gaussian',
	},
	'test_options': {
		'test_mode': 'line',
		'lines': [['NA_H_BETA']],
		'metrics': {'BADASS':0.95, 'ANOVA':0.95, 'CHI2_RATIO':0.10, 'AON':3.0}, # Fitting metrics to use when determining the best model
		'conv_mode': 'all', # or 'any'
		'auto_stop': False,
		'plot_tests': True,
		'force_best': True,
		'continue_fit': True,
	},
	'host_options': {
		'age': [1.0,5.0,10.0],
		'vel_const': {'bool':False, 'val':0.0},
		'disp_const': {'bool': False, 'val':150.0},
	},
	'power_options': {
		'type': 'simple',
	},
	'poly_options': {
		'apoly': {'bool':True, 'order':3},
	},
	'opt_feii_options': {
		'opt_template': {'type':'VC04'},
		'opt_amp_const': {'bool':False, 'br_opt_feii_val':1.0, 'na_opt_feii_val':1.0},
		'opt_disp_const': {'bool':False, 'br_opt_feii_val':3000.0, 'na_opt_feii_val':500.0},
		'opt_voff_const': {'bool':False, 'br_opt_feii_val':0.0, 'na_opt_feii_val':0.0},
	},
	'plot_options': {
		'plot_HTML': True,
	},
}


if __name__ == '__main__':
	main()

