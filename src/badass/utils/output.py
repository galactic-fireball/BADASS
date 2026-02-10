from astropy.io import fits
import numpy as np
import pathlib


class FitType:
	SINGLE = 1
	APERTURE = 2
	SPAXELS = 3
	BINS = 4


class ResultWriter:

	_writer = None

	def __new__(cls, cfg):
		if not cls._writer is None:
			return cls._writer
		cls._writer = super().__new__(cls)
		cls._writer.cfg = cfg

		# TODO: better way to specify
		if not 'fit_area' in cfg.fit:
			cls._writer.fit_type = FitType.SINGLE
			return cls._writer

		if 'aperture' in cfg.fit.fit_area:
			cls._writer.fit_type = FitType.APERTURE
			return cls._writer

		if ('spaxel' in cfg.fit.fit_area) or ('spaxels' in cfg.fit.fit_area):
			cls._writer.fit_type = FitType.SPAXELS
			cls._writer.spaxels = cfg.fit.fit_area.get('spaxels', cfg.fit.fit_area.get('spaxel', None))
			return cls._writer

		if 'bins' in cfg.fit.fit_area:
			cls._writer.fit_type = FitType.BINS
			return cls._writer

		raise Exception('Unknown fit type')


	def add_fit_ctx(self, ctx):
		# TODO: do something with a single fit ctx
		pass


	def compile_results(self):
		# TODO: write output files and plots, rebuild cube if needed
		# TODO: account for relative output_dir
		out_dir = pathlib.Path(self.cfg.io.output_dir)

		def make_maps(fit_dirs, shape):
			result_fits = fits.HDUList()
			result_fits.append(fits.PrimaryHDU()) # TODO: put header info in

			maps = {}
			for fit_dir in fit_dirs:
				x, y = (int(i) for i in fit_dir.name.split('_')[-2:])
				par_table = fit_dir.joinpath('log', 'par_table.fits') # TODO: out file names as constants
				if not par_table.exists():
					continue # TODO: do something else?

				hdu = fits.open(par_table)
				pt = hdu[1].data
				hdu.close()

				for record in pt:
					param_name = record['parameter']
					if not param_name in maps:
						maps[param_name] = np.full(shape, fill_value=np.nan, dtype=float)
					maps[param_name][x,y] = record['best_fit']

			for param, param_map in maps.items():
				result_fits.append(fits.ImageHDU(param_map, name=param))
			result_fits.writeto(out_dir.joinpath('maps.fits'), overwrite=True)
			# TODO: option to generate map pngs

			# TODO: make spectra explorer npz
			fit_data = {}
			for fit_dir in fit_dirs:
				bmc_file = fit_dir.joinpath('log', 'best_model_components.fits')
				if not bmc_file.exists():
					continue

				hdu = fits.open(bmc_file)
				data = hdu[1].data
				fit_data[fit_dir.name] = {
					'WAVE': data['WAVE'],
					'DATA': data['DATA'],
					'MODEL': data['MODEL'],
				}
				hdu.close()

			npz_out = out_dir.joinpath('npz')
			npz_out.mkdir(parents=True, exist_ok=True)

			for fit_name, fit_dict in fit_data.items():
				np.savez_compressed(npz_out.joinpath(fit_name+'.npz'), wave=fit_dict['WAVE'], data=fit_dict['DATA'], model=fit_dict['MODEL'])


		# TODO: make separate functions
		if self.fit_type == FitType.SINGLE:
			pass
		elif self.fit_type == FitType.APERTURE:
			pass
		elif self.fit_type == FitType.SPAXELS:
			outdirs = list(out_dir.glob('spaxel_*_*'))
			if len(outdirs) == 0:
				return
			xmax = max([int(spax_dir.name.split('_')[1]) for spax_dir in outdirs])
			ymax = max([int(spax_dir.name.split('_')[2]) for spax_dir in outdirs])
			shape = (xmax+1,ymax+1)
			fit_dirs = list(out_dir.glob('spaxel_*_*'))
			make_maps(fit_dirs, shape)
		elif self.fit_type == FitType.BINS:
			outdirs = list(out_dir.glob('bin_*_*'))
			if len(outdirs) == 0:
				return
			xmax = max([int(bin_dir.name.split('_')[1]) for bin_dir in outdirs])
			ymax = max([int(bin_dir.name.split('_')[2]) for bin_dir in outdirs])
			shape = (xmax+1,ymax+1)
			fit_dirs = list(out_dir.glob('bin_*_*'))
			make_maps(fit_dirs, shape)


# TODO: save_run_state -> allow to be picked up by a new run
