from astropy.io import fits
import numpy as np
import pathlib


class FitType:
	SINGLE = 1
	SPAXELS = 2
	BINS = 3


class ResultWriter:

	_writer = None

	def __new__(cls, options):
		if not cls._writer is None:
			return cls._writer
		cls._writer = super().__new__(cls)
		cls._writer.options = options

		# TODO: better way to specify
		if not 'fit_area' in options.io_options:
			cls._writer.fit_type = FitType.SINGLE
			return cls._writer

		if 'spaxels' in options.io_options.fit_area:
			cls._writer.fit_type = FitType.SPAXELS
			cls._writer.spaxels = options.io.fit_area.spaxels
			return cls._writer

		if 'bins' in options.io_options.fit_area:
			cls._writer.fit_type = FitType.BINS
			return cls._writer

		raise Exception('Unknown fit type')


	def add_fit_ctx(self, ctx):
		# TODO: do something with a single fit ctx
		pass


	def compile_results(self):
		# TODO: write output files and plots, rebuild cube if needed
		# TODO: account for relative output_dir
		out_dir = pathlib.Path(self.options.io_options.output_dir)

		# TODO: make separate functions
		if self.fit_type == FitType.SINGLE:
			pass
		elif self.fit_type == FitType.SPAXELS:
			pass
		elif self.fit_type == FitType.BINS:
			xmax = max([int(bin_dir.name.split('_')[1]) for bin_dir in out_dir.glob('bin_*_*')])
			ymax = max([int(bin_dir.name.split('_')[2]) for bin_dir in out_dir.glob('bin_*_*')])
			shape = (xmax+1,ymax+1)

			result_fits = fits.HDUList()
			result_fits.append(fits.PrimaryHDU()) # TODO: put header info in

			maps = {}
			for bin_dir in out_dir.glob('bin_*_*'):
				x, y = (int(i) for i in bin_dir.name.split('_')[-2:])
				par_table = bin_dir.joinpath('log', 'par_table.fits') # TODO: out file names as constants
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
			result_fits.writeto(out_dir.joinpath('bin_maps.fits'), overwrite=True)
			# TODO: option to generate map pngs

			# TODO: make spectra explorer npz
			bin_data = {}
			for bin_dir in out_dir.glob('bin_*_*'):
				bmc_file = bin_dir.joinpath('log', 'best_model_components.fits')
				if not bmc_file.exists():
					continue

				hdu = fits.open(bmc_file)
				data = hdu[1].data
				bin_data[bin_dir.name] = {
					'WAVE': data['WAVE'],
					'DATA': data['DATA'],
					'MODEL': data['MODEL'],
				}
				hdu.close()

			npz_out = out_dir.joinpath('npz')
			npz_out.mkdir(parents=True, exist_ok=True)

			for bin_name, bin_dict in bin_data.items():
				np.savez_compressed(npz_out.joinpath(bin_name+'.npz'), wave=bin_dict['WAVE'], data=bin_dict['DATA'], model=bin_dict['MODEL'])


# TODO: save_run_state -> allow to be picked up by a new run
