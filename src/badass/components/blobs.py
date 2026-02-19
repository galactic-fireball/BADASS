import numpy as np
from scipy.interpolate import interp1d
from tabulate import tabulate

import badass.components.spectral_lines.spectral_line as spec_line
import badass.utils.utils as ba_utils


class BlobRegistry:

	_registry = None

	def __new__(cls, ctx=None):
		if cls._registry:
			return cls._registry

		cls._registry = super().__new__(cls)
		cls._registry.ctx = ctx

		return cls._registry


	def __init__(self, ctx):
		self.blobs = {}

		for line in spec_line.SpectralLine.line_list:
			if not line.is_combined:
				continue

			self.register_blob(LineVelBlob(line.name.upper(), self.ctx, line.center))

		IndexBlob.register_index_blobs(self, self.ctx.target.wave[0], self.ctx.target.wave[-1])

		for blob in self.blobs.values():
			if not blob.is_const:
				continue
			blob.compute(self.ctx)


	def register_blob(self, blob):
		self.blobs[blob.name] = blob


	def compute_all(self):
		res = {}
		for name, blob in self.blobs.items():
			if blob.is_const:
				res[name] = blob.cur_val
			else:
				res[name] = blob.compute(self.ctx)
		return res


	def init_store(self, iters):
		self.store = np.zeros((iters, len(self.blobs)))
		self.st_iter = 0


	def do_store(self):
		self.compute_all()
		# TODO: assign idx


	def dump_blobs(self):
		headers = ['Name', 'Type', 'Const?', 'Value']
		table = []

		for blob in self.blobs.values():
			row = []
			row.append(blob.name)
			row.append(blob.__class__.__name__)
			row.append('YES' if blob.is_const else 'NO')
			row.append(blob.cur_val)
			table.append(row)

		self.ctx.log.info('Current Blob Parameters:\n'+tabulate(table, headers, tablefmt='grid'))


class Blob:

	def __init__(self, name, func, kwargs={}, const=False):
		self.name = name
		self.func = func
		self.cur_val = np.nan
		self.kwargs = {}
		self.is_const = const


	def compute(self, ctx):
		self.cur_val = self.func(ctx, self.kwargs)
		return self.cur_val


class LineVelBlob(Blob):

	interp_ftn = None

	def __init__(self, line_name, ctx, center, kwargs={}):
		blob_name = line_name + '_LINE_VEL'

		super().__init__(blob_name, None, kwargs)
		self.is_const = True
		self.center = center

		if (LineVelBlob.interp_ftn is None) and (not ctx is None):
			LineVelBlob.interp_ftn = interp1d(ctx.target.wave, np.arange(len(ctx.target.wave))*ctx.target.velscale, kind='linear', bounds_error=False)


	def compute(self, ctx):
		if (self.center is None) or (LineVelBlob.interp_ftn is None):
			ctx.log.warning('Error computing line velocity blob!')
			return np.nan

		self.cur_val = LineVelBlob.interp_ftn(self.center)
		return self.cur_val


class IndexBlob(Blob):

	WAVES = [1350, 3000, 4000, 5100, 7000]

	@classmethod
	def register_index_blobs(cls, reg, wave_min, wave_max):
		for wave in IndexBlob.WAVES:
			if (wave < wave_min) or (wave > wave_max):
				continue

			reg.register_blob(cls(wave))


	def __init__(self, wave, kwargs={}):
		self.wave = wave
		blob_name = 'INDEX_%d'%self.wave

		super().__init__(blob_name, None, kwargs)
		self.is_const = True


	def compute(self, ctx):
		_, self.cur_val = ba_utils.find_nearest(ctx.target.wave,self.wave)
		return self.cur_val

