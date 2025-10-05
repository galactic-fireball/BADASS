import copy

from input.input import BadassInput

class CubeReader(BadassInput):

	@classmethod
	def parse(cls, input_data, options):
		cube_dict = cls.get_cube_data(input_data, options)
		cube_dict['options'] = options

		area_parsers = {
			'spaxels': cls.spaxel_parse,
			# 'bins': cls.bin_parse,
		}

		fit_area = options.io_options.fit_area
		fit_area_type = list(fit_area.keys())[0]

		if not fit_area_type in area_parsers:
			raise Exception('Fit area type unsupported: %s'%fit_area_type)

		return area_parsers[fit_area_type](cube_dict, input_data, options)


	@classmethod
	def spaxel_parse(cls, cube_dict, input_data, options):
		spaxels = options.io_options.fit_area.spaxels

		if not isinstance(spaxels, (tuple,list)) or (len(spaxels) < 1):
			raise Exception('fit spaxel must be tuple or list!')

		# single spaxel case
		if (len(spaxels) == 2) and (isinstance(spaxels[0], int)):
			x, y = spaxels
			cube_dict['spec'] = cube_dict['spec'][x,y,:]
			cube_dict['noise'] = cube_dict['noise'][x,y,:]
			return cls.from_dict(cube_dict, options)

		if not isinstance(spaxels[0], (tuple,list)):
			raise Exception('fit spaxel invalid')

		# These are the values the subclass Reader told us are spaxel-splitable
		# This way we don't do a deepcopy of large 3D arrays that are going to be cutdown anyway
		split_dict = {split_key:cube_dict.pop(split_key,None) for split_key in cube_dict.pop('splitable', [])}

		inputs = []
		for x,y in spaxels:
			spax_dict = copy.deepcopy(cube_dict)
			spax_dict['options'].io_options.fit_area.spaxels = (x,y)
			spax_dict['options'].io_options.output_dir = '%s/spaxel_%d_%d' % (spax_dict['options'].io_options.output_dir,x,y)

			for key, val in split_dict.items():
				spax_dict[key] = val[x,y,:]

			inputs.append(cls.from_dict(spax_dict))

		return inputs


	@classmethod
	def get_cube_data(cls, input_data, options):
		return {}


# TODO: fit_area options:
# fit_area:
# 	- spaxels: single spaxel tuple or list of spaxels
#	- range: x1 to x2, y1 to y2
# 	- bins:
#		- side_length: int
#	- aperture:
# 		- center, type (Rectangular vs Circular)
# 		- width/radius
