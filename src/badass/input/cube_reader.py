import copy
import matplotlib.pyplot as plt
import numpy as np
from photutils.aperture import CircularAperture, ApertureStats

from badass.input.input import BadassInput

class CubeReader(BadassInput):

    @classmethod
    def parse(cls, input_data, options):
        cube_dict = cls.get_cube_data(input_data, options)
        cube_dict['options'] = options

        # TODO: separate subclasses
        area_parsers = {
            'spaxel': cls.spaxel_parse,
            'spaxels': cls.spaxel_parse,
            'bins': cls.bin_parse,
            'aperture': cls.aperture_parse,
        }

        fit_area = options.fit_options.fit_area
        fit_area_type = list(fit_area.keys())[0]
        fit_area_func = area_parsers.get(fit_area_type, None)

        if fit_area_func is None:
            raise Exception('Fit area type unsupported: %s'%fit_area_type)

        return fit_area_func(cube_dict, input_data, options)


    @classmethod
    def spaxel_parse(cls, cube_dict, input_data, options):
        spaxels = options.fit_options.fit_area.get('spaxels', options.fit_options.fit_area.get('spaxel', None))
        nx = cube_dict.get('nx', cube_dict['spec'].shape[0])
        ny = cube_dict.get('ny', cube_dict['spec'].shape[1])

        if isinstance(spaxels, str):
            if spaxels.lower() != 'all':
                raise Exception('spaxel list invalid: %s'%spaxels)
            fit_spaxels = [(x,y) for x in range(nx) for y in range(ny)]
        elif isinstance(spaxels, dict):
            xs = spaxels.get('x', (0,nx))
            ys = spaxels.get('y', (0,ny))
            fit_spaxels = [(x,y) for x in range(*xs) for y in range(*ys)]
        elif isinstance(spaxels, (tuple,list)):
            # single spaxel case
            if (len(spaxels) == 2) and (isinstance(spaxels[0], int)):
                fit_spaxels = [spaxels]
            # should be list of (x,y) pairs
            elif any([not isinstance(spax, (tuple,list)) for spax in spaxels]):
                raise Exception('spaxel list invalid')
            else:
                fit_spaxels = spaxels
        else:
            raise Exception('spaxel list invalid')

        if options.fit_options.fit_area.get('plot_input', False):
            medcube = np.nanmedian(cube_dict['spec'], axis=2)
            medcube[np.isnan(medcube)] = 0.0

            plt.figure()
            plt.imshow(medcube.T, origin='lower')
            for x,y in fit_spaxels:
                plt.scatter(x, y, color='orange', marker='+', s=22)
            plt.show()

        # These are the values the subclass Reader told us are spaxel-splitable
        # This way we don't do a deepcopy of large 3D arrays that are going to be cutdown anyway
        split_dict = {split_key:cube_dict.pop(split_key,None) for split_key in cube_dict.pop('splitable', ['spec','noise'])}

        inputs = []
        for x,y in fit_spaxels:
            spax_dict = copy.deepcopy(cube_dict)
            spax_dict['options'].fit_options.fit_area.spaxels = (x,y)
            spax_dict['options'].io_options.output_dir = '%s/spaxel_%d_%d' % (spax_dict['options'].io_options.output_dir,x,y)

            for key, val in split_dict.items():
                spax_dict[key] = val[x,y,:]

            inputs.append(cls.from_dict(spax_dict))

        return inputs


    @classmethod
    def bin_parse(cls, cube_dict, input_data, options):
        slength = options.fit_options.fit_area.bins.side_length
        method = options.fit_options.fit_area.bins.get('method','sum')
        plot = options.fit_options.fit_area.get('plot_input', False)

        if plot:
            from matplotlib.patches import Rectangle
            medcube = np.nanmedian(cube_dict['spec'], axis=2)
            medcube[np.isnan(medcube)] = 0.0

            plt.figure()
            plt.imshow(medcube.T, origin='lower')

        nx = cube_dict.get('nx', cube_dict['spec'].shape[0])
        ny = cube_dict.get('ny', cube_dict['spec'].shape[1])
        sx,nx = options.fit_options.fit_area.bins.get('x',(0,nx))
        sy,ny = options.fit_options.fit_area.bins.get('y',(0,ny))

        bxs_r = range(sx, nx, slength)
        bys_r = range(sy, ny, slength)

        cube_spec = cube_dict.pop('spec')
        cube_noise = cube_dict.pop('noise')

        product_name = options.io_options.get('product_name', '')
        if product_name != '': product_name = product_name + '_'

        inputs = []
        bnx = bny = 0
        for bxs in bxs_r:
            for bys in bys_r:
                bxe = min(bxs+slength, nx)
                bye = min(bys+slength, ny)

                if plot:
                    plt.gca().add_patch(Rectangle((bxs,bys), width=bxe-bxs, height=bye-bys, facecolor='none', edgecolor='orange'))

                # print('bin(%d,%d): (%d,%d) ; (%d,%d)'%(bnx,bny,bxs,bxe,bys,bye))
                bin_dict = copy.deepcopy(cube_dict)
                bin_dict['options'].io_options.product_name = product_name + 'BIN(%d,%d)'%(bnx,bny)
                bin_dict['options'].io_options.output_dir = '%s/bin_%d_%d' % (bin_dict['options'].io_options.output_dir,bnx,bny)

                bin_spec = cube_spec[bxs:bxe,bys:bye,:]
                bin_noise = cube_noise[bxs:bxe,bys:bye,:]

                if method == 'sum':
                    bin_spec = np.apply_over_axes(np.nansum, bin_spec, (0,1))
                    bin_noise = np.sqrt(np.apply_over_axes(np.sum, np.square(bin_noise), (0,1)))
                elif method == 'mean':
                    bin_spec = np.apply_over_axes(np.nanmean, bin_spec, (0,1))
                    bin_noise = (np.sqrt(np.apply_over_axes(np.sum, np.square(bin_noise), (0,1)))) / (slength**2)
                else:
                    raise Exception('Unsupport bin method: %s'%method)

                bin_dict['spec'] = bin_spec[0,0,:]
                bin_dict['noise'] = bin_noise[0,0,:]
                inputs.append(cls.from_dict(bin_dict))

                bny += 1
            bny = 0
            bnx += 1

        if plot:
            plt.show()

        return inputs


    @classmethod
    def aperture_parse(cls, cube_dict, input_data, options):
        aperture_options = options.fit_options.fit_area.aperture
        # TODO: RectangularAperture
        # TODO: other methods (mean, etc.)

        def get_circular_aperture():
            ap_center = aperture_options.center
            radius = aperture_options.radius
            aperture = CircularAperture(ap_center, r=radius)

            if options.fit_options.fit_area.get('plot_input', False):
                medcube = np.nanmedian(cube_dict['spec'], axis=2)
                medcube[np.isnan(medcube)] = 0.0

                plt.figure()
                plt.imshow(medcube.T, origin='lower')
                aperture.plot()
                plt.show()

            wave = cube_dict['wave']
            ap_spec = np.zeros(len(wave))
            ap_err = np.zeros(len(wave))

            for i in range(0, len(wave)):
                apstat = ApertureStats(cube_dict['spec'][:,:,i], aperture, error=cube_dict['noise'][:,:,i])
                ap_spec[i] = apstat.sum
                ap_err[i] = apstat.sum_err
            return ap_spec, ap_err

        ap_types = {
            'circular': get_circular_aperture,
        }

        ap_type = aperture_options.type
        ap_func = ap_types.get(ap_type, None)
        if ap_func is None:
            raise Exception('Unsupported aperture type: %s'%ap_type)

        ap_spec, ap_noise = ap_func()
        input_dict = cube_dict
        input_dict['spec'] = ap_spec
        input_dict['noise'] = ap_noise

        return cls.from_dict(input_dict)


    @classmethod
    def get_cube_data(cls, input_data, options):
        return {}

