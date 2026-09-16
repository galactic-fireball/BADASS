from astropy.io import fits
import copy
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from spark.plot import add_ax_labels

from badass.runner.survey import SurveyPipeline, skip_existing
from badass.utils import plotting


@dataclass
class IFUPipeline(SurveyPipeline):
    area_types = ['general',]

    single_sources: dict = field(default_factory=dict)
    source_results: dict = field(default_factory=dict)

    @staticmethod
    def setup_map_spec_axes(cube):
        fig = plt.figure(figsize=(18,10))
        gs = fig.add_gridspec(1, 2, width_ratios=[2,4])
        cube_ax = fig.add_subplot(gs[0,0])
        spec_ax = fig.add_subplot(gs[0,1])
        medcube = cube.get_median_map()
        cube_ax.imshow(medcube, origin='lower', norm=LogNorm())
        return cube_ax, spec_ax


@dataclass
class SpaxelsPipeline(IFUPipeline):
    area_types = ['spaxel','spaxels']

    spaxels: list = field(default_factory=list)

    def initialize_sources(self):
        plot = self.cfg.fit.fit_area.plot_input

        # TODO: 'exclude' option
        self.spaxels = self.cfg.fit.fit_area.spaxels
        nx = self.source.flux.shape[2]
        ny = self.source.flux.shape[1]

        if isinstance(self.spaxels, str):
            if self.spaxels.lower() != 'all':
                # TODO: mark spec or self as invalid?
                raise Exception('spaxel list invalid: %s'%spaxels)
            self.spaxels = [(x,y) for x in range(nx) for y in range(ny)]

        elif isinstance(self.spaxels, dict):
            xs = self.spaxels.get('x', (0,nx))
            ys = self.spaxels.get('y', (0,ny))
            self.spaxels = [(x,y) for x in range(*xs) for y in range(*ys)]

        elif isinstance(self.spaxels, (tuple,list)):
            # single spaxel case
            if (len(self.spaxels) == 2) and (isinstance(self.spaxels[0], int)):
                self.spaxels = [self.spaxels]
            # should be list of (x,y) pairs
            elif any([not isinstance(spax, (tuple,list)) for spax in self.spaxels]):
                raise Exception('spaxel list invalid')
        else:
            xs = (0,nx)
            ys = (0,ny)

        if plot:
            fig, ax = plt.subplots(figsize=(10,10))
            medcube = self.source.get_median_map()
            ax.imshow(medcube, origin='lower', norm=LogNorm())

        for spaxel in self.spaxels:
            # TODO: different cfg (user lines) for each spaxel
            spaxel_cfg = copy.deepcopy(self.cfg)
            source_spax = self.source.spax(*spaxel)

            spaxel_out_dir = spaxel_cfg.io.output_dir.joinpath(self.source.name, source_spax.name)
            if skip_existing(spaxel_out_dir, spaxel_cfg.io.overwrite):
                continue

            spaxel_cfg.io.output_dir = spaxel_out_dir
            spaxel_out_dir.mkdir(parents=True, exist_ok=True)

            if plot:
                ax.scatter(source_spax.x, source_spax.y, color='red', marker='x', s=30)

            self.single_sources[source_spax.name] = (source_spax, spaxel_cfg)

        if plot:
            ax.set_xlabel('X (px)')
            ax.set_ylabel('Y (px)')
            plt.show()
            plt.close()


    def finalize(self):
        self.make_parameter_maps()


    def make_parameter_maps(self):
        maps = {}
        bf_results = {name: res[-1] for name, res in self.source_results.items()}            

        for name, res in bf_results.items():
            x, y = [int(v) for v in name.split('_')[1:]]
            for param in res.final_params.values():
                if not param.name in maps:
                    maps[param.name] = np.full(self.source.shape, np.nan)
                maps[param.name][y,x] = param.best_fit

        result_fits = fits.HDUList()
        result_fits.append(fits.PrimaryHDU()) # TODO: metadata
        for name, data in maps.items():
            result_fits.append(fits.ImageHDU(data, name=name))

        outfile = self.outdir.joinpath('parameter_maps.fits')
        result_fits.writeto(outfile, overwrite=True)


@dataclass
class BinsPipeline(IFUPipeline):
    area_types = ['bin','bins',]

    def initialize_sources(self):
        # TODO: voronoi binning
        slength = self.cfg.fit.fit_area.bins.side_length
        method = self.cfg.fit.fit_area.bins.method
        plot = self.cfg.fit.fit_area.plot_input

        if plot:
            cube_ax, spec_ax = IFUPipeline.setup_map_spec_axes(self.source)

        sx,ex = self.cfg.fit.fit_area.bins.x
        if ex < 0: ex = self.source.flux.shape[2]
        sy,ey = self.cfg.fit.fit_area.bins.y
        if ey < 0: ey = self.source.flux.shape[1]

        bxs_r = range(sx, ex, slength)
        bys_r = range(sy, ey, slength)

        bnx = bny = 0
        for bxs in bxs_r:
            for bys in bys_r:
                bxe = min(bxs+slength, ex)
                bye = min(bys+slength, ey)
                width = bxe - bxs
                height = bye - bys

                # TODO: different cfg (user lines) for each bin
                bin_cfg = copy.deepcopy(self.cfg)
                center = (bxs+(width/2), bys+(height/2))
                bin_name = 'bin_%d_%d'%(bnx,bny)
                source_bin = self.source.aperture('rectangular', center, width=width, height=height, name=bin_name)

                bin_out_dir = bin_cfg.io.output_dir.joinpath(self.source.name, source_bin.name)
                if skip_existing(bin_out_dir, bin_cfg.io.overwrite):
                    continue

                if plot:
                    source_bin.add_to_plot(cube_ax)
                    source_bin.add_spec_plot(spec_ax, norm=True)

                bin_cfg.io.output_dir = bin_out_dir
                bin_out_dir.mkdir(parents=True, exist_ok=True)

                self.single_sources[source_bin.name] = (source_bin, bin_cfg)
                bny += 1
            bny = 0
            bnx += 1

        if plot:
            cube_ax.set_xlabel('X (px)')
            cube_ax.set_ylabel('Y (px)')
            add_ax_labels(spec_ax,'AA')
            spec_ax.set_ylabel('Normalized flux density')
            plt.show()
            plt.close()


@dataclass
class AperturesPipeline(IFUPipeline):
    area_types = ['aperture','apertures',]

    def initialize_sources(self):
        aps = self.cfg.fit.fit_area.apertures
        if not isinstance(aps, list): aps = [aps,]
        plot = self.cfg.fit.fit_area.plot_input

        if plot:
            cube_ax, spec_ax = IFUPipeline.setup_map_spec_axes(self.source)

        for i, ap in enumerate(aps):
            # TODO: different cfg (user lines) for each aperture
            ap_cfg = copy.deepcopy(self.cfg)
            ap_name = 'ap_%d'%i
            kwargs = {k:v for k,v in ap.model_dump().items() if k in ['width','height','radius','a','b','theta']}
            kwargs['name'] = ap_name
            source_ap = self.source.aperture(ap.shape, ap.center, **kwargs)

            ap_out_dir = ap_cfg.io.output_dir.joinpath(self.source.name, source_ap.name)
            if skip_existing(ap_out_dir, ap_cfg.io.overwrite):
                continue

            if plot:
                source_ap.add_to_plot(cube_ax)
                source_ap.add_spec_plot(spec_ax, norm=True)

            ap_cfg.io.output_dir = ap_out_dir
            ap_out_dir.mkdir(parents=True, exist_ok=True)
            self.single_sources[source_ap.name] = (source_ap, ap_cfg)

        if plot:
            cube_ax.set_xlabel('X (px)')
            cube_ax.set_ylabel('Y (px)')
            add_ax_labels(spec_ax,'AA')
            spec_ax.set_ylabel('Normalized flux density')
            plt.show()
            plt.close()


def get_ifu_type(area_type):
    for pipeline in [IFUPipeline, SpaxelsPipeline, BinsPipeline, AperturesPipeline]:
        if area_type in pipeline.area_types:
            return pipeline
    return None

