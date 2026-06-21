import copy
from dataclasses import dataclass, field

from badass.runner.survey import SurveyPipeline, skip_existing
from badass.utils import plotting


@dataclass
class IFUPipeline(SurveyPipeline):
    area_types = ['general',]

    single_sources: dict = field(default_factory=dict)
    source_results: dict = field(default_factory=dict)


    def __post_init__(self):
        pass


@dataclass
class SpaxelsPipeline(IFUPipeline):
    area_types = ['spaxel','spaxels']

    spaxels: list = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()

        # TODO: 'exclude' option
        self.spaxels = self.cfg.fit.fit_area.spaxels
        nx = self.sources.flux.shape[2]
        ny = self.sources.flux.shape[1]

        if isinstance(self.spaxels, str):
            if spaxels.lower() != 'all':
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
            raise Exception('spaxel list invalid')

        for spaxel in self.spaxels:
            # TODO: different cfg (user lines) for each spaxel
            spaxel_cfg = copy.deepcopy(self.cfg)
            source_spax = self.sources.spax(*spaxel)

            spaxel_out_dir = spaxel_cfg.io.output_dir.joinpath(self.sources.name, source_spax.name)
            # if skip_existing(spaxel_out_dir, spaxel_cfg.io.overwrite):
                # continue

            spaxel_cfg.io.output_dir = spaxel_out_dir
            spaxel_out_dir.mkdir(parents=True, exist_ok=True)
            self.single_sources[source_spax.name] = (source_spax, spaxel_cfg)


    def finalize(self):
        self.make_source_plots()


    def make_source_plots(self):
        for res in self.source_results.values():
            res.figures['ml_fit'] = plotting.plot_ml_results(res, self.single_sources[res.name][0])


class BinsPipeline(IFUPipeline):
    area_types = ['bin','bins',]

    def __post_init__(self):
        # TODO: voronoi binning
        slength = self.cfg.fit.fit_area.bins.side_length
        method = self.cfg.fit.fit_area.bins.method
        plot = self.cfg.fit.fit_area.plot_input

        if plot:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            medcube = np.nanmedian(cube_dict['spec'], axis=2)
            medcube[np.isnan(medcube)] = 0.0

            plt.figure()
            plt.imshow(medcube.T, origin='lower')

        sx,ex = self.cfg.fit.fit_area.bins.x
        if ex < 0: ex = self.sources.flux.shape[2]
        sy,ey = self.cfg.fit.fit_area.bins.y
        if ey < 0: ey = self.sources.flux.shape[1]

        bxs_r = range(sx, ex, slength)
        bys_r = range(sy, ey, slength)

        bnx = bny = 0
        for bxs in bxs_r:
            for bys in bys_r:
                bxe = min(bxs+slength, ex)
                bye = min(bys+slength, ey)
                width = bxe - bxs
                height = bye - bys

                if plot:
                    plt.gca().add_patch(Rectangle((bxs,bys), width=bxe-bxs, height=bye-bys, facecolor='none', edgecolor='orange'))

                # TODO: different cfg (user lines) for each bin
                bin_cfg = copy.deepcopy(self.cfg)
                center = (bxs+(width/2), bys+(height/2))
                bin_name = 'bin_%d_%d'%(bnx,bny)
                source_bin = self.sources.aperture('rectangular', center, width=width, height=height, name=bin_name)

                bin_out_dir = bin_cfg.io.output_dir.joinpath(self.sources.name, source_bin.name)
                bin_cfg.io.output_dir = bin_out_dir
                bin_out_dir.mkdir(parents=True, exist_ok=True)

                self.single_sources[source_bin.name] = (source_bin, bin_cfg)
                bny += 1
            bny = 0
            bnx += 1

        if plot:
            plt.show()


class AperturesPipeline(IFUPipeline):
    area_types = ['aperture','apertures',]

    def __post_init__(self):
        aps = self.cfg.fit.fit_area.apertures
        if not isinstance(aps, list): aps = [aps,]
        plot = self.cfg.fit.fit_area.plot_input

        if plot:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            import numpy as np
            from spark.plot import add_ax_labels
            # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18,10), width_ratios=[1,4], sharey=True)
            fig = plt.figure(figsize=(18,10))
            gs = fig.add_gridspec(1, 2, width_ratios=[2,4])
            cube_ax = fig.add_subplot(gs[0,0])
            spec_ax = fig.add_subplot(gs[0,1])
            medcube = self.sources.get_median_map()
            cube_ax.imshow(medcube, origin='lower', norm=LogNorm())

        for i, ap in enumerate(aps):
            # TODO: different cfg (user lines) for each aperture
            ap_cfg = copy.deepcopy(self.cfg)
            ap_name = 'ap_%d'%i
            kwargs = {k:v for k,v in ap.model_dump().items() if k in ['width','height','radius']}
            kwargs['name'] = ap_name
            source_ap = self.sources.aperture(ap.shape, ap.center, **kwargs)

            ap_out_dir = ap_cfg.io.output_dir.joinpath(self.sources.name, source_ap.name)
            if skip_existing(ap_out_dir, ap_cfg.io.overwrite):
                continue

            if plot:
                source_ap.add_to_plot(cube_ax)
                source_ap.add_spec_plot(spec_ax)

            ap_cfg.io.output_dir = ap_out_dir
            ap_out_dir.mkdir(parents=True, exist_ok=True)
            self.single_sources[source_ap.name] = (source_ap, ap_cfg)

        if plot:
            cube_ax.set_xlabel('X (px)')
            cube_ax.set_ylabel('Y (px)')

            clipped = [np.clip(f[0].flux, np.percentile(f[0].flux, 2), np.percentile(f[0].flux, 98)) for f in self.single_sources.values()]
            all_flux = np.concatenate(clipped)
            ymin, ymax = np.min(all_flux), np.max(all_flux)
            spec_ax.set_ylim(ymin, ymax)

            add_ax_labels(spec_ax,'AA')
            plt.show()


def get_ifu_type(area_type):
    for pipeline in [IFUPipeline, SpaxelsPipeline, BinsPipeline, AperturesPipeline]:
        if area_type in pipeline.area_types:
            return pipeline
    return None

