import copy

from badass.runner.pipeline import BadassPipeline
from badass.utils import plotting


class IFUPipeline(BadassPipeline):
    area_type = 'general'

    def __init__(self, target, cfg):
        super().__init__(target, cfg)
        self.single_targets = {}
        self.target_results = {}


# TODO: different spaxels/bins/apertures can have different configs
class SpaxelsPipeline(IFUPipeline):
    area_type = 'spaxels'

    def __init__(self, target, cfg):
        super().__init__(target, cfg)

        self.spaxels = cfg.fit.fit_area.args


    def run(self):
        # for spaxel in self.spaxels:
        for target in self.target:
            target_cfg = copy.deepcopy(self.cfg)

            target_out_dir = target_cfg.io.output_dir.joinpath(target.name)
            # if skip_existing(target_out_dir, target_cfg.io.overwrite):
                # continue

            target_cfg.io.output_dir = target_out_dir
            target_out_dir.mkdir(parents=True, exist_ok=True)
            self.single_targets[target.name] = (target, target_cfg)
            self.target_results[target.name] = BadassPipeline(target, target_cfg, single=False).run()


    def finalize(self):
        self.make_target_plots()


    def make_target_plots(self):
        for res in self.target_results.values():
            res.figures['ml_fit'] = plotting.plot_ml_results(res, self.single_targets[res.name][0])


class BinsPipeline(IFUPipeline):
    area_type = 'bins'


class AperturesPipeline(IFUPipeline):
    area_type = 'apertures'


def get_ifu_type(area_type):
    for pipeline in [IFUPipeline, SpaxelsPipeline, BinsPipeline, AperturesPipeline]:
        if pipeline.area_type == area_type:
            return pipeline
    return None