
# from badass.runner.tests import TestResult, TestRunner
from badass.runner.bootstrap import MLResult, MLRunner
# from badass.runner.mcmc import MCMCContext, MCMCResult, MCMCStage


class BadassPipeline:

    @staticmethod
    def init(targets, cfg):
        print('BadassPipeline init')

        # Batch run IFU areas
        if not cfg.fit.fit_area.type is None:
            pipeline_cls = get_ifu_type(cfg.fit.fit_area.type)
            if pipeline_cls is None:
                raise Exception('Unexpected area type: %s'%cfg.fit.fit_area.type)

            return pipeline_cls(targets, cfg)

        # Multiple non-IFU targets
        if len(targets) > 1:
            return SurveyPipeline(targets, cfg)

        # Single target fitting, no tests
        if isinstance(cfg, list): cfg = cfg[0]
        return BadassPipeline(targets[0], cfg)


    def __init__(self, target, cfg):
        self.target = target
        self.cfg = cfg


    def run(self):
        print('BadassPipeline run')

        # TODO: make BadassPipeline
        # if self.cfg.fit.test_models:
        #     runner = TestRunner(self.target)
        #     runner.run()
        #     runner.finalize()

        if not self.cfg.fit.skip_bootstrap:
            runner = MLRunner(self.target)
            runner.run()
            runner.finalize()
            return runner.result

        # run mcmc

        # collect output


    def finalize(self):
        print('BadassPipeline finalize')


class SurveyPipeline(BadassPipeline):

    def run(self):
        for target in targets:
            BadassPipeline().run()

        gather_data()


class IFUPipeline(BadassPipeline):
    area_type = 'general'

    def run(self):
        for spaxel in spaxels:
            BadassPipeline().run()

        gather_data()


class SpaxelsPipeline(IFUPipeline):
    area_type = 'spaxels'


class BinsPipeline(IFUPipeline):
    area_type = 'bins'


class AperturesPipeline(IFUPipeline):
    area_type = 'apertures'


def get_ifu_type(area_type):
    for pipeline in [IFUPipeline, SpaxelsPipeline, BinsPipeline, AperturesPipeline]:
        if pipeline.area_type == area_type:
            return pipeline
    return None

