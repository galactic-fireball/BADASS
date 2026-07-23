from dataclasses import dataclass

from badass.input.input import BadassSpec
from badass.runner.runner import BadassResult
# from badass.runner.tests import TestResult, TestRunner
from badass.runner.bootstrap import MLRunner
from badass.runner.mcmc import MCMCRunner
# from badass.runner.mcmc import MCMCContext, MCMCResult, MCMCStage
from badass.utils.config import BadassConfig

from badass.utils import plotting

@dataclass
class BadassPipeline:

    sources: BadassSpec = None
    cfg: BadassConfig = None
    single: bool = True
    result: BadassResult = None


    def __post_init__(self):
        pass


    @staticmethod
    def init(sources, cfg):
        print('BadassPipeline init')

        # Batch run IFU areas
        test_cfg = cfg
        if isinstance(test_cfg, list): test_cfg = test_cfg[0]
        if not test_cfg.fit.fit_area.type is None:
            from badass.runner.ifu import get_ifu_type
            pipeline_cls = get_ifu_type(test_cfg.fit.fit_area.type)
            if pipeline_cls is None:
                raise Exception('Unexpected area type: %s'%test_cfg.fit.fit_area.type)

            return pipeline_cls(sources=sources, cfg=cfg)

        # Multiple non-IFU source
        if isinstance(sources, list):
            from badass.runner.survey import SurveyPipeline
            return SurveyPipeline(sources=sources, cfg=cfg)

        # Single source fitting, no tests
        print('Single source')
        if isinstance(cfg, list): cfg = cfg[0]
        return BadassPipeline(sources=sources, cfg=cfg)


    def run(self):
        print('BadassPipeline run')
        # TODO: just set up a list of runners to run

        if not self.cfg.fit.skip_bootstrap:
            runner = MLRunner(source=self.sources, cfg=self.cfg)
            if not runner.source.valid:
                runner.log.error('Invalid source! Skipping!')
                return None
            runner.run()
            runner.finalize()
            self.result = runner.result

        if not self.cfg.mcmc.mcmc_fit:
            return self.result

        # run mcmc
        runner = MCMCRunner(source=self.sources, cfg=self.cfg)
        if not runner.source.valid:
            runner.log.error('Invalid source! Skipping!')
            return None
        runner.run()
        runner.finalize()
        self.result = runner.result
        return self.result


    def finalize(self):
        print('BadassPipeline finalize')
        if self.single:
            plotting.plot_ml_results(self.result, self.sources)




