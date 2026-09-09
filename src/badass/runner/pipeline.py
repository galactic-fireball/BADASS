from dataclasses import dataclass, field
from typing import List

from badass.input.input import BadassSpec
from badass.runner.runner import BadassResult
# from badass.runner.tests import TestResult, TestRunner
from badass.runner.bootstrap import MLRunner
from badass.runner.mcmc import MCMCRunner
from badass.utils.config import BadassConfig

from badass.utils import plotting

@dataclass
class BadassPipeline:

    source: BadassSpec = None
    cfg: BadassConfig = None
    single: bool = True
    results: List[BadassResult] = field(default_factory=list)


    @staticmethod
    def init(source, cfg):
        # Batch run IFU areas
        test_cfg = cfg
        if isinstance(test_cfg, list): test_cfg = test_cfg[0]
        if not test_cfg.fit.fit_area.type is None:
            from badass.runner.ifu import get_ifu_type
            pipeline_cls = get_ifu_type(test_cfg.fit.fit_area.type)
            if pipeline_cls is None:
                raise Exception('Unexpected area type: %s'%test_cfg.fit.fit_area.type)

            return pipeline_cls(source=source, cfg=cfg)

        # Multiple non-IFU source
        if isinstance(source, list):
            from badass.runner.survey import SurveyPipeline
            return SurveyPipeline(source=source, cfg=cfg)

        # Single source fitting, no tests
        if isinstance(cfg, list): cfg = cfg[0]
        return BadassPipeline(source=source, cfg=cfg)


    def run(self):
        print('BadassPipeline run')
        # TODO: just set up a list of runners to run

        if not self.cfg.fit.skip_bootstrap:
            runner = MLRunner(source=self.source, cfg=self.cfg)
            if not runner.source.valid:
                runner.log.error('Invalid source! Skipping! [%s]'%runner.source.err_log)
                return None
            runner.run()
            runner.finalize()
            self.results.append(runner.result)

        if not self.cfg.mcmc.mcmc_fit:
            return self.results

        # run mcmc
        runner = MCMCRunner(source=self.source, cfg=self.cfg, initial_theta=runner.result.final_theta)
        if not runner.source.valid:
            runner.log.error('Invalid source! Skipping!')
            return None
        runner.run()
        runner.finalize()
        self.results.append(runner.result)
        return self.results


    def finalize(self):
        if self.single:
            for result in self.results:
                if result.PLOT_FUNC is None:
                    continue
                result.PLOT_FUNC(self.source)

