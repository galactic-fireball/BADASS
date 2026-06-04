
# from badass.runner.tests import TestResult, TestRunner
from badass.runner.bootstrap import MLResult, MLRunner
# from badass.runner.mcmc import MCMCContext, MCMCResult, MCMCStage

from badass.utils import plotting

class BadassPipeline:

    @staticmethod
    def init(targets, cfg):
        print('BadassPipeline init')

        # Batch run IFU areas
        test_cfg = cfg
        if isinstance(test_cfg, list): test_cfg = test_cfg[0]
        if not test_cfg.fit.fit_area.type is None:
            from badass.runner.ifu import get_ifu_type
            pipeline_cls = get_ifu_type(test_cfg.fit.fit_area.type)
            if pipeline_cls is None:
                raise Exception('Unexpected area type: %s'%test_cfg.fit.fit_area.type)

            return pipeline_cls(targets, cfg)

        # Multiple non-IFU targets
        if isinstance(targets, list):
            from badass.runner.survey import SurveyPipeline
            return SurveyPipeline(targets, cfg)

        # Single target fitting, no tests
        if isinstance(cfg, list): cfg = cfg[0]
        return BadassPipeline(targets, cfg)


    def __init__(self, target, cfg, single=True):
        self.target = target
        self.cfg = cfg
        self.single = single
        self.result = None


    def run(self):
        print('BadassPipeline run')

        # TODO: make BadassPipeline
        # if self.cfg.fit.test_models:
        #     runner = TestRunner(self.target)
        #     runner.run()
        #     runner.finalize()

        if not self.cfg.fit.skip_bootstrap:
            runner = MLRunner(self.target, self.cfg)
            if not runner.target.valid:
                runner.log.error('Invalid target! Skipping!')
                return None
            runner.run()
            runner.finalize()
            self.result = runner.result
            return self.result

        # run mcmc

        # collect output


    def finalize(self):
        print('BadassPipeline finalize')
        if self.single:
            plotting.plot_ml_results(self.result, self.target)




