from dataclasses import dataclass, field
import pathlib
from typing import List

from badass.input.input import BadassSpec
from badass.runner.runner import BadassRunContext, BadassResult
# from badass.runner.tests import TestResult, TestRunner
from badass.runner.bootstrap import MLRunner
from badass.runner.mcmc import MCMCRunner
from badass.utils.config import BadassConfig
from badass.utils.logger import BadassLogger

from badass.utils import plotting

@dataclass
class BadassPipeline:

    source: BadassSpec = None
    cfg: BadassConfig = None
    log: BadassLogger = None
    outdir: pathlib.Path = None
    primary: bool = True
    runners: List[BadassRunContext] = field(default_factory=list)
    results: List[BadassResult] = field(default_factory=list)


    @staticmethod
    def init(source, cfg):
        log = BadassLogger()

        # Batch run IFU areas
        test_cfg = cfg
        if isinstance(test_cfg, list): test_cfg = test_cfg[0]
        if not test_cfg.fit.fit_area.type is None:
            from badass.runner.ifu import get_ifu_type
            pipeline_cls = get_ifu_type(test_cfg.fit.fit_area.type)
            if pipeline_cls is None:
                raise Exception('Unexpected area type: %s'%test_cfg.fit.fit_area.type)

            log.info('Starting %s'%pipeline_cls.__name__)
            return pipeline_cls(source=source, cfg=cfg)

        # Multiple non-IFU source
        if isinstance(source, list):
            from badass.runner.survey import SurveyPipeline
            log.info('Starting Survey Pipeline')
            return SurveyPipeline(source=source, cfg=cfg)

        # Single source fitting, no tests
        if isinstance(cfg, list): cfg = cfg[0]
        log.info('Starting Badass Pipeline')
        return BadassPipeline(source=source, cfg=cfg)


    def __post_init__(self):
        # TODO: self.start_time = time.time()

        if self.outdir is None:
            if not self.cfg.io.output_dir is None:
                self.outdir = self.cfg.io.output_dir
            elif isinstance(self.source, list):
                self.outdir = pathlib.Path('fit_result_%d'%int(time.time() * 1000))
            else:
                if not self.source.file is None:
                    self.outdir = self.source.file.with_suffix('')
                else:
                    self.outdir = pathlib.Path(os.getcwd()).resolve().joinpath(self.source.name)

            if not isinstance(self.source, list) and self.outdir.name != self.source.name:
                self.outdir = self.outdir.joinpath(self.source.name)

        if not self.outdir.is_absolute():
            self.outdir = pathlib.Path(os.getcwd()).resolve().joinpath(self.outdir)

        # TODO: status.json file to check if already completed fit

        self.outdir.mkdir(parents=True, exist_ok=True)
        self.log = BadassLogger(name=self.outdir.name)
        self.log.set_output(self.outdir)

        if self.primary:
            # Write all the buffered logs to a file
            BadassLogger().set_output(self.outdir)
            if not isinstance(self.source, list):
                self.source.log.set_output(self.outdir)

        self.log.info('Pipeline setup complete')


    def run(self):
        # TODO: just set up a list of runners to run
        # runners = []
        # # setup
        # runners[-1].final_runner = True

        # prev_result = None
        # for runner in runners:
        #     runner = Runner(source,cfg,prev_result=prev_result)
        #     runner.run()
        #     prev_result = runner.result

        # for runner in runners:
        #     runner.finalize()

        # results = [runner.result for runner in runners]

        # TODO: TestRunner

        if not self.cfg.fit.skip_bootstrap:
            runner = MLRunner(source=self.source, cfg=self.cfg, log=self.log, outdir=self.outdir)
            if not runner.source.valid:
                runner.log.error('Invalid source! Skipping! [%s]'%runner.source.err_log)
                return None
            runner.run()
            runner.finalize()
            self.results.append(runner.result)

        if not self.cfg.mcmc.mcmc_fit:
            return self.results

        runner = MCMCRunner(source=self.source, cfg=self.cfg, log=self.log, outdir=self.outdir, initial_theta=runner.result.final_theta)
        if not runner.source.valid:
            runner.log.error('Invalid source! Skipping!')
            return None
        runner.run()
        runner.finalize()
        self.results.append(runner.result)
        return self.results


    def finalize(self):
        if not self.primary:
            return

        for result in self.results:
            if result.PLOT_FUNC is None:
                continue
            # result.PLOT_FUNC(self.source)

