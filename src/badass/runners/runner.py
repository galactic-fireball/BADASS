from badass.input.input import BadassInput
from badass.utils.config import BadassConfig


class BadassRunner:

    def __init__(self, inputs, cfg):
        print('RUNNER INIT: %s'%self.__class__.__name__)
        self.inputs = inputs
        self.cfg = cfg
        self.main_out_dir = self.cfg.io.output_dir
        self.run_ctxs = []


    @staticmethod
    def init_target(target):
        if not target.valid:
            return

        # create a new logger for this process
        target.set_new_logger()
        target.postinit()

        if not target.valid:
            return


    def run(self):
        self.target = self.inputs
        self.init_target(self.target)

        from badass import BadassRunContext
        ctx = BadassRunContext(self.target)
        ctx.run()


    def finalize(self):
        pass


class BatchRunner(BadassRunner):
    pass


def run_BADASS(inputs, **kwargs):
    nprocesses = kwargs.get('nprocesses', 1)
    multiprocess = kwargs.get('multiprocess', nprocesses > 1)

    cfg = BadassConfig.get_config_from_args(kwargs)
    targets = BadassInput.get_inputs(inputs, cfg)

    if len(targets) == 0:
        # TODO: master logger
        print('No valid targets to process, returning')
        return

    runner = init_runner(targets, cfg)
    runner.run()
    runner.finalize()


def init_runner(targets, cfg):
    # Decide which Runner should be in charge

    # Batch run model tests
    if cfg.fit.test_models:
        from badass.runners.model_runner import ModelRunner
        return ModelRunner(targets, cfg)


    # Batch run IFU areas
    if not cfg.fit.fit_area.type is None:
        from badass.runners.ifu_runner import get_runner
        runner_cls = get_runner(cfg.fit.fit_area.type)
        if runner_cls is None:
            raise Exception('Unexpected area type: %s'%cfg.fit.fit_area.type)

        return runner_cls(targets, cfg)


    # Multiple non-IFU targets
    if len(targets) > 1:
        from badass.runners.survey_runner import SurveyRunner
        return SurveyRunner(targets, cfg)


    # Single target fitting, no tests
    if isinstance(cfg, list): cfg = cfg[0]
    return BadassRunner(targets[0], cfg)
