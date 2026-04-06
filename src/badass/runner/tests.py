import copy
import numpy as np
import pandas as pd
from tabulate import tabulate


from badass.badass_utils.badass_test_suite import collect_test_metrics
from badass.stage.pipeline import BadassRunContext, BadassResult, BadassRunContext
# from badass.utils.config import SpecLine


class TestResult(BadassResult):

    def __init__(self):
        self.metrics = {}


class TestRunner(BadassRunContext):

    result_cls = TestResult

    def run(self):

        if isinstance(self.inputs, list):
            if len(self.inputs) == 1:
                self.inputs = self.inputs[0]
            else:
                raise Exception('Multiple inputs currently unsupported')

        if isinstance(self.cfg, list):
            if len(self.cfg) == 1:
                self.cfg = self.cfg[0]
            else:
                raise Exception('Multiple configs currently unsupported')

        if self.cfg.test.mode != 'line':
            raise Exception('Currently only line testing is supported')

        from badass import BadassRunContext

        # add null test
        test_target = copy.deepcopy(self.inputs)
        test_cfg = copy.deepcopy(self.cfg)
        test_cfg.io.output_dir = test_cfg.io.output_dir.joinpath('NULL_TEST')
        test_cfg.io.output_dir.mkdir(parents=True, exist_ok=True)
        test_target.outdir = test_cfg.io.output_dir
        test_target.cfg = test_cfg
        # TODO: this should be after multiprocessing for the logger?
        self.init_target(test_target)

        ctx = BadassRunContext(test_target)
        self.run_ctxs.append(ctx)


        for i, test_set in enumerate(self.cfg.test.test_sets):
            test_target = copy.deepcopy(self.inputs)
            test_cfg = copy.deepcopy(self.cfg)
            test_cfg.io.output_dir = test_cfg.io.output_dir.joinpath('TEST_%d'%i)
            test_cfg.io.output_dir.mkdir(parents=True, exist_ok=True)
            test_target.outdir = test_cfg.io.output_dir

            if not isinstance(test_set, list): test_set = [test_set,]
            test_cfg.extend_lines(test_set)
            test_target.cfg = test_cfg

            # TODO: this should be after multiprocessing for the logger?
            self.init_target(test_target)

            ctx = BadassRunContext(test_target)
            self.run_ctxs.append(ctx)

        # TODO: multiprocessing
        for ctx in self.run_ctxs:
            ctx.run()

        # gather metrics

        # if continue_fit:
        #   create new line list
        #   run ctx


    def finalize(self):
        data = pd.DataFrame(columns=['Name', 'ANOVA', 'BADASS', 'CHI2_RATIO', 'AIC', 'BIC', 'F_RATIO', 'SSR_RATIO', 'RCHI2_RATIO'])
        # 'TOT_RCHI2', 'WIN_RCHI2'

        # NULL TEST
        # TODO: add to data with single fit metrics

        for i, ctx in enumerate(self.run_ctxs[1:]):

            test_results = {k:np.nan for k in data.columns.to_list()}
            test_results['Name'] = 'TEST_%d'%i

            # prev_ctx = self.run_ctxs[i] # TODO: TEST_* name and run_ctxs idx is off by one because of NULL_TEST
            prev_ctx = self.run_ctxs[0] # compare with NULL_TEST
            test_results.update(collect_test_metrics(prev_ctx, ctx))
            data.loc[len(data)] = test_results

        data.to_csv(self.main_out_dir.joinpath('test_results.csv'), index=False)
        print(tabulate(data, headers='keys', tablefmt='grid', showindex=False))








# test_sets = []

# oiii_5007 = OIII_5007.copy()
# oiii_5007['children'] = [NA_OIII_5007,]
# test_sets.append([oiii_5007,])

# oiii_5007 = OIII_5007.copy()
# oiii_5007['children'] = [NA_OIII_5007, NA_OIII_5007_BLUE]
# test_sets.append([oiii_5007,])


# test = {
#     'mode': 'line',
#     'test_sets': test_sets,
#     'continue_fit': False,
# }