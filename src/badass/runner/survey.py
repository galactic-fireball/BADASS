import copy
import jinja2
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tabulate import tabulate

from badass.runner.pipeline import BadassPipeline
from badass.utils import plotting



REPORT_HTML_HEADER = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>BADASS Results</title>
    <style>
        * {box-sizing: border-box;}
        thead { background: #9E9EEA; }
        th, td { border: 1px solid lightgrey; padding: 0.25rem 1.25rem; }
        tbody tr:nth-child(even) { background: #D0D0F5}
        .target-report { display: grid; place-items: center; }
        .best-fit-plot { padding: 30px; }
        .params-table { display: flex; flex-direction: row; justify-content: space-around; gap: 5rem; padding: 30px}
    </style>
</head>
<body>
'''

REPORT_HTML_TEMPLATE = '''
<div class="target-report">
    <h2>{{ target_name }}</h2>
    <img class="best-fit-plot" src={{ plot_src }} />
    <div class="params-table">
        {{ params_table_data }}
    </div>
</div>
'''

REPORT_HTML_FOOTER = '''
</body>
</html>
'''


def pipeline_run(target, target_cfg):
    return BadassPipeline(target, target_cfg, single=False).run()


class SurveyPipeline(BadassPipeline):

    def __init__(self, target, cfg):
        super().__init__(target, cfg)

        self.single_targets = {}
        self.target_results = {}


    def run(self):
        print('SurveyPipeline run')

        target_cfgs = self.cfg
        if isinstance(self.cfg, list):
            self.cfg = copy.deepcopy(self.cfg[0])

        for i, target in enumerate(self.target):
            print('Running %s'%target.name)

            if isinstance(target_cfgs, list):
                target_cfg = target_cfgs[i]
            else:
                target_cfg = copy.deepcopy(self.cfg)
            target_cfg.io.output_dir = target_cfg.io.output_dir.joinpath(target.name)
            # TODO: check if already complete
            target_cfg.io.output_dir.mkdir(parents=True, exist_ok=True)
            self.single_targets[target.name] = (target, target_cfg)

        if self.cfg.io.nprocesses == 1:
            for target, target_cfg in list(self.single_targets.values()):
                res = pipeline_run(target, target_cfg)
                if not res is None:
                    self.target_results[target.name] = res
        else:
            p = Pool(processes=self.cfg.io.nprocesses, maxtasksperchild=1)
            run_results = p.starmap(pipeline_run, list(self.single_targets.values()), chunksize=1)
            p.close()

            for res in run_results:
                if not res is None:
                    self.target_results[res.name] = res


    def finalize(self):
        self.make_target_plots()
        self.make_survey_csv()
        self.make_report_html()
        self.make_report_pdf()


    def make_target_plots(self):
        for res in self.target_results.values():
            res.figures['ml_fit'] = plotting.plot_ml_results(res, self.single_targets[res.name][0])


    def make_survey_csv(self):
        data_rows = []
        for res in self.target_results.values():
            row_data = {'target': res.name}
            for param, param_dict in res.params.items():
                std_label = param + '_STD'
                row_data[param] = round(param_dict['med'], 4)
                row_data[std_label] = round(param_dict['std'], 4)
            data_rows.append(row_data)

        df = pd.DataFrame(data_rows)
        df.to_csv(self.cfg.io.output_dir.joinpath('survey_results.csv'), index=False)


    def make_report_html(self):
        table_cols = 3

        html = REPORT_HTML_HEADER
        environment = jinja2.Environment()
        template = environment.from_string(REPORT_HTML_TEMPLATE)

        for res in self.target_results.values():
            df = pd.DataFrame(columns=['Parameter', 'Best Fit', 'Std. Dev.'])

            for param, param_dict in res.params.items():
                row_data = {'Parameter':param, 'Best Fit':round(param_dict['med'],4), 'Std. Dev.':round(param_dict['std'],4)}
                df.loc[len(df)] = row_data

            plot_src = res.out_dir.joinpath('max_likelihood_fit.png').relative_to(self.cfg.io.output_dir)

            table_data = ''
            num_rows = int(np.ceil(len(df) / table_cols))
            for i in range(0, len(df), num_rows):
                ei = min(i+num_rows, len(df))
                tdf = df.iloc[i:ei]
                table_data += tdf.to_html(index=False, justify='center')
                table_data += '\n'

            html += template.render(target_name=res.name, plot_src=str(plot_src), params_table_data=table_data)

        html += REPORT_HTML_FOOTER
        with open(self.cfg.io.output_dir.joinpath('survey_results.html'), 'w') as out:
            out.write(html)


    def make_report_pdf(self):
        pdf = PdfPages(self.cfg.io.output_dir.joinpath('survey_results.pdf'))
        for res in self.target_results.values():
            fit_fig = res.figures.get('ml_fit', plotting.plot_ml_results(res, self.single_targets[res.name][0]))
            pdf.savefig(fit_fig)
        pdf.close()


def main():
    import pathlib
    import sys
    import zipfile

    if len(sys.argv) < 2:
        print('Need output directory')
        return

    out_dir = pathlib.Path(sys.argv[1])
    if not out_dir.is_absolute():
        out_dir = pathlib.Path.cwd().joinpath(out_dir)

    zfile = zipfile.ZipFile(pathlib.Path.cwd().joinpath(out_dir.name+'_res_plots.zip'), 'w')

    zfile.write(out_dir.joinpath('survey_results.html'), arcname='survey_results.html')
    for path in out_dir.glob('**/max_likelihood_fit.png'):
        zfile.write(path, arcname=path.relative_to(out_dir))
    zfile.close()


if __name__ == '__main__':
    main()

