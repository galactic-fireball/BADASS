import copy
from dataclasses import dataclass, field
import jinja2
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
import numpy as np
import pandas as pd
import shutil
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
        .source-report { display: grid; place-items: center; }
        .best-fit-plot { padding: 30px; }
        .params-table { display: flex; flex-direction: row; justify-content: space-around; gap: 5rem; padding: 30px}
    </style>
</head>
<body>
'''

REPORT_HTML_TEMPLATE = '''
<div class="source-report">
    <h2>{{ source_name }}</h2>
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


def pipeline_run(source, source_cfg):
    return BadassPipeline(source, source_cfg, single=False).run()


def skip_existing(outdir, overwrite):
    # TODO: check for fit completed
    if not outdir.joinpath('badass_result', 'mc_result', 'par_table.fits').exists():
        return False

    if overwrite:
        print('Removing old output directory: [%s]'%str(outdir))
        shutil.rmtree(str(outdir))
        return False

    print('Output directory [%s] already exists, not overwriting'%str(outdir))
    return True


@dataclass
class SurveyPipeline(BadassPipeline):

    single_sources: dict = field(default_factory=dict)
    source_results: dict = field(default_factory=dict)

    def __post_init__(self):
        print('SurveyPipeline __post_init__')

        source_cfgs = self.cfg
        if isinstance(self.cfg, list):
            self.cfg = copy.deepcopy(self.cfg[0])

        for i, source in enumerate(self.sources):
            print('Running %s'%source.name)

            if isinstance(source_cfgs, list):
                source_cfg = source_cfgs[i]
            else:
                source_cfg = copy.deepcopy(self.cfg)

            source_out_dir = source_cfg.io.output_dir.joinpath(source.name)
            if skip_existing(source_out_dir, source_cfg.io.overwrite):
                continue

            source_cfg.io.output_dir = source_out_dir
            source_out_dir.mkdir(parents=True, exist_ok=True)
            self.single_sources[source.name] = (source, source_cfg)


    def run(self):
        print('SurveyPipeline run')
        if self.cfg.io.nprocesses == 1:
            for source, source_cfg in list(self.single_sources.values()):
                res = pipeline_run(source, source_cfg)
                if not res is None:
                    self.source_results[source.name] = res
        else:
            p = Pool(processes=self.cfg.io.nprocesses, maxtasksperchild=1)
            run_results = p.starmap(pipeline_run, list(self.single_sources.values()), chunksize=1)
            p.close()

            for res in run_results:
                if not res is None:
                    self.source_results[res.name] = res


    def finalize(self):
        self.make_source_plots()
        self.make_survey_csv()
        self.make_report_html()
        self.make_report_pdf()


    def make_source_plots(self):
        for res in self.source_results.values():
            res.figures['ml_fit'] = plotting.plot_ml_results(res, self.single_sources[res.name][0])


    def make_survey_csv(self):
        data_rows = []
        for res in self.source_results.values():
            row_data = {'source': res.name}
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

        for res in self.source_results.values():
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

            html += template.render(source_name=res.name, plot_src=str(plot_src), params_table_data=table_data)

        html += REPORT_HTML_FOOTER
        with open(self.cfg.io.output_dir.joinpath('survey_results.html'), 'w') as out:
            out.write(html)


    def make_report_pdf(self):
        pdf = PdfPages(self.cfg.io.output_dir.joinpath('survey_results.pdf'))
        for res in self.source_results.values():
            fit_fig = res.figures.get('ml_fit', plotting.plot_ml_results(res, self.single_sources[res.name][0]))
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

