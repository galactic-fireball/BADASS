from astropy.cosmology import LambdaCDM
from astropy.table import Table
from dataclasses import asdict, dataclass, field, fields
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import shutil
from tabulate import tabulate
import time
from typing import Any

from spark.plot import add_ax_labels

from badass.components.params import ParameterRegistry
from badass.components.blobs import BlobRegistry
from badass.components.templates.common import initialize_templates
from badass.components.spectral_lines.spectral_line import SpectralLine
from badass.input.input import BadassSpec
from badass.utils.config import BadassConfig


# TODO: move to BadassLogger class
def make_logger(name, log_file=None):
    log = logging.getLogger('badass.%s'%name)
    log.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.addHandler(sh)

    if not log_file is None:
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setFormatter(formatter)
        log.addHandler(fh)

    return log


@dataclass
class ParamResult:
    name: str
    best_fit: float
    sigma: float
    flag: int

    def to_dict(self):
        res_dict = asdict(self)
        for k in list(res_dict.keys()):
            if isinstance(res_dict[k], dict):
                for kk, v in res_dict[k].items():
                    res_dict[k+'_'+kk] = v
                res_dict.pop(k)
        return res_dict


    @classmethod
    def from_chain(cls, name, chain):
        med = np.nanmedian(chain)
        std = np.nanstd(chain)
        if not np.isfinite(med): med = 0.0
        if not np.isfinite(std): std = 0.0
        return cls(name,med,std,0)


    @classmethod
    def from_data(cls, data):
        return cls(**data)


@dataclass
class MetaComponents:
    wave: np.ndarray
    data: np.ndarray
    noise: np.ndarray
    model: np.ndarray
    mask: np.ndarray
    resid: np.ndarray = None

    def __post_init__(self):
        self.resid = self.data - self.model


    def rescale(self, fit_norm):
        self.data *= fit_norm
        self.noise *= fit_norm
        self.model *= fit_norm
        self.resid *= fit_norm


@dataclass
class BadassResult:
    OUT_NAME = 'badass_result'
    PLOT_FUNC = None
    param_cls = ParamResult
    parameter_file = 'par_table.fits'
    components_file = 'best_model_components.fits'

    ctx: None
    name: str
    out_dir: str | pathlib.Path = None

    final_theta: np.ndarray = None
    final_params: dict[str:ParamResult] = field(default_factory=dict)
    metrics: dict[str:float] = field(default_factory=dict)
    components: dict[str:list[float]] = field(default_factory=dict)
    meta_components: MetaComponents = None

    def __post_init__(self):
        if self.out_dir is None:
            self.out_dir = self.ctx.cfg.io.output_dir.joinpath(self.OUT_NAME)
            self.out_dir.mkdir(parents=True, exist_ok=True)


    @classmethod
    def from_output(cls, out_dir, ctx=None):
        out_dir = pathlib.Path(out_dir).joinpath(cls.OUT_NAME)
        if not out_dir.exists():
            print('Failed to find output directory: %s'%str(out_dir))
            return None

        data = {'name':'','ctx':None,'out_dir':out_dir}
        pt_file = out_dir.joinpath(cls.parameter_file)
        if not pt_file.exists():
            print('Failed to find parameter output file: %s'%str(pt_file))
            return None

        comp_file = out_dir.joinpath(cls.components_file)
        if not comp_file.exists():
            print('Failed to find components file: %s'%str(comp_file))
            return None

        pt = Table.read(pt_file)
        data['final_params'] = {param['name']:cls.param_cls.from_data(dict(param)) for param in pt}
        # TODO: metrics
        metrics = {}

        comps = Table.read(comp_file)
        meta_names = [f.name for f in fields(MetaComponents)]
        data['components'] = {c:np.asarray(comps[c]) for c in comps.colnames if not c in meta_names}
        data['meta_components'] = MetaComponents(**{c:np.asarray(comps[c]) for c in comps.colnames if c in meta_names})

        return cls(**data)


    def dump(self):
        params = list(self.final_params.values())
        if len(params) == 0:
            return
        headers = [k for k in params[0].to_dict()]
        table = [list(p.to_dict().values()) for p in params]
        print(tabulate(table, headers, tablefmt='grid'))


    def quick_view(self):
        fig, ax = plt.subplots()

        ax.step(self.meta_components.wave, self.meta_components.data, color='black', label='Data')
        ax.plot(self.meta_components.wave, self.meta_components.model, color='red', label='Model')

        for label, comp in self.components.items():
            ax.plot(self.meta_components.wave, comp, label=label)
        ax.legend()
        add_ax_labels(ax, 'AA')#, yscale=int(np.log10(flux_norm)))
        plt.show()


    def finalize(self):
        self.collect_final_parameters()
        self.finalize_components()
        self.perform_metrics()
        self.output()


    def set_final_theta(self):
        pass


    def collect_final_parameters(self):
        pass


    def finalize_components(self):
        for key, comp in self.ctx.comps.items():
            self.components[key] = comp * self.ctx.source.fit_norm

        self.meta_components = MetaComponents(self.ctx.fit_wave, self.ctx.fit_flux, self.ctx.fit_err, self.ctx.model, self.ctx.source.fit_mask)
        self.meta_components.rescale(self.ctx.source.fit_norm)


    def perform_metrics(self):
        # METRICS - TODO
        # self.metrics = badass_test_suite.get_fit_test_results(ctx)
        pass


    def output(self):
        self.output_par_table()
        self.output_comps()


    def output_par_table(self):
        table = Table([param.to_dict() for param in self.final_params.values()], meta={
            'z': self.ctx.source.target.z,
            'med_noise': np.nanmedian(self.ctx.fit_err),
            'velscale': self.ctx.source.velscale,
            'fit_norm': self.ctx.source.fit_norm,
            'flux_norm': self.ctx.source.flux_norm,
        })

        table.write(self.out_dir.joinpath(self.parameter_file), overwrite=True)


    def output_comps(self):
        table = Table(self.components | asdict(self.meta_components))
        table.write(self.out_dir.joinpath(self.components_file), overwrite=True)


@dataclass
class BadassRunContext:
    result_cls = BadassResult

    source: BadassSpec
    cfg: BadassConfig
    # log: BadassLogger = None
    outdir: pathlib.Path = None

    fit_wave: np.ndarray = None
    fit_flux: np.ndarray = None
    fit_err: np.ndarray = None
    model: np.ndarray = None

    # current model components
    comps: dict = field(default_factory=dict)

    cosmology: LambdaCDM = None

    param_reg: ParameterRegistry = field(init=False)
    blob_reg: BlobRegistry = field(init=False)
    templates: dict = field(default_factory=dict)
    line_list: list = field(default_factory=list)

    result: BadassResult = field(init=False)


    def __post_init__(self):
        self.start_time = time.time()

        if self.outdir is None:
            if not self.cfg.io.output_dir is None:
                self.outdir = self.cfg.io.output_dir
            elif not source.file is None:
                self.outdir = source.file.with_suffix('')
            else:
                self.outdir = pathlib.Path(os.getcwd()).resolve().joinpath(self.source.name)
        if not self.outdir.is_absolute():
            self.outdir = pathlib.Path(os.getcwd()).resolve().joinpath(self.outdir)

        # TODO: implement fit status files
        if self.outdir.joinpath('results', 'mc_result', 'par_table.fits').exists():
            if self.cfg.io.overwrite:
                # TODO: set up tmp logger
                print('Removing old output directory: [%s]'%str(self.outdir))
                shutil.rmtree(str(self.outdir))
            else:
                self.source.valid = False
                self.source.err_log = 'Output directory [%s] already exists, not overwriting'%str(self.outdir)
                print(self.err_log)
                return

        self.outdir.mkdir(parents=True, exist_ok=True)
        log_dir = self.outdir.joinpath('log')
        log_dir.mkdir(parents=True, exist_ok=True) # TODO: 'log' mkdir eventually happens in separate output class

        self.log = make_logger(self.source.name, log_file=log_dir.joinpath('log.txt'))
        self.source.log = self.log # TODO: separate logger for source?

        self.cosmology = LambdaCDM(**self.cfg.fit.cosmology.dict())

        self.source.postinit()
        if not self.source.valid:
            return

        # The spectral data currently being fit
        if self.fit_wave is None:
            self.fit_wave = self.source.wave.copy()
        if self.fit_flux is None:
            self.fit_flux = self.source.flux.copy()
        if self.fit_err is None:
            self.fit_err = self.source.err.copy()

        max_flux = np.nanmax(self.fit_flux)*1.5
        median_flux = np.nanmedian(self.fit_flux)

        # For use in parameter/hyperpar expressions
        component_args = {
            'median_flux':median_flux, 'max_flux':max_flux,
            'min_wave':np.min(self.fit_wave), 'max_wave':np.max(self.fit_wave),
        }

        self.param_reg = ParameterRegistry(self)
        self.blob_reg = BlobRegistry(self)

        self.templates = initialize_templates(self)
        self.line_list = SpectralLine.initialize_spectral_lines(self, [line.dict() for line in self.cfg.user_lines])

        self.param_reg.initialize(component_args)
        self.param_reg.validate_constraints()

        self.param_reg.dump_parameters()
        self.blob_reg.dump_blobs()

        self.model = np.zeros_like(self.fit_flux)

        self.result = self.result_cls(self, self.source.name)


    def finalize(self):
        # refit the model with the best fit theta
        self.result.set_final_theta()
        self.param_reg.update(self.result.final_theta)
        self.fit_model()

        self.result.finalize()


    def lnprob_wrapper(self, fit_vals):
        if any([np.isnan(v) for v in fit_vals]):
            return np.inf

        self.param_reg.update(fit_vals)
        return -(self.lnprob()[0]) # only care about the first returned value


    def lnprob(self):
        # Log-probability function

        ll = self.lnlike()
        lp = self.param_reg.get_lnpriors()
        if not np.isfinite(lp):
            return -np.inf, ll

        # return log-prob and log-like:
        # bootstrap mode will ignore the latter, mcmc will return it as a blob
        return lp + ll, ll


    def lnlike(self):
        # Log-likelihood function

        self.fit_model()
        fit_mask = self.source.fit_mask
        fit_stat = self.cfg.fit.fit_stat

        data = self.fit_flux[fit_mask]
        model = self.model[fit_mask]
        err = self.fit_err[fit_mask]

        if fit_stat == 'ML':
            return -0.5*np.sum(((data-model)**2/err**2) + np.log(2*np.pi*err**2), axis=0)

        if fit_stat == 'OLS':
            return -np.sum((data - model)**2, axis=0)


    def fit_model(self):
        host_model = np.copy(self.fit_flux)

        self.comps = {}
        extra_comps = {}
        for line in self.line_list:
            host_model = line.add_components(self.comps, host_model, extra_comps)

        for template in self.templates.values():
            host_model = template.add_components(self.comps, host_model)

        # The final model
        self.model = np.sum(list(self.comps.values()), axis=0)

        # Add extra comps after we've computed the model
        self.comps.update(extra_comps)


