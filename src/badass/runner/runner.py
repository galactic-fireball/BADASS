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
from typing import NamedTuple

from spark.plot import add_ax_labels

from badass.components.params import ParameterRegistry
from badass.components.blobs import BlobRegistry
from badass.components.templates.common import initialize_templates
from badass.components.spectral_lines.spectral_line import SpectralLine
from badass.input.input import BadassSpec
from badass.utils import metrics, plotting
from badass.utils.config import BadassConfig
import badass.utils.constants as bc
from badass.utils.logger import BadassLogger, LogObjMixin
from badass.utils.utils import ccm_unred, get_ebv, emline_masker, log_rebin, metal_masker


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


    def to_record(self):
        res_dict = self.to_dict()
        res_dict.pop('name')
        return {self.name+'_'+k:v for k,v in res_dict.items()}


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
    param_cls = ParamResult
    parameter_file = 'par_table.fits'
    components_file = 'best_model_components.fits'
    best_fit_params_file = 'best_fit_params.txt'

    ctx: None
    name: str
    outdir: str | pathlib.Path = None

    final_theta: np.ndarray = None
    final_params: dict[str:ParamResult] = field(default_factory=dict)
    metrics: dict[str:float] = field(default_factory=dict)
    components: dict[str:list[float]] = field(default_factory=dict)
    meta_components: MetaComponents = None

    def __post_init__(self):
        if self.outdir is None and not self.ctx is None:
            self.outdir = self.ctx.cfg.io.output_dir
        self.outdir = self.outdir.joinpath(self.OUT_NAME)
        self.outdir.mkdir(parents=True, exist_ok=True)

        if not self.ctx is None:
            self.ctx.outdir = self.outdir


    @classmethod
    def from_output(cls, outdir, ctx=None):
        outdir = pathlib.Path(outdir).joinpath(cls.OUT_NAME)
        if not outdir.exists():
            print('Failed to find output directory: %s'%str(outdir))
            return None

        data = {'name':'','ctx':None,'outdir':outdir}
        pt_file = outdir.joinpath(cls.parameter_file)
        if not pt_file.exists():
            print('Failed to find parameter output file: %s'%str(pt_file))
            return None

        comp_file = outdir.joinpath(cls.components_file)
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
        tbl_out = tabulate(table, headers, tablefmt='grid')
        if self.ctx is None:
            print(tbl_out)
        else:
            self.ct.log.debug(tbl_out)


    def quick_view(self):
        fig, ax = plt.subplots()

        ax.step(self.meta_components.wave, self.meta_components.data, color='black', label='Data')
        ax.plot(self.meta_components.wave, self.meta_components.model, color='red', label='Model')

        for label, comp in self.components.items():
            ax.plot(self.meta_components.wave, comp, label=label)
        ax.legend()
        # TODO: flux and fit norm??
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
            self.components[key] = comp * self.ctx.fit_norm

        self.meta_components = MetaComponents(self.ctx.fit_wave, self.ctx.fit_flux, self.ctx.fit_err, self.ctx.model, self.ctx.fit_mask)
        self.meta_components.rescale(self.ctx.fit_norm)


    def perform_metrics(self):
        metrics.get_fit_metrics(self)


    def output(self):
        self.output_par_table()
        self.output_comps()
        if self.ctx.cfg.io.outputs.print_results:
            self.print_results()

        self.make_result_plots()


    def output_par_table(self):
        metadata = {
            'z': self.ctx.source.target.z,
            'med_noise': np.nanmedian(self.ctx.fit_err),
            'velscale': self.ctx.source.velscale,
            'fit_norm': self.ctx.fit_norm,
            'flux_norm': self.ctx.source.flux_norm,
        }
        metadata = metadata | self.metrics

        table = Table([param.to_dict() for param in self.final_params.values()], meta=metadata)
        table.write(self.outdir.joinpath(self.parameter_file), overwrite=True)


    def output_comps(self):
        table = Table(self.components | asdict(self.meta_components))
        table.write(self.outdir.joinpath(self.components_file), overwrite=True)


    def print_results(self):
        params = list(self.final_params.values())
        if len(params) == 0:
            return
        headers = [k for k in params[0].to_dict()]
        table = [list(p.to_dict().values()) for p in params]
        pt = tabulate(table, headers, tablefmt='grid')

        headers = ['Metric', 'Value']
        table = [(m[0], '%0.4f'%m[1]) for m in self.metrics.items()]
        mt = tabulate(table, headers, tablefmt='grid')

        with open(self.outdir.joinpath(self.best_fit_params_file), 'w') as outfile:
            outfile.write('**** Best Fit Parameters ****\n\n')
            outfile.write(pt)
            outfile.write('\n\n\n')
            outfile.write('**** Fit Metrics ****\n\n')
            outfile.write(mt)


    def make_result_plots(self):
        if self.ctx.cfg.io.plots.best_model:
            plotting.plot_best_model(self, outdir=self.outdir)


class FitReg(NamedTuple):
    min: float
    max: float

    def __str__(self):
        return f'({self.min}, {self.max})'


    def __repr__(self):
        return self.__str__()


@dataclass
class BadassRunContext(LogObjMixin):
    result_cls = BadassResult

    source: BadassSpec
    cfg: BadassConfig
    log: BadassLogger = None
    outdir: pathlib.Path = None

    fit_reg: FitReg = None
    fit_norm: float = 1.0

    # The spectral data currently being fit
    fit_wave: np.ndarray = None
    fit_flux: np.ndarray = None
    fit_err: np.ndarray = None
    fit_mask: np.ndarray = None
    model: np.ndarray = None

    err_log: str = None

    # current model components
    comps: dict = field(default_factory=dict)

    cosmology: LambdaCDM = None

    param_reg: ParameterRegistry = field(init=False)
    blob_reg: BlobRegistry = field(init=False)
    templates: dict = field(default_factory=dict)
    line_list: list = field(default_factory=list)

    result: BadassResult = field(init=False)


    def __post_init__(self):
        self.cosmology = LambdaCDM(**self.cfg.fit.cosmology.dict())

        if self.fit_wave is None:
            self.fit_wave = self.source.wave
        if self.fit_flux is None:
            self.fit_flux = self.source.flux
        if self.fit_err is None:
            self.fit_err = self.source.err

        self.set_fit_region()

        # Sanitize errors
        med_err = 1.0 if all(np.isnan(self.fit_err)) else np.nanmedian(self.fit_err)
        self.fit_err[(~np.isfinite(self.fit_flux)) | (~np.isfinite(self.fit_err))] = med_err
        self.fit_err[self.fit_err == 0] = med_err


        # Combine fit mask from different sources
        self.fit_mask = np.full(len(self.fit_wave), True)
        self.fit_mask[(~np.isfinite(self.fit_flux)) | (~np.isfinite(self.fit_err))] = False
        for m in self.cfg.user_mask:
            self.fit_mask[(self.fit_wave >= m[0]) & (self.fit_wave <= m[1])] = False
        if self.cfg.fit.mask_bad_pix:
            bad_pix = getattr(self, 'bad_pix', np.array([]))
            self.fit_mask[bad_pix] = False
        if self.cfg.fit.mask_emline:
            emline_mask = emline_masker(self.fit_wave,self.fit_flux,self.fit_err)
            self.fit_mask[emline_mask] = False
        if self.cfg.fit.mask_metal:
            metal_mask = metal_masker(self.fit_wave,self.fit_flux,self.fit_err)
            self.fit_mask[metal_mask] = False

        # Correct for galactic extinction
        ebv = get_ebv(self.source.target.ra, self.source.target.dec, dust_cache=self.cfg.io.dust_cache)
        self.fit_flux = ccm_unred(self.source.obs_wave, self.fit_flux, ebv)

        # Normalize the fit flux
        self.fit_norm = np.nanmax(self.fit_flux)
        self.fit_flux = self.fit_flux / self.fit_norm
        self.fit_err = self.fit_err / self.fit_norm


        # TODO: test
        # if self.cfg.get('pca', {}).get('do_pca',False):
        #     pca_reconstruction(self)


        if all(np.isnan(self.fit_flux)):
            self.err_log = '\'flux\' array is all nans, not running fit'
            return


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

        self.result = self.result_cls(self, self.source.name, outdir=self.outdir)


    def set_fit_region(self):
        self.fit_reg = FitReg(min=self.fit_wave[0], max=self.fit_wave[-1])
        self.log.info('Initial fitting region: {fr}'.format(fr=self.fit_reg))

        user_fit_reg = self.cfg.fit.fit_reg
        if isinstance(user_fit_reg, (tuple,list)):
            user_fit_reg = FitReg(*user_fit_reg)
            if user_fit_reg.min > user_fit_reg.max:
                self.err_log = 'Fitting boundaries overlap!'
                return

            if (user_fit_reg.min > self.fit_reg.max) or (user_fit_reg.max < self.fit_reg.min):
                self.err_log = 'Fitting region not available!'
                return

            if (user_fit_reg.min < self.fit_reg.min) or (user_fit_reg.max > self.fit_reg.max):
                self.log.warn('Input fitting region exceeds available wavelength range. BADASS will adjust your fitting range automatically...')
                self.log.warn('Input fitting range: %s'%str(user_fit_reg))
                self.log.warn('Available wavelength range: %s'%str(self.fit_reg))

            self.fit_reg = FitReg(np.max([user_fit_reg.min, self.fit_reg.min]), np.min([user_fit_reg.max, self.fit_reg.max]))
        elif (isinstance(user_fit_reg, str)) and (user_fit_reg == 'auto'):
            self.log.info('Auto setting fitting region')
            self.fit_reg = FitReg(np.max([user_fit_reg.min, self.fit_reg.min]), np.min([user_fit_reg.max, self.fit_reg.max]))
        else:
            self.err_log = 'Invalid fitting region'
            return

        # The lower limit of the spectrum must be the lower limit of our stellar templates
        # TODO: template function to let each template affect the fitting region?
        if self.cfg.comp.fit_losvd:
            min_losvd = bc.LOSVD_LIBRARIES[self.cfg.losvd.library].min_losvd
            max_losvd = bc.LOSVD_LIBRARIES[self.cfg.losvd.library].max_losvd
            if (self.fit_reg.min < min_losvd) or (self.fit_reg.max > max_losvd):
                self.log.warn('Warning: Fitting LOSVD requires wavelenth range between {mi} Å and {ma} Å for stellar templates. BADASS will adjust your fitting range to fit the LOSVD...'.format(mi=min_losvd, ma=max_losvd))
                self.log.warn('Available wavelength range: %s'%str(self.fit_reg))
            self.fit_reg = FitReg(np.max([min_losvd, self.fit_reg.min]), np.min([max_losvd, self.fit_reg.max]))

        self.log.info('New fitting region is {fr}'.format(fr=self.fit_reg))
        if (self.fit_reg.max - self.fit_reg.min) < bc.MIN_FIT_REGION:
            self.err_log = 'Fitting region too small! The fitting region must be at least {min_reg} A!'.format(min_reg=bc.MIN_FIT_REGION)
            return

        reg_mask = ((self.fit_wave >= self.fit_reg.min) & (self.fit_wave <= self.fit_reg.max))
        self.fit_wave = self.fit_wave[reg_mask]
        self.fit_flux = self.fit_flux[reg_mask]
        self.fit_err = self.fit_err[reg_mask]
        self.source.set_fit_region(self.fit_reg)


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
        fit_stat = self.cfg.fit.fit_stat

        data = self.fit_flux[self.fit_mask]
        model = self.model[self.fit_mask]
        err = self.fit_err[self.fit_mask]

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


