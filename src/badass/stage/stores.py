from astropy.io import fits
from dataclasses import asdict, dataclass, field
import json
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from typing import ClassVar, Dict

from badass.badass_utils import badass_test_suite
import badass.utils.plotting as plotting

@dataclass
class MetaComps:
    data: np.ndarray
    wave: np.ndarray
    noise: np.ndarray
    model: np.ndarray
    resid: np.ndarray


    def finalize(self, fit_norm):
        self.data *= fit_norm
        self.noise *= fit_norm
        self.model *= fit_norm
        self.resid *= fit_norm



@dataclass
class StageStore:

    OUT_NAME: ClassVar[str] = 'store_result'

    ctx: object = None
    params: Dict[str,np.ndarray] = field(default_factory=dict)
    blobs: Dict[str,np.ndarray] = field(default_factory=dict)
    comps: Dict[str,np.ndarray] = field(default_factory=dict)
    meta_comps: MetaComps = None
    metrics: Dict[str,np.ndarray] = field(default_factory=dict)
    # TODO: fit_results should be a dataframe or other data structure?
    fit_results: Dict = field(default_factory=dict)


    def to_dict(self):
        # The dataclass asdict function isn't great, making our own
        return {}


    # TODO: better place for this?
    def update_metrics(self):
        self.metrics['R_SQUARED'] = badass_test_suite.r_squared(self.meta_comps.data, self.meta_comps.model)
        self.metrics['RCHI_SQUARED'] = badass_test_suite.r_chi_squared(self.meta_comps.data, self.meta_comps.model, self.ctx.fit_noise, len(self.ctx.param_reg.get_free_parameters()))


    def save_iter(self):
        pass


    def compile_results(self):
        self.fit_results = {}


    def dump_results(self):
        headers = ['param', 'med', 'std']
        table = []
        for param, param_dict in self.fit_results.items():
            row = [param, param_dict['med'], param_dict['std']]
            table.append(row)

        self.ctx.log.info('Fit Results:\n'+tabulate(table, headers, tablefmt='grid'))


    def get_outfile(self):
        outfile = self.ctx.target.outdir.joinpath('results', self.OUT_NAME)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        return outfile


    def output(self):
        outfile = self.get_outfile()

        with open(outfile.with_suffix('.json'), 'w') as f:
            json.dump(self.to_dict(), f, indent=4)


        plt.style.use('default')
        fig, ax = plt.subplots()
        ax.plot(self.meta_comps.wave, self.meta_comps.data, linewidth=1.0, color='black')
        ax.plot(self.meta_comps.wave, self.meta_comps.model, linewidth=1.0, color='red')
        plt.savefig(outfile.with_suffix('.png'))



@dataclass
class TestStore(StageStore):
    pass


@dataclass
class BasinhopStore(StageStore):
    OUT_NAME: ClassVar[str] = 'basinhop_result'


@dataclass
class MCStore(StageStore):
    OUT_NAME: ClassVar[str] = 'mc_result'

    niters: int = 0
    cur_iter: int = 0
    init_done: bool = False

    params_chain: Dict[str,np.ndarray] = field(default_factory=dict)
    blobs_chain: Dict[str,np.ndarray] = field(default_factory=dict)
    metrics_chain: Dict[str,np.ndarray] = field(default_factory=dict)


    def save_iter(self, result):

        self.ctx.param_reg.update_vals(result['x'])
        self.params = self.ctx.param_reg.get_param_dict()
        self.ctx.fit_model()
        self.blobs = self.ctx.blob_reg.compute_all()
        self.metrics['LOG_LIKE'] = result['fun']
        self.update_metrics()


        chain_pairs = [
            (self.params, self.params_chain),
            (self.blobs, self.blobs_chain),
            (self.metrics, self.metrics_chain),
        ]

        if not self.init_done:
            for container, chain in chain_pairs:
                for item in container.keys():
                    chain[item] = np.zeros(self.niters+1)

            self.init_done = True

        # TODO: better way to do this?
        for container, chain in chain_pairs:
            for item_name, item_val in container.items():
                chain[item_name][self.cur_iter] = item_val

        self.cur_iter += 1


    def compile_results(self):
        super().compile_results()        

        def add_fit_result(key, vals):
            med = np.nanmedian(vals)
            std = np.nanstd(vals)
            if not np.isfinite(med): med = 0.0
            if not np.isfinite(std): std = 0.0
            self.fit_results[key] = {'med':med, 'std':std}
            return med, std


        for key, vals in self.params_chain.items():
            med, std = add_fit_result(key, vals)

            param = self.ctx.param_reg.get_param(key)
            if not param.is_free:
                continue

            flag = 0
            if med-std <= param.plim[0]: flag += 1
            if med+std >= param.plim[1]: flag += 1
            self.fit_results[key]['flag'] = flag


        for key, vals in self.blobs_chain.items():
            add_fit_result(key, vals)

        self.fit_results.update(self.ctx.blob_reg.get_postfits(self.fit_results))

        for key, vals in self.metrics_chain.items():
            add_fit_result(key, vals)


        # Rescale amplitudes
        for pname, param_dict in self.fit_results.items():
            if pname[-4:] != '_AMP':
                continue
            param_dict['med'] *= self.ctx.target.fit_norm
            param_dict['std'] *= self.ctx.target.fit_norm

        for comp in self.comps.values():
            comp *= self.ctx.target.fit_norm

        self.meta_comps.finalize(self.ctx.target.fit_norm)


    def output(self):
        super().output()

        outdir = self.get_outfile()
        outdir.mkdir(parents=True, exist_ok=True)

        col1 = fits.Column(name='parameter', format='30A', array=list(self.fit_results.keys()))
        col2 = fits.Column(name='best_fit', format='E', array=[v['med'] for v in self.fit_results.values()])
        col3 = fits.Column(name='sigma', format='E', array=[v['std'] for v in self.fit_results.values()])
        cols = fits.ColDefs([col1,col2,col3])
        table_hdu = fits.BinTableHDU.from_columns(cols)

        hdr = fits.Header()
        hdr['z'] = self.ctx.target.z
        hdr['med_noise'] = np.nanmedian(self.ctx.fit_noise)
        hdr['velscale'] = self.ctx.target.velscale
        hdr['fit_norm'] = self.ctx.target.fit_norm
        hdr['flux_norm'] = self.ctx.target.flux_norm

        primary = fits.PrimaryHDU(header=hdr)
        hdu = fits.HDUList([primary, table_hdu])
        hdu.writeto(outdir.joinpath('par_table.fits'), overwrite=True)

        cols = []
        for key, val in self.comps.items():
            cols.append(fits.Column(name=key.upper(), format='E', array=val))

        for key, val in asdict(self.meta_comps).items():
            cols.append(fits.Column(name=key.upper(), format='E', array=val))

        mask = np.zeros(len(self.meta_comps.wave), dtype=bool)
        mask[self.ctx.target.fit_mask] = True
        cols.append(fits.Column(name='MASK', format='E', array=mask))

        cols = fits.ColDefs(cols)
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.writeto(outdir.joinpath('best_model_components.fits'), overwrite=True)

        plotting.plot_ml_results(self)
        self.ctx.log.info('Done ML fitting %s!' % self.ctx.target.cfg.io.output_dir)





@dataclass
class MCMCStore(StageStore):
    pass
