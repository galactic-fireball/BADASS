from dataclasses import dataclass, field
import json
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from typing import ClassVar, Dict

from badass.badass_utils import badass_test_suite


@dataclass
class MetaComps:
    data: np.ndarray
    wave: np.ndarray
    noise: np.ndarray
    model: np.ndarray
    resid: np.ndarray


@dataclass
class StageStore:

    OUT_NAME: ClassVar[str] = 'store_result'

    ctx: object = None
    params: Dict[str,np.ndarray] = field(default_factory=dict)
    blobs: Dict[str,np.ndarray] = field(default_factory=dict)
    comps: Dict[str,np.ndarray] = field(default_factory=dict)
    meta_comps: MetaComps = None
    metrics: Dict[str,np.ndarray] = field(default_factory=dict)
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


    def output(self):
        outfile = self.ctx.target.outdir.joinpath('results', self.OUT_NAME)
        outfile.parent.mkdir(parents=True, exist_ok=True)

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

        for key, vals in self.params_chain.items():
            med = np.nanmedian(vals)
            std = np.nanstd(vals)
            if not np.isfinite(med): med = 0.0
            if not np.isfinite(std): std = 0.0
            self.fit_results[key] = {'med':med, 'std':std}

            param = self.ctx.param_reg.get_param(key)
            if not param.is_free:
                continue

            flag = 0
            if med-std <= param.plim[0]: flag += 1
            if med+std >= param.plim[1]: flag += 1
            self.fit_results[key]['flag'] = flag


        for key, vals in self.blobs_chain.items():
            med = np.nanmedian(vals)
            std = np.nanstd(vals)
            if not np.isfinite(med): med = 0.0
            if not np.isfinite(std): std = 0.0
            self.fit_results[key] = {'med':med, 'std':std}


        self.fit_results.update(self.ctx.blob_reg.get_postfits(self.fit_results))



@dataclass
class MCMCStore(StageStore):
    pass
