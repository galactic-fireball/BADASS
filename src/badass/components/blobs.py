from dataclasses import dataclass, field, InitVar
import numpy as np
from scipy.interpolate import interp1d
from tabulate import tabulate
from typing import Callable, ClassVar, Dict, List

import badass.utils.utils as ba_utils


# components that make up the continuum
cont_comps = ['POWER', 'HOST_GALAXY', 'BALMER_CONT', 'APOLY', 'MPOLY',]


class BlobRegistry:

    def __init__(self, ctx):
        self.ctx = ctx
        self.blobs = []

        IndexBlob.register_index_blobs(self, self.ctx.target.wave[0], self.ctx.target.wave[-1])

        # compute the values for all the constant blobs
        for blob in self.blobs:
            if not blob.is_const:
                continue
            blob.compute(self.ctx, {})


    @staticmethod
    def get_component(ctx, comp):
        comp_store = ctx.store.comps
        if not comp in comp_store:
            return None
        return comp_store[comp]


    def register_blob(self, blob):
        self.blobs.append(blob)


    def get_blobs(self):
        return self.blobs


    def get_blob_names(self):
        return [blob.name for blob in self.blobs]


    def calc_cont(self):
        cont = np.zeros(len(self.ctx.fit_wave))
        for comp_key in cont_comps:
            comp_val = BlobRegistry.get_component(self.ctx, comp_key)
            if comp_val is None:
                continue
            cont += comp_val
        return cont


    def compute_all(self):

        kwargs = {
            'continuum': self.calc_cont()
        }

        res = {}
        for blob in self.blobs:
            val = blob.cur_val
            if not blob.is_const:
                val = blob.compute(self.ctx, kwargs)

            if isinstance(val, float):
                res[blob.name] = val
            elif isinstance(val, dict):
                res.update(val)

        return res


    def dump_blobs(self):
        headers = ['Name', 'Type', 'Const?', 'Value']
        table = []

        for blob in self.blobs:
            row = []
            row.append(blob.name)
            row.append(blob.__class__.__name__)
            row.append('YES' if blob.is_const else 'NO')

            if isinstance(blob.cur_val, (float,int)):
                row.append(blob.cur_val)
                table.append(row)
                continue

            row.append('MULT')
            table.append(row)

            # cur_val is a dict, add children rows
            for ckey, cval in blob.cur_val.items():
                child = []
                child.append('\t'+ckey)
                child.append('----')
                child.append('YES' if blob.is_const else 'NO')
                child.append(cval)
                table.append(child)


        self.ctx.log.info('Current Blob Parameters:\n'+tabulate(table, headers, tablefmt='grid'))


@dataclass
class Blob:

    name: str = None
    func: Callable | None = None
    cur_val: float | Dict[str,float] = 0.0
    is_const: bool = False


    def compute(self, ctx, kwargs):
        self.cur_val = self.func(ctx, self.kwargs)
        return self.cur_val


@dataclass
class LineVelBlob(Blob):

    is_const: bool = True
    center: float = 0.0

    interp_ftn: ClassVar[Callable] = None
    ctx: InitVar = None

    def __post_init__(self, ctx):
        self.name = self.name + '_LINE_VEL'

        if (LineVelBlob.interp_ftn is None) and (not ctx is None):
            LineVelBlob.interp_ftn = interp1d(ctx.target.wave, np.arange(len(ctx.target.wave))*ctx.target.velscale, kind='linear', bounds_error=False)


    def compute(self, ctx, kwargs):
        if (self.center is None) or (LineVelBlob.interp_ftn is None):
            ctx.log.warning('Error computing line velocity blob!')
            return np.nan

        self.cur_val = float(LineVelBlob.interp_ftn(self.center))
        return self.cur_val


@dataclass
class IndexBlob(Blob):

    is_const: bool = True
    wave: int = 0

    WAVES: ClassVar[List] = [1350, 3000, 4000, 5100, 7000]

    @classmethod
    def register_index_blobs(cls, reg, wave_min, wave_max):
        for wave in IndexBlob.WAVES:
            if (wave < wave_min) or (wave > wave_max):
                continue

            reg.register_blob(cls(wave=wave))


    def __post_init__(self):
        self.name = 'INDEX_%d'%self.wave


    def compute(self, ctx, kwargs):
        self.cur_val = float(ba_utils.find_nearest(ctx.target.wave,self.wave)[1])
        return self.cur_val


@dataclass
class ComponentBlob(Blob):

    component: str = ''

    def __post_init__(self):
        self.component = self.component.upper()
        self.name = self.component

        self.cur_val = {
            'FLUX': 0.0,
            'LUM': 0.0,
            'EW': 0.0,
        }


    @classmethod
    def register_comp_blobs(cls, reg, comps):
        for comp in comps:
            reg.register_blob(cls(component=comp))


    def compute(self, ctx, kwargs):
        obs_wave = ba_utils.redden(ctx.fit_wave, z=ctx.target.z)
        spec = BlobRegistry.get_component(ctx, self.component)

        if np.all([spec == 0.0]):
            self.cur_val['FLUX'] = 0.0
            self.cur_val['LUM'] = 0.0
            self.cur_val['FLUX'] = 0.0
            return self.cur_val

        flux = np.trapz(spec, obs_wave)
        flux = np.abs(flux)*ctx.target.flux_norm*ctx.target.fit_norm
        self.cur_val['FLUX'] = np.log10(flux) if flux != 0.0 else flux

        self.cur_val['LUM'] = np.log10(ctx.flux_to_lum(flux)) if flux != 0.0 else 0.0

        cont = kwargs['continuum']
        ew = np.trapz(spec/cont, obs_wave)
        self.cur_val['EW'] = ew if np.isfinite(ew) else 0.0

        return self.cur_val

