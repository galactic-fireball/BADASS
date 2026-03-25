from dataclasses import dataclass, field, InitVar
import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from tabulate import tabulate
from typing import Callable, ClassVar, Dict, List

import badass.utils.utils as ba_utils
from badass.components.spectral_lines.utils import calculate_fwhm, calculate_w80


# components that make up the continuum
cont_comps = ['POWER', 'HOST_GALAXY', 'BALMER_CONT', 'APOLY', 'MPOLY',]


class BlobRegistry:

    def __init__(self, ctx):
        self.ctx = ctx
        self.blobs = []

        IndexBlob.register_index_blobs(self, self.ctx.target.wave[0], self.ctx.target.wave[-1])
        ContinuumBlob.register_cont_blobs(self)
        ContFracBlob.register_contfrac_blobs(self)

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
        self.ctx.log.debug('Registered %s: %s'%(blob.__class__.__name__, blob.name))


    def get_blobs(self):
        return self.blobs


    def get_blob_names(self):
        return [blob.name for blob in self.blobs]


    def get_blob(self, blob_name):
        for blob in self.blobs:
            if blob.name == blob_name:
                return blob
        return None


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
            'continuum': self.calc_cont(),
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


    def get_postfits(self, fit_results):
        results = {}
        for blob in self.blobs:
            results.update(blob.compute_postfit(self.ctx, fit_results))
        return results


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
                child.append('%0.04f'%cval)
                table.append(child)


        self.ctx.log.info('Current Blob Parameters:\n'+tabulate(table, headers, tablefmt='grid'))


@dataclass
class Blob:

    name: str = None
    func: Callable | None = None
    cur_val: float | Dict[str,float] = 0.0
    is_const: bool = False
    data: Dict = field(default_factory=dict)


    def compute(self, ctx, kwargs):
        self.cur_val = self.func(ctx, self.kwargs, self.data)
        return self.cur_val


    def compute_postfit(self, ctx, fit_results):
        return {}


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


# TODO: get rid of IndexBlob and store the needed index in the ContinuumBlob instance
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

            reg.register_blob(cls(name='INDEX_%d'%wave, wave=wave))


    def compute(self, ctx, kwargs):
        self.cur_val = float(ba_utils.find_nearest(ctx.target.wave,self.wave)[1])
        return self.cur_val


@dataclass
class ContinuumBlob(Blob):

    wave: int = 0
    idx: int = 0

    WAVES: ClassVar[List] = [1350, 3000, 5100]

    def __post_init__(self):
        self.idx = int(self.idx)

        self.cur_val = {
            'F_CONT_TOT_%d'%self.wave: 0.0,
            'F_CONT_AGN_%d'%self.wave: 0.0,
            'F_CONT_HOST_%d'%self.wave: 0.0,
            'L_CONT_TOT_%d'%self.wave: 0.0,
            'L_CONT_AGN_%d'%self.wave: 0.0,
            'L_CONT_HOST_%d'%self.wave: 0.0,
        }


    @classmethod
    def register_cont_blobs(cls, reg):
        for wave in ContinuumBlob.WAVES:
            idx = reg.get_blob('INDEX_%d'%wave)
            if idx is None:
                continue

            reg.register_blob(cls(name='L_CONT_%d'%wave, wave=wave, idx=idx.cur_val))


    @staticmethod
    def get_conts_at_idx(ctx, idx):
        cont_comps = {
            'TOT': ['POWER', 'HOST_GALAXY', 'BALMER_CONT', 'APOLY', 'MPOLY',],
            'AGN': ['POWER', 'BALMER_CONT', 'APOLY', 'MPOLY',],
            'HOST': ['HOST_GALAXY', 'APOLY', 'MPOLY',],
        }

        res = {}
        for type, comps in cont_comps.items():
            cont_at_idx = 0.0
            for key in comps:
                if not key in ctx.store.comps:
                    continue
                cont_at_idx += ctx.store.comps[key][idx]
            res[type] = cont_at_idx
        return res


    def compute(self, ctx, kwargs):
        conts = ContinuumBlob.get_conts_at_idx(ctx, self.idx)

        for segment in ['TOT', 'AGN', 'HOST']:
            flux = conts[segment]*ctx.target.flux_norm*ctx.target.fit_norm
            lum = 0.0
            if flux != 0.0:
                lum = np.log10(ctx.flux_to_lum(flux))
                flux = np.log10(flux)
            self.cur_val['F_CONT_%s_%d'%(segment, self.wave)] = flux
            self.cur_val['L_CONT_%s_%d'%(segment, self.wave)] = lum

        return self.cur_val


@dataclass
class ContFracBlob(ContinuumBlob):
    WAVES: ClassVar[List] = [4000, 7000]

    def __post_init__(self):
        self.idx = int(self.idx)

        self.cur_val = {
            'AGN_FRAC_%d'%self.wave: 0.0,
            'HOST_FRAC_%d'%self.wave: 0.0,
        }


    @classmethod
    def register_contfrac_blobs(cls, reg):
        for wave in ContFracBlob.WAVES:
            idx = reg.get_blob('INDEX_%d'%wave)
            if idx is None:
                continue

            reg.register_blob(cls(name='CONT_FRAC_%d'%wave, wave=wave, idx=idx.cur_val))


    def compute(self, ctx, kwargs):
        conts = ContinuumBlob.get_conts_at_idx(ctx, self.idx)
        for type in ['AGN', 'HOST']:
            self.cur_val[type+'_FRAC_%d'%self.wave] = conts[type] / conts['TOT']
        return self.cur_val


@dataclass
class ComponentBlob(Blob):

    comp_spec: np.ndarray = None

    obs_wave: ClassVar[np.ndarray] = None

    def __post_init__(self):
        self.cur_val = {
            self.name+'_FLUX': 0.0,
            self.name+'_LUM': 0.0,
            self.name+'_EW': 0.0,
        }


    @classmethod
    def register_comp_blobs(cls, reg, comps):
        ComponentBlob.obs_wave = ba_utils.redden(reg.ctx.fit_wave, z=reg.ctx.target.z)
        for comp in comps:
            reg.register_blob(cls(name=comp.upper()))


    def compute(self, ctx, kwargs):

        self.comp_spec = BlobRegistry.get_component(ctx, self.name)

        if np.all([self.comp_spec == 0.0]):
            self.cur_val[self.name+'_FLUX'] = 0.0
            self.cur_val[self.name+'_LUM'] = 0.0
            self.cur_val[self.name+'_EW'] = 0.0
            return self.cur_val

        flux = np.trapz(self.comp_spec, ComponentBlob.obs_wave)
        flux = np.abs(flux)*ctx.target.flux_norm*ctx.target.fit_norm
        self.cur_val[self.name+'_FLUX'] = np.log10(flux) if flux != 0.0 else flux

        self.cur_val[self.name+'_LUM'] = np.log10(ctx.flux_to_lum(flux)) if flux != 0.0 else 0.0

        cont = kwargs['continuum']
        ew = np.trapz(self.comp_spec/cont, ComponentBlob.obs_wave)
        self.cur_val[self.name+'_EW'] = ew if np.isfinite(ew) else 0.0

        return self.cur_val


@dataclass
class LineComponentBlob(ComponentBlob):

    line: object = None

    def __post_init__(self):
        super().__post_init__()
        self.cur_val.update({
            self.line.name+'_FWHM': 0.0,
            self.line.name+'_W80': 0.0,
            self.line.name+'_NPIX': 0.0,
            self.line.name+'_SNR': 0.0,
        })


    def compute(self, ctx, kwargs):
        super().compute(ctx, kwargs)

        self.cur_val[self.line.name+'_FWHM'] = calculate_fwhm(ctx.fit_wave, self.comp_spec, ctx.target.velscale)
        self.cur_val[self.line.name+'_W80'] = calculate_w80(ctx.fit_wave, self.comp_spec, self.line.center)

        # compute number of pixels (NPIX)
        # - the number of pixels of the line model that are above the raw noise
        self.cur_val[self.line.name+'_NPIX'] = len(np.where(np.abs(self.comp_spec) > ctx.fit_noise)[0])

        # compute the signal-to-noise ratio (SNR)
        # - the maximum value of the line model above the MEAN value of the noise within the channels
        self.cur_val[self.line.name+'_SNR'] = np.nanmax(np.abs(self.comp_spec)) / np.nanmean(ctx.fit_noise)

        return self.cur_val
        # TODO: add line window metrics


    def compute_postfit(self, ctx, fit_results):
        results = {}

        disp_res = self.line.disp_res_kms
        results[self.line.name+'_DISP_RES'] = {'med': disp_res, 'std': 0.0}

        disp_dict = fit_results[self.line.name+'_DISP']
        disp_corr = np.nanmax((0.0, np.sqrt(disp_dict['med']**2-disp_res**2)))
        results[self.line.name+'_DISP_CORR'] = {'med': disp_corr, 'std': disp_dict['std'], 'flag': disp_dict.get('flag',0)}

        fwhm_dict = fit_results[self.line.name+'_FWHM']
        fwhm_corr = np.nanmax((0.0, np.sqrt(fwhm_dict['med']**2-(2.3548*disp_res)**2)))
        results[self.line.name+'_FWHM_CORR'] = {'med': fwhm_corr, 'std': fwhm_dict['std'], 'flag': fwhm_dict.get('flag',0)}

        w80_dict = fit_results[self.line.name+'_W80']
        w80_corr = np.nanmax((0.0, np.sqrt(w80_dict['med']**2-(2.567*disp_res)**2)))
        results[self.line.name+'_W80_CORR'] = {'med': w80_corr, 'std': w80_dict['std'], 'flag': w80_dict.get('flag',0)}

        return results



@dataclass
class CombinedLineComponentBlob(LineComponentBlob):

    def __post_init__(self):
        super().__post_init__()
        self.cur_val.update({
            self.line.name+'_VOFF': 0.0,
            self.line.name+'_DISP': 0.0,
        })


    def compute(self, ctx, kwargs):
        super().compute(ctx, kwargs)

        # get the LineVel blob associated with this line
        line_vel = ctx.blob_reg.get_blob(self.name+'_LINE_VEL').cur_val
        vel = np.arange(len(ctx.fit_wave))*ctx.target.velscale - line_vel
        full_profile = np.abs(self.comp_spec)
        norm_profile = full_profile/np.sum(full_profile)
        voff = np.trapz(vel*norm_profile,vel)/simpson(norm_profile,vel)
        self.cur_val[self.line.name+'_VOFF'] = voff if np.isfinite(voff) else 0.0

        disp = np.sqrt(np.trapz(vel**2*norm_profile,vel)/np.trapz(norm_profile,vel) - (voff**2))
        self.cur_val[self.line.name+'_DISP'] = disp if np.isfinite(disp) else 0.0
        return self.cur_val

