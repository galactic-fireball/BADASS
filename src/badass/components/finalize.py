from dataclasses import dataclass, field
from typing import Callable, List


def scale_fit_norm_finalize(ctx, val):
    return val * ctx.source.fit_norm


def scale_flux_norm_finalize(ctx, val):
    return val * ctx.source.flux_norm


def log_finalize(ctx, val):
    # TODO: support chains, so check for 0.0 in array
    if val == 0.0: return val
    return np.log10(val)


def redden_finalize(ctx, val):
    pass


def deredden_finalize(ctx, val):
    pass


@dataclass
class Finalizable:
    name: str
    finalizers: List[Callable] = field(default_factory=list)

    @staticmethod
    def get_finalizers(pname):
        if pname is None: return []

        # amplitudes
        if pname[-3:] == 'AMP':
            return [scale_fit_norm_finalize]

        # TODO: blob params: flux, lum, etc.
        return []


    def finalize(self, ctx, val=None):
        res_val = val
        for finalizer in self.finalizers:
            res_val = finalizer(ctx, val)
        print('finalizing: %s [%f -> %f]'%(self.name,val,res_val))
        return res_val

