from dataclasses import dataclass, field
import numexpr as ne
import numpy as np
import scipy.optimize as op
from tabulate import tabulate
from typing import Dict, List

from badass.components.priors import lnprior_gaussian, lnprior_halfnorm, lnprior_jeffreys, lnprior_flat
prior_map = {'gaussian': lnprior_gaussian, 'halfnorm': lnprior_halfnorm, 'jeffreys': lnprior_jeffreys, 'flat': lnprior_flat}


# TODO: handle all parameter finalization here (such as adjusting for fit_norm, etc.)
#   include mixin classes for various functionality
#   include a staticmethod to determine the parameter's class
#   Parameter -> FreeParameter, ConstParameter, ExprParameter, FitNormMixin, ReddenMixin, etc.
# Each class handles its own, initialization (class flag for "satisfied"), updating, etc.
# MetricParameter


# TODO: blobs are parameters as well, handle their own stuff:
#   class BlobParameter(FitParameter)


# TODO: Hyperpar NamedTuple with inner PLim Named Tuple


@dataclass
class FitParameter:

    name: str = None
    expr: dict | str | float | int = None
    source: str = None

    is_free: bool = False
    value: [float,int] = None # the current value being used in the model

    # for free parameters
    idx: int = -1
    init: str | float | int = None
    plim: List[str | float | int] = field(default_factory=list)
    prior: Dict = field(default_factory=dict)
    has_prior: bool = False

    # names of fit parameters whose value depends on self
    dependents: List[str] = field(default_factory=list)
    is_shared: bool = False
    # if self is a dependent on another fit parameter, self's parent's name
    parent: str = None


    def __post_init__(self):
        self.is_free = isinstance(self.expr, dict)
        self.has_prior = self.prior != {}
        self.is_shared = len(self.dependents) != 0
        if self.expr is None:
            self.expr = self.init
        if (self.value is None) and (isinstance(self.expr, (int,float))):
            self.value = self.expr

        if isinstance(self.plim, tuple):
            self.plim = list(self.plim) # mutability for expr -> numbers


    def add_dependent(self, dep_name):
        self.dependents.append(dep_name)
        self.is_shared = True


class ParameterRegistry:

    def __init__(self, ctx):
        super().__init__()

        self.ctx = ctx
        self.params = {}
        self.free_count = 0 # TODO: actually update and use this
        self.expr_dict = {}
        self.constraints = []


    def add_param(self, **kwargs) -> FitParameter:
        if ('name' in kwargs) and (kwargs['name'] in self.params):
            self.ctx.log.info('%s already in parameter registry, not adding'%kwargs['name'])
            return self.params[kwargs['name']]

        expr = kwargs.get('expr', None)
        if isinstance(expr, dict):
            # free parameter: add init, plim, prior to kwargs
            kwargs.update(expr)

        fp = FitParameter(**kwargs)
        if fp.name is None:
            fp.name = 'PARAM_%d'%len(self.params)

        self.params[fp.name] = fp
        return fp


    def get_free_parameters(self) -> list[FitParameter]:
        return [p for p in self.params.values() if p.is_free]


    @property
    def free_param_count(self):
        return len(self.get_free_parameters())


    @property
    def param_names(self):
        return list(self.params.keys())


    def get_prior_parameters(self) -> list[FitParameter]:
        return [p for p in self.params.values() if p.has_prior]


    def is_free(self, param_name) -> bool:
        if not param_name in self.params:
            return False
        return self.params[param_name].is_free


    @property
    def ntotal(self):
        return len(self.params)


    def _update_all(self, todo=[]) -> None:
        rerun = [] # if we find an expr that can't be fulfilled yet, rerun

        self.expr_dict.update({v.name:v.value for v in self.params.values() if not v.value is None})

        for param in self.params.values():
            if (len(todo) > 0) and (not param.name in todo):
                continue

            if param.expr is None:
                # param has already been resolved, skip
                pass

            elif isinstance(param.expr, (int,float)): # const parameter
                param.value = param.expr
                self.expr_dict[param.name] = param.value
                continue

            elif isinstance(param.expr, dict): # free parameter
                if isinstance(param.init, (int,float)):
                    # set value to the init hyperpar
                    param.value = param.init
                    self.expr_dict[param.name] = param.init
                    param.expr = None # no longer need to evaluate

                elif isinstance(param.init, str):
                    if ne.validate(param.init, local_dict=self.expr_dict):
                        # a term in the expression hasn't been resolved yet
                        rerun.append(param.name)
                        continue

                    param.init = ne.evaluate(param.init, local_dict=self.expr_dict).item()
                    param.value = param.init
                    self.expr_dict[param.name] = param.value
                    param.expr = None # no longer need to evaluate

            elif isinstance(param.expr, str): # parameter that needs to be evaluated
                if ne.validate(param.expr, local_dict=self.expr_dict):
                    # a term in the expression hasn't been resolved yet
                    rerun.append(param.name)
                    continue

                param.value = ne.evaluate(param.expr, local_dict=self.expr_dict).item()
                self.expr_dict[param.name] = param.value
                if param.is_free:
                    param.expr = None # no longer need to evaluate

            if not param.is_free:
                continue

            for i in [0,1]:
                if isinstance(param.plim[i], str):
                    if ne.validate(param.plim[i], local_dict=self.expr_dict):
                        rerun.append(param.name)
                        continue

                    param.plim[i] = ne.evaluate(param.plim[i], local_dict=self.expr_dict).item()

            # TODO: need to numexpr any prior values?

        if len(rerun) == 0:
            # all params have been evaluated successfully
            return

        self._update_all(todo=rerun)


    def init_values(self, expr_dict:dict) -> None:
        self.expr_dict = expr_dict
        valid_dict = self.expr_dict.copy()
        valid_dict.update({k:0 for k in self.params.keys()})

        for param in self.params.values():
            if (isinstance(param.expr, str)) and (ne.validate(param.expr, local_dict=valid_dict)):
                self.ctx.log.error('Parameter [%s] expr value: %s is invalid!'%(param.name,param.expr))
                param.expr = param.init
                # TODO: something more drastic? make free parameter?

            if not param.is_free:
                continue

            if (isinstance(param.init, str)) and (ne.validate(param.init, local_dict=valid_dict)):
                breakpoint()
                self.ctx.log.error('Parameter [%s] init value: %s is invalid!'%(param.name,param.init))
                param.init = 0.0
                param.expr = 0.0
                param.value = 0.0

            for i in [0,1]:
                if (isinstance(param.plim[i], str)) and (ne.validate(param.plim[i], local_dict=valid_dict)):
                    self.ctx.log.error('Parameter [%s] plim %d value: %s is invalid!'%(param.name,i,param.plim[i]))
                    param.plim[i] = 0.0

        self._update_all()

        for idx, p in enumerate(self.get_free_parameters()):
            p.idx = idx


    def fit_vector(self) -> np.ndarray:
        fp = self.get_free_parameters()
        theta = np.zeros(len(fp))
        for p in fp:
            theta[p.idx] = p.value
        return theta


    def update_vals(self, theta:np.ndarray) -> None:
        for param in self.params.values():
            param.value = None

        fp = self.get_free_parameters()
        for p in fp:
            p.value = theta[p.idx]

        self._update_all()


    def get_fit_bounds(self):
        fp = self.get_free_parameters()
        lo = [p.plim[0] for p in fp]
        hi = [p.plim[1] for p in fp]
        return op.Bounds(lo, hi, keep_feasible=True)


    def get_lnpriors(self):
        fp = self.get_free_parameters()
        lp_arr = [0.0 if p.plim[0] <= p.value <= p.plim[1] else -np.inf for p in fp]

        # Loop through soft constraints
        local_dict = self.get_param_dict()
        for expr1, expr2 in self.constraints:
            con_pass = ne.evaluate(expr1, local_dict=local_dict).item() - ne.evaluate(expr2, local_dict=local_dict).item() >= 0
            lp_arr.append(0.0 if con_pass else -np.inf)

        # Loop through parameters with priors on them
        for param in self.get_prior_parameters():
            prior_type = param.prior['type']
            if not prior_type in prior_map: # TODO: validate elsewhere
                continue

            lp_arr += prior_map[prior_type](param.value, **self.get_param_hyperdict(param.name))

        return np.sum(lp_arr)


    def get_param(self, param_name):
        if not param_name in self.params:
            return None
        return self.params[param_name]


    def get_param_val(self, param_name):
        param = self.get_param(param_name)
        if (param is None) or (param.value is None):
            return np.nan
        return param.value


    def get_param_hyperdict(self, param_name):
        param = self.get_param(param_name)
        if (param is None) or (not param.is_free):
            return {}

        return {
            'init': param.init,
            'plim': param.plim,
            'prior': param.prior,
        }


    def get_param_dict(self):
        return {param.name:param.value for param in self.params.values()}


    def validate_constraints(self):
        local_dict = self.get_param_dict()

        self.constraints = []
        for con in self.ctx.cfg.user_constraints:
            if any([ne.validate(c, local_dict=local_dict) for c in con]):
                self.ctx.log.info('%s constraint removed because one or more free parameters not available'%con)
                continue

            val1 = ne.evaluate(con[0],local_dict=local_dict).item()
            val2 = ne.evaluate(con[1],local_dict=local_dict).item()
            if val1 < val2:
                self.ctx.log.info('%s constraint removed because it is violated by the initial values'%con)
                continue

            self.constraints.append(con)
        self.ctx.cfg.user_constraints = self.constraints


    def get_constraints(self):
        def eval_con(x, self, expr1, expr2):
            self.update_vals(x)
            local_dict = self.get_param_dict()
            r1 = ne.evaluate(expr1, local_dict=local_dict).item()
            r2 = ne.evaluate(expr2, local_dict=local_dict).item()
            return r1 - r2

        return [{'type':'ineq', 'fun':eval_con, 'args':(self, con[0], con[1])} for con in self.constraints]


    def dump_parameters(self) -> None:
        headers = ['Parameter', 'Source', 'Free?', 'Expr', 'Current Value', 'init', 'plim', 'prior']
        table = []

        for param in self.params.values():
            row = []
            row.append(param.name)
            row.append(param.source if not param.source is None else 'UNK')
            row.append('YES' if param.is_free else 'NO')
            row.append(param.expr)
            row.append(param.value)

            if (param.is_free) and (not param.init is None):
                if isinstance(param.init, (float,int)):
                    row.append('%0.04f'%param.init)
                elif isinstance(param.init, str):
                    row.append('\'%s\''%param.init)
            else:
                row.append('----')

            if (param.is_free) and (not param.plim is None):
                plimstr = '('
                for i in [0,1]:
                    if isinstance(param.plim[i], (float,int)):
                        plimstr += '%0.04f' % param.plim[i]
                    elif isinstance(param.plim[i], str):
                        plimstr += '\'%s\'' % param.plim[i]
                    if i == 0:
                        plimstr += ', '
                plimstr += ')'
                row.append(plimstr)
            else:
                row.append('----')

            if param.has_prior:
                row.append(param.prior.get('type', 'UNK'))
            else:
                row.append('----')

            table.append(row)

        self.ctx.log.info('Current Parameters:\n'+tabulate(table, headers, tablefmt='grid'))
        self.ctx.log.info('Total Free Parameters: %d'%len(self.get_free_parameters()))
