from dataclasses import dataclass, field, fields
from graphlib import TopologicalSorter, CycleError
import numexpr as ne
import numpy as np
import scipy.optimize as op
from tabulate import tabulate
from typing import Callable, Dict, List, NamedTuple

from badass.components.priors import lnprior_gaussian, lnprior_halfnorm, lnprior_jeffreys, lnprior_flat
prior_map = {'gaussian': lnprior_gaussian, 'halfnorm': lnprior_halfnorm, 'jeffreys': lnprior_jeffreys, 'flat': lnprior_flat}

from badass.components.finalize import Finalizable


@dataclass
class Parameter(Finalizable):
    pr: 'ParameterRegistry' = None
    source: str = ''
    value: [float,int] = np.nan # the current value being used in the model

    @property
    def is_free(self):
        return isinstance(self, FreeParameter)


    @staticmethod
    def create(**kwargs):
        expr = kwargs.get('expr', None)
        if expr is None:
            return None

        if not 'finalizers' in kwargs:
            pname = kwargs.get('name', None)
            kwargs['finalizers'] = Finalizable.get_finalizers(pname)

        # dict containing and init and plim values -> FreeParameter
        if isinstance(expr, dict):
            kwargs.update(expr)
            return FreeParameter.from_dict(**kwargs)

        # str expression to evaluate -> ExprParameter
        if isinstance(expr, str):
            return ExprParameter.from_dict(**kwargs)

        # known, constant value -> ConstParameter
        if isinstance(expr, (int,float,np.floating)):
            kwargs['value'] = float(expr)
            return ConstParameter.from_dict(**kwargs)

        return None


    @classmethod
    def from_dict(cls, **kwargs):
        valid_fields = {f.name for f in fields(cls)}
        cls_data = {k:v for k,v in kwargs.items() if k in valid_fields}
        return cls(**cls_data)


    def initialize(self, expr_dict, params):
        return True


    def finalize(self, ctx, val=None):
        tval = val if not val is None else self.value
        res_val = super().finalize(ctx,val=tval)
        if val is None:
            self.value = res_val
        return res_val


class PLim(NamedTuple):
    min: int | float
    max: int | float


@dataclass
class FreeParameter(Parameter):
    idx: int = -1 # index into theta

    init: str | float | int = None
    plim: PLim | List[str | float | int] = field(default_factory=list)
    prior: Dict = None
    has_prior: bool = False


    def initialize(self, expr_dict, params):
        if isinstance(self.init, str):
            self.init = ne.evaluate(self.init, expr_dict).item()

        if not isinstance(self.plim, PLim):
            mi, ma = self.plim
            if isinstance(mi, str):
                mi = ne.evaluate(mi, expr_dict).item()
            if isinstance(ma, str):
                ma = ne.evaluate(ma, expr_dict).item()
            self.plim = PLim(min=mi, max=ma)

        self.value = self.init
        return True


@dataclass
class ConstParameter(Parameter):
    pass


@dataclass
class ExprParameter(Parameter):
    expr: str = None

    # names of parameters needed for expression resolution
    dependencies: List[str] = field(default_factory=list)

    def initialize(self, expr_dict, params):
        self.dependencies = ne.necompiler.getExprNames(self.expr, {})[0]
        for dep in self.dependencies:
            if not dep in params:
                self.pr.ctx.log.error('Unknown dependency parameter: [%s], removing parameter: %s'%(dep,self.name))
                return False
        return True


    def update(self, params):
        param_dict = {pname:params[pname].value for pname in self.dependencies}
        self.value = ne.evaluate(self.expr, param_dict).item()


    def evaluate_chains(self, param_chains):
        return ne.evaluate(self.expr, param_chains)


class ParameterRegistry:

    def __init__(self, ctx):
        super().__init__()

        self.ctx = ctx
        self.params = {}

        self.free_params = {}
        self.theta = []
        self.free_count = 0

        self.expr_params = {}
        self.expr_order = []

        self.constraints = []


    def add_param(self, **kwargs) -> Parameter:
        param_name = kwargs.get('name', 'PARAM_%d'%len(self.params))

        if param_name in self.params:
            self.ctx.log.info('%s already in parameter registry, not adding'%param_name)
            return self.params[param_name]

        param = Parameter.create(pr=self, **kwargs)
        if param is None:
            self.ctx.log.error('Parameter creation failed')
            return None

        self.ctx.log.debug('Adding parameter: %s'%param)
        self.params[param.name] = param

        if param.is_free:
            param.idx = self.free_count
            self.free_params[param.name] = param
            self.free_count += 1

        if isinstance(param, ExprParameter):
            self.expr_params[param.name] = param

        return param


    def initialize(self, expr_dict):
        invalid = []
        for p in self.params.values():
            if not p.initialize(expr_dict, self.params):
                invalid.append(p.name)
        [self.params.pop(pname,None) for pname in invalid]

        expr_param_deps = {p.name:p.dependencies for p in self.expr_params.values()}
        ts = TopologicalSorter(expr_param_deps)
        try:
            self.expr_order = list(ts.static_order())
        except CycleError as e:
            self.ctx.error(e)
            return False

        # make sure only ExprParameters are in here
        self.expr_order = [ep for ep in self.expr_order if ep in self.expr_params]
        for pname in self.expr_order:
            self.params[pname].update(self.params)

        self.validate_constraints()


    def update(self, theta):
        for p in self.free_params.values():
            p.value = theta[p.idx]

        for pname in self.expr_order:
            self.params[pname].update(self.params)


    def fit_vector(self) -> np.ndarray:
        theta = np.zeros(self.free_count)
        for p in self.free_params.values():
            theta[p.idx] = p.value
        return theta


    def get_fit_bounds(self):
        lo = [p.plim.min for p in self.free_params.values()]
        hi = [p.plim.max for p in self.free_params.values()]
        return op.Bounds(lo, hi, keep_feasible=True)


    def get_param_dict(self):
        return {param.name:param.value for param in self.params.values()}


    def evaluate_chains(self, fp_chains, finalize=False):
        param_chains = {p.name:fp_chains[p.idx] for p in self.free_params.values()}
        for pname in self.expr_order:
            param_chains[pname] = self.params[pname].evaluate_chains(param_chains)

        for param in self.params.values():
            if param.name in param_chains:
                continue
            # should only be ConstParameters at this point
            param_chains[param.name] = np.full(len(fp_chains[0]), param.value)

        if finalize:
            for pname, chain in param_chains.items():
                param_chains[pname] = self.params[pname].finalize(self.ctx,val=chain)

        return param_chains


    def finalize(self):
        for param in self.params.values():
            param.finalize(self.ctx)


    @property
    def param_names(self):
        return list(self.params.keys())


    def get_lnpriors(self):
        lp_arr = [0.0 if p.plim[0] <= p.value <= p.plim[1] else -np.inf for p in self.free_params.values()]

        # Loop through soft constraints
        local_dict = self.get_param_dict()
        for expr1, expr2 in self.constraints:
            con_pass = ne.evaluate(expr1, local_dict=local_dict).item() - ne.evaluate(expr2, local_dict=local_dict).item() >= 0
            lp_arr.append(0.0 if con_pass else -np.inf)

        # Loop through parameters with priors on them
        for param in self.free_params.values():
            if param.prior is None:
                continue

            prior_type = param.prior['type']
            if not prior_type in prior_map: # TODO: validate elsewhere
                continue

            lp_arr += prior_map[prior_type](param.value, **self.get_param_hyperdict(param.name))

        return np.sum(lp_arr)


    def is_free(self, param_name) -> bool:
        return param_name in self.free_params


    def get_param(self, param_name):
        return self.params.get(param_name, None)


    def get_param_val(self, param_name):
        param = self.params.get(param_name, None)
        if (param is None) or (param.value is None):
            return np.nan
        return param.value


    def get_param_hyperdict(self, param_name):
        param = self.free_params.get(param_name, None)
        if param is None:
            return {}

        return {
            'init': param.init,
            'plim': param.plim,
            'prior': param.prior,
        }


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
            self.update(x)
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
            row.append(param.expr if isinstance(param, ExprParameter) else '')
            row.append(param.value)

            if not param.is_free:
                row.extend(['','','',])
                table.append(row)
                continue

            if not param.init is None:
                if isinstance(param.init, (float,int)):
                    row.append('%0.04f'%param.init)
                elif isinstance(param.init, str):
                    row.append('\'%s\''%param.init)
            else:
                row.append('----')

            if not param.plim is None:
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
        self.ctx.log.info('Total Free Parameters: %d' % self.free_count)

