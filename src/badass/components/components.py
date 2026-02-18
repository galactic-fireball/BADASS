from dataclasses import dataclass, field
import numexpr as ne
from tabulate import tabulate
from typing import Dict, List

# TODO: make BaseModel instead?
@dataclass
class FitParameter:

    name: str = None
    expr: dict | str | float | int = None
    source: str = None

    is_free: bool = False
    value: [float,int] = None # the current value being used in the model

    # for free parameters
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

        if isinstance(self.plim, tuple):
            self.plim = list(self.plim) # mutability for expr -> numbers


    def add_dependent(self, dep_name):
        self.dependents.append(dep_name)
        self.is_shared = True


class ParameterRegistry:

    _registry = None

    def __new__(cls, ctx=None):

        if cls._registry:
            return cls._registry

        cls._registry = super().__new__(cls)

        cls._registry.ctx = ctx
        cls._registry.params = {}
        cls._registry.cur_idxs = {}
        return cls._registry


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
        return [p for p in self.params if p.is_free]


    def is_free(self, param_name) -> bool:
        if not param_name in self.params:
            return False
        return self.params[param_name].is_free


    def _init_values(self, expr_dict:dict, todo=[]) -> None:
        rerun = [] # if we find an expr that can't be fulfilled yet, rerun

        expr_dict.update({v.name:v.value for v in self.params.values() if not v.value is None})

        for param in self.params.values():
            if (len(todo) > 0) and (not param.name in todo):
                continue

            # do this first in case init is a str expr we need to eval each round
            if isinstance(param.expr, dict):
                param.expr = param.init # free parameter

            if isinstance(param.expr, (int,float)):
                param.value = param.expr # const parameter
                continue

            if isinstance(param.expr, str): # const parameter that needs evaluated
                if ne.validate(param.expr, local_dict=expr_dict):
                    # a term in the expression hasn't been evaluated yet
                    rerun.append(param.name)
                    continue

                param.value = ne.evaluate(param.expr, local_dict=expr_dict).item()
                expr_dict[param.name] = param.value

            if not param.is_free:
                continue

            for i in [0,1]:
                if isinstance(param.plim[i], str):
                    if ne.validate(param.plim[i], local_dict=expr_dict):
                        rerun.append(param.name)
                        continue

                    param.plim[i] = ne.evaluate(param.plim[i], local_dict=expr_dict).item()

            # TODO: need to numexpr any prior values?

        if len(rerun) == 0:
            # all params have been evaluated successfully
            return

        self._init_values(expr_dict, todo=rerun)


    def init_values(self, expr_dict:dict) -> None:
        valid_dict = expr_dict.copy()
        valid_dict.update({k:0 for k in self.params.keys()})

        for param in self.params.values():
            if (isinstance(param.expr, str)) and (ne.validate(param.expr, local_dict=valid_dict)):
                self.ctx.log.error('Parameter [%s] expr value: %s is invalid!'%(param.name,param.expr))
                param.expr = param.init

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

        self._init_values(expr_dict)


    def get_fit_values(self) -> list[float]:
        fp = self.get_free_parameters()

        self.cur_idxs = {}
        vals = []
        for i, param in enumerate(fp):
            vals.append(param.value)
            self.cur_idxs[param.name] = i


    def update_model_params(self, vals:list[float]) -> None:
        if len(vals) != len(self.cur_idxs):
            self.ctx.log.critical('Number of model parameters is mismatched from expected')

        for param_name, idx in self.cur_idxs.items():
            self.params[param_name].value = vals[idx]

        for param in self.params.values():
            if not isinstance(param.expr, str):
                continue

            # TODO: numexpr param.expr and put in param.value


    def get_param(self, param_name):
        if not param_name in self.params:
            return None
        return self.params[param_name]


    def get_param_val(self, param_name):
        param = self.get_param(param_name)
        if (param is None) or (param.value is None):
            return np.nan
        return param.value


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


class BadassComponent:

    def __init__(self, ctx):
        self.ctx = ctx
        self.pr = self.ctx.param_reg
        self.comp_params = []
        self.initialize_parameters()


    @classmethod
    def initialize_component(cls, ctx):
        return None


    def initialize_parameters(self):
        pass


    def add_components(self, params, comp_dict, host_model):
        return comp_dict, host_model


    def get_param(self, param_name):
        return np.nan

