import numpy as np

from templates.common import BadassTemplate

class PolynomialTemplate(BadassTemplate):

    @classmethod
    def initialize_template(cls, ctx):
        if not ctx.options.comp_options.fit_poly:
            return None

        return cls(ctx)


    def __init__(self, ctx):
        self.ctx = ctx

        self.fit_apoly = False
        self.fit_mpoly = False

        if (self.ctx.options.poly_options.apoly.bool) and (self.ctx.options.poly_options.apoly.order >= 0):
            self.fit_apoly = True
            self.apoly_order = self.ctx.options.poly_options.apoly.order

        if (self.ctx.options.poly_options.mpoly.bool) and (self.ctx.options.poly_options.mpoly.order >= 0):
            self.fit_mpoly = True
            self.mpoly_order = self.ctx.options.poly_options.mpoly.order


    def initialize_parameters(self, params, args):
        if self.fit_apoly:
            self.ctx.log.info('- Fitting additive legendre polynomial component')
            for n in range(1, int(self.apoly_order)+1):
                    params['APOLY_COEFF_%d' % n] = {
                                                    'init':0.0,
                                                    'plim':(-1.0e2,1.0e2),
                                                   }
        if self.fit_mpoly:
            self.ctx.log.info('- Fitting multiplicative legendre polynomial component')
            for n in range(1, int(self.mpoly_order)+1):
                params['MPOLY_COEFF_%d' % n] = {
                                                'init':0.0,
                                                'plim':(-1.0e2,1.0e2),
                                               }


    def add_components(self, params, comp_dict, host_model):
        if self.fit_apoly:
            nw = np.linspace(-1, 1, len(self.ctx.fit_wave))
            coeff = np.empty(self.apoly_order+1)
            coeff[0] = 0.0
            for n in range(1, len(coeff)):
                coeff[n] = params['APOLY_COEFF_%d'%n]
            apoly = np.polynomial.legendre.legval(nw, coeff)

            comp_dict['APOLY'] = apoly
            host_model = host_model - apoly

        if self.fit_mpoly:
            nw = np.linspace(-1, 1, len(self.ctx.fit_wave))
            coeff = np.empty(mpoly_order+1)
            for n in range(1, len(coeff)):
                coeff[n] = params['MPOLY_COEFF_%d'%n]
            mpoly = np.polynomial.legendre.legval(nw, coeff)

            comp_dict['MPOLY'] = mpoly
            host_model = host_model * mpoly

        return host_model
