import numpy as np

from badass.components.templates.common import BadassTemplate

class PolynomialTemplate(BadassTemplate):

    OPTION_NAME = 'poly'
    PARAM_PREFIX = '' # APOLY_ or MPOLY_ will be used below
    TEMPLATE_PARAMS = []

    @classmethod
    def initialize_component(cls, ctx):
        if not ctx.cfg.comp.fit_poly:
            return None

        if (ctx.cfg.poly.apoly_order <= 0) and (ctx.cfg.poly.mpoly_order <= 0):
            return None

        return cls(ctx)


    def __init__(self, ctx):
        self.fit_apoly = False
        self.fit_mpoly = False

        # Since we don't know the polynomial orders ahead of time to populate TEMPLATE_PARAMS,
        # do that first and then call the common __init__
        poly_cfg = ctx.cfg.poly
        self.apoly_order = int(poly_cfg.apoly_order)
        if self.apoly_order > 0:
            self.fit_apoly = True

            # If the coefficients were supplied in a list, unwrap them to multiple parameter entries
            if isinstance(poly_cfg.apoly_coeff,(list,tuple)):
                coeffs = poly_cfg.apoly_coeff
                if len(coeffs) != self.apoly_order:
                    ctx.log.warning('Provided %d apoly coeff vals, but order is %d. Updating order'%(len(coeffs),self.apoly_order))
                    self.apoly_order = len(coeffs)

                for n in range(1,self.apoly_order+1):
                    poly_cfg['apoly_coeff_%d'%n] = coeffs[n-1]
            else: # otherwise, make them all the same value (hyperpar dict or Number)
                for n in range(1,self.apoly_order+1):
                    poly_cfg['apoly_coeff_%d'%n] = poly_cfg.apoly_coeff

            self.TEMPLATE_PARAMS.extend(['apoly_coeff_%d'%n for n in range(1,self.apoly_order+1)])

        self.mpoly_order = int(poly_cfg.mpoly_order)
        if self.mpoly_order > 0:
            self.fit_mpoly = True

            # If the coefficients were supplied in a list, unwrap them to multiple parameter entries
            if isinstance(poly_cfg.mpoly_coeff,(list,tuple)):
                coeffs = poly_cfg.mpoly_coeff
                if len(coeffs) != self.mpoly_order:
                    ctx.log.warning('Provided %d mpoly coeff vals, but order is %d. Updating order'%(len(coeffs),self.mpoly_order))
                    self.mpoly_order = len(coeffs)

                for n in range(1,self.mpoly_order+1):
                    poly_cfg['mpoly_coeff_%d'%n] = coeffs[n-1]
            else: # otherwise, make them all the same value (hyperpar dict or Number)
                for n in range(1,self.mpoly_order+1):
                    poly_cfg['mpoly_coeff_%d'%n] = poly_cfg.mpoly_coeff

            self.TEMPLATE_PARAMS.extend(['mpoly_coeff_%d'%n for n in range(1,self.mpoly_order+1)])

        # common template in initialization
        super().__init__(ctx)


    def add_components(self, comp_dict, host_model):
        if self.fit_apoly:
            nw = np.linspace(-1, 1, len(self.ctx.fit_wave))
            coeff = np.empty(self.apoly_order+1)
            coeff[0] = 0.0
            for n in range(1, len(coeff)):
                coeff[n] = self.get_param('apoly_coeff_%d'%n)
            apoly = np.polynomial.legendre.legval(nw, coeff)

            comp_dict['APOLY'] = apoly
            host_model = host_model - apoly

        if self.fit_mpoly:
            nw = np.linspace(-1, 1, len(self.ctx.fit_wave))
            coeff = np.empty(self.mpoly_order+1)
            coeff[0] = 0.0
            for n in range(1, len(coeff)):
                coeff[n] = self.get_param('mpoly_coeff_%d'%n)
            mpoly = np.polynomial.legendre.legval(nw, coeff)

            comp_dict['MPOLY'] = mpoly
            host_model = host_model * mpoly

        return host_model
