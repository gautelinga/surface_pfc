import dolfin as df
from .cmd import mpi_rank
import random
import sympy as sp
import numpy as np
from ufl import max_value, min_value


# Tweaked from Oasis
class NdFunction(df.Function):
    """Vector function used for postprocessing.

    Assign data from ListTensor components using FunctionAssigner.
    """

    def __init__(self, u, name="Assigned Vector Function"):

        self.u = u
        S = u[0].function_space()
        mesh = S.mesh()
        constrained_domain = S.dofmap().constrained_domain
        Sd = df.FunctionSpace(mesh,
                              df.MixedElement(
                                  [_u.function_space().ufl_element()
                                   for _u in u]),
                              constrained_domain=constrained_domain)

        df.Function.__init__(self, Sd, name=name)
        self.fa = [df.FunctionAssigner(Sd.sub(i), S) for i, _u in enumerate(u)]

    def __call__(self):
        for i, _u in enumerate(self.u):
            self.fa[i].assign(self.sub(i), _u)


# Class representing the intial conditions.
class RandomInitialConditions(df.UserExpression):
    def __init__(self, u_, **kwargs):
        self.size = len(u_)
        super().__init__(**kwargs)

    def eval(self, values, x):
        for i in range(self.size):
            values[i] = 0.0
        values[0] = 2*random.random()-1

    def value_shape(self):
        return (self.size,)

# Class representing the intial conditions. Wavenumber hardcoded for now.
class AroundInitialConditions(df.UserExpression):
    def __init__(self, u_, **kwargs):
        self.size = len(u_)
        super().__init__(**kwargs)

    def eval(self, values, x):
        for i in range(self.size):
            values[i] = 0.0
        values[0] = 0.5*np.sin(x[1]/np.sqrt(2))

    def value_shape(self):
        return (self.size,)

# Class representing the intial conditions. Wavenumber hardcoded for now.
class AlongInitialConditions(df.UserExpression):
    def __init__(self, u_, **kwargs):
        self.size = len(u_)
        super().__init__(**kwargs)

    def eval(self, values, x):
        for i in range(self.size):
            values[i] = 0.0
        values[0] = 0.5*np.sin(x[0]/np.sqrt(2))

    def value_shape(self):
        return (self.size,)

# Class representing the intial conditions. Wavenumber hardcoded for now.
class CircularInitialConditions(df.UserExpression):
    def __init__(self, u_, **kwargs):
        self.size = len(u_)
        super().__init__(**kwargs)

    def eval(self, values, x):
        for i in range(self.size):
            values[i] = 0.0
        values[0] = np.sin(np.sqrt(x[0]**2+x[1]**2)/np.sqrt(2))
        #values[0] = np.sin(1.15*np.sqrt(x[0]**2+x[1]**2)/np.sqrt(2))

    def value_shape(self):
        return (self.size,)

# Class representing the intial conditions for manufactured solution
class MMSInitialConditions(df.UserExpression):
    def __init__(self, u_, geo_map, **kwargs):
        self.size = len(u_)
        self.map = geo_map
        super().__init__(**kwargs)

    def eval(self, values, x):
        for i in range(self.size):
            values[i] = 0.0
        #values[0] = 0.5*np.sin(x[0]/np.sqrt(2))
        #values[0] = self.map.psiMMS
        values[0] = (np.sin(x[1]/np.sqrt(2)))**2 + (np.sin(x[0]/np.sqrt(2)))**2

    def value_shape(self):
        return (self.size,)


class QuarticPotential:
    def __init__(self):
        self.Psi, self.Tau = sp.symbols('psi tau', real=True)
        w = self.Tau/2*self.Psi**2 + self.Psi**4/4
        dw = sp.diff(w, self.Psi)
        ddw = sp.diff(dw, self.Psi)
        self.f_w = sp.lambdify([self.Psi, self.Tau], w)
        self.f_dw = sp.lambdify([self.Psi, self.Tau], dw)
        self.f_ddw = sp.lambdify([self.Psi, self.Tau], ddw)

    def derivative_linearized(self, c_, c_1, tau):
        return self.f_dw(c_1, tau) + self.f_ddw(c_1, tau)*(c_-c_1)

    def derivative_stab(self, c_, c_1, tau):
        w_2 = self.Psi**2/2
        w_4 = self.Psi**4/4
        dw_2 = sp.lambdify([self.Psi], sp.diff(w_2, self.Psi))
        dw_4 = sp.lambdify([self.Psi], sp.diff(w_4, self.Psi))
        tau_pos = max_value(0., tau)
        tau_neg = min_value(0., tau)
        return dw_4(c_) + tau_pos * dw_2(c_) + tau_neg * dw_2(c_1)

    def __call__(self, c_, tau):
        return self.f_w(c_, tau)
