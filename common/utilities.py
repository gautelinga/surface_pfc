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


class TimeStepSelector(df.Constant):
    def __init__(self, value):
        self.chop_factor = 2
        df.Constant.__init__(self, value)

    def get(self):
        return float(self.values())

    def set(self, value):
        self.assign(value)

    def chop(self):
        self.assign(self.get()/self.chop_factor)


# Set tau
def anneal_func(t, tau_0, tau_ramp, t_ramp):
    dtau = (tau_0 - tau_ramp)/2
    tau_avg = (tau_0 + tau_ramp)/2
    k = np.pi/t_ramp
    return dtau*np.cos(k*t) + tau_avg
