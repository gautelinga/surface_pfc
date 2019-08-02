import dolfin as df
from .cmd import mpi_rank
import random
import sympy as sp


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


# Class representing the intial conditions
class RandomInitialConditions(df.UserExpression):
    def __init__(self, u_, **kwargs):
        random.seed(2 + mpi_rank())
        self.size = len(u_)
        super().__init__(**kwargs)

    def eval(self, values, x):
        for i in range(self.size):
            values[i] = 0.0
        values[0] = 2*random.random()-1

    def value_shape(self):
        return (self.size,)


class QuarticPotential:
    def __init__(self):
        Psi, Tau = sp.symbols('psi tau')
        w = Tau/2*Psi**2 + Psi**4/4
        dw = sp.diff(w, Psi)
        ddw = sp.diff(dw, Psi)
        self.f_w = sp.lambdify([Psi, Tau], w)
        self.f_dw = sp.lambdify([Psi, Tau], dw)
        self.f_ddw = sp.lambdify([Psi, Tau], ddw)

    def derivative_linearized(self, c_, c_1, tau):
        return self.f_dw(c_1, tau) + self.f_ddw(c_1, tau)*(c_-c_1)

    def __call__(self, c_, tau):
        return self.f_w(c_, tau)
