import dolfin as df
from ufl.tensors import ListTensor
from .cmd import mpi_rank
import random


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
