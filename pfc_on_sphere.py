import dolfin as df
from maps import SphereMap, EllipsoidMap
from common.io import dump_xdmf, Timeseries
import random
import numpy as np


# Class representing the intial conditions
class InitialConditions(df.UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + df.MPI.rank(df.MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = (0.5 - random.random())
        values[1] = 0.0

    def value_shape(self):
        return (2,)


R = 30.0
res = 100
dt = 0.1
tau = 0.2
ell = 1.0  # 10.0/(2*np.pi*np.sqrt(2))

geo_map = EllipsoidMap(0.75*R, 0.75*R, 1*R)
geo_map.initialize_ref_space(res)
# ref_mesh = geo_map.ref_mesh
geo_map.initialize_metric()
xyz = geo_map.coords()
dump_xdmf(xyz)

W = geo_map.mixed_space((geo_map.ref_el, geo_map.ref_el))

# Define trial and test functions
du = df.TrialFunction(W)
q, v = df.TestFunctions(W)

# Define functions
u = df.TrialFunction(W)
u_ = df.Function(W, name="u_")  # current solution
u_1 = df.Function(W, name="u_1")  # solution from previous converged step

# Split mixed functions
dc, dmu = df.split(du)
c,  mu = df.split(u)
c_1, mu_1 = df.split(u_1)

# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
u_1.interpolate(u_init)


def w_lin(vc_, vc_1, vtau):
    return vtau*vc_ + vc_1**3 + 3*vc_1**2*(vc_-vc_1)


# Brazovskii-Swift (non-conserved PFC with dc/dt = -delta F/delta c)
# Semi-covariantized version (zero thickness) (i.e. the inner products now involve the metric a0)
F_c_L = geo_map.form(
    1/dt * q * (c - c_1) - 4 * ell**2 * geo_map.dotgrad(mu, q))
F_c_NL = geo_map.form(w_lin(c, c_1, tau) * q)
F_mu = geo_map.form(ell**2 * geo_map.dotgrad(c, v) + (mu - c)*v)
F = F_c_L + F_c_NL + F_mu

a = df.lhs(F)
L = df.rhs(F)

# SOLVER
problem = df.LinearVariationalProblem(a, L, u_)
solver = df.LinearVariationalSolver(problem)

#solver.parameters["linear_solver"] = "gmres"
#solver.parameters["preconditioner"] = "jacobi"

df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True

# Output file
cfile = Timeseries("c")
mufile = Timeseries("mu")

# Step in time
t = 0.0
it = 0
T = 100.0

c_, mu_ = u_1.split()
c_.rename("c", "tmp")
mu_.rename("mu", "tmp")
cfile.write(c_, it)
mufile.write(mu_, it)

while t < T:
    it += 1
    t += dt

    solver.solve()

    u_1.assign(u_)
    if it % 1 == 0:
        c_, mu_ = u_.split()
        c_.rename("c", "tmp")
        mu_.rename("mu", "tmp")
        cfile.write(c_, it)
        mufile.write(mu_, it)
