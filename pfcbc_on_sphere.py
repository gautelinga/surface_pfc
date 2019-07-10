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
        return (5,)


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

W = geo_map.mixed_space((geo_map.ref_el,
                         geo_map.ref_el,
                         geo_map.ref_el,
                         geo_map.ref_el,
                         geo_map.ref_el))

# Define trial and test functions
du = df.TrialFunction(W)
xi, etatt, etats, etast, etass = df.TestFunctions(W)

# Define functions
u = df.TrialFunction(W)
u_ = df.Function(W, name="u_")  # current solution
u_1 = df.Function(W, name="u_1")  # solution from previous converged step

# Split mixed functions
dpsi, dmu_tt, dmu_ts, dmu_st, dmu_ss = df.split(du)
psi,  mu_tt, mu_ts, mu_st, mu_ss = df.split(u)
psi_, mu_tt_, mu_ts_, mu_st_, mu_ss_ = df.split(u_)
psi_1, mu_tt_1, mu_ts_1, mu_st_1, mu_ss_1 = df.split(u_1)

# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
u_1.interpolate(u_init)
u_.assign(u_1)


def w_lin(c_, c_1, vtau):
    return vtau*c_ + c_1**3 + 3*c_1**2*(c_-c_1)


# Brazovskii-Swift (non-conserved PFC with dc/dt = -delta F/delta c)
# Semi-covariantized version (zero thickness) (i.e. the inner products now involve the metric a0)
Mu = (mu_tt, mu_ts, mu_st, mu_ss)
mu = geo_map.trace(Mu, index_pos="ll")
Mu_ = (mu_tt_, mu_ts_, mu_st_, mu_ss_)
mu_ = geo_map.trace(Mu_, index_pos="ll")

Eta = (etatt, etats, etast, etass)
eta_times_mu = geo_map.inner(Eta, Mu, index_pos="uull")
Deta = geo_map.grad_matrix(Eta, grad_index_pos="l", matrix_index_pos="uu")
v = geo_map.trace_tensor(Deta, index_pos="uul", sum_index=1)  # => upper vector

F_psi_L = geo_map.form(
    1/dt * (psi - psi_1) * xi
    - 4 * ell**2 * mu*xi
    + 4 * ell**4 * geo_map.dotgrad(mu, xi))
F_psi_NL = geo_map.form(w_lin(psi, psi_1, tau) * xi)
F_mu = geo_map.form(eta_times_mu + geo_map.dot(v, geo_map.grad(psi)))
F = F_psi_L + F_psi_NL + F_mu

a = df.lhs(F)
L = df.rhs(F)

# SOLVER
problem = df.LinearVariationalProblem(a, L, u_)
solver = df.LinearVariationalSolver(problem)

# solver.parameters["linear_solver"] = "gmres"
# solver.parameters["preconditioner"] = "jacobi"

df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True

# Output file
psifile = Timeseries("psi", name="psi")
mufile = Timeseries("mu", name="mu", space=geo_map.S_ref)

# Step in time
t = 0.0
it = 0
T = 100.0

psi_q = u_.split()[0]
psifile.write(psi_q, it)
mufile.write(mu_, it)

while t < T:
    it += 1
    t += dt

    solver.solve()

    u_1.assign(u_)
    if it % 1 == 0:
        psi_q = u_.split()[0]
        psifile.write(psi_q, it)
        mufile.write(mu_, it)
