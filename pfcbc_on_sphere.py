import dolfin as df
from maps import SphereMap, EllipsoidMap
from common.io import dump_xdmf, Timeseries
import random
import numpy as np
import sympy as sp


# Class representing the intial conditions
class InitialConditions(df.UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + df.MPI.rank(df.MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = (0.0 - 0.00001*random.random())
        values[1] = 0.0

    def value_shape(self):
        return (3,)
        # return (2,)


R = 30.0  # Radius
res = 200  # Resolution
dt = 0.5
tau = 0.2
ell = 1.0  # 10.0/(2*np.pi*np.sqrt(2))
h = df.Constant(1.1)
M = df.Constant(1.0)  # Mobility

geo_map = EllipsoidMap(0.75*R, 0.75*R, 1.25*R)
geo_map.initialize(res)

W = geo_map.mixed_space((geo_map.ref_el,
                         geo_map.ref_el,
                         geo_map.ref_el))

# Define trial and test functions
du = df.TrialFunction(W)
xi, eta, etahat = df.TestFunctions(W)

# Define functions
u = df.TrialFunction(W)
u_ = df.Function(W, name="u_")  # current solution
u_1 = df.Function(W, name="u_1")  # solution from previous converged step

# Split mixed functions
dpsi, dnu, dnuhat = df.split(du)
psi,  nu, nuhat = df.split(u)
psi_, nu_, nuhat_ = df.split(u_)
psi_1, nu_1, nuhat_1 = df.split(u_1)

# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
u_1.interpolate(u_init)
u_.assign(u_1)

Psi, Tau = sp.symbols('psi tau')
w = Tau/2*Psi**2 + Psi**4/4
dw = sp.diff(w, Psi)
ddw = sp.diff(dw, Psi)
f_dw = sp.lambdify([Psi, Tau], dw)
f_ddw = sp.lambdify([Psi, Tau], ddw)


def w_lin(c_, c_1, vtau):
    return f_dw(c_1, vtau) + f_ddw(c_1, vtau)*(c_-c_1)


# Brazovskii-Swift (non-conserved PFC with dc/dt = -delta F/delta c)
m_NL = F_psi_NL = (1 + geo_map.K * h**2/12) * w_lin(psi, psi_1, tau) * xi
m_0 = 4 * ell**2 * nu*xi - 4 * ell**4 * geo_map.dotgrad(nu, xi)
m_2 = (2 * (geo_map.H * nuhat - geo_map.K*nu)*eta
       - 4 * geo_map.dotcurvgrad(nuhat, eta)
       + 5 * geo_map.K * geo_map.dotgrad(nu, eta)
       - 2 * geo_map.H * (geo_map.dotgrad(nuhat, eta)
                          + geo_map.dotcurvgrad(nu, eta)))/3
m = m_NL + m_0 + h**2 * m_2

F_psi = geo_map.form(1/dt * (psi - psi_1) * xi + M*m)
F_nu = geo_map.form(nu*eta + geo_map.dotgrad(psi, eta))
F_nuhat = geo_map.form(nuhat*etahat + geo_map.dotcurvgrad(psi, etahat))

F = F_psi + F_nu + F_nuhat

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
ts = Timeseries("results_pfcbc", u_, ("psi", "nu", "nuhat"), geo_map, 0)
ts.add_scalar_field(psi_**2, "psi_squared")

# Step in time
t = 0.0
it = 0
T = 100.0

ts.dump(it)

while t < T:
    it += 1
    t += dt

    solver.solve()

    u_1.assign(u_)
    if it % 1 == 0:
        ts.dump(it)
