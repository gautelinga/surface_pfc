import dolfin as df
from maps import EllipsoidMap
from common.io import Timeseries, save_checkpoint, load_checkpoint, load_mesh, \
    load_parameters
from common.cmd import mpi_max, parse_command_line
from common.utilities import RandomInitialConditions
import sympy as sp
import os

parameters = dict(
    R=30.0,  # Radius
    res=140,  # Resolution
    dt=0.5,
    tau=0.2,
    h=1.1,
    M=1.0,  # Mobility
    restart_folder=None,
    t_0=0.0,
    tstep=0,
    T=1000.0,
    checkpoint_intv=50,
)
cmd_kwargs = parse_command_line()
parameters.update(**cmd_kwargs)
if parameters["restart_folder"]:
    load_parameters(parameters, os.path.join(
        parameters["restart_folder"], "parameters.dat"))
    parameters.update(**cmd_kwargs)

R = parameters["R"]
res = parameters["res"]
dt = parameters["dt"]
tau = parameters["tau"]
h = parameters["h"]
M = parameters["M"]

geo_map = EllipsoidMap(0.75*R, 0.75*R, 1.25*R)
geo_map.initialize(res, restart_folder=parameters["restart_folder"])

W = geo_map.mixed_space((geo_map.ref_el,)*4)

# Define trial and test functions
du = df.TrialFunction(W)
chi, xi, eta, etahat = df.TestFunctions(W)

# Define functions
u = df.TrialFunction(W)
u_ = df.Function(W, name="u_")  # current solution
u_1 = df.Function(W, name="u_1")  # solution from previous converged step

# Split mixed functions
dpsi, dmu, dnu, dnuhat = df.split(du)
psi,  mu, nu, nuhat = df.split(u)
psi_, mu_, nu_, nuhat_ = df.split(u_)
psi_1, mu_1, nu_1, nuhat_1 = df.split(u_1)

# Create intial conditions
if parameters["restart_folder"] is None:
    u_init = RandomInitialConditions(u_, degree=1)
    u_1.interpolate(u_init)
    u_.assign(u_1)
else:
    load_checkpoint(parameters["restart_folder"], u_, u_1)

Psi, Tau = sp.symbols('psi tau')
w = Tau/2*Psi**2 + Psi**4/4
dw = sp.diff(w, Psi)
ddw = sp.diff(dw, Psi)
f_w = sp.lambdify([Psi, Tau], w)
f_dw = sp.lambdify([Psi, Tau], dw)
f_ddw = sp.lambdify([Psi, Tau], ddw)


def w_lin(c_, c_1, vtau):
    return f_dw(c_1, vtau) + f_ddw(c_1, vtau)*(c_-c_1)


# Brazovskii-Swift (non-conserved PFC with dc/dt = -delta F/delta c)
m_NL = F_psi_NL = (1 + geo_map.K * h**2/12) * w_lin(psi, psi_1, tau) * xi
m_0 = 4 * nu * xi - 4 * geo_map.dotgrad(nu, xi)
m_2 = (2 * (geo_map.H * nuhat - geo_map.K*nu)*eta
       - 4 * geo_map.dotcurvgrad(nuhat, eta)
       + 5 * geo_map.K * geo_map.dotgrad(nu, eta)
       - 2 * geo_map.H * (geo_map.dotgrad(nuhat, eta)
                          + geo_map.dotcurvgrad(nu, eta)))/3
m = m_NL + m_0 + h**2 * m_2

F_psi = geo_map.form(1/dt * (psi - psi_1) * chi + M*geo_map.dotgrad(mu, chi))
F_mu = geo_map.form(mu*xi - m)
F_nu = geo_map.form(nu*eta + geo_map.dotgrad(psi, eta))
F_nuhat = geo_map.form(nuhat*etahat + geo_map.dotcurvgrad(psi, etahat))

F = F_psi + F_mu + F_nu + F_nuhat

a = df.lhs(F)
L = df.rhs(F)

# SOLVER
problem = df.LinearVariationalProblem(a, L, u_)
solver = df.LinearVariationalSolver(problem)

#solver.parameters["linear_solver"] = "gmres"
#solver.parameters["preconditioner"] = "jacobi"

df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True

#
t = parameters["t_0"]
tstep = parameters["tstep"]
T = parameters["T"]

# Output file
ts = Timeseries("results_pfcbc_conserved", u_,
                ("psi", "mu", "nu", "nuhat"), geo_map, tstep,
                restart_folder=parameters["restart_folder"])
E_0 = (2*nu_**2 - 2*geo_map.dotgrad(psi_, psi_) + f_w(psi_, tau))
ts.add_scalar_field(E_0, "E_0")
ts.add_scalar_field(df.sqrt(geo_map.dotgrad(mu_, mu_)), "abs_grad_mu")

# Step in time
ts.dump(tstep)

while t < T:
    tstep += 1
    t += dt

    u_1.assign(u_)
    solver.solve()

    if tstep % 1 == 0:
        ts.dump(tstep)
        grad_mu = ts.get_function("abs_grad_mu")
        grad_mu_max = mpi_max(grad_mu.vector().get_local())
        dt = 0.4/grad_mu_max
        ts.dump_stats(t, [grad_mu_max, dt], "data")

    if tstep % parameters["checkpoint_intv"] == 0 or t >= T:
        save_checkpoint(tstep, t, geo_map.ref_mesh,
                        u_, u_1, ts.folder, parameters)
