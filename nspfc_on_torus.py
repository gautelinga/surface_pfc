import dolfin as df
from maps import TorusMap
from common.io import Timeseries, save_checkpoint, load_checkpoint, \
    load_parameters
from common.cmd import mpi_max, parse_command_line, info_blue, info_cyan
from common.utilities import QuarticPotential, TimeStepSelector
from ics import StripedIC, RandomIC
import os
import ufl
import numpy as np


parameters = dict(
    R=30.,  # Radius
    r=10.,
    res=100,  # Resolution
    dt=2e-1,
    rho=.01,
    eta=.1,
    tau=0.2,
    M=1.0,
    restart_folder=None,
    folder="results_nspfc_torus",
    t_0=0.0,
    tstep=0,
    T=2000,
    dump_intv=5,
    stats_intv=1,
    checkpoint_intv=50,
    verbose=True,
    init_mode="random",
    alpha=0.0,
)
cmd_kwargs = parse_command_line()
parameters.update(**cmd_kwargs)
if parameters["restart_folder"]:
    load_parameters(parameters, os.path.join(
        parameters["restart_folder"], "parameters.dat"))
    parameters.update(**cmd_kwargs)

R = parameters["R"]
r = parameters["r"]
res = parameters["res"]
dt = TimeStepSelector(parameters["dt"])
rho = df.Constant(parameters["rho"])
eta = df.Constant(parameters["eta"])
tau = df.Constant(parameters["tau"])
M = df.Constant(parameters["M"])

geo_map = TorusMap(R, r)
geo_map.initialize(res, restart_folder=parameters["restart_folder"])

W = geo_map.mixed_space((geo_map.ref_el, geo_map.ref_el, geo_map.ref_el,
                         geo_map.ref_vel, geo_map.ref_el))
field_names = ("psi", "mu", "nu", "u", "p")
# W = geo_map.mixed_space((geo_map.ref_el, geo_map.ref_el, geo_map.ref_el))
# field_names = ("psi", "mu", "nu")

# Define trial and test functions
chi, xi, beta, v, q = df.TestFunctions(W)
# chi, xi, beta = df.TestFunctions(W)

# Define functions
w = df.TrialFunction(W)
w_ = df.Function(W, name="u_")  # current solution
w_1 = df.Function(W, name="u_1")  # solution from previous converged step

# Split mixed functions
psi, mu, nu, u,  p = df.split(w)
psi_, mu_, nu_, u_, p_ = df.split(w_)
psi_1, mu_1, nu_1, u_1, p_1 = df.split(w_1)

# psi, mu, nu = df.split(w)
# psi_, mu_, nu_ = df.split(w_)
# psi_1, mu_1, nu_1 = df.split(w_1)

# Create intial conditions
if parameters["restart_folder"] is None:
    init_mode = parameters["init_mode"]
    if init_mode == "random":
        w_init = RandomIC(w_, degree=1)
    else:
        exit("Unknown IC")
    w_1.interpolate(w_init)
    w_.assign(w_1)
else:
    load_checkpoint(parameters["restart_folder"], w_, w_1)

w_pot = QuarticPotential()

dw_stab = w_pot.derivative_stab(psi_, psi_1, tau)

f = df.Constant((0., 0.))

# Define some UFL indices:
i, j, k, l = ufl.Index(), ufl.Index(), ufl.Index(), ufl.Index()

# Brazovskii-Swift (conserved PFC with dc/dt = grad^2 delta F/delta c)
m_NL = dw_stab * xi
m_0 = 4 * nu_ * xi - 4 * geo_map.gab[i, j]*nu_.dx(i)*xi.dx(j)
m = m_NL + m_0
m_NS = (rho / dt * geo_map.g_ab[i, j] * (u_[i]-u_1[i]) * v[j]
        + rho * geo_map.g_ab[i, j] * u_[k] *
        geo_map.CovD10(u_)[k, i] * v[j]
        + eta * geo_map.g_ab[i, k] * geo_map.gab[j, l] *
        geo_map.CovD10(u_)[i, j] * geo_map.CovD10(v)[k, l]
        + eta * geo_map.K * geo_map.g_ab[i, j] * u_[i] * v[j]
        - p_ * geo_map.CovD10(v)[i, i]
        - q * geo_map.CovD10(u_)[i, i]
        + psi_ * mu_.dx(i) * v[i]
        - f[i]*v[i])

F_psi = geo_map.form(1/dt * (psi_ - psi_1) * chi
                     - psi_ * u_[i] * chi.dx(i)
                     + M * geo_map.gab[i, j]*mu_.dx(i)*chi.dx(j))
F_mu = geo_map.form(mu_*xi - m)
F_nu = geo_map.form(nu_*beta + geo_map.gab[i, j]*psi_.dx(i)*beta.dx(j))
F_PFC = F_psi + F_mu + F_nu
F_NS = geo_map.form(m_NS)
F = F_PFC + F_NS

J = df.derivative(F, w_, du=w)

problem = df.NonlinearVariationalProblem(F, w_, J=J)
solver = df.NonlinearVariationalSolver(problem)
solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-8
solver.parameters["newton_solver"]["relative_tolerance"] = 1e-5
solver.parameters["newton_solver"]["maximum_iterations"] = 16
#solver.parameters["newton_solver"]["linear_solver"] = "gmres"
#solver.parameters["newton_solver"]["preconditioner"] = "default"
# solver.parameters["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True
# solver.parameters["newton_solver"]["krylov_solver"]["absolute_tolerance"] = 1e-8
# solver.parameters["newton_solver"]["krylov_solver"]["monitor_convergence"] = False
#solver.parameters["newton_solver"]["krylov_solver"]["maximum_iterations"] = 1000

# solver.parameters["linear_solver"] = "gmres"
# solver.parameters["preconditioner"] = "jacobi"

df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True

#
t = parameters["t_0"]
tstep = parameters["tstep"]
T = parameters["T"]

# Output file
ts = Timeseries(parameters["folder"], w_,
                field_names, geo_map, tstep,
                parameters=parameters,
                restart_folder=parameters["restart_folder"])

E_kin = 0.5*rho*geo_map.g_ab[i, j]*u_[i]*u_[j]
divu = geo_map.CovD10(u_)[i, i]
U = [sum([geo_map.get_function(xi + "_," + vj)*u_[dj]
          for dj, vj in enumerate(geo_map.AXIS_REF)])
     for xi in geo_map.AXIS]
GradMu = [sum([geo_map.get_function(xi + "_," + vj)*geo_map.gab[i, dj]*mu_.dx(i)
               for dj, vj in enumerate(geo_map.AXIS_REF)])
          for xi in geo_map.AXIS]

E_psi = (2*nu_**2 - 2 * geo_map.gab[i, j]*psi_.dx(i)*psi_.dx(j)
         + w_pot(psi_, tau))
grad_mu_ufl = df.sqrt(geo_map.gab[i, j]*mu_.dx(i)*mu_.dx(j))
u_norm_ufl = df.sqrt(geo_map.g_ab[i, j]*u_[i]*u_[j])

ts.add_field(E_psi, "E_psi")
ts.add_field(GradMu, "GradMu")
ts.add_field(grad_mu_ufl,
             "abs_grad_mu")

ts.add_field(E_kin, "E_kin")
ts.add_field(divu, "divu")
ts.add_field(U, "U")
ts.add_field(u_norm_ufl, "u_norm")

# Step in time
ts.dump(tstep)

while t < T:
    tstep += 1
    info_cyan("tstep = {}, time = {}".format(tstep, t))

    w_1.assign(w_)

    converged = False
    while not converged:
        try:
            solver.solve()
            converged = True
        except:
            info_blue("Did not converge. Chopping timestep.")
            dt.chop()
            info_blue("New timestep is: dt = {}".format(dt.get()))

    # Update time with final dt value
    t += dt.get()

    if tstep % parameters["dump_intv"] == 0:
        ts.dump(t)

    # Assigning timestep size according to grad_mu_max:

    # grad_mu = ts.get_function("abs_grad_mu")
    grad_mu = df.project(grad_mu_ufl, geo_map.S_ref)
    grad_mu_max = mpi_max(grad_mu.vector().get_local())
    # u_norm = ts.get_function("u_norm")
    u_norm = df.project(u_norm_ufl, geo_map.S_ref)
    u_norm_max = mpi_max(u_norm.vector().get_local())

    vel_max = max(u_norm_max, grad_mu_max)

    dt_prev = dt.get()
    dt.set(min(0.2/vel_max, T-t))

    info_blue("dt = {}".format(dt.get()))
    if tstep % parameters["stats_intv"] == 0:
        E_kin_out = df.assemble(geo_map.form(E_kin))
        E_psi_out = df.assemble(geo_map.form(E_psi))
        divu_out = df.assemble(geo_map.form(divu))

        ts.dump_stats(t, [E_kin_out, E_psi_out, divu_out, grad_mu_max,
                          u_norm_max],
                      "data")

    if tstep % parameters["checkpoint_intv"] == 0 or t >= T:
        save_checkpoint(tstep, t, geo_map.ref_mesh,
                        w_, w_1, ts.folder, parameters)
