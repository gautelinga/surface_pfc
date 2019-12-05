import dolfin as df
from surfaise import CylinderMap
from surfaise.common.io import (
    Timeseries, save_checkpoint, load_checkpoint,
    load_parameters)
from surfaise.common.cmd import (
    mpi_max, parse_command_line, info_blue,
    info_cyan, info_red, mpi_any)
from surfaise.common.utilities import (
    QuarticPotential, TimeStepSelector, anneal_func)
from surfaise.ics import StripedIC, RandomIC
import os
import ufl
import numpy as np


parameters = dict(
    R=20*np.sqrt(2),  # Radius
    Rz=40.,
    # res=130,  # Resolution
    res=200,  # Resolution
    dt=1e-1,
    tau=0.2,
    t_ramp=1000.,
    tau_ramp=0.99,
    h=20.0,
    M=1.0,  # Mobility
    restart_folder=None,
    folder="results_cylinder",
    t_0=0.0,
    tstep=0,
    T=6000,
    checkpoint_intv=50,
    verbose=True,
    anneal=True,
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
Rz = parameters["Rz"]
res = parameters["res"]
dt = TimeStepSelector(parameters["dt"])
tau = df.Constant(parameters["tau"])
h = df.Constant(parameters["h"])
M = df.Constant(parameters["M"])

#geo_map = EllipsoidMap(R, R, Rz)
geo_map = CylinderMap(R, 2*np.pi*R)

geo_map.initialize(res, restart_folder=parameters["restart_folder"])

W = geo_map.mixed_space(4)

# Define trial and test functions
du = df.TrialFunction(W)
chi, xi, eta, etahat = df.TestFunctions(W)

# Define functions
u = df.TrialFunction(W)
u_ = df.Function(W, name="u_")  # current solution
u_1 = df.Function(W, name="u_1")  # solution from previous converged step

# Split mixed functions
psi,  mu, nu, nuhat = df.split(u)
psi_, mu_, nu_, nuhat_ = df.split(u_)
psi_1, mu_1, nu_1, nuhat_1 = df.split(u_1)

# Create intial conditions
if parameters["restart_folder"] is None:
    init_mode = parameters["init_mode"]
    if init_mode == "random":
        u_init = RandomIC(u_, amplitude=1e-1, degree=1)
    elif init_mode == "striped":
        u_init = StripedIC(u_, alpha=parameters["alpha"]*np.pi/180.0, degree=1)
    else:
        exit("No init_mode set.")
    u_1.interpolate(u_init)
    u_.assign(u_1)
else:
    load_checkpoint(parameters["restart_folder"], u_, u_1)

w = QuarticPotential()

#dw_lin = w.derivative_linearized(psi, psi_1, tau)
dw_stab = w.derivative_stab(psi_, psi_1, tau)

# Define some UFL indices:
i, j, k, l = ufl.Index(), ufl.Index(), ufl.Index(), ufl.Index()

# Brazovskii-Swift (conserved PFC with dc/dt = grad^2 delta F/delta c)
m_NL = F_psi_NL = (1 + geo_map.K * h**2/12) * dw_stab * xi
m_0 = (4 * nu_ * xi
       - 4 * geo_map.gab[i, j]*nu_.dx(i)*xi.dx(j))
m_2 = (2 * (geo_map.H * nuhat_ - geo_map.K*nu_)*xi
       - 4 * geo_map.Kab[i, j]*nuhat_.dx(i)*xi.dx(j)
       + 5 * geo_map.K * geo_map.gab[i, j]*nu_.dx(i)*xi.dx(j)
       - 2 * geo_map.H * (geo_map.gab[i, j]*nuhat_.dx(i)*xi.dx(j)
                          + geo_map.Kab[i, j]*nu_.dx(i)*xi.dx(j)))/3
m = m_NL + m_0 + h**2 * m_2

F_psi = geo_map.form(1/dt * (psi_ - psi_1) * chi
                     + M * geo_map.gab[i, j]*mu_.dx(i)*chi.dx(j))

# Enable/disable Manufactured Solution by choosing one of the two lines below:
F_mu = geo_map.form(mu_*xi - m)

F_nu = geo_map.form(nu_*eta + geo_map.gab[i, j]*psi_.dx(i)*eta.dx(j))
F_nuhat = geo_map.form(nuhat_*etahat
                       + geo_map.Kab[i, j]*psi_.dx(i)*etahat.dx(j))

F = F_psi + F_mu + F_nu + F_nuhat

# a = df.lhs(F)
# L = df.rhs(F)
J = df.derivative(F, u_, du=u)

# SOLVER
# problem = df.LinearVariationalProblem(a, L, u_)
# solver = df.LinearVariationalSolver(problem)

problem = df.NonlinearVariationalProblem(F, u_, J=J)
solver = df.NonlinearVariationalSolver(problem)
solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-8
solver.parameters["newton_solver"]["relative_tolerance"] = 1e-6
solver.parameters["newton_solver"]["maximum_iterations"] = 6
# solver.parameters["newton_solver"]["linear_solver"] = "gmres"
# solver.parameters["newton_solver"]["preconditioner"] = "default"
# solver.parameters["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True
# solver.parameters["newton_solver"]["krylov_solver"]["absolute_tolerance"] = 1e-8
# solver.parameters["newton_solver"]["krylov_solver"]["monitor_convergence"] = False
# solver.parameters["newton_solver"]["krylov_solver"]["maximum_iterations"] = 1000

# solver.parameters["linear_solver"] = "gmres"
# solver.parameters["preconditioner"] = "jacobi"

df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True

#
t = parameters["t_0"]
tstep = parameters["tstep"]
T = parameters["T"]

# Output file
ts = Timeseries(parameters["folder"], u_,
                ("psi", "mu", "nu", "nuhat"), geo_map, tstep,
                parameters=parameters,
                restart_folder=parameters["restart_folder"])

# Shorthand notation:
H = geo_map.H
K = geo_map.K
gab = geo_map.gab

E_0 = (2*nu_**2 - 2 * geo_map.gab[i, j]*psi_.dx(i)*psi_.dx(j) + w(psi_, tau))
E_2 = (h**2/12)*(2*(4*nuhat_**2 + 4*H*nuhat_*nu_ - 5*K*nu_**2)
                 - 2 * (2*H*nuhat_ - 2*K*gab[i, j]*psi_.dx(i)*psi_.dx(j))
                 + (tau/2)*K*psi_**2 + (1/4)*K*psi_**4)
ts.add_field(E_0, "E_0")
ts.add_field(E_2, "E_2")
ts.add_field(df.sqrt(geo_map.gab[i, j]*mu_.dx(i)*mu_.dx(j)),
             "abs_grad_mu")

# Step in time
ts.dump(tstep)

initial_step = bool(parameters["restart_folder"] is None)
t_prev = t
while t < T:
    tstep += 1
    info_cyan("tstep = {}, time = {}".format(tstep, t))

    u_1.assign(u_)

    converged = False
    while not converged:
        if parameters["anneal"]:
            tau.assign(
                anneal_func(
                    t+dt.get(),
                    parameters["tau"],
                    parameters["tau_ramp"],
                    parameters["t_ramp"]))

        # Compute energy
        u_.assign(u_1)
        Eout_0 = df.assemble(geo_map.form(E_0))
        Eout_2 = df.assemble(geo_map.form(E_2))
        E_before = Eout_0 + Eout_2

        try:
            solver.solve()
            converged = True
        except:
            info_blue("Did not converge. Chopping timestep.")
            dt.chop()
            info_blue("New timestep is: dt = {}".format(dt.get()))

        Eout_0 = df.assemble(geo_map.form(E_0))
        Eout_2 = df.assemble(geo_map.form(E_2))
        E_after = Eout_0 + Eout_2
        dE = E_after - E_before
        if not initial_step and dE > 0.0:
            dt.chop()
            converged = False

    initial_step = False

    # Update time with final dt value
    t += dt.get()
    # Set dt:
    grad_mu_ufl = df.sqrt(geo_map.gab[i, j]*mu_.dx(i)*mu_.dx(j))
    grad_mu = df.project(grad_mu_ufl, geo_map.S_ref)
    grad_mu_max = mpi_max(grad_mu.vector().get_local())
    dt_prev = dt.get()
    dt.set(min(min(0.25/grad_mu_max, T-t),parameters["t_ramp"]/200) )
    info_blue("dt = {}".format(dt.get()))

    if tstep % 100 == 0 or np.floor(t/1000)-np.floor(t_prev/1000) > 0:
        ts.dump(t)

        # Assigning timestep size according to grad_mu_max:
        #grad_mu = ts.get_function("abs_grad_mu")
        #grad_mu_max = mpi_max(grad_mu.vector().get_local())
        #dt_prev = dt.get()
        #dt.set(min(min(0.25/grad_mu_max, T-t),parameters["t_ramp"]/100) )
        #info_blue("dt = {}".format(dt.get()))

        ts.dump_stats(t,
                      [grad_mu_max, dt_prev, dt.get(),
                       float(h.values()),
                       Eout_0, Eout_2,
                       Eout_0 + Eout_2, float(tau.values()),
                       dE],
                      "data")

    if tstep % parameters["checkpoint_intv"] == 0 or t >= T:
        save_checkpoint(tstep, t, geo_map.ref_mesh,
                        u_, u_1, ts.folder, parameters)
    t_prev = t
