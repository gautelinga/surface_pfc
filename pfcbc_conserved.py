import dolfin as df
from maps import EllipsoidMap, CylinderMap
from common.io import Timeseries, save_checkpoint, load_checkpoint, \
    load_parameters
from common.cmd import mpi_max, parse_command_line
from common.utilities import RandomInitialConditions, QuarticPotential, AroundInitialConditions, AlongInitialConditions, MMSInitialConditions
import os
import ufl
import numpy as np
from common.MMS import ManufacturedSolution
import sympy as sp

parameters = dict(
    #R=10*np.sqrt(2),  # Radius
    R=4*np.sqrt(2),  # Radius
    #res=140,  # Resolution
    res=200,  # Resolution
    dt=1e-1,
    tau=0.2,
    h=1.1,
    #h=0.0,
    M=1.0,  # Mobility
    restart_folder=None,
    t_0=0.0,
    tstep=0,
    T=1e6,
    # T=10000.0,
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
dt = df.Constant(parameters["dt"])
tau = parameters["tau"]
h = df.Constant(parameters["h"])
M = parameters["M"]

#geo_map = EllipsoidMap(0.75*R, 0.75*R, 1.25*R)
geo_map = CylinderMap(R, 2*np.pi*R)
geo_map.initialize(res, restart_folder=parameters["restart_folder"])

# Initialize the Method of Manufactured Solutions:
psi_mms_input = (sp.sin(geo_map.s/sp.sqrt(2)))**2 + (sp.sin(geo_map.t/sp.sqrt(2)))**2 # Note that the Initial Condition must also be separately specified.
my_mms = ManufacturedSolution(geo_map, psi_mms_input)
my_mms.initialize_fields()
psiMMS = my_mms.psi
nuMMS =  my_mms.laplacian
nuhatMMS =  my_mms.curvplacian

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
    #u_init = RandomInitialConditions(u_, degree=1)
    #u_init = AroundInitialConditions(u_, degree=1)
    #u_init = AlongInitialConditions(u_, degree=1)
    #u_init = AlongInitialConditions(u_, degree=1)
    u_init = MMSInitialConditions(u_, geo_map, degree=1)
    u_1.interpolate(u_init)
    u_.assign(u_1)
else:
    load_checkpoint(parameters["restart_folder"], u_, u_1)

w = QuarticPotential()

dw_lin = w.derivative_linearized(psi, psi_1, tau)

# Define some UFL indices:
i, j, k, l = ufl.Index(), ufl.Index(), ufl.Index(), ufl.Index()

# dotgrad(u,v) -> geo_map.gab[i,j]*u.dx(i)*v.dx(j)
# dotcurvgrad(u,v) -> geo_map.Kab[i,j]*u.dx(i)*v.dx(j)

# Functionals for Manufactured solution:
dw_linMMS = w.derivative_linearized(psiMMS, psiMMS, tau)
m_NLMMS = (1 + geo_map.K * h**2/12) * dw_linMMS * xi
m_0MMS = 4 * nuMMS * xi - 4 * geo_map.gab[i, j]*nuMMS.dx(i)*xi.dx(j)

m_2MMS = (2 * (geo_map.H * nuhatMMS - geo_map.K * nuMMS)*xi
       - 4 * geo_map.Kab[i, j]*nuhatMMS.dx(i)*xi.dx(j)
       + 5 * geo_map.K * geo_map.gab[i, j] * nuMMS.dx(i)*xi.dx(j)
       - 2 * geo_map.H * (geo_map.gab[i, j]*nuhatMMS.dx(i)*xi.dx(j)
                          + geo_map.Kab[i, j]*nuMMS.dx(i)*xi.dx(j)))/3

mMMS = m_NLMMS + m_0MMS + h**2 * m_2MMS

# Brazovskii-Swift (conserved PFC with dc/dt = grad^2 delta F/delta c)
m_NL = F_psi_NL = (1 + geo_map.K * h**2/12) * dw_lin * xi
m_0 = 4 * nu * xi - 4 * geo_map.gab[i, j]*nu.dx(i)*xi.dx(j)
m_2 = (2 * (geo_map.H * nuhat - geo_map.K*nu)*xi
        - 4 * geo_map.Kab[i, j]*nuhat.dx(i)*xi.dx(j)
        + 5 * geo_map.K * geo_map.gab[i, j]*nu.dx(i)*xi.dx(j)
        - 2 * geo_map.H * (geo_map.gab[i, j]*nuhat.dx(i)*xi.dx(j)
                           + geo_map.Kab[i, j]*nu.dx(i)*xi.dx(j)))/3
m = m_NL + m_0 + h**2 * m_2

F_psi = geo_map.form(1/dt * (psi - psi_1) * chi
                     + M * geo_map.gab[i, j]*mu.dx(i)*chi.dx(j))

# Enable/disable Manufactured Solution by choosing one of the two lines below:
#F_mu = geo_map.form(mu*xi - m)
F_mu = geo_map.form(mu*xi - m + mMMS)

F_nu = geo_map.form(nu*eta + geo_map.gab[i, j]*psi.dx(i)*eta.dx(j))
F_nuhat = geo_map.form(nuhat*etahat + geo_map.Kab[i, j]*psi.dx(i)*etahat.dx(j))

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
                parameters=parameters,
                restart_folder=parameters["restart_folder"])
# Shorthand notation:
H = geo_map.H
K = geo_map.K
gab = geo_map.gab

E_0 = (2*nu_**2 - 2 * geo_map.gab[i, j]*psi_.dx(i)*psi_.dx(j) + w(psi_, tau))
ts.add_scalar_field(E_0, "E_0")
ts.add_scalar_field(df.sqrt(geo_map.gab[i, j]*mu_.dx(i)*mu_.dx(j)),
                    "abs_grad_mu")

# Step in time
ts.dump(tstep)

while t < T:
    tstep += 1
    t += float(dt.values())

    u_1.assign(u_)
    solver.solve()

    if tstep % 1 == 0:
        # ts.dump(tstep)
        ts.dump(t)
        E_0 = (2*nu_**2 - 2 * geo_map.gab[i, j]*psi_.dx(i)*psi_.dx(j) + w(psi_, tau))
        #E_nonK = df.assemble(geo_map.form(geo_map.sqrt_g*(2*nu_**2-2*geo_map.gab[i,j]*psi_.dx(i)*psi_.dx(j) + (tau/2) * psi_**2 + (1/4) *psi_**4))) # Double checked for correctness
        E_nonK = df.assemble(geo_map.form(E_0))
        E_K = df.assemble(geo_map.form((h**2/12)*(2*(4*nuhat_**2 + 4*H*nuhat_*nu_ - 5*K*nu_**2) - 2 * (2*H*nuhat_ - 2*K*gab[i,j]*psi_.dx(i)*psi_.dx(j)) + (tau/2)*K*psi_**2 + (1/4)*K*psi_**4))) # Double checked for correctness
        # if tstep > 20:
        #     dh = 0.001
        #     h.assign(max(0,float(h.values())-dh))
        #     #dt.assign(0.0001)
        grad_mu = ts.get_function("abs_grad_mu")
        grad_mu_max = mpi_max(grad_mu.vector().get_local())
        ts.dump_stats(t, [grad_mu_max, float(dt.values()), float(h.values()), E_nonK, E_K], "data")

    if tstep % parameters["checkpoint_intv"] == 0 or t >= T:
        save_checkpoint(tstep, t, geo_map.ref_mesh,
                        u_, u_1, ts.folder, parameters)
