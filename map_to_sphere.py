import dolfin as df
import numpy as np
import sympy as sp

res = 50
Lx = 2*np.pi
Ly = np.pi

ref_mesh = df.RectangleMesh.create(
    [df.Point(0., 0.), df.Point(Lx, Ly)],
    [2*res, res], df.cpp.mesh.CellType.Type.triangle)


def top(x, on_boundary):
    return on_boundary and x[1] > Ly-100*df.DOLFIN_EPS


def btm(x, on_boundary):
    return on_boundary and x[1] < 100*df.DOLFIN_EPS


class PeriodicBC(df.SubDomain):

    def inside(self, x, on_boundary):
        return bool(df.near(x[0], 0.0) and on_boundary)

    # Left side is master
    def map(self, x, y):
        y[0] = x[0]
        if x[0] > Lx - 100*df.DOLFIN_EPS:
            y[0] = 0.0
        y[1] = x[1]


class SphereMap:
    def __init__(self, R):
        self.t, self.s = sp.symbols('t s')
        self.map = dict()
        # self.map["x"] = R * sp.cos(self.t) * sp.sin(self.s)
        # self.map["y"] = R * sp.sin(self.t) * sp.sin(self.s)
        # self.map["z"] = R * sp.cos(self.s)
        # self.map["x"] = R * sp.cos(self.t)
        # self.map["y"] = R * sp.sin(self.t)
        # self.map["z"] = R * self.s
        self.map["x"] = self.t
        self.map["y"] = self.s
        self.map["z"] = 5 * R * sp.exp(-(
            (self.t-Lx/2)**2+(self.s-Ly/2)**2)/(2*0.5**2))

        self.map["xs"] = sp.diff(self.map["x"], self.s)
        self.map["xt"] = sp.diff(self.map["x"], self.t)
        self.map["ys"] = sp.diff(self.map["y"], self.s)
        self.map["yt"] = sp.diff(self.map["y"], self.t)
        self.map["zs"] = sp.diff(self.map["z"], self.s)
        self.map["zt"] = sp.diff(self.map["z"], self.t)

        self.map["g_ss"] = (self.map["xs"]**2
                            + self.map["ys"]**2
                            + self.map["zs"]**2)
        self.map["g_st"] = (self.map["xs"]*self.map["xt"]
                            + self.map["ys"]*self.map["yt"]
                            + self.map["zs"]*self.map["zt"])
        self.map["g_tt"] = (self.map["xt"]**2
                            + self.map["yt"]**2
                            + self.map["zt"]**2)
        self.map["g_ss"] = sp.simplify(self.map["g_ss"])
        self.map["g_st"] = sp.simplify(self.map["g_st"])
        self.map["g_tt"] = sp.simplify(self.map["g_tt"])
        
        self.map["g_det"] = (self.map["g_ss"]*self.map["g_tt"]
                             - self.map["g_st"]**2)
        self.map["g_det"] = sp.simplify(self.map["g_det"])
        self.map["g"] = abs(self.map["g_det"])

        self.map["gT_ss"] = sp.simplify(self.map["g_tt"]/self.map["g_det"])
        self.map["gT_st"] = sp.simplify(-self.map["g_st"]/self.map["g_det"])
        self.map["gT_tt"] = sp.simplify(self.map["g_ss"]/self.map["g_det"])

        self.evalf = dict()
        for key in self.map.keys():
            self.evalf[key] = sp.lambdify([self.t, self.s],
                                          self.map[key], "numpy")

    def eval(self, key, t_vals, s_vals):
        v = self.evalf[key](t_vals, s_vals)
        if isinstance(v, int) or isinstance(v, float):
            return v*np.ones_like(t_vals)
        else:
            return v


pbc = PeriodicBC()
pbc = None
dS_ref = df.Measure("dx", domain=ref_mesh)

ref_el = df.FiniteElement("Lagrange", "triangle", 1)
S_ref = df.FunctionSpace(ref_mesh, ref_el, constrained_domain=pbc)
S3_ref = df.FunctionSpace(ref_mesh,
                          df.MixedElement((ref_el, ref_el, ref_el)),
                          constrained_domain=pbc)

#ref_coords = ref_mesh.coordinates()
Theta = df.interpolate(df.Expression("x[0]", degree=1), S_ref)
Phi = df.interpolate(df.Expression("x[1]", degree=1), S_ref)

R = 1.0
theta = Theta.vector().get_local()
phi = Phi.vector().get_local()

geo_map = SphereMap(R)
#X = R*np.cos(theta)*np.sin(phi)
#Y = R*np.sin(theta)*np.sin(phi)
#Z = R*np.cos(phi)
X = geo_map.eval("x", theta, phi)
Y = geo_map.eval("y", theta, phi)
Z = geo_map.eval("z", theta, phi)

x = df.Function(S_ref)
y = df.Function(S_ref)
z = df.Function(S_ref)
x.rename("x", "tmp")
y.rename("y", "tmp")
z.rename("z", "tmp")

x.vector().set_local(X)
y.vector().set_local(Y)
z.vector().set_local(Z)
xyz = df.project(df.as_vector((x, y, z)), S3_ref)
xyz.rename("xyz", "tmp")

xdmff = df.XDMFFile(ref_mesh.mpi_comm(), "test.xdmf")
xdmff.write(xyz)
xdmff.close()

Sqrt_g = np.sqrt(geo_map.eval("g", theta, phi))
sqrt_g = df.Function(S_ref)  # Could be Expression!
sqrt_g.vector().set_local(Sqrt_g)
sqrt_g.rename("sqrt_g", "tmp")
dS = sqrt_g*dS_ref

area_ref = df.assemble(df.Constant(1.0)*dS_ref)
area = df.assemble(df.Constant(1.0)*dS)
print(area_ref, area)

local_area = df.project(sqrt_g*df.CellVolume(ref_mesh), S_ref)
local_area.rename("localarea", "tmp")
xdmff_loc = df.XDMFFile(ref_mesh.mpi_comm(), "localarea.xdmf")
xdmff_loc.write(local_area)
xdmff_loc.close()

u = df.TrialFunction(S_ref)
v = df.TestFunction(S_ref)

u_0 = df.Expression("exp(-(pow(x[0]-x01, 2)+pow(x[1]-y01, 2))/(2*pow(sigma, 2)))",
                    x01=Lx/8, y01=Ly/2,
                    sigma=Lx/20,
                    degree=1)
f = df.Expression("0.0", degree=1)

u_ = df.Function(S_ref, name="u")
u_1 = df.interpolate(u_0, S_ref)
u_.assign(u_1)

a = df.dot(df.grad(u), df.grad(v))*df.dx
L = f*v*df.dx

g_tt = df.Function(S_ref)
g_st = df.Function(S_ref)
g_ss = df.Function(S_ref)
g_tt.vector().set_local(geo_map.eval("g_tt", theta, phi))
g_st.vector().set_local(geo_map.eval("g_st", theta, phi))
g_ss.vector().set_local(geo_map.eval("g_ss", theta, phi))

dt = 0.01

a_dt = 1./dt*u*v*sqrt_g*dS_ref
a_diff = (g_tt*u.dx(0)*v.dx(0)
          + g_ss*u.dx(1)*v.dx(1)
          + g_st*u.dx(0)*v.dx(1)
          + g_st*u.dx(1)*v.dx(0))*sqrt_g*dS_ref
a_surf = a_dt + a_diff
L_surf = f*v*sqrt_g*dS_ref + 1./dt*u_1*v*sqrt_g*dS_ref

#U = df.Function(S_ref, name="U")

A = df.assemble(a_surf)

solver = df.PETScKrylovSolver("gmres")  # gmres
solver.set_operator(A)

# null_vec = df.Vector(U.vector())
# S_ref.dofmap().set(null_vec, 1.0)
# null_vec *= 1.0/null_vec.norm("l2")
# null_space = df.VectorSpaceBasis([null_vec])
# df.as_backend_type(A).set_nullspace(null_space)
# null_space.orthogonalize(b)
t = 0.0
T = 1.0
xdmfff = df.XDMFFile(ref_mesh.mpi_comm(), "u_.xdmf")
xdmfff.parameters["rewrite_function_mesh"] = False
xdmfff.parameters["flush_output"] = True
it = 0
xdmfff.write(u_, float(it))
while t < T:
    it += 1
    b = df.assemble(L_surf)
    
    solver.solve(u_.vector(), b)

    xdmfff.write(u_, float(it))

    u_1.assign(u_)
    t += dt
xdmfff.close()

    

Top = df.AutoSubDomain(top)
Btm = df.AutoSubDomain(btm)
facets = df.MeshFunction("size_t", ref_mesh, ref_mesh.topology().dim()-1, 0)
Top.mark(facets, 1)
Btm.mark(facets, 2)
flx_top = df.assemble(u_.dx(1)*df.ds(1, domain=ref_mesh, subdomain_data=facets))
flx_btm = df.assemble(u_.dx(1)*df.ds(2, domain=ref_mesh, subdomain_data=facets))
print(flx_top)
print(flx_btm)









