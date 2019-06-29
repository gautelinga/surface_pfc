import dolfin as df
import numpy as np
import sympy as sp


def dump_xdmf(f):
    filename = "{}.xdmf".format(f.name())
    with df.XDMFFile(ref_mesh.mpi_comm(), filename) as xdmff:
        xdmff.write(f)


class PBC(df.SubDomain):
    def __init__(self, ts_min, ts_max):
        self.t_min = ts_min[0]
        self.t_max = ts_max[0]
        self.s_min = ts_min[1]
        self.s_max = ts_max[1]
        df.SubDomain.__init__(self)


class SpherePBC(PBC):
    def inside(self, x, on_boundary):
        return bool(df.near(x[0], self.t_min) and on_boundary)

    # Left side is master
    def map(self, x, y):
        y[0] = self.t_min
        y[1] = x[1]


class CylinderPBC(PBC):
    def inside(self, x, on_boundary):
        return bool(df.near(x[0], self.t_min) and on_boundary)

    # Left side is master
    def map(self, x, y):
        if x[0] > self.t_max - 100*df.DOLFIN_EPS:
            y[0] = self.t_min
            y[1] = x[1]


class GeoMap:
    def __init__(self, xyz, ts, ts_min, ts_max):
        self.t = ts[0]
        self.s = ts[1]
        self.map = dict()
        self.map["x"] = xyz[0]
        self.map["y"] = xyz[1]
        self.map["z"] = xyz[2]
        self.t_min = ts_min[0]
        self.t_max = ts_max[0]
        self.s_min = ts_min[1]
        self.s_max = ts_max[1]
        self.compute_geometry()

    def compute_geometry(self):
        # Simple derivatives
        self.map["xs"] = sp.diff(self.map["x"], self.s)
        self.map["xt"] = sp.diff(self.map["x"], self.t)
        self.map["ys"] = sp.diff(self.map["y"], self.s)
        self.map["yt"] = sp.diff(self.map["y"], self.t)
        self.map["zs"] = sp.diff(self.map["z"], self.s)
        self.map["zt"] = sp.diff(self.map["z"], self.t)

        # Double derivatives
        self.map["xss"] = sp.diff(self.map["xs"], self.s)
        self.map["xst"] = sp.diff(self.map["xs"], self.t)
        self.map["xtt"] = sp.diff(self.map["xt"], self.t)
        self.map["yss"] = sp.diff(self.map["ys"], self.s)
        self.map["yst"] = sp.diff(self.map["ys"], self.t)
        self.map["ytt"] = sp.diff(self.map["yt"], self.t)
        self.map["zss"] = sp.diff(self.map["zs"], self.s)
        self.map["zst"] = sp.diff(self.map["zs"], self.t)
        self.map["ztt"] = sp.diff(self.map["zt"], self.t)

        for key in self.map.keys():
            self.map[key] = sp.simplify(self.map[key])

        # The metric tensor
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
        self.map["sqrt_g"] = sp.sqrt(abs(self.map["g"]))

        self.map["gss"] = sp.simplify(self.map["g_tt"]/self.map["g_det"])
        self.map["gst"] = sp.simplify(-self.map["g_st"]/self.map["g_det"])
        self.map["gtt"] = sp.simplify(self.map["g_ss"]/self.map["g_det"])

        self.map["g_ss_s"] = sp.simplify(sp.diff(self.map["g_ss"], self.s))
        self.map["g_ss_t"] = sp.simplify(sp.diff(self.map["g_ss"], self.t))
        self.map["g_st_s"] = sp.simplify(sp.diff(self.map["g_st"], self.s))
        self.map["g_st_t"] = sp.simplify(sp.diff(self.map["g_st"], self.t))
        self.map["g_tt_s"] = sp.simplify(sp.diff(self.map["g_tt"], self.s))
        self.map["g_tt_t"] = sp.simplify(sp.diff(self.map["g_tt"], self.t))
        
        # The normal
        cross_x = sp.simplify(
            self.map["yt"]*self.map["zs"] - self.map["ys"]*self.map["zt"])
        cross_y = sp.simplify(
            self.map["zt"]*self.map["xs"] - self.map["zs"]*self.map["xt"])
        cross_z = sp.simplify(
            self.map["xt"]*self.map["ys"] - self.map["xs"]*self.map["yt"])
        cross_mag = sp.simplify(
            sp.sqrt(cross_x**2 + cross_y**2 + cross_z**2))
        self.map["nx"] = sp.simplify(cross_x/cross_mag)
        self.map["ny"] = sp.simplify(cross_y/cross_mag)
        self.map["nz"] = sp.simplify(cross_z/cross_mag)

        # Curvature tensor
        self.map["K_ss"] = sp.simplify(self.map["nx"]*self.map["xss"]
                                       + self.map["ny"]*self.map["yss"]
                                       + self.map["nz"]*self.map["zss"])
        self.map["K_st"] = sp.simplify(self.map["nx"]*self.map["xst"]
                                       + self.map["ny"]*self.map["yst"]
                                       + self.map["nz"]*self.map["zst"])
        self.map["K_tt"] = sp.simplify(self.map["nx"]*self.map["xtt"]
                                       + self.map["ny"]*self.map["ytt"]
                                       + self.map["nz"]*self.map["ztt"])
        self.map["Ks_s"] = sp.simplify(self.map["gss"]*self.map["K_ss"]
                                       + self.map["gst"]*self.map["K_st"])
        self.map["Ks_t"] = sp.simplify(self.map["gss"]*self.map["K_st"]
                                       + self.map["gst"]*self.map["K_tt"])
        self.map["Kt_s"] = sp.simplify(self.map["gst"]*self.map["K_ss"]
                                       + self.map["gtt"]*self.map["K_st"])
        self.map["Kt_t"] = sp.simplify(self.map["gst"]*self.map["K_st"]
                                       + self.map["gtt"]*self.map["K_tt"])

        # Mean and Gaussian curvature
        self.map["H"] = sp.simplify((self.map["Ks_s"] + self.map["Kt_t"])/2)
        self.map["K"] = sp.simplify(self.map["Ks_s"]*self.map["Kt_t"]
                                    - self.map["Ks_t"]*self.map["Kt_s"])

        # Cristoffel symbols
        self.map["Gs_ss"] = sp.simplify((
            self.map["gss"]*self.map["g_ss_s"]
            + self.map["gst"]*(
                2*self.map["g_st_s"]-self.map["g_ss_t"]))/2)
        self.map["Gs_st"] = sp.simplify((
            self.map["gss"]*self.map["g_ss_t"]
            + self.map["gst"]*self.map["g_tt_s"])/2)
        self.map["Gs_tt"] = sp.simplify((
            self.map["gss"]*(
                2*self.map["g_st_t"] - self.map["g_tt_s"])
            + self.map["gst"]*self.map["g_tt_t"])/2)
        self.map["Gt_ss"] = sp.simplify((
            self.map["gtt"]*(
                2*self.map["g_st_s"] - self.map["g_ss_t"])
            + self.map["gst"]*self.map["g_ss_s"])/2)
        self.map["Gt_st"] = sp.simplify((
            self.map["gtt"]*self.map["g_tt_s"]
            + self.map["gst"]*self.map["g_ss_t"])/2)
        self.map["Gt_tt"] = sp.simplify((
            self.map["gtt"]*self.map["g_tt_t"]
            + self.map["gst"]*(
                2*self.map["g_st_t"]-self.map["g_tt_s"]))/2)

        self.evalf = dict()
        for key in self.map.keys():
            self.evalf[key] = sp.lambdify([self.t, self.s],
                                          self.map[key], "numpy")

    def eval(self, key):  # , t_vals, s_vals):
        v = self.evalf[key](self.t_vals, self.s_vals)
        if isinstance(v, int) or isinstance(v, float):
            return v*np.ones_like(self.t_vals)
        else:
            return v

    def get_function(self, key):  # , space, t_vals, s_vals):
        f = df.Function(self.S_ref)
        f.rename(key, "tmp")
        F = self.eval(key)
        f.vector().set_local(F)
        return f

    def initialize_metric(self):  # , S_ref, t_vals, s_vals):
        self.gtt = self.get_function("gtt")
        self.gst = self.get_function("gst")
        self.gss = self.get_function("gss")
        self.sqrt_g = self.get_function("sqrt_g")

    def dot(self, u, v):
        return (self.gtt*u.dx(0)*v.dx(0)
                + self.gss*u.dx(1)*v.dx(1)
                + self.gst*u.dx(0)*v.dx(1)
                + self.gst*u.dx(1)*v.dx(0))

    def form(self, integrand):
        return integrand*self.sqrt_g*self.dS_ref

    def coords(self):
        x = geo_map.get_function("x")
        y = geo_map.get_function("y")
        z = geo_map.get_function("z")
        xyz = df.project(df.as_vector((x, y, z)), self.S3_ref)
        xyz.rename("xyz", "tmp")
        return xyz

    def normal(self):
        nx = geo_map.get_function("nx")
        ny = geo_map.get_function("ny")
        nz = geo_map.get_function("nz")
        n = df.project(df.as_vector((nx, ny, nz)), self.S3_ref)
        n.rename("n", "tmp")
        return n

    def compute_mesh(self, res):
        N = int((self.t_max-self.t_min)/(self.s_max-self.s_min))
        ref_mesh = df.RectangleMesh.create(
            [df.Point(self.t_min, self.s_min),
             df.Point(self.t_max, self.s_max)],
            [N*res, res], df.cpp.mesh.CellType.Type.triangle)
        self.ref_mesh = ref_mesh

    def compute_pbc(self):
        self.pbc = None

    def initialize_ref_space(self, res):
        self.compute_mesh(100)
        self.compute_pbc()

        self.dS_ref = df.Measure("dx", domain=self.ref_mesh)

        ref_el = df.FiniteElement("Lagrange", self.ref_mesh.ufl_cell(), 1)
        self.S_ref = df.FunctionSpace(self.ref_mesh, ref_el,
                                      constrained_domain=self.pbc)
        self.S3_ref = df.FunctionSpace(self.ref_mesh,
                                       df.MixedElement(
                                           (ref_el, ref_el, ref_el)),
                                       constrained_domain=self.pbc)

        T_vals = df.interpolate(df.Expression("x[0]", degree=1),
                                self.S_ref)
        S_vals = df.interpolate(df.Expression("x[1]", degree=1),
                                self.S_ref)

        self.t_vals = T_vals.vector().get_local()
        self.s_vals = S_vals.vector().get_local()

    def local_area(self):
        local_area = df.project(self.sqrt_g*df.CellVolume(self.ref_mesh),
                                self.S_ref)
        local_area.rename("localarea", "tmp")
        return local_area


class SphereMap(GeoMap):
    def __init__(self, R):
        t, s = sp.symbols('t s')
        x = R * sp.cos(t) * sp.sin(s)
        y = R * sp.sin(t) * sp.sin(s)
        z = R * sp.cos(s)

        t_min = 0.
        s_min = 0.
        t_max = 2*np.pi
        s_max = np.pi

        ts = (t, s)
        xyz = (x, y, z)
        ts_min = (t_min, s_min)
        ts_max = (t_max, s_max)
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max)

    def compute_mesh(self, res, eps=1e-2):
        self.eps = eps
        N = int((self.t_max-self.t_min)/(self.s_max-self.s_min))
        ref_mesh = df.RectangleMesh.create(
            [df.Point(self.t_min, self.s_min + eps),
             df.Point(self.t_max, self.s_max - eps)],
            [N*res, res], df.cpp.mesh.CellType.Type.triangle)
        self.ref_mesh = ref_mesh

    def compute_pbc(self):
        ts_min = (self.t_min, self.s_min)
        ts_max = (self.t_max, self.s_max)
        self.pbc = SpherePBC(ts_min, ts_max)


class CylinderMap(GeoMap):
    def __init_(self, R, L):
        t, s = sp.symbols('t s')
        x = R * sp.cos(t)
        y = R * sp.sin(t)
        z = s
        #self.map["x"] = self.t
        #self.map["y"] = self.s
        #self.map["z"] = 2 * R * sp.exp(-(
        #    (self.t-Lx/2)**2+(self.s-Ly/2)**2)/(2*0.5**2))
        t_min = 0.
        t_max = 2*np.pi
        s_min = 0.
        s_max = L

        ts = (t, s)
        xyz = (x, y, z)
        ts_min = (t_min, s_min)
        ts_max = (t_max, s_max)
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max)

    def compute_pbc(self):
        ts_min = (self.t_min, self.s_min)
        ts_max = (self.t_max, self.s_max)
        self.pbc = CylinderPBC(ts_min, ts_max)

R = 1.0

geo_map = SphereMap(R)
geo_map.initialize_ref_space(100)
# S_ref = geo_map.S_ref
# S3_ref = geo_map.S3_ref
# theta = geo_map.t_vals
# phi = geo_map.s_vals
ref_mesh = geo_map.ref_mesh
geo_map.initialize_metric()

n = geo_map.normal()
xyz = geo_map.coords()

K = geo_map.get_function("K")
H = geo_map.get_function("H")
local_area = geo_map.local_area()

dump_xdmf(xyz)
dump_xdmf(n)
dump_xdmf(K)
dump_xdmf(H)
dump_xdmf(local_area)

area_ref = df.assemble(df.Constant(1.0)*geo_map.dS_ref)
area = df.assemble(geo_map.form(df.Constant(1.0)))
print(area_ref, area)

u = df.TrialFunction(geo_map.S_ref)
v = df.TestFunction(geo_map.S_ref)

Lx = geo_map.t_max
Ly = geo_map.s_max
u_0 = df.Expression("exp(-(pow(x[0]-x01, 2)+pow(x[1]-y01, 2))/"
                    "(2*pow(sigma, 2)))",
                    x01=Lx/2, y01=Ly/2,
                    sigma=Lx/30,
                    degree=1)
f = df.Expression("0.0", degree=1)

u_ = df.Function(geo_map.S_ref, name="u")
u_1 = df.interpolate(u_0, geo_map.S_ref)
u_.assign(u_1)

dt = 0.1

F_t = geo_map.form(1./dt*(u-u_1)*v)  # sqrt_g*dS_ref
F_diff = geo_map.form(geo_map.dot(u, v))  # sqrt_g*dS_ref
F = F_t + F_diff

a_surf = df.lhs(F)
L_surf = df.rhs(F)

A = df.assemble(a_surf)

solver = df.PETScKrylovSolver("gmres")  # gmres
solver.set_operator(A)


def top(x, on_boundary):
    return on_boundary and x[1] > geo_map.s_max-geo_map.eps-100*df.DOLFIN_EPS


def btm(x, on_boundary):
    return on_boundary and x[1] < geo_map.s_min+geo_map.eps+100*df.DOLFIN_EPS


Top = df.AutoSubDomain(top)
Btm = df.AutoSubDomain(btm)
facets = df.MeshFunction("size_t", ref_mesh, ref_mesh.topology().dim()-1, 0)
Top.mark(facets, 1)
Btm.mark(facets, 2)

t = 0.0
T = 10.0
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

    print("int T =", df.assemble(geo_map.form(u_)))
    flx_top = df.assemble(u_.dx(1)*df.ds(
        1, domain=ref_mesh, subdomain_data=facets))
    flx_btm = df.assemble(u_.dx(1)*df.ds(
        2, domain=ref_mesh, subdomain_data=facets))
    print("flux  =", flx_top, flx_btm)

    u_1.assign(u_)
    t += dt

xdmfff.close()








