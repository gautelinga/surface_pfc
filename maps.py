import sympy as sp
import numpy as np
import dolfin as df
from bcs import SpherePBC, CylinderPBC


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
        x = self.get_function("x")
        y = self.get_function("y")
        z = self.get_function("z")
        xyz = df.project(df.as_vector((x, y, z)), self.S3_ref)
        xyz.rename("xyz", "tmp")
        return xyz

    def normal(self):
        nx = self.get_function("nx")
        ny = self.get_function("ny")
        nz = self.get_function("nz")
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


class EllipsoidMap(GeoMap):
    def __init__(self, Rx, Ry, Rz):
        t, s = sp.symbols('t s')
        x = Rx * sp.cos(t) * sp.sin(s)
        y = Ry * sp.sin(t) * sp.sin(s)
        z = Rz * sp.cos(s)

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
            [N*res, res], df.cpp.mesh.CellType.Type.quadrilateral)
        self.ref_mesh = ref_mesh

    def compute_pbc(self):
        ts_min = (self.t_min, self.s_min)
        ts_max = (self.t_max, self.s_max)
        self.pbc = SpherePBC(ts_min, ts_max)


class SphereMap(EllipsoidMap):
    def __init__(self, R):
        EllipsoidMap.__init__(self, R, R, R)


class CylinderMap(GeoMap):
    def __init_(self, R, L):
        t, s = sp.symbols('t s')
        x = R * sp.cos(t)
        y = R * sp.sin(t)
        z = s

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


class GaussianBumpMap(GeoMap):
    def __init__(self, Lx, Ly, H, sigma):
        t, s = sp.symbols('t s')
        x = t
        y = s
        z = H * sp.exp(-((t-Lx/2)**2+(s-Ly/2)**2)/(2*sigma**2))

        t_min = 0.
        t_max = Lx
        s_min = 0.
        s_max = Ly

        ts = (t, s)
        xyz = (x, y, z)
        ts_min = (t_min, s_min)
        ts_max = (t_max, s_max)
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max)
