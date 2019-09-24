import sympy as sp
import numpy as np
import dolfin as df
from bcs import EllipsoidPBC, CylinderPBC
from common.mesh_refinement import densified_ellipsoid_mesh
from common.utilities import NdFunction
from common.io import load_mesh
from common.cmd import info_red, info_cyan
import os
import ufl


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
        print("Metric computed")

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
        self.map["K_ts"] = self.map["K_st"]
        self.map["K_tt"] = sp.simplify(self.map["nx"]*self.map["xtt"]
                                       + self.map["ny"]*self.map["ytt"]
                                       + self.map["nz"]*self.map["ztt"])
        print('K_ij computed, computing K^i_j ... ')
        self.map["Ks_s"] = sp.simplify(self.map["gss"]*self.map["K_ss"]
                                       + self.map["gst"]*self.map["K_ts"])
        self.map["Ks_t"] = sp.simplify(self.map["gss"]*self.map["K_st"]
                                       + self.map["gst"]*self.map["K_tt"])
        self.map["Kt_s"] = sp.simplify(self.map["gst"]*self.map["K_ss"]
                                       + self.map["gtt"]*self.map["K_ts"])
        self.map["Kt_t"] = sp.simplify(self.map["gst"]*self.map["K_st"]
                                       + self.map["gtt"]*self.map["K_tt"])
        print('K^i_j computed, computing K^ij ... ')

        # Skipping "simplify" because it is exceedingly slow:
        self.map["Kss"] = (self.map["gss"]*self.map["Ks_s"]
                                      + self.map["gst"]*self.map["Ks_t"])
        self.map["Kst"] = (self.map["gst"]*self.map["Ks_s"]
                                      + self.map["gtt"]*self.map["Ks_t"])
        self.map["Kts"] = (self.map["gss"]*self.map["Kt_s"]
                                      + self.map["gst"]*self.map["Kt_t"])
        self.map["Ktt"] = (self.map["gst"]*self.map["Kt_s"]
                                      + self.map["gtt"]*self.map["Kt_t"])
        print('K^ij computed.')
        # Curvature tensor K^i_j as a matrix
        self.map["Kmat"] = sp.Matrix([[self.map["Kt_t"],self.map["Kt_s"]], [self.map["Kt_s"],self.map["Ks_s"]]])
        # self.map["Keigenvects"] = self.map["Kmat"].eigenvects()
        # self.map["Keigenvals"] = self.map["Kmat"].eigenvals()
        # # Save principal curvatures and principal directions:
        # self.map["k1"] = self.map["Keigenvects"][0][0]
        # self.map["u1"] = self.map["Keigenvects"][0][2][0]
        # self.map["u1t"] = self.map["u1"][0]
        # self.map["u1s"] = self.map["u1"][1]
        # self.map["k2"] = self.map["Keigenvects"][1][0]
        # self.map["u2"] = self.map["Keigenvects"][1][2][0]
        # self.map["u2t"] = self.map["u2"][0]
        # self.map["u2s"] = self.map["u2"][1]
        print("Curvature tensor computed")

        # Mean and Gaussian curvature (no simplify, too heavy)
        print('Computing scalar curvatures ...')
        self.map["H"] = ((self.map["Ks_s"] + self.map["Kt_t"])/2)
        self.map["K"] = (self.map["Ks_s"]*self.map["Kt_t"]
                                    - self.map["Ks_t"]*self.map["Kt_s"])

        print('Computing Christoffel symbols ...')
        # Christoffel symbols
        self.map["Gs_ss"] = ((
            self.map["gss"]*self.map["g_ss_s"]
            + self.map["gst"]*(
                2*self.map["g_st_s"]-self.map["g_ss_t"]))/2)
        self.map["Gs_st"] = ((
            self.map["gss"]*self.map["g_ss_t"]
            + self.map["gst"]*self.map["g_tt_s"])/2)
        self.map["Gs_tt"] = ((
            self.map["gss"]*(
                2*self.map["g_st_t"] - self.map["g_tt_s"])
            + self.map["gst"]*self.map["g_tt_t"])/2)
        self.map["Gt_ss"] = ((
            self.map["gtt"]*(
                2*self.map["g_st_s"] - self.map["g_ss_t"])
            + self.map["gst"]*self.map["g_ss_s"])/2)
        self.map["Gt_st"] = ((
            self.map["gtt"]*self.map["g_tt_s"]
            + self.map["gst"]*self.map["g_ss_t"])/2)
        self.map["Gt_tt"] = ((
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
        f.vector()[:] = F
        return f

    def initialize(self, res, restart_folder=None):
        if restart_folder is None:
            self.compute_mesh(res)
            self.compute_pbc()
            isgood = False
            while not isgood:
                self.initialize_ref_space(res)
                self.initialize_metric()
                isgood = self.recompute_mesh(res)
        else:
            info_red("Load mesh from checkpoint")
            self.ref_mesh = load_mesh(os.path.join(restart_folder,
                                                   "fields.h5"),
                                      use_partition_from_file=True)
            self.compute_pbc()
            self.initialize_ref_space(res)
            self.initialize_metric()

    def initialize_metric(self):  # , S_ref, t_vals, s_vals):
        self.g_tt = self.get_function("g_tt")
        self.g_st = self.get_function("g_st")
        self.g_ss = self.get_function("g_ss")
        self.gtt = self.get_function("gtt")
        self.gst = self.get_function("gst")
        self.gss = self.get_function("gss")
        self.sqrt_g = self.get_function("sqrt_g")

        self.Ktt = self.get_function("Ktt")
        self.Kts = self.get_function("Kts")
        self.Kst = self.get_function("Kst")
        self.Kss = self.get_function("Kss")

        self.K_tt = self.get_function("K_tt")
        self.K_ts = self.get_function("K_ts")
        self.K_st = self.get_function("K_st")
        self.K_ss = self.get_function("K_ss")

        self.Kt_t = self.get_function("Kt_t")
        self.Kt_s = self.get_function("Kt_s")
        self.Ks_t = self.get_function("Ks_t")
        self.Ks_s = self.get_function("Ks_s")

        # Curvature tensor as matrix
        #self.Kmat = self.get_function("Kmat")

        # Principal curvatures and directions:
        # self.k1 = self.get_function("k1")
        # self.k2 = self.get_function("k2")
        # self.u1t = self.get_function("u1t")
        # self.u1s = self.get_function("u1s")
        # self.u1 = ufl.as_tensor([[self.u1t],
        #                           [self.u1s]])
        # self.u2t = self.get_function("u2t")
        # self.u2s = self.get_function("u2s")
        # self.u2 = ufl.as_tensor([[self.u2t],
        #                           [self.u2s]])
        self.K = self.get_function("K")
        self.H = self.get_function("H")

        self.Gs_ss = self.get_function("Gs_ss")
        self.Gs_st = self.get_function("Gs_st")
        self.Gs_tt = self.get_function("Gs_tt")
        self.Gt_ss = self.get_function("Gt_ss")
        self.Gt_st = self.get_function("Gt_st")
        self.Gt_tt = self.get_function("Gt_tt")

        self.gab = ufl.as_tensor([[self.gtt, self.gst],
                                  [self.gst, self.gss]])  # Inverse metric, g^ij
        self.Kab = ufl.as_tensor([[self.Ktt, self.Kst],
                                  [self.Kst, self.Kss]])  # K^{ij}
        self.K_ab = ufl.as_tensor([[self.K_tt, self.K_st],
                                   [self.K_st, self.K_ss]])  # K_{ij}
        self.Ka_b = ufl.as_tensor([[self.Kt_t, self.Kt_s],
                                   [self.Ks_t, self.Ks_s]])  # K^i_j
        self.Ga_bc = ufl.as_tensor([[[self.Gt_tt, self.Gt_st],
                                     [self.Gt_st, self.Gt_ss]],
                                    [[self.Gs_tt, self.Gs_st],
                                     [self.Gs_st, self.Gs_ss]]])  # Christoffel symbols

        # We assume the format
        # ufl.as_tensor([[[ttt, tts], [tst, tss]], [[stt, sts], [sst, sss]]])

    def CovD10(self, V):
        """ Takes covariant derivative of a (1,0) tensor V -- a vector. """
        # Christoffel symbols: Ga_bc[i,j,k]
        i, j, k = ufl.Index(), ufl.Index(), ufl.Index(), ufl.Index()
        nablaV = ufl.as_tensor(V[i].dx(j) + self.Ga_bc[i, j, k]*V[k], (i, j))
        return nablaV

    def CovD01(self, W):
        """Takes covariant derivative of a (0,1) tensor W -- a co-vector or
        one-form."""
        # Christoffel symbols: Ga_bc[i,j,k]
        i, j, k = ufl.Index(), ufl.Index(), ufl.Index()
        nablaW = ufl.as_tensor(W[i].dx(j) - self.Ga_bc[k, j, i]*W[k], (i, j))
        return nablaW

    def CovD02(self, T):
        """ Takes covariant derivative of a (0,2) tensor T. """
        # Christoffel symbols: Ga_bc[i,j,k]
        i, j, k, l = ufl.Index(), ufl.Index(), ufl.Index(), ufl.Index()
        nablaT = ufl.as_tensor(T[i, j].dx(k) - self.Ga_bc[l, k, i]*T[l, j]
                               - self.Ga_bc[l, k, j]*T[i, l], (i, j, k))
        return nablaT

    def CovD11(self, T):
        """ Takes covariant derivative of a (1,1) tensor T. """
        # Christoffel symbols: Ga_bc[i,j,k]
        i, j, k, l = ufl.Index(), ufl.Index(), ufl.Index(), ufl.Index()
        nablaT = ufl.as_tensor(T[i, j].dx(k) + self.Ga_bc[i, k, l]*T[l, j]
                               - self.Ga_bc[l, k, j]*T[i, l], (i, j, k))
        return nablaT

    def CovD20(self, T):
        """ Takes covariant derivative of a (2,0) tensor T. """
        # Christoffel symbols: Ga_bc[i,j,k]
        i, j, k, l = ufl.Index(), ufl.Index(), ufl.Index(), ufl.Index()
        nablaT = ufl.as_tensor(T[i, j].dx(k) + self.Ga_bc[i, k, l]*T[l, j]
                               + self.Ga_bc[j, k, l]*T[i, l], (i, j, k))
        return nablaT

    def LB00(self, psi):
        """ Applies Laplace-Beltrami operator to a scalar """
        i, j = ufl.Index(), ufl.Index()
        gradpsi = ufl.as_tensor(psi.dx(i), (i))
        ddpsi = self.CovD01(gradpsi)
        LBpsi = self.gab[i, j] * ddpsi[i, j]
        return LBpsi

    def surfdot(self, u, v):
        i, j = ufl.Index(), ufl.Index()
        return self.gab[i, j]*u[i]*v[j]

    def dotgrad(self, u, v):
        i, j = ufl.Index(), ufl.Index()
        return self.gab[i, j]*u.dx(i)*v.dx(j)

    def dotcurvgrad(self, u, v):
        i, j = ufl.Index(), ufl.Index()
        return self.Kab[i, j]*u.dx(i)*v.dx(j)

    def form(self, integrand):
        return integrand*self.sqrt_g*self.dS_ref

    def coords(self):
        # NOTE: Doesn't work for geometris that are periodic in 3d
        x = self.get_function("x")
        y = self.get_function("y")
        z = self.get_function("z")
        xyz = NdFunction([x, y, z], name="xyz")
        xyz()
        return xyz

    def get_curvaturematrix(self):
        Kt_t = self.get_function("Kt_t")
        Kt_s = self.get_function("Kt_s")
        Ks_t = self.get_function("Ks_t")
        Ks_s = self.get_function("Ks_s")

        Kt_t_arr = Kt_t.vector().get_local()
        Kt_s_arr = Kt_s.vector().get_local()
        Ks_t_arr = Ks_t.vector().get_local()
        Ks_s_arr = Ks_s.vector().get_local()

        kappa1_arr = Ks_s.vector().get_local() # Fiducial initiation
        kappa2_arr = Ks_s.vector().get_local() # Fiducial initiation
        u1t_arr = Ks_s.vector().get_local() # Fiducial initiation
        u1s_arr = Ks_s.vector().get_local() # Fiducial initiation
        u2t_arr = Ks_s.vector().get_local() # Fiducial initiation
        u2s_arr = Ks_s.vector().get_local() # Fiducial initiation
        theta1_arr = Ks_s.vector().get_local() # Fiducial initiation
        theta2_arr = Ks_s.vector().get_local() # Fiducial initiation

        i = 0
        for Kt_t_loc, Kt_s_loc, Ks_t_loc, Ks_s_loc in zip(Kt_t_arr, Kt_s_arr, Ks_t_arr, Ks_s_arr):
            try:
                Kmat_loc = np.array([[Kt_t_loc,Kt_s_loc],[Ks_t_loc,Ks_s_loc]])
                kappa_loc, u_loc = np.linalg.eig(Kmat_loc)
                # Possible orderings of curvatures:
                # Absolute curvature: Gives discontinuous kappa_i, but is useful
                # when only the most curved direction is relevant.
                # Signed curvature: Gives continuous kappa_i, but places emphasis
                # on something rather irrelevant, namly the sign of the curvature.
                #if abs(kappa_loc[0]) > abs(kappa_loc[1]):
                if kappa_loc[0] > kappa_loc[1]:
                    kappa1_arr[i] = kappa_loc[0]
                    kappa2_arr[i] = kappa_loc[1]
                    u1t_arr[i] = u_loc[0,0]
                    u1s_arr[i] = u_loc[1,0]
                    u2t_arr[i] = u_loc[0,1]
                    u2s_arr[i] = u_loc[1,1]
                else:
                    kappa1_arr[i] = kappa_loc[1]
                    kappa2_arr[i] = kappa_loc[0]
                    u1t_arr[i] = u_loc[0,1]
                    u1s_arr[i] = u_loc[1,1]
                    u2t_arr[i] = u_loc[0,0]
                    u2s_arr[i] = u_loc[1,0]
                theta1_arr[i] = np.arctan2(u1s_arr[i],u1t_arr[i])
                theta2_arr[i] = np.arctan2(u2s_arr[i],u2t_arr[i])
            except:
                kappa1_arr[i] = 0
                kappa2_arr[i] = 0

                u1t_arr[i] = 0
                u1s_arr[i] = 0
                u2t_arr[i] = 0
                u2s_arr[i] = 0

                theta1_arr[i] = 0
                theta2_arr[i] = 0
            i += 1
        kappa1 = df.Function(self.S_ref)
        kappa1.vector().set_local(kappa1_arr)
        kappa2 = df.Function(self.S_ref)
        kappa2.vector().set_local(kappa2_arr)

        theta1 = df.Function(self.S_ref)
        theta1.vector().set_local(theta1_arr)
        theta2 = df.Function(self.S_ref)
        theta2.vector().set_local(theta2_arr)

        return kappa1, kappa2, theta1, theta2

    def normal(self):
        nx = self.get_function("nx")
        ny = self.get_function("ny")
        nz = self.get_function("nz")
        n = NdFunction([nx, ny, nz], name="n")
        n()
        return n

    def compute_mesh(self, res):
        N = int(np.ceil((self.t_max-self.t_min)/(self.s_max-self.s_min)))
        ref_mesh = df.RectangleMesh.create(
            [df.Point(self.t_min, self.s_min),
             df.Point(self.t_max, self.s_max)],
            [N*res, res], df.cpp.mesh.CellType.Type.triangle)
        import matplotlib.pyplot as plt
        #df.plot(ref_mesh)
        #plt.show()
        self.ref_mesh = ref_mesh

    def recompute_mesh(self, res):
        return True

    def compute_pbc(self):
        self.pbc = None

    def mixed_space(self, mixed_el):
        return df.FunctionSpace(self.ref_mesh,
                                df.MixedElement(mixed_el),
                                constrained_domain=self.pbc)

    def initialize_ref_space(self, res):
        self.dS_ref = df.Measure("dx", domain=self.ref_mesh)

        self.ref_el = df.FiniteElement("Lagrange", self.ref_mesh.ufl_cell(), 1)
        self.S_ref = df.FunctionSpace(self.ref_mesh, self.ref_el,
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

    def is_periodic_in_3d(self):
        return False


class EllipsoidMap(GeoMap):
    def __init__(self, Rx, Ry, Rz):
        t, s = sp.symbols('t s', real=True)
        x = Rx * sp.cos(t) * sp.sin(s)
        y = Ry * sp.sin(t) * sp.sin(s)
        z = Rz * sp.cos(s)
        self.Rx = Rx
        self.Ry = Ry
        self.Rz = Rz

        t_min = 0.
        s_min = 0.
        t_max = 2*np.pi
        s_max = np.pi

        ts = (t, s)
        xyz = (x, y, z)
        ts_min = (t_min, s_min)
        ts_max = (t_max, s_max)
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max)

    def compute_mesh(self, res, eps=1e-3):
        self.eps = eps
        self.ref_mesh = densified_ellipsoid_mesh(
            4*0.25*res**2, self.Rx, self.Ry, self.Rz, eps=eps)

    def recompute_mesh(self, res):
        # disable refinement
        return True

        S_tot = df.assemble(self.form(df.Constant(1.)))
        deltaS_max = S_tot/res**2

        cell_markers = df.MeshFunction("bool", self.ref_mesh,
                                       self.ref_mesh.topology().dim())
        cell_markers.set_all(False)
        isgood = True

        for cell in df.cells(self.ref_mesh):
            deltaS_ref = cell.volume()
            sqrt_g_loc = self.sqrt_g(cell.midpoint())
            deltaS = sqrt_g_loc*deltaS_ref
            if deltaS >= deltaS_max:
                isgood = False
                cell_markers[cell] = True

        if not isgood:
            self.ref_mesh = df.refine(self.ref_mesh, cell_markers)

        return isgood

    def compute_pbc(self):
        ts_min = (self.t_min, self.s_min)
        ts_max = (self.t_max, self.s_max)
        self.pbc = EllipsoidPBC(ts_min, ts_max)


class SphereMap(EllipsoidMap):
    def __init__(self, R):
        EllipsoidMap.__init__(self, R, R, R)

class CylinderMap(GeoMap):
    def __init__(self, R, L, double_periodic=True):
        t, s = sp.symbols('t s', real=True)
        x = R * sp.cos(t/R)
        y = R * sp.sin(t/R)
        z = s

        t_min = 0.
        t_max = 2*np.pi*R
        s_min = 0.
        s_max = L

        ts = (t, s)
        xyz = (x, y, z)
        ts_min = (t_min, s_min)
        ts_max = (t_max, s_max)
        self.double_periodic = double_periodic
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max)

    def compute_pbc(self):
        ts_min = (self.t_min, self.s_min)
        ts_max = (self.t_max, self.s_max)
        self.pbc = CylinderPBC(ts_min, ts_max,
                               double_periodic=self.double_periodic)

    def is_periodic_in_3d(self):
        return self.double_periodic

class GaussianBumpMap(GeoMap):
    def __init__(self, Lx, Ly, h, sigma):
        t, s = sp.symbols('t s', real=True)
        x = t
        y = s
        z = h * sp.exp(-(t**2+s**2)/(2*sigma**2))

        t_min = -Lx/2
        t_max = Lx/2
        s_min = -Ly/2
        s_max = Ly/2

        ts = (t, s)
        xyz = (x, y, z)
        ts_min = (t_min, s_min)
        ts_max = (t_max, s_max)
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max)
    def compute_mesh(self, res):
        import mshr
        print("Using overloaded compute_mesh for GaussianBump")
        N = int(np.ceil((self.t_max-self.t_min)/(self.s_max-self.s_min)))
        rect = mshr.Rectangle(df.Point(self.t_min, self.s_min), df.Point(self.t_max, self.s_max))
        ref_mesh = mshr.generate_mesh(rect, res, "cgal")
        self.ref_mesh = ref_mesh

class SaddleMap(GeoMap):
    def __init__(self, Lx, Ly, a, b):
        t, s = sp.symbols('t s', real=True)
        x = t
        y = s
        z = a*t**2-b*s**2

        t_min = -Lx/2
        t_max = Lx/2
        s_min = -Lx/2
        s_max = Lx/2

        ts = (t, s)
        xyz = (x, y, z)
        ts_min = (t_min, s_min)
        ts_max = (t_max, s_max)
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max)

    def compute_mesh(self, res):
        import mshr
        print("Using overloaded compute_mesh for Saddle geometry")
        rect = mshr.Rectangle(df.Point(self.t_min, self.s_min), df.Point(self.t_max, self.s_max))
        ref_mesh = mshr.generate_mesh(rect, res, "cgal")
        self.ref_mesh = ref_mesh
