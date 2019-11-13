import sympy as sp
import numpy as np
import dolfin as df
from bcs import EllipsoidPBC, CylinderPBC, TorusPBC
from common.mesh_refinement import densified_ellipsoid_mesh
from common.utilities import NdFunction, AssignedTensorFunction, \
    determinant, inverse
from common.io import load_mesh
from common.cmd import info_red, info_cyan, info_blue
from itertools import product
import os
import ufl
import mshr
import cloudpickle as pickle


class GeoMap:
    def __init__(self, xyz, ts, ts_min, ts_max, verbose=False):
        self.AXIS_REF = [tsi.name for tsi in ts]
        self.dim_ref = len(self.AXIS_REF)
        self.AXIS = ["x", "y", "z"]
        self.r_ref = dict()
        for d, ts_d in zip(self.AXIS_REF, ts):
            self.r_ref[d] = ts_d
        self.map = dict()
        for d, xyz_d in zip(self.AXIS, xyz):
            self.map[d] = xyz_d
        self.r_ref_min = dict()
        self.r_ref_max = dict()
        for d, ts_min_d, ts_max_d in zip(self.AXIS_REF, ts_min, ts_max):
            self.r_ref_min[d] = ts_min_d
            self.r_ref_max[d] = ts_max_d
        self.verbose = verbose
        self.evalf = dict()

    def compute_geometry(self):
        # Simple derivatives
        self.info_verbose("Computing derivatives")
        for xi, j in product(self.AXIS, self.AXIS_REF):
            self.map[xi + "_," + j] = sp.diff(self.map[xi], self.r_ref[j])

        # Double derivatives
        self.info_verbose("Computing double derivatives")
        for xi, j, k in product(self.AXIS, self.AXIS_REF, self.AXIS_REF):
            self.map[xi + "_," + j + k] = sp.diff(
                self.map[xi + "_," + j], self.r_ref[k])

        # The metric tensor
        self.info_verbose("Computing metric tensor")
        for j, k in product(self.AXIS_REF, self.AXIS_REF):
            # g_ab is symmetric
            g_jk_id = "g_" + j + k
            g_kj_id = "g_" + k + j
            if g_kj_id in self.map:
                self.map[g_jk_id] = self.map[g_kj_id]
            else:
                self.map[g_jk_id] = sum(
                    [self.map[xi + "_," + j] * self.map[xi + "_," + k]
                     for xi in self.AXIS])

        self.info_verbose("Computing determinant symbolically")
        _g_ab = [[self.map["g_" + j + k]
                  for k in self.AXIS_REF]
                 for j in self.AXIS_REF]
        self.map["g_det"] = determinant(_g_ab)

        self.map["g"] = abs(self.map["g_det"])
        self.map["sqrt_g"] = sp.sqrt(abs(self.map["g"]))

        self.info_verbose("Computing inverse metric tensor")
        _gab = inverse(_g_ab, det=self.map["g_det"])

        self.info_verbose("Getting the elements of inverse metric tensor")
        for (dj, j), (dk, k) in product(enumerate(self.AXIS_REF),
                                        enumerate(self.AXIS_REF)):
            # g^ab is symmetric
            gjk_id = "g^" + j + k
            gkj_id = "g^" + k + j
            if gjk_id in self.map:
                self.map[gjk_id] = self.map[gkj_id]
            else:
                self.map[gjk_id] = _gab[dj][dk]

        self.info_verbose("Computing derivatives of metric tensor")
        for j, k, l in product(self.AXIS_REF, self.AXIS_REF, self.AXIS_REF):
            # g_jk,l is symmetric wrt. j <-> k
            g_jkl_id = "g_" + j + k + "," + l
            g_kjl_id = "g_" + k + j + "," + l
            if g_jkl_id in self.map:
                self.map[g_jkl_id] = self.map[g_kjl_id]
            else:
                self.map[g_jkl_id] = sp.diff(
                    self.map["g_" + j + k], self.r_ref[l])

        # Curvature related quantities are
        # not available for higher dimensions
        if self.dim_ref <= 2:
            # The normal
            self.info_verbose("Computing normal")

            _v = [None, None]
            for dj, j in enumerate(self.AXIS_REF):
                _v[dj] = sp.Matrix([self.map[xi + "_," + j]
                                    for xi in self.AXIS])
            cross = _v[0].cross(_v[1])
            cross_mag = cross.norm()
            for di, xi in enumerate(self.AXIS):
                self.map["n_" + xi] = cross[di]/cross_mag

            # Curvature tensor
            self.info_verbose("Computing curvature tensor")
            self.info_verbose("Computing K_ij")
            for j, k in product(self.AXIS_REF, self.AXIS_REF):
                # K_ab is symmetric
                K_jk_id = "K_" + j + k
                K_kj_id = "K_" + k + j
                if K_kj_id in self.map:
                    self.map[K_jk_id] = self.map[K_kj_id]
                else:
                    self.map[K_jk_id] = sum(
                        [self.map["n_" + xi]*self.map[xi + "_," + j + k]
                         for xi in self.AXIS])

            self.info_verbose("Computing K^i_j")
            for j, k in product(self.AXIS_REF, self.AXIS_REF):
                self.map["K^" + j + "_" + k] = sum(
                    [self.map["g^" + j + l]*self.map["K_" + l + k]
                     for l in self.AXIS_REF])

            self.info_verbose("Computing K^ij")
            for j, k in product(self.AXIS_REF, self.AXIS_REF):
                Kjk_id = "K^" + j + k
                Kkj_id = "K^" + k + j
                if Kkj_id in self.map:
                    self.map[Kjk_id] = self.map[Kkj_id]
                else:
                    self.map[Kjk_id] = sum(
                        [self.map["g^" + l + k]*self.map["K^" + j + "_" + l]
                         for l in self.AXIS_REF])

            self.info_verbose("Curvature tensor computed")

            # Mean and Gaussian curvature (no simplify, too heavy)
            self.info_verbose("Computing scalar curvatures")
            self.map["H"] = sum([self.map["K^" + j + "_" + j]
                                for j in self.AXIS_REF])/2
            self.map["K"] = determinant([[self.map["K^" + j + "_" + k]
                                        for k in self.AXIS_REF]
                                        for j in self.AXIS_REF])

        # Christoffel symbols
        self.info_verbose("Computing Christoffel symbols")
        for j, k, l in product(self.AXIS_REF, self.AXIS_REF, self.AXIS_REF):
            # Ga_bc is symmetric wrt. b <-> c
            Gj_kl_id = "G^" + j + "_" + k + l
            Gj_lk_id = "G^" + j + "_" + l + k
            if Gj_kl_id in self.map:
                self.map[Gj_lk_id] = self.map[Gj_kl_id]
            else:
                self.map[Gj_lk_id] = sum(
                    [self.map["g^" + j + m]*(
                        self.map["g_" + m + k + "," + l]
                        + self.map["g_" + m + l + "," + k]
                        - self.map["g_" + k + l + "," + m])
                     for m in self.AXIS_REF])

    def eval(self, key):
        if key not in self.evalf:
            self.info_verbose("Lambdifying: {}".format(key))
            self.evalf[key] = sp.lambdify(
                [self.r_ref[j] for j in self.AXIS_REF],
                self.map[key], "numpy")

        # v = self.evalf[key](self.t_vals, self.s_vals)
        v = self.evalf[key](*[self.r_ref_vals[j] for j in self.AXIS_REF])
        if isinstance(v, int) or isinstance(v, float):
            # length = len(self.t_vals)
            length = len(self.r_ref_vals[self.AXIS_REF[0]])
            return v*np.ones(length)
        else:
            return v

    def make_function(self, key):
        f = df.Function(self.S_ref)
        f.rename(key, "tmp")
        return f

    def get_function(self, key):  # , space, t_vals, s_vals):
        f = self.make_function(key)
        F = self.eval(key)
        f.vector()[:] = F
        return f

    def initialize(self, res, restart_folder=None):
        if restart_folder is not None:
            evalf_filename = os.path.join(restart_folder, "evalf.pkl")
            if os.path.exists(evalf_filename):
                self.info_verbose("Loading stored evalf from checkpoint")
                with open(evalf_filename, "rb") as f:
                    self.evalf = pickle.load(f)
            map_filename = os.path.join(restart_folder, "map.pkl")
            if os.path.exists(map_filename):
                self.info_verbose("Loading stored map from checkpoint")
                with open(map_filename, "rb") as f:
                    self.map = pickle.load(f)
        self.compute_geometry()
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

    def _dot_pointwise(self, a, b, key):
        self.info_verbose("Computing pointwise: {}".format(key))
        f = self.make_function(key)
        f_vec = np.zeros_like(f.vector().get_local())
        for ai, bi in zip(a, b):
            f_vec[:] += ai.vector().get_local()*bi.vector().get_local()
        f.vector()[:] = f_vec
        return f

    def _det_pointwise_vec(self, A_vec):
        dims = np.shape(A_vec)
        f_vec = np.zeros(dims[2])
        for i in range(dims[2]):
            f_vec[i] = np.linalg.det(A_vec[:, :, i])
        return f_vec

    def _det_pointwise(self, A, key):
        self.info_verbose("Computing determinant pointwise: {}".format(key))
        dims = np.shape(A)
        assert(dims[0] == dims[1])
        A_vec = [[None for _ in range(dims[0])] for _ in range(dims[0])]
        for di in range(dims[0]):
            for dj in range(dims[0]):
                A_vec[di][dj] = A[di][dj].vector().get_local()
        f = self.make_function(key)
        f.vector()[:] = self._det_pointwise_vec(np.array(A_vec))
        return f

    def initialize_metric(self):  # , S_ref, t_vals, s_vals):
        self._g = dict()
        for j, k in product(self.AXIS_REF, self.AXIS_REF):
            # g_ab is symmetric
            if "_" + k + j in self._g:
                self._g["_" + j + k] = self._g["_" + k + j]
            else:
                self._g["_" + j + k] = self.get_function("g_" + j + k)
            # gab is symmetric
            if "^" + k + j in self._g:
                self._g["^" + j + k] = self._g["^" + k + j]
            else:
                self._g["^" + j + k] = self.get_function("g^" + j + k)

        self.sqrt_g = self.get_function("sqrt_g")

        _g_ab = [[self._g["_" + j + k]
                  for k in self.AXIS_REF]
                 for j in self.AXIS_REF]
        _gab = [[self._g["^" + j + k]
                 for k in self.AXIS_REF]
                for j in self.AXIS_REF]

        self.g_ab = ufl.as_tensor(_g_ab)  # Metric g_ij
        self.gab = ufl.as_tensor(_gab)  # Inverse metric, g^ij

        # Curvature quantities are not supported in higher dimensions
        if self.dim_ref <= 2:
            self._K = dict()
            for j, k in product(self.AXIS_REF, self.AXIS_REF):
                # K_ab is symmetric
                if "_" + k + j in self._K:
                    self._K["_" + j + k] = self._K["_" + k + j]
                else:
                    self._K["_" + j + k] = self.get_function("K_" + j + k)

            # Raising indices pointwise (faster than symbolically)
            for j, k in product(self.AXIS_REF, self.AXIS_REF):
                self._K["^" + j + "_" + k] = self._dot_pointwise(
                    [self._g["^" + j + m] for m in self.AXIS_REF],
                    [self._K["_" + m + k] for m in self.AXIS_REF],
                    "K^" + j + "_" + k)

            for j, k in product(self.AXIS_REF, self.AXIS_REF):
                # Kab is symmetric
                if "^" + k + j in self._K:
                    self._K["^" + j + k] = self._K["^" + k + j]
                else:
                    self._K["^" + j + k] = self._dot_pointwise(
                        [self._g["^" + k + m] for m in self.AXIS_REF],
                        [self._K["^" + j + "_" + m] for m in self.AXIS_REF],
                        "K^" + j + k)

            _K_ab = [[self._K["_" + j + k]
                      for k in self.AXIS_REF]
                     for j in self.AXIS_REF]
            _Ka_b = [[self._K["^" + j + "_" + k]
                      for k in self.AXIS_REF]
                     for j in self.AXIS_REF]
            _Kab = [[self._K["^" + j + k]
                     for k in self.AXIS_REF]
                    for j in self.AXIS_REF]

            self.K = self._det_pointwise(_Ka_b, "K")

            self.H = self.make_function("H")
            self.H.vector()[:] = sum(
                [self._K["^" + m + "_" + m].vector().get_local()
                 for m in self.AXIS_REF])/2

            self.Kab = ufl.as_tensor(_Kab)  # K^{ij}
            self.K_ab = ufl.as_tensor(_K_ab)  # K_{ij}
            self.Ka_b = ufl.as_tensor(_Ka_b)  # K^i_j

        self._G = dict()
        # for j, k, l in product(self.AXIS_REF, self.AXIS_REF, self.AXIS_REF):
        #     self._G["^" + j + "_" + k + l] = self.make_function("G" + j + "_" + k + l)
        #     self._G["^" + j + "_" + k + l].vector()[:] = sum(
        #         [self._g["^" + j + m].vector().get_local()*(
        #             self._g["_" + m + k + "_" + l].vector().get_local()
        #             + self._g["_" + m + l + "_" + k].vector().get_local()
        #             - self._g["_" + k + l + "_" + m].vector().get_local())
        #          for m in self.AXIS_REF])
        for j, k, l in product(self.AXIS_REF, self.AXIS_REF, self.AXIS_REF):
            # Ga_bc is symmetric wrt. b <-> c
            if "^" + j + "_" + l + k in self._G:
                self._G["^" + j + "_" + k + l] = self._G["^" + j + "_" + l + k]
            else:
                self._G["^" + j + "_" + k + l] = \
                  self.get_function("G^" + j + "_" + k + l)

        _Ga_bc = [[[self._G["^" + j + "_" + k + l]
                    for l in self.AXIS_REF]
                   for k in self.AXIS_REF]
                  for j in self.AXIS_REF]
        self.Ga_bc = ufl.as_tensor(_Ga_bc)  # Christoffel symbols

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
        # NOTE: Doesn't work for geometries that are periodic in 3d
        # x = self.get_function("x")
        # y = self.get_function("y")
        # z = self.get_function("z")
        # xyz = NdFunction([x, y, z], name="xyz")
        xyz = NdFunction([self.get_function(xi) for xi in self.AXIS],
                         name="xyz")
        xyz()
        return xyz

    def dcoords(self):
        # xt = self.get_function("xt")
        # yt = self.get_function("yt")
        # zt = self.get_function("zt")
        # xs = self.get_function("xs")
        # ys = self.get_function("ys")
        # zs = self.get_function("zs")
        # xyzt = NdFunction([xt, yt, zt], name="xyzt")
        # xyzs = NdFunction([xs, ys, zs], name="xyzs")
        # xyzt()
        # xyzs()
        dxyz = [None]*len(self.AXIS_REF)
        for dj, j in enumerate(self.AXIS_REF):
            dxyz[dj] = NdFunction([self.get_function(xi + "_," + j)
                                   for xi in self.AXIS],
                                  name="xyz_,{}".format(dj))
            dxyz[dj]()
        return dxyz

    def metric_tensor(self):
        # g_ab = NdFunction([self.g_tt, self.g_st, self.g_ss],
        #                   name="g_ab")
        # g_ab = NdFunction([self._g["_tt"], self._g["_st"], self._g["_ss"]],
        #                  name="g_ab")
        g_ab = AssignedTensorFunction([self._g["_" + j + k]
                                       for j, k in product(
                                           self.AXIS_REF, self.AXIS_REF)],
                                      name="g_ab")
        g_ab()
        return g_ab

    def metric_tensor_inv(self):
        # gab = NdFunction([self.gtt, self.gst, self.gss],
        #                  name="gab")
        # gab = NdFunction([self._g["^tt"], self._g["^st"], self._g["^ss"]],
        #                  name="gab")
        gab = AssignedTensorFunction([self._g["^" + j + k]
                                      for j, k in product(
                                          self.AXIS_REF, self.AXIS_REF)],
                                     name="g^ab")
        gab()
        return gab

    def curvature_tensor(self):
        # K_ab = NdFunction([self.K_tt, self.K_st, self.K_ss],
        #                   name="K_ab")
        # K_ab = NdFunction([self._K["_tt"], self._K["_st"], self._K["_ss"]],
        #                   name="K_ab")
        K_ab = AssignedTensorFunction([self._K["_" + j + k]
                                       for j, k in product(
                                               self.AXIS_REF, self.AXIS_REF)],
                                      name="K_ab")
        K_ab()
        return K_ab

    def normal(self):
        # nx = self.get_function("nx")
        # ny = self.get_function("ny")
        # nz = self.get_function("nz")
        # n = NdFunction([nx, ny, nz], name="n")
        n = NdFunction([self.get_function("n_" + xi) for xi in self.AXIS],
                       name="n")
        n()
        return n

    def compute_mesh(self, res):
        N = [0]*len(self.AXIS_REF)
        for dj, j in enumerate(self.AXIS_REF):
            N[dj] = int(np.ceil((self.r_ref_max[j]-self.r_ref_min[j])/(
                             self.r_ref_max[self.AXIS_REF[0]]
                             - self.r_ref_min[self.AXIS_REF[0]])))
        ref_mesh = df.RectangleMesh.create(
            [df.Point(*[self.r_ref_min[j] for j in self.AXIS_REF]),
             df.Point(*[self.r_ref_max[j] for j in self.AXIS_REF])],
            [Nd*res for Nd in N], df.cpp.mesh.CellType.Type.triangle)
        # N = int(np.ceil((self.t_max-self.t_min)/(self.s_max-self.s_min)))
        # ref_mesh = df.RectangleMesh.create(
        #     [df.Point(self.t_min, self.s_min),
        #      df.Point(self.t_max, self.s_max)],
        #     [N*res, res], df.cpp.mesh.CellType.Type.triangle)
        self.ref_mesh = ref_mesh

    def recompute_mesh(self, res):
        return True

    def compute_pbc(self):
        self.pbc = None

    def mixed_space(self, arg):
        if isinstance(arg, int):
            el_list = (self.ref_el,)*arg
        else:
            el_list = arg
        return df.FunctionSpace(self.ref_mesh,
                                df.MixedElement(el_list),
                                constrained_domain=self.pbc)

    def initialize_ref_space(self, res):
        self.dS_ref = df.Measure("dx", domain=self.ref_mesh)

        self.ref_el = df.FiniteElement("Lagrange", self.ref_mesh.ufl_cell(), 1)
        self.S_ref = df.FunctionSpace(self.ref_mesh, self.ref_el,
                                      constrained_domain=self.pbc)

        self.r_ref_vals = dict()
        for dj, j in enumerate(self.AXIS_REF):
            Rj_ref_vals = df.interpolate(
                df.Expression("x[{}]".format(dj), degree=1),
                self.S_ref)
            self.r_ref_vals[j] = Rj_ref_vals.vector().get_local()

        # T_vals = df.interpolate(df.Expression("x[0]", degree=1),
        #                         self.S_ref)
        # S_vals = df.interpolate(df.Expression("x[1]", degree=1),
        #                         self.S_ref)

        # self.t_vals = T_vals.vector().get_local()
        # self.s_vals = S_vals.vector().get_local()

    def local_area(self):
        local_area = df.project(self.sqrt_g*df.CellVolume(self.ref_mesh),
                                self.S_ref)
        local_area.rename("localarea", "tmp")
        return local_area

    def is_periodic_in_3d(self):
        return False

    def info_verbose(self, message):
        if self.verbose:
            info_cyan(message)


class EllipsoidMap(GeoMap):
    def __init__(self, Rx, Ry, Rz, verbose=False):
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
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max, verbose=verbose)

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
        # ts_min = (self.t_min, self.s_min)
        # ts_max = (self.t_max, self.s_max)
        ts_min = [self.r_ref_min[j] for j in self.AXIS_REF]
        ts_max = [self.r_ref_max[j] for j in self.AXIS_REF]
        self.pbc = EllipsoidPBC(ts_min, ts_max)


class SphereMap(EllipsoidMap):
    def __init__(self, R, verbose=False):
        EllipsoidMap.__init__(self, R, R, R, verbose=verbose)


class CylinderMap(GeoMap):
    def __init__(self, R, L, double_periodic=True, verbose=False):
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
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max, verbose=verbose)

    def compute_pbc(self):
        # ts_min = (self.t_min, self.s_min)
        # ts_max = (self.t_max, self.s_max)
        ts_min = [self.r_ref_min[j] for j in self.AXIS_REF]
        ts_max = [self.r_ref_max[j] for j in self.AXIS_REF]
        self.pbc = CylinderPBC(ts_min, ts_max,
                               double_periodic=self.double_periodic)

    def is_periodic_in_3d(self):
        return self.double_periodic


class GaussianBumpMap(GeoMap):
    def __init__(self, Lx, Ly, h, sigma, verbose=False):
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
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max, verbose=verbose)

    def compute_mesh(self, res):
        self.info_verbose("Using overloaded compute_mesh for GaussianBump")
        t_min = self.r_ref_min["t"]
        t_max = self.r_ref_max["t"]
        s_min = self.r_ref_min["s"]
        s_max = self.r_ref_max["s"]
        # N = int(np.ceil((t_max-t_min)/(s_max-s_min)))
        rect = mshr.Rectangle(df.Point(t_min, s_min),
                              df.Point(t_max, s_max))
        ref_mesh = mshr.generate_mesh(rect, res, "cgal")
        self.ref_mesh = ref_mesh


class GaussianBumpMapPBC(GeoMap):
    def __init__(self, Lx, Ly, h, sigma, double_periodic=True, verbose=False):
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
        self.double_periodic = double_periodic
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max, verbose=verbose)

    def compute_pbc(self):
        # ts_min = (self.t_min, self.s_min)
        # ts_max = (self.t_max, self.s_max)
        ts_min = [self.r_ref_min[j] for j in self.AXIS_REF]
        ts_max = [self.r_ref_max[j] for j in self.AXIS_REF]
        self.pbc = CylinderPBC(ts_min, ts_max,
                               double_periodic=self.double_periodic)

    def is_periodic_in_3d(self):
        return self.double_periodic


class GaussianBumpMapRound(GeoMap):
    def __init__(self, R, h, sigma, verbose=False):
        t, s = sp.symbols('t s', real=True)
        x = t
        y = s
        z = h * sp.exp(-(t**2+s**2)/(2*sigma**2))

        t_min = -R
        t_max = R
        s_min = -R
        s_max = R

        ts = (t, s)
        xyz = (x, y, z)
        ts_min = (t_min, s_min)
        ts_max = (t_max, s_max)
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max, verbose=verbose)

    def compute_mesh(self, res):
        self.info_verbose("Using overloaded compute_mesh for GaussianBump")
        t_min = self.r_ref_min["t"]
        t_max = self.r_ref_max["t"]
        circ = mshr.Circle(df.Point(0, 0), (t_max-t_min)/2)
        ref_mesh = mshr.generate_mesh(circ, res, "cgal")
        self.ref_mesh = ref_mesh


class SaddleMap(GeoMap):
    def __init__(self, Lx, Ly, a, b, verbose=False):
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
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max, verbose=verbose)

    def compute_mesh(self, res):
        self.info_verbose("Using overloaded compute_mesh for Saddle geometry")
        t_min = self.r_ref_min["t"]
        t_max = self.r_ref_max["t"]
        s_min = self.r_ref_min["s"]
        s_max = self.r_ref_max["s"]
        rect = mshr.Rectangle(df.Point(t_min, s_min),
                              df.Point(t_max, s_max))
        ref_mesh = mshr.generate_mesh(rect, res, "cgal")
        self.ref_mesh = ref_mesh


class BumpyMap(GeoMap):
    def __init__(self, Lx, Ly, amplitudes, wavenumbers, verbose=False):
        # Lx, Ly, maximum amplitude, maximum wavenumber
        t, s = sp.symbols('t s', real=True)
        x = t
        y = s
        # Generate the height function:
        z = 0
        n_k = 0
        for a in amplitudes:
            # k -> 2*(np.random.random()-0.5)*k
            k1 = wavenumbers[n_k]
            k2 = wavenumbers[n_k+1]
            k3 = wavenumbers[n_k+2]
            k4 = wavenumbers[n_k+3]
            z += a*sp.cos(k1*t+k2*s)*sp.cos(k3*t+k4*s)
            n_k += 4

        t_min = -Lx/2
        t_max = Lx/2
        s_min = -Lx/2
        s_max = Lx/2

        ts = (t, s)
        xyz = (x, y, z)
        ts_min = (t_min, s_min)
        ts_max = (t_max, s_max)
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max, verbose=verbose)

    # def compute_mesh(self, res):
    #     self.info_verbose("Using overloaded compute_mesh for Bumpy geometry")
    #     rect = mshr.Rectangle(df.Point(self.t_min, self.s_min),
    #                           df.Point(self.t_max, self.s_max))
    #     ref_mesh = mshr.generate_mesh(rect, res, "cgal")
    #     self.ref_mesh = ref_mesh


class RoughMap(GeoMap):
    def __init__(self, Lx, Ly, amplitude, modes_file,
                 double_periodic=True,
                 verbose=False):
        # Lx, Ly, maximum amplitude, maximum wavenumber
        t, s = sp.symbols('t s', real=True)
        x = t
        y = s

        t_min = -Lx/2
        t_max = Lx/2
        s_min = -Lx/2
        s_max = Lx/2

        # Generate the height function
        z = 0
        modes_data = np.loadtxt(modes_file)
        i_ = modes_data[:, 0].astype(int)
        k_ = modes_data[:, 1].astype(int)
        a_ = modes_data[:, 2]
        b_ = modes_data[:, 3]
        abnorm = np.sqrt(np.sum(a_**2 + b_**2))
        a_ /= abnorm
        b_ /= abnorm

        for i, k, a, b in zip(i_, k_, a_, b_):
            phi_i = 2*np.pi*i*t/(t_max-t_min)
            phi_k = 2*np.pi*k*s/(s_max-s_min)
            z += amplitude*(a * sp.cos(phi_i + phi_k)
                            + b * sp.sin(phi_i + phi_k))

        ts = (t, s)
        xyz = (x, y, z)
        ts_min = (t_min, s_min)
        ts_max = (t_max, s_max)
        self.double_periodic = True
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max, verbose=verbose)

    def compute_pbc(self):
        # ts_min = (self.t_min, self.s_min)
        # ts_max = (self.t_max, self.s_max)
        ts_min = [self.r_ref_min[j] for j in self.AXIS_REF]
        ts_max = [self.r_ref_max[j] for j in self.AXIS_REF]
        self.pbc = CylinderPBC(ts_min, ts_max,
                               double_periodic=self.double_periodic)

    def is_periodic_in_3d(self):
        return self.double_periodic


class SaddleMapRound(GeoMap):
    def __init__(self, R, a, b):
        t, s = sp.symbols('t s', real=True)
        x = t
        y = s
        z = a*t**2-b*s**2

        t_min = -R
        t_max = R
        s_min = -R
        s_max = R

        ts = (t, s)
        xyz = (x, y, z)
        ts_min = (t_min, s_min)
        ts_max = (t_max, s_max)
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max)

    def compute_mesh(self, res):
        import mshr
        print("Using overloaded compute_mesh for Saddle geometry")
        t_min = self.r_ref_min["t"]
        t_max = self.r_ref_max["t"]
        circ = mshr.Circle(df.Point(0, 0), (t_max-t_min)/2)
        ref_mesh = mshr.generate_mesh(circ, res, "cgal")
        self.ref_mesh = ref_mesh


class TorusMap(GeoMap):
    def __init__(self, R, r, verbose=False):
        t, s = sp.symbols('t s', real=True)
        x = (R + r*sp.cos(t/r)) * sp.cos(s/R)
        y = (R + r*sp.cos(t/r)) * sp.sin(s/R)
        z = r*sp.sin(t/r)

        t_min = 0.
        t_max = 2*np.pi*r
        s_min = 0.
        s_max = 2*np.pi*R

        ts = (t, s)
        xyz = (x, y, z)
        ts_min = (t_min, s_min)
        ts_max = (t_max, s_max)
        GeoMap.__init__(self, xyz, ts, ts_min, ts_max, verbose=verbose)
        self.R = R
        self.r = r

    def compute_pbc(self):
        ts_min = [self.r_ref_min[j] for j in self.AXIS_REF]
        ts_max = [self.r_ref_max[j] for j in self.AXIS_REF]
        # ts_min = (self.t_min, self.s_min)
        # ts_max = (self.t_max, self.s_max)
        self.pbc = TorusPBC(ts_min, ts_max)

    def compute_mesh(self, res):
        factor = np.sqrt(self.R/self.r)
        Nt = int(res/factor)
        Ns = int(factor*res)

        t_min = self.r_ref_min["t"]
        t_max = self.r_ref_max["t"]
        s_min = self.r_ref_min["s"]
        s_max = self.r_ref_max["s"]

        ref_mesh = df.RectangleMesh.create(
            [df.Point(t_min, s_min),
             df.Point(t_max, s_max)],
            [Nt, Ns], df.cpp.mesh.CellType.Type.triangle)
        self.ref_mesh = ref_mesh
