import sympy as sp
import numpy as np
import dolfin as df


class ManufacturedSolution:
    def __init__(self, geo_map, f):
        self.t = geo_map.t
        self.s = geo_map.s
        self.map = dict()
        self.geo_map = geo_map
        self.geodict = geo_map.map
        self.f = f
        self.initialize()

    def initialize(self):
        # Manufactured solution:
        self.map["psi"] = self.f
        self.map["nu"] = (
            self.geodict["gtt"] * sp.diff(self.f, self.t, self.t)
            + 2 * self.geodict["gst"] * sp.diff(self.f,
                                                self.s, self.t)
            + self.geodict["gss"] * sp.diff(self.f, self.s, self.s)
            + (1/self.geodict["sqrt_g"]) * (
                self.geodict["gtt"]
                * sp.diff(self.geodict["sqrt_g"], self.t)
                * sp.diff(self.f, self.t)
                + self.geodict["gst"]
                * sp.diff(self.geodict["sqrt_g"], self.t)
                * sp.diff(self.f, self.s)
                + self.geodict["gst"]
                * sp.diff(self.geodict["sqrt_g"], self.s)
                * sp.diff(self.f, self.t)
                + self.geodict["gss"]
                * sp.diff(self.geodict["sqrt_g"], self.s)
                * sp.diff(self.f, self.s)))
        self.map["nuhat"] = (
            self.geodict["Ktt"] * sp.diff(self.f, self.t, self.t)
            + 2 * self.geodict["Kts"] * sp.diff(self.f,
                                                self.s, self.t)
            + self.geodict["Kss"]*sp.diff(self.f, self.s, self.t)
            - self.geodict["Ktt"]
            * self.geodict["Gt_tt"] * sp.diff(self.f, self.t)
            - 2 * self.geodict["Kst"]
            * self.geodict["Gt_st"] * sp.diff(self.f, self.t)
            - self.geodict["Kss"]
            * self.geodict["Gt_ss"] * sp.diff(self.f, self.t)
            - self.geodict["Ktt"]
            * self.geodict["Gs_tt"] * sp.diff(self.f, self.s)
            - 2 * self.geodict["Kst"]
            * self.geodict["Gs_st"] * sp.diff(self.f, self.s)
            - self.geodict["Kss"]
            * self.geodict["Gs_ss"] * sp.diff(self.f, self.s)
        )

        self.evalf = dict()
        for key in self.map.keys():
            self.evalf[key] = sp.lambdify([self.t, self.s],
                                          self.map[key], "numpy")

    def eval(self, key):
        v = self.evalf[key](self.geo_map.t_vals,
                            self.geo_map.s_vals)
        if isinstance(v, int) or isinstance(v, float):
            return v*np.ones_like(self.t_vals)
        else:
            return v

    def get_function(self, key):
        f = df.Function(self.geo_map.S_ref)
        f.rename("{}MMS".format(key), "tmp")
        F = self.eval(key)
        f.vector()[:] = F
        return f

    def psi(self):
        return self.get_function("psi")

    def laplacian(self):
        return self.get_function("nu")

    def curvplacian(self):
        return self.get_function("nuhat")
