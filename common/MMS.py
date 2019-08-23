import sympy as sp
import numpy as np
import dolfin as df


class ManufacturedSolution:
    def __init__(self, geo_map, psi_in):
        self.t = geo_map.t
        self.s = geo_map.s
        self.map = dict()
        self.S_ref = geo_map.S_ref
        self.t_vals = geo_map.t_vals
        self.s_vals = geo_map.s_vals
        self.geodict = geo_map.map
        self.psi_in = psi_in
        self.compute_MMS()

    def compute_MMS(self):
        # Manufactured solution:
        self.map["psiMMS"] = self.psi_in
        self.map["nuMMS"] = (self.geodict["gtt"] * sp.diff(sp.diff(self.map["psiMMS"], self.t), self.t)
                                + 2 * self.geodict["gst"] * sp.diff(sp.diff(self.map["psiMMS"], self.s), self.t)
                                + self.geodict["gss"] * sp.diff(sp.diff(self.map["psiMMS"], self.s), self.s)
                                + (1/self.geodict["sqrt_g"]) * (self.geodict["gtt"] * sp.diff(self.geodict["sqrt_g"], self.t) * sp.diff(self.map["psiMMS"], self.t)
                                + self.geodict["gst"] * sp.diff(self.geodict["sqrt_g"], self.t) * sp.diff(self.map["psiMMS"], self.s)
                                + self.geodict["gst"] * sp.diff(self.geodict["sqrt_g"], self.s) * sp.diff(self.map["psiMMS"], self.t)
                                + self.geodict["gss"] * sp.diff(self.geodict["sqrt_g"], self.s) * sp.diff(self.map["psiMMS"], self.s)))
        self.map["nuhatMMS"] = (self.geodict["Ktt"]*sp.diff(sp.diff(self.map["psiMMS"], self.t), self.t)
                                + 2 * self.geodict["Kts"]*sp.diff(sp.diff(self.map["psiMMS"], self.s), self.t)
                                + self.geodict["Kss"]*sp.diff(sp.diff(self.map["psiMMS"], self.s), self.t)
                                - self.geodict["Ktt"] * self.geodict["Gt_tt"] * sp.diff(self.map["psiMMS"], self.t)
                                - 2 * self.geodict["Kst"] * self.geodict["Gt_st"] * sp.diff(self.map["psiMMS"], self.t)
                                - self.geodict["Kss"] * self.geodict["Gt_ss"] * sp.diff(self.map["psiMMS"], self.t)
                                - self.geodict["Ktt"] * self.geodict["Gs_tt"] * sp.diff(self.map["psiMMS"], self.s)
                                - 2 * self.geodict["Kst"] * self.geodict["Gs_st"] * sp.diff(self.map["psiMMS"], self.s)
                                - self.geodict["Kss"] * self.geodict["Gs_ss"] * sp.diff(self.map["psiMMS"], self.s))

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

    def initialize_fields(self):
        self.psi = self.get_function("psiMMS")
        self.laplacian = self.get_function("nuMMS")
        self.curvplacian = self.get_function("nuhatMMS")
