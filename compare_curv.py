import argparse
from utilities.InterpolatedTimeSeries import InterpolatedTimeSeries
from utilities.plot import plot_any_field
from common.io import dump_xdmf
from common.utilities import NdFunction
from postprocess import get_step_and_info
from fenicstools import interpolate_nonmatching_mesh
import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


class MyProbes():
    def __init__(self, x):
        self.x = x
        self.vals = []
        
    def __call__(self, func):
        rank = func.value_rank()
        if rank == 0:
            dim = 1
        elif rank == 1:
            dim = func.value_shape()[0]  # for now
        else:
            exit("Tensors not supported")
        Z = np.zeros((len(self.x), dim))
        for i, xi in enumerate(self.x):
            Z[i, :] = func(df.Point(*xi))
        self.vals.append(Z)

    def array(self):
        if len(self.vals) == 0:
            return None
        elif len(self.vals) == 1:
            return self.vals
        else:
            return self.vals


def to_vector(A):
    A_ = dict()
    for idx, Aij in A.items():
        A_[idx] = Aij.vector().get_local()
    return A_


def pick_random_points(p_cell, x_cell, n):
    i = np.random.choice(np.arange(len(p_cell)),
                         n, p=p_cell)
    sqrt_r1 = np.sqrt(np.random.uniform(0, 1, (n, 1)))
    r2 = np.random.uniform(0, 1, (n, 1))
    x_loc = x_cell[i, :]
    return ((1-sqrt_r1)*x_loc[:, 0:2]
            + sqrt_r1*(1-r2)*x_loc[:, 2:4]
            + r2*sqrt_r1*x_loc[:, 4:6])


def main():
    parser = argparse.ArgumentParser(description="Average various files")
    parser.add_argument("-l", "--list", nargs="+", help="List of folders",
                        required=True)
    parser.add_argument("-f", "--fields", nargs="+", default=None,
                        help="Sought fields")
    parser.add_argument("-t", "--time", type=float, default=0, help="Time")
    parser.add_argument("--show", action="store_true", help="Show")
    parser.add_argument("--hist", action="store_true", help="Histogram")
    args = parser.parse_args()

    tss = []
    for folder in args.list:
        ts = InterpolatedTimeSeries(folder, sought_fields=args.fields)
        tss.append(ts)
    Ntss = len(tss)

    all_fields_ = []
    for ts in tss:
        all_fields_.append(set(ts.fields))
    all_fields = list(set.intersection(*all_fields_))
    if args.fields is None:
        fields = all_fields
    else:
        fields = list(set.intersection(set(args.fields), set(all_fields)))

    f_in = []
    for ts in tss:
        f_in.append(ts.functions())

    # Using the first timeseries to define the spaces
    # Could be redone to e.g. a finer, structured mesh.
    # ref_mesh = tss[0].mesh
    ref_spaces = dict([(field, f.function_space()) for
                       field, f in f_in[0].items()])

    if "psi" not in fields:
        exit("No psi")

    # Loading geometry
    g_ab = []
    gab = []
    K_ab = []
    dxyz = []

    var_names = ["t", "s"]
    index_names = ["tt", "st", "ss"]
    dim_names = ["x", "y", "z"]

    for ts in tss:
        gab_loc = dict([(idx, ts.function("g{}".format(idx)))
                        for idx in index_names])
        g_ab_loc = dict([(idx, ts.function("g_{}".format(idx)))
                         for idx in index_names])
        K_ab_loc = dict([(idx, ts.function("K_{}".format(idx)))
                         for idx in index_names])
        dxyz_loc = [None]*3
        for d in range(3):
            dxyz_loc[d] = dict([(i, ts.function("{}{}".format(
                dim_names[d], i))) for i in var_names])
            ts.set_val(dxyz_loc[d]["t"], ts.xyzt[:, d])
            ts.set_val(dxyz_loc[d]["s"], ts.xyzs[:, d])

        for ij, idx in enumerate(index_names):
            ts.set_val(g_ab_loc[idx], ts.g_ab[:, ij])
            # Could compute the gab locally instead of loading
            ts.set_val(gab_loc[idx], ts.gab[:, ij])
            ts.set_val(K_ab_loc[idx], ts.K_ab[:, ij])

        g_ab_loc["ts"] = g_ab_loc["st"]
        gab_loc["ts"] = gab_loc["st"]
        K_ab_loc["ts"] = K_ab_loc["st"]

        g_ab.append(g_ab_loc)
        gab.append(gab_loc)
        K_ab.append(K_ab_loc)
        dxyz.append(dxyz_loc)

    v_ = []
    for g_ab_loc, gab_loc, K_ab_loc, dxyz_loc, ts in zip(
            g_ab, gab, K_ab, dxyz, tss):
        g_ab_ = to_vector(g_ab_loc)
        gab_ = to_vector(gab_loc)
        K_ab_ = to_vector(K_ab_loc)
        dxyz_ = [to_vector(dxyz_loc_i) for dxyz_loc_i in dxyz_loc]

        Ka_b_ = dict()
        Ka_b_loc = dict()
        for i in var_names:
            for j in var_names:
                Ka_b_[i + j] = sum([gab_[i + k] * K_ab_[k + j]
                                    for k in var_names])

                Ka_b_loc[i + j] = ts.function("K{}_{}".format(i, j))
                Ka_b_loc[i + j].vector()[:] = Ka_b_[i + j]

        kappa1 = ts.function("kappa1")
        kappa2 = ts.function("kappa2")
        v1 = dict([(i, ts.function("v1{}".format(i))) for i in var_names])
        v2 = dict([(i, ts.function("v2{}".format(i))) for i in var_names])
        kappa1_ = kappa1.vector().get_local()
        kappa2_ = kappa2.vector().get_local()
        v1_ = dict([(i, vi.vector().get_local()) for i, vi in v1.items()])
        v2_ = dict([(i, vi.vector().get_local()) for i, vi in v2.items()])
        for idof in range(len(kappa1_)):
            M = np.array([[Ka_b_["tt"][idof], Ka_b_["ts"][idof]],
                          [Ka_b_["st"][idof], Ka_b_["ss"][idof]]])
            kappas, vs = np.linalg.eig(M)
            kappa1_[idof] = kappas[0]
            kappa2_[idof] = kappas[1]
            for ind, i in enumerate(var_names):
                v1_[i][idof] = vs[ind, 0]
                v2_[i][idof] = vs[ind, 1]

        kappa1.vector()[:] = kappa1_
        kappa2.vector()[:] = kappa2_
        for i in var_names:
            v1[i].vector()[:] = v1_[i]
            v2[i].vector()[:] = v2_[i]

        v_.append(v1_)

        v1_out = NdFunction([v1["t"], v1["s"]], name="v1")
        v1_out()
        dump_xdmf(v1_out)

        v2_out = NdFunction([v2["t"], v2["s"]], name="v2")
        v2_out()
        dump_xdmf(v2_out)

        kappa = NdFunction([kappa1, kappa2], name="kappa")
        kappa()
        dump_xdmf(kappa, folder=ts.geometry_folder)

        tau1_vec = [ts.function("tau1_{}".format(dim_names[i]))
                    for i in range(3)]
        tau2_vec = [ts.function("tau2_{}".format(dim_names[i]))
                    for i in range(3)]
        for d in range(3):
            tau1_vec[d].vector()[:] = sum([dxyz_[d][i] * v1_[i]
                                           for i in var_names])
            tau2_vec[d].vector()[:] = sum([dxyz_[d][i] * v2_[i]
                                           for i in var_names])
        tau1 = NdFunction(tau1_vec, name="tau1")
        tau2 = NdFunction(tau2_vec, name="tau2")
        tau1()
        tau2()
        dump_xdmf(tau1, folder=ts.geometry_folder)
        dump_xdmf(tau2, folder=ts.geometry_folder)

    sqrt_g = df.Function(ref_spaces["psi"], name="sqrt_g")
    costheta = df.Function(ref_spaces["psi"], name="costheta")
    for its, ts in enumerate(tss):
        if args.time is not None:
            step, time = get_step_and_info(ts, args.time)

        ts.update(f_in[its]["psi"], "psi", step)
        psi = f_in[its]["psi"]
        dpsi_a = dict()
        dpsi_a_ = dict()
        for ind, i in enumerate(var_names):
            dpsi_a[i] = df.project(psi.dx(ind), ts.function_space)
            dpsi_a_[i] = dpsi_a[i].vector().get_local()

        gab_ = to_vector(gab[its])
        g_ab_ = to_vector(g_ab[its])

        dpsi_norm_ = np.sqrt(
            sum([gab_[i+j]*dpsi_a_[i]*dpsi_a_[j]
                 for i, j in product(var_names, var_names)]))

        v1a_ = v_[its]
        v1_norm_ = np.sqrt(
            sum([g_ab_[i+j]*v1a_[i]*v1a_[j]
                 for i, j in product(var_names, var_names)]))

        v1_dot_dpsi_ = sum([v1a_[i]*dpsi_a_[i]
                            for i in var_names])

        costheta_loc = ts.function("costheta")
        costheta_loc.vector()[:] = abs(v1_dot_dpsi_ /
                                       (v1_norm_*dpsi_norm_+1e-8))

        sqrt_g_loc = ts.function("sqrt_g")
        sqrt_g_loc.vector()[:] = np.sqrt(g_ab_["tt"]*g_ab_["ss"]
                                         - g_ab_["ts"]*g_ab_["st"])

        dump_xdmf(costheta_loc, folder=ts.geometry_folder)

        costheta_intp = interpolate_nonmatching_mesh(costheta_loc,
                                                     ref_spaces["psi"])
        costheta.vector()[:] += costheta_intp.vector().get_local()/Ntss

        sqrt_g_intp = interpolate_nonmatching_mesh(sqrt_g_loc,
                                                   ref_spaces["psi"])
        sqrt_g.vector()[:] += sqrt_g_intp.vector().get_local()/Ntss

    dump_xdmf(costheta)

    p_cell = np.zeros((ref_spaces["psi"].mesh().num_cells()))
    x_cell = np.zeros((ref_spaces["psi"].mesh().num_cells(), 6))
    for i, ic in enumerate(df.cells(ref_spaces["psi"].mesh())):
        x_cell[i, :] = np.array(ic.get_coordinate_dofs())
        p_cell[i] = sqrt_g(ic.midpoint())*ic.volume()

    p_cell /= p_cell.sum()
    x_pts = pick_random_points(p_cell, x_cell, 10000)

    if args.show:
        fig = df.plot(costheta)
        plt.scatter(x_pts[:, 0], x_pts[:, 1], s=0.2, color='k')
        plt.colorbar(fig)

        plt.show()

    if args.hist:
        probes = MyProbes(x_pts)
        probes(costheta)

        ct = probes.array()[0]
        theta = np.arccos(ct)

        plt.figure()
        plt.hist(theta, bins=100, density=True)
        plt.xlabel("theta")
        plt.ylabel("P(theta)")
        plt.show()
        
        

if __name__ == "__main__":
    main()
