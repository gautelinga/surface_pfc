import argparse
from utilities.InterpolatedTimeSeries import InterpolatedTimeSeries
from utilities.plot import plot_any_field
from common.io import dump_xdmf
from postprocess import get_step_and_info
from fenicstools import interpolate_nonmatching_mesh
import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import random


def main():
    parser = argparse.ArgumentParser(description="Average various files")
    parser.add_argument("-l", "--list", nargs="+", help="List of folders",
                        required=True)
    parser.add_argument("-f", "--fields", nargs="+", default=None,
                        help="Sought fields")
    parser.add_argument("-t", "--time", type=float, default=0, help="Time")
    parser.add_argument("--show", action="store_true", help="Show")
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
    rad_t = []
    rad_s = []
    g = []
    g_inv = []
    for ts in tss:
        # Should compute these from the curvature tensor
        rad_t.append(df.interpolate(
                df.Expression("x[0]", degree=2), ts.function_space))
        rad_s.append(df.interpolate(
                df.Expression("x[1]", degree=2), ts.function_space))

        g_loc = [ts.function(name) for name in ["gtt", "gst", "gss"]]
        g_inv_loc = [ts.function(name) for name in ["g_tt", "g_st", "g_ss"]]

        for ij in range(3):
            ts.set_val(g_loc[ij], ts.g[:, ij])
            # Could compute the following locally instead of loading
            ts.set_val(g_inv_loc[ij], ts.g_inv[:, ij])

        g.append(g_loc)
        g_inv.append(g_inv_loc)

    costheta = df.Function(ref_spaces["psi"], name="costheta")
    # Number of nodes:
    arrsiz = int(len(costheta.vector().get_local()))
    # Array to hold costheta^2 values for all runs:
    costhetas = np.zeros([arrsiz,Ntss])
    # Number of random samples to perform per simulation
    nsamples=1000
    # Array for randomly sampled costheta^2 values:
    costhetasamp=np.zeros([nsamples,Ntss])
    #xmin=
    for its, ts in enumerate(tss):
        if args.time is not None:
            step, time = get_step_and_info(ts, args.time)

        ts.update(f_in[its]["psi"], "psi", step)
        psi = f_in[its]["psi"]
        psi_t = df.project(psi.dx(0), ts.function_space)
        psi_s = df.project(psi.dx(1), ts.function_space)

        gp_t = psi_t.vector().get_local()
        gp_s = psi_s.vector().get_local()
        gtt, gst, gss = [g_ij.vector().get_local() for g_ij in g[its]]
        g_tt, g_st, g_ss = [g_ij.vector().get_local() for g_ij in g_inv[its]]
        rht = rad_t[its].vector().get_local()
        rhs = rad_s[its].vector().get_local()
        rh_norm = np.sqrt(g_tt*rht**2 + g_ss*rhs**2 + 2*g_st*rht*rhs)
        gp_norm = np.sqrt(gtt*gp_t**2 + gss*gp_s**2 + 2*gst*gp_t*gp_s)

        costheta_loc = df.Function(ts.function_space)
        costheta_loc.vector()[:] = abs(gp_t/(gp_norm+1e-8))**2

        costheta_intp = interpolate_nonmatching_mesh(costheta_loc,
                                                     ref_spaces["psi"])
        costheta.vector()[:] += costheta_intp.vector().get_local()/Ntss
        costhetas[:, its] = costheta_intp.vector().get_local()
        # print(costheta(df.Point(0,0)))
        t_coords = df.interpolate(df.Expression("x[0]", degree=2), ts.function_space)
        s_coords = df.interpolate(df.Expression("x[1]", degree=2), ts.function_space)
        t_min = min(t_coords.vector())
        t_max = max(t_coords.vector())
        s_min = min(s_coords.vector())
        s_max = max(s_coords.vector())
        for k in range(0,nsamples-1):
            Pt = t_min+(t_max-t_min)*random.random()
            Ps = s_min+(s_max-s_min)*random.random()
            P = df.Point(Pt,Ps)
            costhetasamp[k,its] = costheta_loc(P)

    dump_xdmf(costheta)

    if args.show:
        #fig = df.plot(costheta)
        #plt.colorbar(fig)
        rc('text', usetex=True)
        plt.hist(costhetas,bins=20,density=True,stacked=True)
        #plt.hist(costhetasamp,bins=10,density=True,stacked=True)
        ax = plt.gca()
        #ax.set_aspect(0.7)
        ax.set_xlabel('$\\cos^2(\\theta)$')
        ax.set_ylabel('Relative frequency')
        plt.show()


if __name__ == "__main__":
    main()
