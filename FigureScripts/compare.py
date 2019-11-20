import argparse
from utilities.InterpolatedTimeSeries import InterpolatedTimeSeries
from utilities.plot import plot_any_field
from common.io import dump_xdmf
from postprocess import get_step_and_info
from fenicstools import interpolate_nonmatching_mesh, Probes
import dolfin as df
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Average various files")
    parser.add_argument("-l", "--list", nargs="+", help="List of folders",
                        required=True)
    parser.add_argument("-f", "--fields", nargs="+", default=None,
                        help="Sought fields")
    parser.add_argument("-t", "--time", type=float, default=0, help="Time")
    parser.add_argument("--show", action="store_true", help="Show")
    parser.add_argument("-R", "--radius", type=float, default=None, help="Radial distance")
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
        # abs(cos(theta)):
        costheta_loc.vector()[:] = abs((rht*gp_t + rhs*gp_s)/(rh_norm*gp_norm+1e-8))
        # cos(theta)**2:
        #costheta_loc.vector()[:] = ((rht*gp_t + rhs*gp_s)/(rh_norm*gp_norm+1e-8))**2
        # sin(theta):
        #costheta_loc.vector()[:] = np.sin(np.arccos((rht*gp_t + rhs*gp_s)/(rh_norm*gp_norm+1e-8)))
        # sin(theta)**2:
        # costheta_loc.vector()[:] = np.sin(np.arccos((rht*gp_t + rhs*gp_s)/(rh_norm*gp_norm+1e-8)))**2
        # abs(theta):
        #costheta_loc.vector()[:] = abs(np.arccos((rht*gp_t + rhs*gp_s)/(rh_norm*gp_norm+1e-8)))


        costheta_intp = interpolate_nonmatching_mesh(costheta_loc,
                                                     ref_spaces["psi"])
        costheta.vector()[:] += costheta_intp.vector().get_local()/Ntss

    dump_xdmf(costheta)

    if args.show:
        JET = plt.get_cmap('jet')
        RYB = plt.get_cmap('RdYlBu')
        RYB_r = plt.get_cmap('RdYlBu_r')
        mgm = plt.get_cmap('magma')
        fig, ax = plt.subplots()
        fig.set_size_inches(4.7,4.7)
        rc('text', usetex=True)
        rc('font',**{'family':'serif','serif':['Palatino']})
        # First, dump a hires PNG version of the data:
        plot = df.plot(costheta, cmap=mgm)
        plt.axis('off')
        plt.savefig('anglogram_hires.png', format="png", bbox_inches='tight',  pad_inches = 0, dpi=500)
        mainfs = 10 # Fontsize
        titlefs = 12 # Fontsize
        #plt.set_cmap('jet')
        #cbar = fig.colorbar(plot, ticks=[0, 0.5, 1], orientation='vertical')
        #cbar.ax.set_yticklabels(['Radial', 'Intermediate', 'Azimuthal'])
        #cbar = fig.colorbar(plot, ticks=[0, 0.5, 0.95], orientation='horizontal', fraction=0.046, pad=0.04)
        #cbar = fig.colorbar(plot, ticks=[0, 0.5, 0.995], orientation='horizontal', fraction=0.046, pad=0.04)
        cbar = fig.colorbar(plot, ticks=[0, 0.5, 0.9995], orientation='horizontal', fraction=0.046, pad=0.04)
        cbar.ax.set_xticklabels(['Radial', 'Intermediate', 'Azimuthal'])
        #ax.set_title('Stripe orientation -- $\\cos^2(\\theta)$', fontsize=mainfs)
        #ax.set_title(r'Stripe orientation -- $\cos^2(\theta)$')
        plt.text(0.5, 1.05, r'Stripe orientation -- $\left\vert\cos(\theta)\right\vert$', fontsize=titlefs, horizontalalignment='center', transform=ax.transAxes)
        #ax.set_title('Stripe orientation', fontsize=mainfs)
        #plt.text(0.2, 0.2, '$\\textcolor{red}{\\mathbf{p}(u,w,\\xi)}=\\textcolor{blue}{\\tilde{\\mathbf{p}}(u,w)}  + \\xi \\tilde{\\mathbf{n}}(u,w)$', fontsize=mainfs)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #plt.show()
        plt.savefig('anglogram.pdf', format="pdf", bbox_inches='tight',  pad_inches = 0)
        #plt.axis('on')
        #ax.get_xaxis().set_visible(True)
        #ax.get_yaxis().set_visible(True)
        #plt.colorbar(plot)
        #plt.show()

    if args.radius is not None and args.radius > 0:
        Nr, Nphi = 256, 256
        r_lin = np.linspace(0., args.radius, Nr)
        phi_lin = np.linspace(0, 2*np.pi, Nphi, endpoint=False)
        R, Phi = np.meshgrid(r_lin, phi_lin)
        r = R.reshape((Nr*Nphi, 1))
        phi = Phi.reshape((Nr*Nphi, 1))

        xy = np.hstack((r*np.cos(phi), r*np.sin(phi)))

        pts = xy.flatten()
        probes = Probes(pts, ref_spaces["psi"])
        probes(costheta)

        ct = probes.array()
        CT = ct.reshape(R.shape)

        g_r = CT.mean(axis=0)
        g_phi = CT.mean(axis=1)

        plt.figure()
        plt.plot(r_lin, g_r)
        plt.ylabel("g(r)")
        plt.xlabel("r")

        plt.figure()
        plt.plot(phi_lin, g_phi)
        plt.xlabel("phi")
        plt.ylabel("g(phi)")
        plt.show()



if __name__ == "__main__":
    main()
