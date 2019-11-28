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

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex
    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)
    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


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
    arrsiz = int(len(costheta.vector().get_local()))
    print("Function vector size: ", arrsiz )
    costhetas = np.zeros([arrsiz,Ntss])
    nsamples=1000
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
        # For theta (hat x, gradpsi):
        #costheta_loc.vector()[:] = abs(gp_t/(gp_norm+1e-8))**2
        #costheta_loc.vector()[:] = abs(gp_t/(gp_norm+1e-8))
        # Plot cos(theta):
        #costheta_loc.vector()[:] = gp_t/(gp_norm+1e-8)
        # Plot theta = acos(cos(theta)):
        costheta_loc.vector()[:] = np.arccos(gp_t/(gp_norm+1e-8))

        # For theta (hat r, gradpsi):
        #costheta_loc.vector()[:] = abs((rht*gp_t + rhs*gp_s)/(rh_norm*gp_norm+1e-8))

        costheta_intp = interpolate_nonmatching_mesh(costheta_loc,
                                                     ref_spaces["psi"])
        costheta.vector()[:] += costheta_intp.vector().get_local()/Ntss
        costhetas[:,its] = costheta_intp.vector().get_local()
        #print(costheta(df.Point(0,0)))
        t_coords=df.interpolate(df.Expression("x[0]", degree=2), ts.function_space)
        s_coords=df.interpolate(df.Expression("x[1]", degree=2), ts.function_space)
        t_min = min(t_coords.vector())
        t_max = max(t_coords.vector())
        s_min = min(s_coords.vector())
        s_max = max(s_coords.vector())
        for k in range(0,nsamples-1):
            Pt = t_min+(t_max-t_min)*random.random()
            Ps = s_min+(s_max-s_min)*random.random()
            P = df.Point(Pt,Ps)
            try:
                costhetasamp[k,its] = costheta_loc(P)
            except:
                print("Error occurred at (t,s) = ", Pt, "," ,Ps)
                print("Geometric parameters: ")
                print("t_min = ", t_min)
                print("t_max = ", t_max)
                print("s_min = ", s_min)
                print("s_max = ", s_max)

    dump_xdmf(costheta)

    if args.show:
        #fig = df.plot(costheta)
        #plt.colorbar(fig)
        rc('text', usetex=True)
        fig = plt.figure(figsize=(1.75,2.5))
        #plt.hist(costhetas,bins=10,density=True,stacked=True)
        # Stacked version:
        # Create color palette manually:
        #pal = []
        #for k in range(0,Ntss):
        #    pal.append('#e34a33')
        #print("Color palette: ", pal)
        #plt.hist(costhetas,bins=100,density=True,stacked=True,color=pal,rasterized=True)
        #plt.hist(costhetasamp,bins=10,density=True,stacked=True)
        # Unstacked version:
        costhetas_coll = np.reshape(costhetas, [arrsiz*Ntss,1])
        print(costhetas_coll)
        plt.hist(costhetas_coll,bins=100,density=True,color='#e34a33',rasterized=True)
        ax = plt.gca()
        #ax.set_aspect(0.7)
        ax.set_xlabel('$\\theta$', fontsize=10)
        #ax.set_xlabel('$\\theta$')
        ax.set_ylabel('Relative frequency', fontsize=10)
        #plt.xticks(np.arange(0,3.2,np.pi))
        #plt.yticks(np.arange(0,2.1,1))
        plt.yticks([],[])
        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        ax.set_ylim([0,3])
        x1 = np.linspace(0,np.pi,num=200)
        x2 = np.linspace(0,np.pi,num=200)
        #f = (2/np.pi)*(1-(2/np.pi)*abs(x1-np.pi/2))
        fz = (1/np.pi)*x2/x2
        #plt.plot(x1,f,lw=2)
        #plt.plot(x2,fz,lw=2)
        L = 2*np.pi*20*np.sqrt(2)
        lam = 2*np.pi*np.sqrt(2)
        k=0
        xs = []
        ys = []
        while k*lam/L <= 1:
            xk = np.arccos(k*lam/L)
            yk = np.arcsin(k*lam/L)
            xs.append(xk)
            ys.append(yk)
            k += 1
        #plt.vlines(xs,0,0.3)
        #plt.vlines(ys,0.3,0.6)
        plt.savefig('hist_cyl_h20.pdf', format="pdf", bbox_inches='tight',  pad_inches = 0)
        plt.show()


if __name__ == "__main__":
    main()
