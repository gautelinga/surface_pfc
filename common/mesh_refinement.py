import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from pynverse import inversefunc
import dolfin as df
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD

EPS = 0.01


def f(phi, kappa):
    return np.sqrt(1./kappa * np.sin(phi)**2 + kappa * np.cos(phi)**2)


def f_reg(phi, kappa):
    if phi < EPS:
        return f(EPS, kappa)
    elif phi > 2*np.pi-EPS:
        return f(2*np.pi-EPS. kappa)
    else:
        return f(phi, kappa)


def F_loc(phi_loc, kappa):
    return integrate.quad(f, 0., phi_loc, args=kappa)[0]


def F(phi, kappa):
    return np.array([F_loc(phi_loc, kappa) for phi_loc in phi])


def Finv(s, kappa):
    invfunc = inversefunc(F_loc, args=(kappa), domain=[0., 2*np.pi])
    return np.array([invfunc(s_loc) for s_loc in s])


def g(theta, alpha):
    return np.sqrt(alpha/np.tan(theta)**2 + 1)


def g_reg(theta, alpha):
    if theta < EPS:
        return g(EPS, alpha)
    elif theta > np.pi-EPS:
        return g(np.pi-EPS, alpha)
    else:
        return g(theta, alpha)


def G_loc(theta_loc, alpha):
    return integrate.quad(g_reg, 0., theta_loc, args=alpha)[0]


def G(theta, alpha):
    return np.array([G_loc(theta_loc, alpha) for theta_loc in theta])


def Ginv(t, alpha):
    invfunc = inversefunc(G_loc, args=(alpha), domain=[0.0, np.pi])
    return np.array([invfunc(t_loc) for t_loc in t])


def phi(s, kappa):
    return Finv(F_loc(2*np.pi, kappa)*s/(2*np.pi), kappa)


def theta(t, alpha):
    return Ginv(G_loc(np.pi, alpha)*t/np.pi, alpha)


def phi_spline(s, kappa, N=20):
    if comm.Get_rank() == 0:
        s_intp = np.linspace(0., np.pi/2, N)
        phi_intp = phi(s_intp, kappa)-s_intp
        y = InterpolatedUnivariateSpline(s_intp, phi_intp)
    else:
        y = None
    y = comm.bcast(y, root=0)
    sign = -np.sign(np.remainder(s, np.pi)-np.pi/2)
    s_mod = sign*(np.remainder(s, np.pi/2)-np.pi/4) + np.pi/4
    return sign*y(s_mod)+s


def theta_spline(t, alpha, N=20):
    if comm.Get_rank() == 0:
        t_intp = np.linspace(0., np.pi/2, N)
        theta_intp = theta(t_intp, alpha)-t_intp
        y = InterpolatedUnivariateSpline(t_intp, theta_intp)
    else:
        y = None
    y = comm.bcast(y, root=0)
    sign = -np.sign(np.remainder(t, np.pi)-np.pi/2)
    t_mod = sign*(np.remainder(t, np.pi/2) - np.pi/4) + np.pi/4
    return sign*y(t_mod)+t


def densified_ellipsoid_mesh(res, Rx, Ry, Rz, eps=1e-2, return_details=False):
    N = res
    kappa = Rx/Ry
    alpha = 0.5*((Rx/Rz)**2 + (Ry/Rz)**2)
    kNt = np.sqrt(N*Rz/np.sqrt(Rx*Ry)*G_loc(np.pi, alpha) /
                  F_loc(2*np.pi, kappa))
    Nt = int(np.round(kNt))
    Ns = int(np.round(N*1.0/Nt))
    mesh = df.RectangleMesh.create(
        [df.Point(0., 0.+eps), df.Point(2*np.pi, np.pi-eps)],
        [Ns, Nt],
        df.cpp.mesh.CellType.Type.triangle
        # df.cpp.mesh.CellType.Type.quadrilateral
    )
    x = mesh.coordinates()[:]
    x[:, 0] = phi_spline(x[:, 0], kappa)
    x[:, 1] = theta_spline(x[:, 1], alpha)

    if return_details:
        return mesh, (kappa, alpha, Nt, Ns)

    return mesh


if __name__ == "__main__":

    Rx = 1.0
    Ry = 5.0
    Rz = 10.0
    res = 200

    mesh, (kappa, alpha, Nt, Ns) = densified_ellipsoid_mesh(
        res, Rx, Ry, Rz, return_details=True)

    print("Kappa =", kappa)
    print("Alpha =", alpha)

    print("Nt =", Nt)
    print("Ns =", Ns)
    print("N  =", Nt*Ns)

    plt.figure()
    df.plot(mesh)

    plt.show()
