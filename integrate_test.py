import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from pynverse import inversefunc


def f(phi, kappa):
    return np.sqrt(1./kappa * np.sin(phi)**2 + kappa * np.cos(phi)**2)


def F_loc(phi_loc, kappa):
    return integrate.quad(f, 0., phi_loc, args=kappa)[0]


def Finv_loc(s_loc, kappa):
    invfunc = inversefunc(F_loc, args=(kappa), domain=[0., 2*np.pi])
    return invfunc(s_loc)


def F(phi, kappa):
    return np.array([F_loc(phi_loc, kappa) for phi_loc in phi])


def Finv(s, kappa):
    return np.array([Finv_loc(s_loc, kappa) for s_loc in s])


if __name__ == "__main__":
    s = np.linspace(0., 2*np.pi, 50)
    
    for kappa in [0.01, 0.1, 0.5, .9, 1/0.9, 2.0, 10., 100.]:
        #Fhat = (2*np.pi*F(phi, kappa)/F_loc(2*np.pi, kappa) - phi)*kappa
        #print(Fhat)
        beta = s/(2*np.pi)
        g = Finv(F_loc(0, kappa)*(1.-beta) + F_loc(2*np.pi, kappa)*beta, kappa)
        gres = g - s
        plt.plot(s, gres)

    plt.show()
