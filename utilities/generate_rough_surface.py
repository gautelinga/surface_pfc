import pyfftw
import numpy as np
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import h5py
import argparse


def generate_surface_fourier(Ni, Nk, H, seed=0):
    np.random.seed(seed)
    A = np.zeros((Ni, Nk), dtype=complex)
    for i, k in product(range(0, Ni//2+1), range(0, Nk//2+1)):
        phase = np.random.uniform(0., np.pi)
        if i != 0 or k != 0:
            rad = (np.power(i**2 + k**2, -(H+1.)/2.) *
                   np.random.normal(0., 1.))
        else:
            rad = 0.
        A[i, k] = rad*np.exp(phase*1.j)
        if i == 0:
            i0 = 0
        else:
            i0 = Ni - i
        if k == 0:
            k0 = 0
        else:
            k0 = Nk - k
        A[i0, k0] = rad*np.exp(-phase*1.j)
    A[Ni//2, 0] = A[Ni//2, 0].real
    A[0, Nk//2] = A[0, Nk//2].real
    A[Ni//2, Nk//2] = A[Ni//2, Nk//2].real
    for i, k in product(range(1, Ni//2), range(1, Nk//2)):
        phase = np.random.uniform(0., np.pi)
        rad = (np.power(i**2 + k**2, -(H+1.)/2.) *
               np.random.normal(0., 1.))
        A[i, Nk-k] = rad * np.exp(phase*1.j)
        A[Ni-i, k] = rad * np.exp(-phase*1.j)
    X = ifft2d(A)
    return X, A


def ifft2d(A):
    Ni, Nk = A.shape
    a = pyfftw.empty_aligned((Ni, Nk), dtype='complex128')
    b = pyfftw.empty_aligned((Ni, Nk), dtype='complex128')
    ifft_obj = pyfftw.FFTW(a, b, axes=(0, 1), direction='FFTW_BACKWARD')
    a[:] = A
    ifft_a = ifft_obj()
    return np.array(ifft_a).real


def power_spectrum(z_in, dim=0):
    if dim == 0:
        z = z_in
    elif dim == 1:
        z = z_in.T
    else:
        exit("Wrong dim")
    Ni, Nk = z.shape
    a = pyfftw.empty_aligned(Ni, dtype='float64')
    b = pyfftw.empty_aligned(Ni//2 + 1, dtype='complex128')
    power = np.zeros(Ni//2+1)
    fft_obj = pyfftw.FFTW(a, b)
    for k in range(Nk):
        a[:] = z[:, k]
        fft_a = np.asarray(fft_obj())
        power += np.real(fft_a * np.conj(fft_a))/(N//2)
    return power


def parse_args():
    parser = argparse.ArgumentParser(description='Create a rough surface.')
    parser.add_argument('H', type=float, help='Hurst exponent')
    parser.add_argument('N', type=int, help='Linear size of surface')
    parser.add_argument('--seed', type=int, default=32, help='Random seed')
    parser.add_argument('-o', type=str, default="rough_surf.h5",
                        help='Name of output file')
    parser.add_argument('-v', action='store_true',
                        help='View plots')
    parser.add_argument("--plot_intv", type=int, default=1, help="Plot interval")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    H = args.H
    N = args.N
    Ni = N
    Nk = N
    # A = np.zeros((N, N), dtype=complex)
    seed = args.seed
    outfilename = args.o

    x, y = np.meshgrid(range(Nk), range(Ni))
    plot_intv = args.plot_intv

    z, ftz = generate_surface_fourier(Ni, Nk, H, seed=seed)
    with h5py.File(outfilename, "w") as h5f:
        h5f.create_dataset("surface/z", data=z)

    a = ftz.real
    b = ftz.imag

    a[0, 0] = 0
    b[0, 0] = 0
    
    n_modes = 10

    z_rec2 = np.zeros((Ni, Nk))
    I, K = np.meshgrid(np.arange(Ni), np.arange(Nk), indexing="ij")
    ofile = open("modes.dat", "w")
    for m in range(-n_modes, n_modes+1):
        phi_m = 2*np.pi*m*I/Ni
        for n in range(-n_modes, n_modes+1):
            phi_n = 2*np.pi*n*K/Nk
            z_rec2[:, :] += 1.0/(Ni*Nk) * (a[m, n] * np.cos(phi_m + phi_n)
                                           - b[m, n] * np.sin(phi_m + phi_n))
            ofile.write("{} {} {} {}\n".format(m, n, a[m, n], -b[m, n]))
    ofile.close()
            
    plt.figure()
    plt.imshow(z_rec2)

    power = power_spectrum(z_rec2, 0)
    print(power)
    power = power[power > 1e-30]
    print(power)

    f = np.fft.fftfreq(power.size, 1.)
    logf = np.log10(f[f > 0.])
    logpower = np.log10(power[f > 0.])
    p, v = np.polyfit(logf[:len(logf)], logpower[:len(logf)],
                      1, cov=True)

    Hf = -(p[0]+1)/2
    dHf = np.sqrt(v[0, 0])/2
    print("H = {}, Hf = {} +/- {}".format(H, Hf, dHf))

    if args.v:
        plt.figure(0)
        plt.plot(logf, logpower)

        np.savetxt("scaling.dat", np.vstack((logf, logpower)).T)

        plt.plot(logf, p[1] + p[0]*logf, label="f^{}".format(p[0]))
        plt.xlabel("log k")
        plt.ylabel("log P(k)")
        plt.legend()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x[::plot_intv, ::plot_intv],
                               y[::plot_intv, ::plot_intv],
                               z_rec2[::plot_intv, ::plot_intv],
                               rstride=1, cstride=1, cmap=cm.viridis,
                               linewidth=0, antialiased=False)
        ax.auto_scale_xyz([0., N], [0, N], [np.min(z), np.max(z)])

        plt.show()
