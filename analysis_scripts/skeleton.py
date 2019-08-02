""" plot script """
from common.cmd import info, info_cyan, mpi_is_root
from postprocess import get_step_and_info, get_steps
from utilities.plot import plot_any_field
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import measurements


def description(ts, **kwargs):
    info("Get topological skeleton.")


def extract_defects(nodes, z):
    z = np.asarray(z > 0, dtype=float)
    data = dict()
    for xi, yi, zi in zip(nodes[:, 0], nodes[:, 1], z):
        key = tuple((xi, yi))
        data[key] = zi

    x = np.unique(nodes[:, 0])
    y = np.unique(nodes[:, 1])
    nx = len(x)
    ny = len(y)

    Z = np.zeros((ny, 3*nx))
    for i, x_loc in enumerate(x):
        for j, y_loc in enumerate(y):
            val = data[(x_loc, y_loc)]
            Z[j, i] = val
            Z[j, nx+i] = val
            Z[j, 2*nx+i] = val

    s = skeletonize(Z)
    s = s[:, nx:2*nx]
    Z = Z[:, nx:2*nx]

    adjacents = {(-1, 1), (0, 1), (1, 1),
                 (-1, 0), (1, 0),
                 (-1, -1), (0, -1), (1, -1)}

    B = np.zeros(s.shape)
    for i, row in enumerate(s):
        for j, val in enumerate(row):
            if s[i, j]:
                for di, dj in adjacents:
                    imod = (i+di) % ny
                    jmod = (j+dj) % nx
                    B[i, j] += s[imod, jmod]

    I, J = np.meshgrid(range(nx), range(ny))

    B2 = np.asarray(np.logical_and(B > 2, B < 4))
    lw, num = measurements.label(B2)
    T = np.zeros_like(lw)
    for c in range(1, lw.max()+1):
        ids = (lw == c)
        ic = int(np.round(np.mean(I[ids])))
        jc = int(np.round(np.mean(J[ids])))
        T[jc, ic] = 1

    Q = B == 1

    return (Q, T, Z)


def method(ts, dt=0, save=True, **kwargs):
    """ Get topological skeleton. """
    info_cyan("Plotting at given time/step using Matplotlib.")

    params = ts.get_parameters()
    steps = get_steps(ts, dt)

    data = np.zeros((len(steps), 4))

    if save:
        plt.figure()
    for i, step in enumerate(steps):
        info("Step {} of {}".format(step, len(ts)))

        if mpi_is_root():
            for field in ts.fields:
                Q, T, Z = extract_defects(ts.nodes, ts[field, step])

                data[i, 0] = step
                data[i, 1] = ts.times[step]
                data[i, 2] = np.sum(Q)
                data[i, 3] = np.sum(T)

                # plt.figure()
                if save:
                    plt.imshow(T - Q + 0.5*Z)
                    plt.savefig(os.path.join(ts.plots_folder, "defects_{}.png".format(step)))
                # plt.show()

    if mpi_is_root():
        with open(os.path.join(ts.analysis_folder,
                               "defects_in_time.dat"),
                  "w") as outfile:
            np.savetxt(outfile, data, header="Step\tTime\tN-\tN+")
