""" plot script """
from common.cmd import info, info_cyan, mpi_is_root
from postprocess import get_step_and_info
from utilities.plot import plot_any_field
import os


def description(ts, **kwargs):
    info("Plot at any given timestep or time using Matplotlib.")


def method(ts, time=None, step=0, show=True, save=False, latex=False, **kwargs):
    """ Plot at given timestep using matplotlib. """
    info_cyan("Plotting at given time/step using Matplotlib.")
    if time is not None:
        step, time = get_step_and_info(ts, time)
    if mpi_is_root():
        for field in ts.fields:
            if save:
                save_fig_file = os.path.join(
                    ts.plots_folder, "{}_{:06d}.png".format(field, step))
            else:
                save_fig_file = None

            plot_any_field(ts.nodes, ts.elems, ts[field, step],
                           save=save_fig_file, show=show, label=field,
                           latex=latex)
