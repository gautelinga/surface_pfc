import dolfin as df
import os
from .cmd import mpi_is_root, mpi_barrier, mpi_comm, \
    mpi_size, mpi_rank, info_red, info_cyan, info_on_red
import numpy as np
import json
import h5py
import cloudpickle as pickle
from .utilities import NdFunction


def dump_xdmf(f, folder=""):
    filename = os.path.join(folder, "{}.xdmf".format(f.name()))
    with df.XDMFFile(mpi_comm(), filename) as xdmff:
        xdmff.parameters["rewrite_function_mesh"] = False
        xdmff.parameters["flush_output"] = True
        xdmff.write(f)


def dump_coords(geo_map, folder="", name="xyz"):
    filename = os.path.join(folder, "{}.xdmf".format(name))
    xyz = geo_map.coords()
    with df.XDMFFile(mpi_comm(), filename) as xdmff:
        xdmff.parameters["rewrite_function_mesh"] = False
        xdmff.parameters["flush_output"] = True
        xdmff.write(xyz)
    mpi_barrier()
    if mpi_is_root() and geo_map.is_periodic_in_3d():
        with h5py.File(os.path.join(
                folder, "{}.h5".format(name)), "r+") as h5f:
            ts = np.array(h5f["Mesh/mesh/geometry"])
            t = ts[:, 0]
            s = ts[:, 1]
            xyz_new = np.vstack((geo_map.evalf["x"](t, s),
                                 geo_map.evalf["y"](t, s),
                                 geo_map.evalf["z"](t, s))).T
            xyz = h5f["VisualisationVector/0"]
            xyz[:, :] = xyz_new


def dump_dcoords(geo_map, folder="", name="dxyz"):
    dxyz = geo_map.dcoords()
    for i, xyzi in enumerate(dxyz):
        filename = os.path.join(folder, "{}{}.xdmf".format(name, i))
        with df.XDMFFile(mpi_comm(), filename) as xdmff:
            xdmff.parameters["rewrite_function_mesh"] = False
            xdmff.parameters["flush_output"] = True
            xdmff.write(xyzi)


def dump_metric_tensor(geo_map, folder="", name="g_ab"):
    filename = os.path.join(folder, "{}.xdmf".format(name))
    g = geo_map.metric_tensor()
    with df.XDMFFile(mpi_comm(), filename) as xdmff:
        xdmff.parameters["rewrite_function_mesh"] = False
        xdmff.parameters["flush_output"] = True
        xdmff.write(g)


def dump_metric_tensor_inv(geo_map, folder="", name="gab"):
    # Is it necessary to dump both?
    filename = os.path.join(folder, "{}.xdmf".format(name))
    g_inv = geo_map.metric_tensor_inv()
    with df.XDMFFile(mpi_comm(), filename) as xdmff:
        xdmff.parameters["rewrite_function_mesh"] = False
        xdmff.parameters["flush_output"] = True
        xdmff.write(g_inv)


def dump_curvature_tensor(geo_map, folder="", name="K_ab"):
    filename = os.path.join(folder, "{}.xdmf".format(name))
    K_ab = geo_map.curvature_tensor()
    with df.XDMFFile(mpi_comm(), filename) as xdmff:
        xdmff.parameters["rewrite_function_mesh"] = False
        xdmff.parameters["flush_output"] = True
        xdmff.write(K_ab)


def dump_map(geo_map, folder=""):
    filename = os.path.join(folder, "map.pkl")
    if mpi_is_root():
        with open(filename, "wb") as f:
            pickle.dump(geo_map.map, f)


def dump_evalf(geo_map, folder=""):
    filename = os.path.join(folder, "evalf.pkl")
    if mpi_is_root():
        with open(filename, "wb") as f:
            pickle.dump(geo_map.evalf, f)


def makedirs_safe(folder):
    """ Make directory in a safe way. """
    if mpi_is_root() and not os.path.exists(folder):
        os.makedirs(folder)


def remove_safe(path):
    """ Remove file in a safe way. """
    if mpi_is_root() and os.path.exists(path):
        os.remove(path)


def create_directories(results_folder):
    """ Create folders. """
    makedirs_safe(results_folder)
    mpi_barrier()

    info_red("Creating initial folders.")

    # GL: Add option to restart.

    previous_list = os.listdir(results_folder)
    if len(previous_list) == 0:
        folder = os.path.join(results_folder, "1")
    else:
        previous = max([int(entry) if entry.isdigit() else 0
                        for entry in previous_list])
        folder = os.path.join(results_folder, str(previous+1))

    mpi_barrier()

    geofolder = os.path.join(folder, "Geometry")
    tsfolder = os.path.join(folder, "Timeseries")
    statsfolder = os.path.join(folder, "Statistics")
    checkpointfolder = os.path.join(folder, "Checkpoint")
    settingsfolder = os.path.join(folder, "Settings")
    makedirs_safe(geofolder)
    makedirs_safe(tsfolder)
    makedirs_safe(statsfolder)
    makedirs_safe(checkpointfolder)
    makedirs_safe(settingsfolder)
    # GL: add more?
    return folder


class Timeseries:
    def __init__(self, results_folder, u_, field_names, geo_map, tstep0=0,
                 parameters=None, restart_folder=None):
        self.u_ = u_  # Pointer
        self.tstep0 = tstep0
        num_sub_el = u_.function_space().ufl_element().num_sub_elements()
        self.fields = field_names
        if num_sub_el > 0:
            assert(num_sub_el == len(field_names))
        else:
            assert(len(field_names) == 1 or isinstance(field_names, str))
            if isinstance(field_names, str):
                self.fields = (field_names,)

        if restart_folder is None:
            self.folder = create_directories(results_folder)
        else:
            self.folder = restart_folder.split("Checkpoint")[0]
        geofolder = os.path.join(self.folder, "Geometry")
        checkpointfolder = os.path.join(self.folder, "Checkpoint")
        dump_coords(geo_map, folder=geofolder)
        dump_dcoords(geo_map, folder=geofolder)
        dump_xdmf(geo_map.normal(), folder=geofolder)
        dump_metric_tensor(geo_map, folder=geofolder)
        dump_metric_tensor_inv(geo_map, folder=geofolder)
        dump_curvature_tensor(geo_map, folder=geofolder)
        dump_map(geo_map, folder=checkpointfolder)
        dump_evalf(geo_map, folder=checkpointfolder)

        self.files = dict()
        for field in self.fields:
            filename = os.path.join(self.folder, "Timeseries",
                                    "{}_from_tstep_{}".format(field,
                                                              self.tstep0))
            self.files[field] = self._create_file(filename)

        if parameters:
            parametersfile = os.path.join(
                self.folder, "Settings",
                "parameters_from_tstep_{}.dat".format(self.tstep0))
            dump_parameters(parameters, parametersfile)

        self.S_ref = geo_map.S_ref

        self.extra_fields = dict()
        self.extra_field_functions = dict()

    def dump(self, tstep):
        q_ = self._unpack()
        for field, qi_ in zip(self.fields, q_):
            qi_.rename(field, "tmp")
            self.files[field].write(qi_, tstep)

        # Dumping extra fields
        if len(self.extra_fields) > 0:
            # S = q_[0].function_space().collapse()
            S = self.S_ref
            for field, ufl_expression in self.extra_fields.items():
                if isinstance(ufl_expression, list):
                    v = [df.project(expr_i, S) for expr_i in ufl_expression]
                    vf = NdFunction(v, name=field)
                    vf()
                    self.extra_field_functions[field] = vf
                    self.files[field].write(vf, tstep)
                else:
                    self.extra_field_functions[field] = df.project(ufl_expression,
                                                                   S)
                    self.extra_field_functions[field].rename(field, "tmp")
                    self.files[field].write(self.extra_field_functions[field],
                                            tstep)

    def _unpack(self):
        num_fields = len(self.fields)
        if num_fields == 1:
            return (self.u_,)
        else:
            return self.u_.split()

    def _create_file(self, filename):
        f = df.XDMFFile(mpi_comm(), "{}.xdmf".format(filename))
        f.parameters["rewrite_function_mesh"] = False
        f.parameters["flush_output"] = True
        return f

    def close(self):
        for field in self.files.keys():
            self.files[field].close()

    def add_field(self, ufl_expression, field_name):
        filename = os.path.join(self.folder, "Timeseries",
                                "{}_from_tstep_{}".format(field_name,
                                                          self.tstep0))
        self.extra_fields[field_name] = ufl_expression
        self.files[field_name] = self._create_file(filename)

    def get_function(self, field):
        return self.extra_field_functions[field]

    def dump_stats(self, t, data_at_t, name):
        if mpi_is_root():
            filename = os.path.join(self.folder, "Statistics",
                                    "{}.dat".format(name))
            data_at_t = np.array(data_at_t).flatten()
            with open(filename, "a+") as outfile:
                outfile.write("{}".format(t))
                for d in data_at_t:
                    outfile.write("\t {}".format(d))
                outfile.write("\n")


def save_checkpoint(tstep, t, mesh, w_, w_1, folder, parameters, name=""):
    """ Save checkpoint files.
    A part of this is taken from the Oasis code."""
    checkpointfolder = os.path.join(folder, "Checkpoint")
    parameters["num_processes"] = mpi_size()
    parameters["t_0"] = t
    parameters["tstep"] = tstep
    parametersfile = os.path.join(checkpointfolder, "parameters.dat")
    parametersfile_old = parametersfile + ".old"
    if mpi_is_root():
        # In case of failure, keep old file.
        if os.path.exists(parametersfile):
            os.system("mv {0} {1}".format(parametersfile,
                                          parametersfile_old))
        dump_parameters(parameters, parametersfile)

    mpi_barrier()
    h5filename = os.path.join(checkpointfolder, "fields.h5")
    h5filename_old = h5filename + ".old"
    # In case of failure, keep old file.
    if mpi_is_root() and os.path.exists(h5filename):
        os.system("mv {0} {1}".format(h5filename, h5filename_old))
    h5file = df.HDF5File(mpi_comm(), h5filename, "w")
    h5file.flush()
    info_red("Storing mesh")
    h5file.write(mesh, "mesh")
    mpi_barrier()
    info_red("Storing current solution")
    h5file.write(w_, "{}/current".format(name))
    info_red("Storing previous solution")
    h5file.write(w_1, "{}/previous".format(name))
    mpi_barrier()
    h5file.close()
    # Since program is still running, delete the old files.
    remove_safe(h5filename_old)
    mpi_barrier()
    remove_safe(parametersfile_old)


def load_checkpoint(checkpointfolder, w_, w_1, name=""):
    if checkpointfolder:
        h5filename = os.path.join(checkpointfolder, "fields.h5")
        h5file = df.HDF5File(mpi_comm(), h5filename, "r")
        info_red("Loading current solution")
        h5file.read(w_, "{}/current".format(name))
        info_red("Loading previous solution")
        h5file.read(w_1, "{}/previous".format(name))
        h5file.close()


def load_mesh(filename, subdir="mesh",
              use_partition_from_file=False):
    """ Loads the mesh specified by the argument filename. """
    info_cyan("Loading mesh: " + filename)
    if not os.path.exists(filename):
        info_red("Couldn't find file: " + filename)
        exit()
    mesh = df.Mesh()
    h5file = df.HDF5File(mesh.mpi_comm(), filename, "r")
    h5file.read(mesh, subdir, use_partition_from_file)
    h5file.close()
    return mesh


def dump_parameters(parameters, settingsfilename):
    """ Dump parameters to file """
    with open(settingsfilename, "w") as settingsfile:
        json.dump(parameters, settingsfile, indent=4*' ', sort_keys=True)


def load_parameters(parameters, settingsfilename):
    if not os.path.exists(settingsfilename):
        info_on_red("File " + settingsfilename + " does not exist.")
        exit()
    with open(settingsfilename, "r") as settingsfile:
        parameters.update(json.load(settingsfile))
