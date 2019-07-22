import dolfin as df
import os
from .cmd import mpi_is_root, mpi_barrier, mpi_comm, info_red


def dump_xdmf(f, folder=""):
    filename = os.path.join(folder, "{}.xdmf".format(f.name()))
    with df.XDMFFile(mpi_comm(), filename) as xdmff:
        xdmff.parameters["rewrite_function_mesh"] = False
        xdmff.parameters["flush_output"] = True
        xdmff.write(f)


def makedirs_safe(folder):
    """ Make directory in a safe way. """
    if mpi_is_root() and not os.path.exists(folder):
        os.makedirs(folder)


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
    makedirs_safe(geofolder)
    makedirs_safe(tsfolder)
    # GL: add more?
    return folder


class Timeseries:
    def __init__(self, results_folder, u_, field_names, geo_map, tstep0):
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

        self.folder = create_directories(results_folder)
        geofolder = os.path.join(self.folder, "Geometry")
        dump_xdmf(geo_map.coords(), folder=geofolder)
        dump_xdmf(geo_map.normal(), folder=geofolder)

        self.files = dict()
        for field in self.fields:
            filename = os.path.join(self.folder, "Timeseries",
                                    "{}_from_tstep_{}".format(field,
                                                              self.tstep0))
            self.files[field] = self._create_file(filename)

        self.extra_fields = dict()

    def dump(self, tstep):
        q_ = self._unpack()
        for field, qi_ in zip(self.fields, q_):
            qi_.rename(field, "tmp")
            self.files[field].write(qi_, tstep)

        for field, ufl_expression in self.extra_fields.items():
            q_tmp = df.project(ufl_expression, q_[0].function_space().collapse())
            q_tmp.rename(field, "tmp")
            self.files[field].write(q_tmp, tstep)

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

    def add_scalar_field(self, ufl_expression, field_name):
        filename = os.path.join(self.folder, "Timeseries",
                                "{}_from_tstep_{}".format(field_name, self.tstep0))
        self.extra_fields[field_name] = ufl_expression
        self.files[field_name] = self._create_file(filename)
