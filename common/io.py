import dolfin as df


def dump_xdmf(f):
    filename = "{}.xdmf".format(f.name())
    with df.XDMFFile(df.MPI.comm_world, filename) as xdmff:
        xdmff.write(f)


class Timeseries:
    def __init__(self, filename, name=None, space=None):
        self.xdmfff = df.XDMFFile(df.MPI.comm_world, "{}.xdmf".format(filename))
        self.xdmfff.parameters["rewrite_function_mesh"] = False
        self.xdmfff.parameters["flush_output"] = True
        self.name = name
        self.space = space

    def write(self, f, it):
        if isinstance(f, df.function.function.Function):
            if self.name is not None:
                f.rename(self.name, "tmp")
            self.xdmfff.write(f, it)
        else:
            ff = df.project(f, self.space)
            if self.name is not None:
                ff.rename(self.name, "tmp")
            self.xdmfff.write(ff, it)

    def close(self):
        self.xdmfff.close()


def export_xdmf():
    pass
