import dolfin as df


def dump_xdmf(f):
    filename = "{}.xdmf".format(f.name())
    with df.XDMFFile(df.MPI.comm_world, filename) as xdmff:
        xdmff.write(f)


class Timeseries:
    def __init__(self, name):
        self.xdmfff = df.XDMFFile(df.MPI.comm_world, "{}.xdmf".format(name))
        self.xdmfff.parameters["rewrite_function_mesh"] = False
        self.xdmfff.parameters["flush_output"] = True

    def write(self, f, it):
        self.xdmfff.write(f, it)

    def close(self):
        self.xdmfff.close()
