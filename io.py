import dolfin as df


def dump_xdmf(f):
    filename = "{}.xdmf".format(f.name())
    with df.XDMFFile(df.MPI.comm_world, filename) as xdmff:
        xdmff.write(f)
