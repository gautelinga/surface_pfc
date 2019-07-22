import dolfin as df


class PBC(df.SubDomain):
    def __init__(self, ts_min, ts_max):
        self.t_min = ts_min[0]
        self.t_max = ts_max[0]
        self.s_min = ts_min[1]
        self.s_max = ts_max[1]
        df.SubDomain.__init__(self)


class EllipsoidPBC(PBC):
    def inside(self, x, on_boundary):
        return bool(df.near(x[0], self.t_min) and on_boundary)

    # Left side is master
    def map(self, x, y):
        y[0] = self.t_min
        y[1] = x[1]


class CylinderPBC(PBC):
    def inside(self, x, on_boundary):
        return bool(df.near(x[0], self.t_min) and on_boundary)

    # Left side is master
    def map(self, x, y):
        if x[0] > self.t_max - 100*df.DOLFIN_EPS:
            y[0] = self.t_min
            y[1] = x[1]
