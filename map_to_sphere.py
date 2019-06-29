import dolfin as df
from maps import SphereMap
from io import dump_xdmf


R = 1.0
res = 100

geo_map = SphereMap(R)
geo_map.initialize_ref_space(res)
ref_mesh = geo_map.ref_mesh
geo_map.initialize_metric()

n = geo_map.normal()
xyz = geo_map.coords()

K = geo_map.get_function("K")
H = geo_map.get_function("H")
try:
    local_area = geo_map.local_area()
    dump_xdmf(local_area)
except:
    pass


dump_xdmf(xyz)
dump_xdmf(n)
dump_xdmf(K)
dump_xdmf(H)

area_ref = df.assemble(df.Constant(1.0)*geo_map.dS_ref)
area = df.assemble(geo_map.form(df.Constant(1.0)))
print(area_ref, area)

u = df.TrialFunction(geo_map.S_ref)
v = df.TestFunction(geo_map.S_ref)

Lx = geo_map.t_max
Ly = geo_map.s_max
u_0 = df.Expression("exp(-(pow(x[0]-x01, 2)+pow(x[1]-y01, 2))/"
                    "(2*pow(sigma, 2)))",
                    x01=Lx/2, y01=Ly/2,
                    sigma=Lx/30,
                    degree=1)
f = df.Expression("0.0", degree=1)

u_ = df.Function(geo_map.S_ref, name="u")
u_1 = df.interpolate(u_0, geo_map.S_ref)
u_.assign(u_1)

dt = 0.1

F_t = geo_map.form(1./dt*(u-u_1)*v)
F_diff = geo_map.form(geo_map.dot(u, v))
F = F_t + F_diff

a_surf = df.lhs(F)
L_surf = df.rhs(F)

A = df.assemble(a_surf)

solver = df.PETScKrylovSolver("gmres")  # gmres
solver.set_operator(A)


def top(x, on_boundary):
    return on_boundary and x[1] > geo_map.s_max-geo_map.eps-100*df.DOLFIN_EPS


def btm(x, on_boundary):
    return on_boundary and x[1] < geo_map.s_min+geo_map.eps+100*df.DOLFIN_EPS


Top = df.AutoSubDomain(top)
Btm = df.AutoSubDomain(btm)
facets = df.MeshFunction("size_t", ref_mesh, ref_mesh.topology().dim()-1, 0)
Top.mark(facets, 1)
Btm.mark(facets, 2)

t = 0.0
T = 10.0
xdmfff = df.XDMFFile(ref_mesh.mpi_comm(), "u_.xdmf")
xdmfff.parameters["rewrite_function_mesh"] = False
xdmfff.parameters["flush_output"] = True

it = 0
xdmfff.write(u_, float(it))
while t < T:
    it += 1
    b = df.assemble(L_surf)

    solver.solve(u_.vector(), b)

    xdmfff.write(u_, float(it))

    print("int T =", df.assemble(geo_map.form(u_)))
    flx_top = df.assemble(u_.dx(1)*df.ds(
        1, domain=ref_mesh, subdomain_data=facets))
    flx_btm = df.assemble(u_.dx(1)*df.ds(
        2, domain=ref_mesh, subdomain_data=facets))
    print("flux  =", flx_top, flx_btm)

    u_1.assign(u_)
    t += dt

xdmfff.close()
