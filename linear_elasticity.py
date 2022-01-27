import fenics as fe
from matplotlib.pyplot import show

L = 1
W = 0.2
mu = 1
rho = 1
delta = W / L
gamma = 0.4 * delta ** 2
beta = 1.25
lambda_ = beta
g = gamma


mesh = fe.BoxMesh(fe.Point(0, 0, 0), fe.Point(L, W, W), 10, 3, 3)
V = fe.VectorFunctionSpace(
    mesh, "P", 1
)  # The function space is now a vector field instead of a scalar field!

tol = 1e-14


def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol


bc = fe.DirichletBC(V, fe.Constant((0, 0, 0)), clamped_boundary)

# Define stress ans strain


def epsilon(u):
    return 0.5 * (fe.nabla_grad(u) + fe.nabla_grad(u).T)


def sigma(u):
    return lambda_ * fe.nabla_grad(u) * fe.Identity(d) + 2 * mu * epsilon(u)


# Variational problem
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
d = u.geometric_dimension()
f = fe.Constant((0, 0, -rho * g))
T = fe.Constant((0, 0, 0))
a = fe.inner(sigma(u), epsilon(v)) * fe.dx
l = fe.dot(f, v) * fe.dx + fe.dot(T, v) * fe.ds

# Compute solution
u = fe.Function(V)
fe.solve(a == l, u, bc)

# Plot solution
fe.plot(u, title="Displacement", mode="Displacement")

s = sigma(u) - (1.0 / 3) * fe.tr(sigma(u)) * \
    fe.Identity(d)  # Deviatoric stress
von_Mises = fe.sqrt(3.0 / 2 * fe.inner(s, s))
V = fe.FunctionSpace(mesh, "P", 1)
von_Mises = fe.project(von_Mises, V)
fe.plot(von_Mises, title="Stress intensity")


# Compute magnitude of displacement
u_magnitude = fe.sqrt(fe.dot(u, u))
u_magnitude = fe.project(u_magnitude, V)
fe.plot(u_magnitude, title="Displacement magnitude")
print(
    "Min/max u: ",
    u_magnitude.vector().get_local().min(),
    u_magnitude.vector().get_local().max(),
)
show()
