#!/usr/bin/env python


# get_ipython().run_line_magic('matplotlib', 'notebook')

from fenics import *
from dolfin_adjoint import *
from ufl_dnn.neural_network import ANN
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# Algorithmic parameters
niternp = 30  # number of non-penalized iterations
niter = 30  # total number of iterations
pmax = 4  # maximum SIMP exponent
exponent_update_frequency = 4  # minimum number of steps between exponent update
tol_mass = 1e-4  # tolerance on mass when finding Lagrange multiplier
thetamin = 0.001  # minimum density modeling void


# Problem parameters
thetamoy = 0.4  # target average material density
E = Constant(1)
nu = Constant(0.3)
lamda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / (2 * (1 + nu))
f = Constant((0, -1))  # vertical downwards force

# Mesh
# mesh = RectangleMesh(Point(-2, 0), Point(2, 1), 50, 30, "crossed")
Nx, Ny = 20, 20
mesh = UnitSquareMesh(Nx, Ny)
x, y = SpatialCoordinate(mesh)

# Boundaries
def left(x, on_boundary):
    return near(x[0], -2) and on_boundary


def load(x, on_boundary):
    return near(x[0], 2) and near(x[1], 0.5, 0.05)


def bottom(x, on_boundary):
    return near(x[1], 0) and on_boundary


def load_top(x, on_boundary):
    return near(x[1], 1)


facets = MeshFunction("size_t", mesh, 1)
AutoSubDomain(load_top).mark(facets, 1)
ds = Measure("ds", subdomain_data=facets)

# Function space for density field
V0 = FunctionSpace(mesh, "DG", 0)
# Function space for displacement
V2 = VectorFunctionSpace(mesh, "CG", 2)
# Fixed boundary condtions
bc = DirichletBC(V2, Constant((0, 0)), bottom)

p = Constant(1)  # SIMP penalty exponent
exponent_counter = 0  # exponent update counter
lagrange = Constant(1)  # Lagrange multiplier for volume constraint

thetaold = Function(V0, name="Density")
thetaold.interpolate(Constant(thetamoy))
coeff = thetaold ** p
theta = Function(V0)

volume = assemble(Constant(1.0) * dx(domain=mesh))
avg_density_0 = assemble(thetaold * dx) / volume  # initial average density
avg_density = 0.0


# We now define some useful functions for formulating the linear elastic variational problem.


def eps(v):
    return sym(grad(v))


def sigma(v):
    return coeff * (lamda * div(v) * Identity(2) + 2 * mu * eps(v))


def energy_density(u, v):
    return inner(sigma(u), eps(v))


# Inhomogeneous elastic variational problem
u_ = TestFunction(V2)
du = TrialFunction(V2)
a = inner(sigma(du), eps(u_)) * dx
L = dot(f, u_) * ds(1)


def local_project(v, V):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_) * dx
    b_proj = inner(v, v_) * dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    u = Function(V)
    solver.solve_local_rhs(u)
    return u


def update_theta():
    theta.assign(
        local_project(
            (p * coeff * energy_density(u, u) / lagrange) ** (1 / (p + 1)), V0
        )
    )
    thetav = theta.vector().get_local()
    theta.vector().set_local(np.maximum(np.minimum(1, thetav), thetamin))
    theta.vector().apply("insert")
    avg_density = assemble(theta * dx) / volume
    return avg_density


def update_lagrange_multiplier(avg_density):
    avg_density1 = avg_density
    # Initial bracketing of Lagrange multiplier
    if avg_density1 < avg_density_0:
        lagmin = float(lagrange)
        while avg_density < avg_density_0:
            lagrange.assign(Constant(lagrange / 2))
            avg_density = update_theta()
        lagmax = float(lagrange)
    elif avg_density1 > avg_density_0:
        lagmax = float(lagrange)
        while avg_density > avg_density_0:
            lagrange.assign(Constant(lagrange * 2))
            avg_density = update_theta()
        lagmin = float(lagrange)
    else:
        lagmin = float(lagrange)
        lagmax = float(lagrange)

    # Dichotomy on Lagrange multiplier
    inddico = 0
    while (abs(1.0 - avg_density / avg_density_0)) > tol_mass:
        lagrange.assign(Constant((lagmax + lagmin) / 2))
        avg_density = update_theta()
        inddico += 1
        if avg_density < avg_density_0:
            lagmin = float(lagrange)
        else:
            lagmax = float(lagrange)
    print("   Dichotomy iterations:", inddico)


def update_exponent(exponent_counter):
    exponent_counter += 1
    if i < niternp:
        p.assign(Constant(1))
    elif i >= niternp:
        if i == niternp:
            print("\n Starting penalized iterations\n")
        if (abs(compliance - old_compliance) < 0.01 * compliance_history[0]) and (
            exponent_counter > exponent_update_frequency
        ):
            # average gray level
            gray_level = assemble((theta - thetamin) * (1.0 - theta) * dx) * 4 / volume
            p.assign(
                Constant(min(float(p) * (1 + 0.3 ** (1.0 + gray_level / 2)), pmax))
            )
            exponent_counter = 0
            print("   Updated SIMP exponent p = ", float(p))
    return exponent_counter


u = Function(V2, name="Displacement")
old_compliance = 1e30
ffile = XDMFFile("topology_optimization.xdmf")
ffile.parameters["flush_output"] = True
ffile.parameters["functions_share_mesh"] = True


compliance_history = []
for i in range(niter):
    solve(
        a == L,
        u,
        bc,
        solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"},
    )
    ffile.write(thetaold, i)
    ffile.write(u, i)

    compliance = assemble(action(L, u))
    compliance_history.append(compliance)
    print("Iteration {}: compliance =".format(i), compliance)

    avg_density = update_theta()

    update_lagrange_multiplier(avg_density)

    exponent_counter = update_exponent(exponent_counter)

    # Update theta field and compliance
    thetaold.assign(theta)
    old_compliance = compliance


plot(theta, cmap="bone_r")
plt.title("Final density")
plt.show()

plt.figure()
plt.plot(np.arange(1, niter + 1), compliance_history)
ax = plt.gca()
ymax = ax.get_ylim()[1]
plt.plot([niternp, niternp], [0, ymax], "--k")
plt.annotate(
    r"$\leftarrow$ Penalized iterations $\rightarrow$",
    xy=[niternp + 1, ymax * 0.02],
    fontsize=14,
)
plt.xlabel("Number of iterations")
plt.ylabel("Compliance")
plt.title("Convergence history", fontsize=16)
plt.show()


# Inhomogeneous elastic variational problem
u_ = TestFunction(V2)
du = TrialFunction(V2)
coeff = theta

E_ex = inner(sigma(du), eps(u_)) * dx - dot(f, u_) * ds(1)
u_obs = Function(V2, name="Displacement")

solve(
    lhs(E_ex) == rhs(E_ex),
    u_obs,
    bc,
    solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"},
)


plot(u_obs)
plt.show()
coeff = 1

bias = [True, True]
layers = [2, 10, 1]
net = ANN(layers, bias=bias, mesh=mesh)
theta_pred = project(net(x, y), V0)
plot(theta_pred, cmap="bone_r")
plt.show()


E = inner(net(x, y) * sigma(du), eps(u_)) * dx - dot(f, u_) * ds(1)

u_hat = Function(V2, name="Displacement")

# Solve PDE
solve(lhs(E) == rhs(E), u_hat, bc)

plot(u_hat, title="u_hat")
plt.show()


# L ^ 2 error as loss
loss = assemble((u_hat - u_obs) ** 2 * dx)  # Loss function

# Define reduced formulation of problem
hat_loss = ReducedFunctional(loss, net.weights_ctrls())

# Use scipy L - BFGS optimiser
opt_theta = minimize(
    hat_loss, options={"disp": True, "gtol": 1e-12, "ftol": 1e-12, "maxiter": 100}
)
print(opt_theta)
net.set_weights(opt_theta)


u_test = Function(V2)

E_test = inner(net(x, y) * sigma(u_), eps(du)) * dx - dot(f, u_) * ds(1)

solve(lhs(E_test) == rhs(E_test), u_test, bc)

theta_pred = project(net(x, y), V0)
plot(theta_pred, cmap="bone_r")


plt.show()


plot(u_test)

plt.show()
