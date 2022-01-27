
from fenics import *
from dolfin_adjoint import *
from ufl_dnn.neural_network import ANN
import matplotlib.pyplot as plt
import numpy as np

# Problem parameters
thetamoy = 0.4 # target average material density
E = Constant(1)
nu = Constant(0.3)
lamda = E*nu/(1+nu)/(1-2*nu)
mu = E/(2*(1+nu))
f = Constant((0, -1)) # vertical downwards force

# Mesh
mesh = RectangleMesh(Point(-2, 0), Point(2, 1), 50, 30, "crossed")
Nx, Ny = 20, 20
mesh = UnitSquareMesh(Nx, Ny)
x, y = SpatialCoordinate(mesh)

plot(mesh)
plt.show()

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

p = Constant(1) # SIMP penalty exponent
exponent_counter = 0 # exponent update counter
lagrange = Constant(1) # Lagrange multiplier for volume constraint

thetaold = Function(V0, name="Density")
thetaold.interpolate(Expression("(1-x[1])*0.4+0.6",degree=2))
coeff = thetaold**p
theta = Function(V0)
plot(thetaold)
plt.show()

volume = assemble(Constant(1.)*dx(domain=mesh))
avg_density_0 = assemble(thetaold*dx)/volume # initial average density
avg_density = 0.


def eps(v):
    return sym(grad(v))
def sigma(v):
    return lamda*div(v)*Identity(2)+2*mu*eps(v)
def energy_density(u, v):
    return inner(sigma(u), eps(v))

# Inhomogeneous elastic variational problem
u_ = TestFunction(V2)
du = TrialFunction(V2)

E_ex = inner(coeff*sigma(du), eps(u_))*dx - dot(f, u_)*ds(1)
u_obs = Function(V2, name="Displacement")

solve(lhs(E) == rhs(E), u_obs, bc, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})

plot(u_obs)
plt.show()

bias = [True, True]
layers = [2, 10, 1]
net = ANN(layers, bias=bias, mesh=mesh)
theta_pred = project(net(x,y),V0)


coeff = theta_pred**p
theta = Function(V0)
plot(theta_pred)


plt.show()

# Inhomogeneous elastic variational problem
u_ = TestFunction(V2)
du = TrialFunction(V2)
E = inner(net(x,y)**p*sigma(u_), eps(du))*dx - dot(f, u_)*ds(1)

u_hat = Function(V2, name="Displacement")

# Solve PDE
solve(lhs(E) == rhs(E), u_hat, bc)

plot(u_hat)
plt.show()



# L ^ 2 error as loss
loss = assemble((u_hat - u_obs) ** 2 * dx)  # Loss function

# Define reduced formulation of problem
hat_loss = ReducedFunctional(loss, net.weights_ctrls())

#Use scipy L - BFGS optimiser
opt_theta = minimize(
    hat_loss, options={"disp": True, "gtol": 1e-12, "ftol": 1e-12, "maxiter": 100}
)
print(opt_theta)
net.set_weights(opt_theta)


u_test = Function(V2)

E_test = inner(net(x,y)*sigma(u_), eps(du))*dx - dot(f, u_)*ds(1)

solve(lhs(E_test) == rhs(E_test),u_test,bc)

theta_pred = project(net(x,y),V0)
plot(theta_pred)


plt.show()


plot(u_test)

plt.show()
