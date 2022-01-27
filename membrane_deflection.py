from __future__ import print_function
import numpy as np
from fenics import *
from matplotlib import interactive
from mshr import *

import matplotlib.pyplot as plt


# generate domain
domain = Circle(Point(0, 0), 1)
mesh = generate_mesh(domain, 64)

V = FunctionSpace(mesh, "P", 1)


# check for boundary
tol = 1e-14


def boundary(x, on_boundary):
    return on_boundary


# boundary condition
D_b = Constant(0.0)
bc = DirichletBC(V, D_b, boundary)


# Point load
beta = 4
R0 = 0.6
p = Expression(
    "4*exp(-beta*beta*(x[0]*x[0]+(x[1]-R0)*(x[1]-R0)))", degree=2, beta=beta, R0=R0
)

# Arguments can be changed in runtime
p.beta = 12
p.R0 = 0.3

w = TrialFunction(V)
v = TestFunction(V)

# PDE
a = dot(grad(w), grad(v))*dx
l = p*v*dx

w = Function(V)
solve(a == l, w, bc)

# Post processing

# Interpolation of the load p onto the finite element function space
p_i = interpolate(p, V)

plt.figure(1)
plot(w, title="Deflection")
plt.savefig("membrane_deflection_results/deflection.pdf")
plt.savefig("membrane_deflection_results/deflection.png")
vtkfile_w = File("membrane_deflection_results/deflection.pvd")
vtkfile_w << w

plt.figure(2)
plot(p_i, title="Load")
plt.savefig("membrane_deflection_results/load.pdf")
plt.savefig("membrane_deflection_results/load.png")
vtkfile_p = File("membrane_deflection_results/load.pvd")
vtkfile_p << p_i

# Curve plot along x = 0 comparing p and w

tol = 0.001  # avoid hitting points outside the domain
y = np.linspace(-1 + tol, 1 - tol, 101)
points = [(0, y_) for y_ in y]  # 2D points
w_line = np.array([w(point) for point in points])
p_line = np.array([p_i(point) for point in points])

plt.figure(3)
plt.plot(y, 50 * w_line, "k", linewidth=2)  # magnify w
plt.plot(y, p_line, "b--", linewidth=2)
plt.grid(True)
plt.xlabel("$y$")
plt.legend(["Deflection ($\\times 50$)", "Load"], loc="upper left")
plt.savefig("membrane_deflection_results/curves.pdf")
# plt.savefig('poisson_membrane/curves.png')


plt.show()
