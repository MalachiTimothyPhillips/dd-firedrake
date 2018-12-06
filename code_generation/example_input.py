"""
Simple poission problem on the unit square [0,1]^2.
In particular, we formulate this out of the idea that
u = sin(2 \pi x) sin (2 \pi u)

Thus we have that \dfrac{\partial^2 u}{\partial x^2} = -4 \pi^2 sin(2\pix)sin(2\piy).
Similarly, the term \dfrac{\partial^2 u}{\partial y^2} is the same term.
Notice also that this analytical u respects the boundary condition, and thus with
the proper forcing function f(x,y):

    f(x,y) = 8\pi^2cos(2\pix)cos(2\piy)

We are confident that we must have the correct solution for this particular problem.

"""

from firedrake import *
import numpy as np
mesh=Mesh("2x2Overlap.msh")
V=FunctionSpace(mesh, "CG", 1) # piecewise linear elements

u=TrialFunction(V)
v=TestFunction(V)
f=Function(V)
x,y=SpatialCoordinate(mesh)
f.interpolate((8*pi*pi)*sin(2*pi*x)*sin(2*pi*y))
params={'ksp_type':'preonly','pc_type':'lu'}

BEGIN PROGRAM HERE
mesh_name ~ mesh
form ~ (dot(grad(u), grad(v))) * dx
rhs ~ f * v * dx
space_variable ~ coord
functionSpace ~ V
dirichletBCs={1:0,2:0,10:0,11:0,15:0,16:0,8:0,5:0}
# domain number to id mapping
domains={"Omega1":1,"Omega2":2,"Omega3":3,"Omega4":4}
# [domain number, domain number]->interface id mapping
interfaces={"dOmega1nOmega2":4,"dOmega1nOmega3":3,"dOmega2nOmega1":6,"dOmega2nOmega4":7,"dOmega3nOmega1":9,"dOmega3nOmega4":12,"dOmega4nOmega2":13,"dOmega4nOmega3":14}
# solver selection, parameters, etc
# respects everything past bcs (i.e., just keeps it the same)
solution~u

# always of the form: form == rhs, solution
solver_settings ~ solver_parameters=params

END PROGRAM HERE
# does this line exist?
# perhaps some other lines after the fact?
