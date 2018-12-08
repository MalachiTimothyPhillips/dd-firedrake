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
import time
mesh=Mesh(MY_INPUT_MESH)
N=MY_NX
N_subdom=N*N
V=FunctionSpace(mesh, "CG", 1) # piecewise linear elements

u=TrialFunction(V)
v=TestFunction(V)
f=Function(V)
x,y=SpatialCoordinate(mesh)
f.interpolate((8*pi*pi)*sin(2*pi*x)*sin(2*pi*y))
params={'ksp_type':'preonly','pc_type':'lu'}

a=dot(grad(u),grad(v))*dx
L=f*v*dx
u_entire=Function(V)

bcNums=BC_MAP
myBCs=[]
for surface_face, value in bcNums.items():
    myBCs.append(DirichletBC(V,value,surface_face))
start_time1=time.time()
solve(a==L,u_entire,bcs=myBCs,solver_parameters=params)
end_time1=time.time()

# for our purposes, we need to also generate timing data for the regular solve

start_time2=time.time()
BEGIN_PROGRAM_HERE
mesh_name ~ mesh
form ~ (dot(grad(u), grad(v))) * dx
rhs ~ f * v * dx
space_variable ~ coord
functionSpace ~ V
dirichletBCs=BC_MAP
# domain number to id mapping
domains=SUBDOMAIN_MAP
# [domain number, domain number]->interface id mapping
interfaces=INTERFACE_MAP
# solver selection, parameters, etc
# respects everything past bcs (i.e., just keeps it the same)
solution~u

# always of the form: form == rhs, solution
solver_settings ~ solver_parameters=params

END_PROGRAM_HERE
end_time2=time.time()
print("{},{},{},{}".format(N,N_subdom,end_time1-start_time1,end_time2-start_time2))
# does this line exist?
# perhaps some other lines after the fact?
