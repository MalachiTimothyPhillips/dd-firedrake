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
mesh=Mesh("../meshing/4x4Overlap.msh")
N=4
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

bcNums={1:0,2:0,16:0,51:0,5:0,18:0,32:0,55:0,9:0,34:0,48:0,59:0,13:0,50:0,64:0,63:0,}
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
dirichletBCs={1:0,2:0,16:0,51:0,5:0,18:0,32:0,55:0,9:0,34:0,48:0,59:0,13:0,50:0,64:0,63:0,}
# domain number to id mapping
domains={"Omega1":1,"Omega2":2,"Omega3":3,"Omega4":4,"Omega5":5,"Omega6":6,"Omega7":7,"Omega8":8,"Omega9":9,"Omega10":10,"Omega11":11,"Omega12":12,"Omega13":13,"Omega14":14,"Omega15":15,"Omega16":16,}
# [domain number, domain number]->interface id mapping
interfaces={"dOmega6nOmega7":24,"dOmega6nOmega5":22,"dOmega6nOmega10":23,"dOmega6nOmega2":21,"dOmega10nOmega11":40,"dOmega10nOmega9":38,"dOmega10nOmega14":39,"dOmega10nOmega6":37,"dOmega7nOmega8":28,"dOmega7nOmega6":26,"dOmega7nOmega11":27,"dOmega7nOmega3":25,"dOmega11nOmega12":44,"dOmega11nOmega10":42,"dOmega11nOmega15":43,"dOmega11nOmega7":41,"dOmega5nOmega6":20,"dOmega5nOmega9":19,"dOmega5nOmega1":17,"dOmega9nOmega10":36,"dOmega9nOmega13":35,"dOmega9nOmega5":33,"dOmega8nOmega7":30,"dOmega8nOmega12":31,"dOmega8nOmega4":29,"dOmega12nOmega11":46,"dOmega12nOmega16":47,"dOmega12nOmega8":45,"dOmega2nOmega3":8,"dOmega2nOmega1":6,"dOmega2nOmega6":7,"dOmega3nOmega4":12,"dOmega3nOmega2":10,"dOmega3nOmega7":11,"dOmega14nOmega15":56,"dOmega14nOmega13":54,"dOmega14nOmega10":53,"dOmega15nOmega16":60,"dOmega15nOmega14":58,"dOmega15nOmega11":57,"dOmega1nOmega2":4,"dOmega1nOmega5":3,"dOmega13nOmega14":52,"dOmega13nOmega9":49,"dOmega4nOmega3":14,"dOmega4nOmega8":15,"dOmega16nOmega15":62,"dOmega16nOmega12":61,}
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
