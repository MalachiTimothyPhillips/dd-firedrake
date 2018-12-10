from firedrake import *
mesh=Mesh(...)
V=FunctionSpace(mesh, "CG", 1)
u=TrialFunction(V)
v=TestFunction(V)
f=Function(V)
x,y=SpatialCoordinate(mesh)
f.interpolate((8*pi*pi)*sin(2*pi*x)*sin(2*pi*y))
params={'ksp_type':'preonly','pc_type':'lu'}
a=dot(grad(u),grad(v))*dx
L=f*v*dx
myBCSurfaces=[...]
myBCs=[]
for surface in myBCSurfaces:
    myBCs.append(DirichletBC(V,0,surface))
BEGIN PROGRAM HERE
mesh_name ~ mesh
form ~ (dot(grad(u), grad(v))) * dx
rhs ~ f * v * dx
space_variable ~ coord
functionSpace ~ V
dirichletBCs={...}
# domain number to id mapping
domains={...}
# [domain number, domain number]->interface id mapping
interfaces={...}
solution~u
solver_settings ~ solver_parameters=params
END PROGRAM HERE
