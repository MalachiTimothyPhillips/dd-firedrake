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
mesh = UnitSquareMesh(20,20)
V=FunctionSpace(mesh, "CG", 1) # piecewise linear elements

u=TrialFunction(V)
v=TestFunction(V)
f=Function(V)
x,y=SpatialCoordinate(mesh)
f.interpolate((8*pi*pi)*sin(2*pi*x)*sin(2*pi*y))
a = (dot(grad(v), grad(u))) * dx
L = f * v * dx
u=Function(V)

"""

Homogeneous dirichlet boundary conditions throughout mean that faces 1,2,3,4 must
have zero on their own boundary conditions

"""
bc1=DirichletBC(V,0.0,1)
bc2=DirichletBC(V,0.0,2)
bc3=DirichletBC(V,0.0,3)
bc4=DirichletBC(V,0.0,4) # without these particular BCs, we would be solving the Neumann problem instead (which is not what we want!)

# form as linear variational problem to avoid re-assembly
problem=LinearVariationalProblem(a,L,u,bcs=[bc1,bc2,bc3,bc4])
solver=LinearVariationalSolver(problem)
solver.solve()
File("poisson.pvd").write(u)
f.interpolate(sin(2*pi*x)*sin(2*pi*y))
print("Error in solution of: {}".format(sqrt(assemble(dot(u-f,u-f)*dx))))
