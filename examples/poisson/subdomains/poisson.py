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
mesh=UnitSquareMesh(10,10)
V=FunctionSpace(mesh, "CG", 1) # piecewise linear elements

x,y=SpatialCoordinate(mesh)
omega1 = SubDomainData(x < 0.6)
omega2 = SubDomainData(x > 0.4)

u1=TrialFunction(V)
u2=TrialFunction(V)
v1=TestFunction(V)
v2=TestFunction(V)
f=Function(V)
g=Function(V)

g.interpolate(0*x*y+1)
f.interpolate((8*pi*pi)*sin(2*pi*x)*sin(2*pi*y))
a1 = (dot(grad(v1), grad(u1))) * dx
a2 = (dot(grad(v2), grad(u2))) * dx
L1 = f * v1 * dx(subdomain_data=omega1)
L2 = f * v1 * dx(subdomain_data=omega2)
u1=Function(V)
u2=Function(V)
# initialize for doing A.S.
u1.interpolate(0*x*y)
u2.interpolate(0*x*y)

nSchwarzIter=2;




solve(a1 == L1, u1)
solve(a2 == L2, u2)


