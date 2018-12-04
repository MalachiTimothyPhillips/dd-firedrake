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
mesh=Mesh("TwoDomain.msh")
V=FunctionSpace(mesh, "CG", 1) # piecewise linear elements

omega1_id=1
omega2_id=2

omega2_face_id=2
omega1_face_id=3

u=TrialFunction(V)
v=TestFunction(V)
f=Function(V)
x,y=SpatialCoordinate(mesh)
f.interpolate((8*pi*pi)*sin(2*pi*x)*sin(2*pi*y))
a1 = (dot(grad(v), grad(u))) * dx
L1 = f * v * dx(omega1_id)
a2 = (dot(grad(v), grad(u))) * dx
L2 = f * v * dx(omega2_id)

u1=Function(V)
u2=Function(V)
u1.interpolate(0*x*y)
u2.interpolate(0*x*y)

"""

Homogeneous dirichlet boundary conditions throughout mean that faces 1,2,3,4 must
have zero on their own boundary conditions

"""

class MyBC(DirichletBC):
    def __init__(self, V, value, markers):
        # Call superclass init
        # We provide a dummy subdomain id.
        super(MyBC, self).__init__(V, value, 0)
        # Override the "nodes" property which says where the boundary
        # condition is to be applied.
        self.nodes = np.unique(np.where(markers.dat.data_ro_with_halos == 0)[0])

x=SpatialCoordinate(mesh)
bc1=DirichletBC(V,0,1)
bc2=DirichletBC(V,0,4)
bc3=DirichletBC(V,0,5)
bc4=DirichletBC(V,0,6)
bc5=DirichletBC(V,0,7)
bc6=DirichletBC(V,0,8)

bc=DirichletBC(V,1,omega1_face_id)
q=Function(V)
q=u2
bc.apply(q)
bc_dOmega1_n_Omega2=bc

bc=DirichletBC(V,1,omega2_face_id)
q=Function(V)
q=u2
bc.apply(q)

bc_dOmega2_n_Omega1=bc

nSchwarzIter=2

p_omega_1 = LinearVariationalProblem(a1,L1,u1,bcs=[bc1,bc2,bc3,bc4,bc5,bc6,bc_dOmega1_n_Omega2])
p_omega_2 = LinearVariationalProblem(a2,L2,u2,bcs=[bc1,bc2,bc3,bc4,bc5,bc6,bc_dOmega2_n_Omega1])
solver1 = LinearVariationalSolver(p_omega_1)
solver2 = LinearVariationalSolver(p_omega_2)


for iteration in range(nSchwarzIter):
    solver1.solve()
    solver2.solve()
