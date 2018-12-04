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

V_DG0_O1 = FunctionSpace(mesh, "DG", 0)
V_DG0_O2 = FunctionSpace(mesh, "DG", 0)

# Heaviside step function in omega 1
I_O1 = Function( V_DG0_O1 )
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(omega1_id), {'f': (I_O1, WRITE)} )
I_cg_O1 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',
         dx, {'A' : (I_cg_O1, RW), 'B': (I_O1, READ)} )

# Heaviside step function in omega 2
I_O2 = Function( V_DG0_O2 )
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.;', dx(omega2_id), {'f': (I_O2, WRITE)} )
I_cg_O2 = Function(V)
par_loop( ' for (int i=0; i<A.dofs; i++) for(int j=0; j<2; j++) A[i][j] = fmax(A[i][j], B[0][0]);',
         dx, {'A' : (I_cg_O2, RW), 'B': (I_O2, READ)} )


class MyBC(DirichletBC):
    def __init__(self, V, value, markers):
        # Call superclass init
        # We provide a dummy subdomain id.
        super(MyBC, self).__init__(V, value, 0)
        # Override the "nodes" property which says where the boundary
        # condition is to be applied.
        self.nodes = np.unique(np.where(markers.dat.data_ro_with_halos == 0)[0])

# magical B.C. generator to allow us not to have singular system
BC_include_O2_only = MyBC(V,0,I_cg_O2)
BC_include_O1_only = MyBC(V,0,I_cg_O1)



a1 = (dot(grad(u), grad(v))) * dx(omega1_id)
L1 = f * v * dx(omega1_id)
a2 = (dot(grad(v), grad(u))) * dx(omega2_id)
L2 = f * v * dx(omega2_id)

u1=Function(V)
u2=Function(V)
u1.interpolate(0*x*y)
u2.interpolate(0*x*y)

a = (dot(grad(v), grad(u))) * dx
L = f * v * dx
u=Function(V)

bc1=DirichletBC(V,0,1)
bc2=DirichletBC(V,0,4)
bc3=DirichletBC(V,0,5)
bc4=DirichletBC(V,0,6)
bc5=DirichletBC(V,0,7)
bc6=DirichletBC(V,0,8)
solve(a == L, u, bcs=[bc1,bc2,bc3,bc4,bc5,bc6])
f.interpolate(sin(2*pi*x)*sin(2*pi*y))
original_solution_error=sqrt(assemble(dot(u-f,u-f)*dx))
print(original_solution_error)


bc1=DirichletBC(V,0,1)
bc2=DirichletBC(V,0,4)
bc3=DirichletBC(V,0,5)
bc4=DirichletBC(V,0,6)
bc5=DirichletBC(V,0,7)
bc6=DirichletBC(V,0,8)

e1=u2
e2=u1
bc_dOmega1_n_Omega2=DirichletBC(V,e1,omega1_face_id)
bc_dOmega2_n_Omega1=DirichletBC(V,e2,omega2_face_id)

nSchwarzIter=100

params={'ksp_type':'preonly','pc_type':'lu'}

for iteration in range(nSchwarzIter):
    solve(a1 == L1,u1,bcs=[bc1,bc2,bc3,bc4,bc5,bc6,bc_dOmega1_n_Omega2,BC_include_O1_only], solver_parameters=params)
    solve(a2 == L2,u2,bcs=[bc1,bc2,bc3,bc4,bc5,bc6,bc_dOmega2_n_Omega1,BC_include_O2_only], solver_parameters=params)

e1=assemble(dot(u1-f,u1-f)*dx(omega1_id))
e2=assemble(dot(u2-f,u2-f)*dx(omega2_id))
new_solution_error=sqrt(e1**2+e2**2)
print(new_solution_error)
