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

## TODO: This part generated from list
# Boundary identities
O1=1
O2=2
O3=3
O4=4

# Interface identities
dO1nO2=4
dO1nO3=3
dO2nO1=6
dO2nO4=7
dO3nO1=9
dO3nO4=12
dO4nO2=13
dO4nO3=14


u=TrialFunction(V)
v=TestFunction(V)
f=Function(V)
x,y=SpatialCoordinate(mesh)
f.interpolate((8*pi*pi)*sin(2*pi*x)*sin(2*pi*y))

## TODO: This part also generated from list
hO1 = FunctionSpace(mesh, "DG", 0)
hO2 = FunctionSpace(mesh, "DG", 0)
hO3 = FunctionSpace(mesh, "DG", 0)
hO4 = FunctionSpace(mesh, "DG", 0)



I_O1 = Function( hO1 )
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(O1), {'f': (I_O1, WRITE)} )
I_cg_O1 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',
         dx, {'A' : (I_cg_O1, RW), 'B': (I_O1, READ)} )

I_O2 = Function( hO2 )
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.;', dx(O2), {'f': (I_O2, WRITE)} )
I_cg_O2 = Function(V)
par_loop( ' for (int i=0; i<A.dofs; i++) for(int j=0; j<2; j++) A[i][j] = fmax(A[i][j], B[0][0]);',
         dx, {'A' : (I_cg_O2, RW), 'B': (I_O2, READ)} )

I_O3 = Function( hO3 )
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.;', dx(O3), {'f': (I_O3, WRITE)} )
I_cg_O3 = Function(V)
par_loop( ' for (int i=0; i<A.dofs; i++) for(int j=0; j<2; j++) A[i][j] = fmax(A[i][j], B[0][0]);',
         dx, {'A' : (I_cg_O3, RW), 'B': (I_O3, READ)} )
I_O4 = Function( hO4 )
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.;', dx(O4), {'f': (I_O4, WRITE)} )
I_cg_O4 = Function(V)
par_loop( ' for (int i=0; i<A.dofs; i++) for(int j=0; j<2; j++) A[i][j] = fmax(A[i][j], B[0][0]);',
         dx, {'A' : (I_cg_O4, RW), 'B': (I_O4, READ)} )

# TODO: This part can always be statically pasted in
class MyBC(DirichletBC):
    def __init__(self, V, value, markers):
        # Call superclass init
        # We provide a dummy subdomain id.
        super(MyBC, self).__init__(V, value, 0)
        # Override the "nodes" property which says where the boundary
        # condition is to be applied.
        self.nodes = np.unique(np.where(markers.dat.data_ro_with_halos == 0)[0])

# TODO: generate from list of domains
# magical B.C. generator to allow us not to have singular system
BC_O1_only = MyBC(V,0,I_cg_O1)
BC_O2_only = MyBC(V,0,I_cg_O2)
BC_O3_only = MyBC(V,0,I_cg_O3)
BC_O4_only = MyBC(V,0,I_cg_O4)


#TODO: generate this from the description of the form itself?

a = (dot(grad(u), grad(v))) * dx
L = f * v * dx

a1 = (dot(grad(u), grad(v))) * dx(O1)
L1 = f * v * dx(O1)
a2 = (dot(grad(v), grad(u))) * dx(O2)
L2 = f * v * dx(O2)
a3 = (dot(grad(v), grad(u))) * dx(O3)
L3 = f * v * dx(O3)
a4 = (dot(grad(v), grad(u))) * dx(O4)
L4 = f * v * dx(O4)

# TODO: generate these, 2 for each domain
u1=Function(V)
u2=Function(V)
u3=Function(V)
u4=Function(V)
u1_old=Function(V)
u2_old=Function(V)
u3_old=Function(V)
u4_old=Function(V)
u1.interpolate(0*x*y)
u2.interpolate(0*x*y)
u3.interpolate(0*x*y)
u4.interpolate(0*x*y)
u1_old.interpolate(0*x*y)
u2_old.interpolate(0*x*y)
u3_old.interpolate(0*x*y)
u4_old.interpolate(0*x*y)


# TODO: generate from list of exterior BCs
# zero B.C. on all the exterior lines
exterior_line_ids=[1,2,10,11,15,16,8,5]
bcs=[]
for surface in exterior_line_ids:
    bcs.append(DirichletBC(V,0.0,surface))

e1=u2_old+u3_old
e2=u1_old+u4_old
e3=u1_old+u4_old
e4=u2_old+u3_old

bc_dO1nO2=DirichletBC(V,e1,dO1nO2)
bc_dO1nO3=DirichletBC(V,e1,dO1nO3)
bc1=bcs.copy()
bc1.append(bc_dO1nO2)
bc1.append(bc_dO1nO3)
bc1.append(BC_O1_only)

bc_dO2nO1=DirichletBC(V,e2,dO2nO1)
bc_dO2nO4=DirichletBC(V,e2,dO2nO4)
bc2=bcs.copy()
bc2.append(bc_dO2nO1)
bc2.append(bc_dO2nO4)
bc2.append(BC_O2_only)

bc_dO3nO1=DirichletBC(V,e3,dO3nO1)
bc_dO3nO4=DirichletBC(V,e3,dO3nO4)
bc3=bcs.copy()
bc3.append(bc_dO3nO1)
bc3.append(bc_dO3nO4)
bc3.append(BC_O3_only)

bc_dO4nO2=DirichletBC(V,e4,dO4nO2)
bc_dO4nO3=DirichletBC(V,e4,dO4nO3)
bc4=bcs.copy()
bc4.append(bc_dO4nO2)
bc4.append(bc_dO4nO3)
bc4.append(BC_O4_only)

nSchwarzIter=10

params={'ksp_type':'preonly','pc_type':'lu'}

# TODO: will need to generate these, too
for iteration in range(nSchwarzIter):
    u1_old=u1
    u2_old=u2
    u3_old=u3
    u4_old=u4
    solve(a1 == L1,u1,bcs=bc1, solver_parameters=params)
    solve(a2 == L2,u2,bcs=bc2, solver_parameters=params)
    solve(a3 == L3,u3,bcs=bc3, solver_parameters=params)
    solve(a4 == L4,u4,bcs=bc4, solver_parameters=params)

e1=assemble(dot(u1-f,u1-f)*dx(O1))
e2=assemble(dot(u2-f,u2-f)*dx(O2))
e3=assemble(dot(u3-f,u3-f)*dx(O3))
e4=assemble(dot(u4-f,u4-f)*dx(O4))

new_solution_error=sqrt(e1**2+e2**2+e3**2+e4**2)
print(new_solution_error)
