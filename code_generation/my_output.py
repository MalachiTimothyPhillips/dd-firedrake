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

dirichletBCs={1:0,2:0,10:0,11:0,15:0,16:0,8:0,5:0}

domains={"Omega1":1,"Omega2":2,"Omega3":3,"Omega4":4}

interfaces={"dOmega1nOmega2":4,"dOmega1nOmega3":3,"dOmega2nOmega1":6,"dOmega2nOmega4":7,"dOmega3nOmega1":9,"dOmega3nOmega4":12,"dOmega4nOmega2":13,"dOmega4nOmega3":14}

Omega1=1
dOmega1nOmega2=4
dOmega1nOmega3=3
Omega2=2
dOmega2nOmega1=6
dOmega2nOmega4=7
Omega3=3
dOmega3nOmega1=9
dOmega3nOmega4=12
Omega4=4
dOmega4nOmega2=13
dOmega4nOmega3=14

class MyBC(DirichletBC):
    def __init__(self, V, value, markers):
        super(MyBC, self).__init__(V, value, 0)
        self.nodes = np.unique(np.where(markers.dat.data_ro_with_halos == 0)[0])
    
hOmega1=FunctionSpace(mesh,'DG', 0)
I_Omega1 = Function(hOmega1)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega1), {'f': (I_Omega1, WRITE)} )
I_cg_Omega1 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega1, RW), 'B': (I_Omega1, READ)} )
hOmega2=FunctionSpace(mesh,'DG', 0)
I_Omega2 = Function(hOmega2)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega2), {'f': (I_Omega2, WRITE)} )
I_cg_Omega2 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega2, RW), 'B': (I_Omega2, READ)} )
hOmega3=FunctionSpace(mesh,'DG', 0)
I_Omega3 = Function(hOmega3)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega3), {'f': (I_Omega3, WRITE)} )
I_cg_Omega3 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega3, RW), 'B': (I_Omega3, READ)} )
hOmega4=FunctionSpace(mesh,'DG', 0)
I_Omega4 = Function(hOmega4)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega4), {'f': (I_Omega4, WRITE)} )
I_cg_Omega4 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega4, RW), 'B': (I_Omega4, READ)} )

bcs=[]
for surface, value in dirichletBCs.items():
    bcs.append(DirichletBC(V,value,surface))
    
BC_Omega1_only=MyBC(V,0,I_cg_Omega1)
BC_Omega2_only=MyBC(V,0,I_cg_Omega2)
BC_Omega3_only=MyBC(V,0,I_cg_Omega3)
BC_Omega4_only=MyBC(V,0,I_cg_Omega4)
coord=SpatialCoordinate(mesh)
uOmega1=Function(V)
uOmega1.interpolate(1*coord[0]*coord[1])
uOmega1_old=Function(V)
uOmega1_old.interpolate(1*coord[0]*coord[1])
uOmega2=Function(V)
uOmega2.interpolate(1*coord[0]*coord[1])
uOmega2_old=Function(V)
uOmega2_old.interpolate(1*coord[0]*coord[1])
uOmega3=Function(V)
uOmega3.interpolate(1*coord[0]*coord[1])
uOmega3_old=Function(V)
uOmega3_old.interpolate(1*coord[0]*coord[1])
uOmega4=Function(V)
uOmega4.interpolate(1*coord[0]*coord[1])
uOmega4_old=Function(V)
uOmega4_old.interpolate(1*coord[0]*coord[1])
aOmega1 = (dot(grad(u), grad(v))) * dx(Omega1)
LOmega1 = f * v * dx(Omega1)
aOmega2 = (dot(grad(u), grad(v))) * dx(Omega2)
LOmega2 = f * v * dx(Omega2)
aOmega3 = (dot(grad(u), grad(v))) * dx(Omega3)
LOmega3 = f * v * dx(Omega3)
aOmega4 = (dot(grad(u), grad(v))) * dx(Omega4)
LOmega4 = f * v * dx(Omega4)
bcOmega1=bcs.copy()
e1=uOmega1_old+uOmega2_old
bcOmega1.append(DirichletBC(V,e1,dOmega1nOmega2))
e2=uOmega1_old+uOmega3_old
bcOmega1.append(DirichletBC(V,e2,dOmega1nOmega3))
bcOmega1.append(BC_Omega1_only)
bcOmega2=bcs.copy()
e3=uOmega2_old+uOmega1_old
bcOmega2.append(DirichletBC(V,e3,dOmega2nOmega1))
e4=uOmega2_old+uOmega4_old
bcOmega2.append(DirichletBC(V,e4,dOmega2nOmega4))
bcOmega2.append(BC_Omega2_only)
bcOmega3=bcs.copy()
e5=uOmega3_old+uOmega1_old
bcOmega3.append(DirichletBC(V,e5,dOmega3nOmega1))
e6=uOmega3_old+uOmega4_old
bcOmega3.append(DirichletBC(V,e6,dOmega3nOmega4))
bcOmega3.append(BC_Omega3_only)
bcOmega4=bcs.copy()
e7=uOmega4_old+uOmega2_old
bcOmega4.append(DirichletBC(V,e7,dOmega4nOmega2))
e8=uOmega4_old+uOmega3_old
bcOmega4.append(DirichletBC(V,e8,dOmega4nOmega3))
bcOmega4.append(BC_Omega4_only)
nSchwarz=10
for iteration in range(nSchwarz):
	uOmega1_old=uOmega1
	uOmega2_old=uOmega2
	uOmega3_old=uOmega3
	uOmega4_old=uOmega4
	solve(aOmega1==LOmega1,uOmega1,bcs=bcOmega1,solver_parameters=params)
	solve(aOmega2==LOmega2,uOmega2,bcs=bcOmega2,solver_parameters=params)
	solve(aOmega3==LOmega3,uOmega3,bcs=bcOmega3,solver_parameters=params)
	solve(aOmega4==LOmega4,uOmega4,bcs=bcOmega4,solver_parameters=params)
# does this line exist?
# perhaps some other lines after the fact?
