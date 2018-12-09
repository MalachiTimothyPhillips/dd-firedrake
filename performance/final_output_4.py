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

ref_mesh=Mesh("../meshing/4x4Comparison.msh")
V_ref=FunctionSpace(ref_mesh, "CG", 1)
u=TrialFunction(V_ref)
v=TestFunction(V_ref)
f_ref=Function(V_ref)
x,y=SpatialCoordinate(ref_mesh)
f_ref.interpolate((8*pi*pi)*sin(2*pi*x)*sin(2*pi*y))
f=Function(V)
x,y=SpatialCoordinate(mesh)
f.interpolate((8*pi*pi)*sin(2*pi*x)*sin(2*pi*y))
params={'ksp_type':'preonly','pc_type':'lu'}

a=dot(grad(u),grad(v))*dx
L=f_ref*v*dx
u_entire=Function(V_ref)

bc1=DirichletBC(V_ref,0.0,1)
bc2=DirichletBC(V_ref,0.0,2)
bc3=DirichletBC(V_ref,0.0,3)
bc4=DirichletBC(V_ref,0.0,4)
start_time1=time.time()
solve(a==L,u_entire,bcs=[bc1,bc2,bc3,bc4],solver_parameters=params)
end_time1=time.time()

# for our purposes, we need to also generate timing data for the regular solve
u=TrialFunction(V)
v=TestFunction(V)

start_time2=time.time()
dirichletBCs={1:0,2:0,16:0,51:0,5:0,18:0,32:0,55:0,9:0,34:0,48:0,59:0,13:0,50:0,64:0,63:0,}

domains={"Omega1":1,"Omega2":2,"Omega3":3,"Omega4":4,"Omega5":5,"Omega6":6,"Omega7":7,"Omega8":8,"Omega9":9,"Omega10":10,"Omega11":11,"Omega12":12,"Omega13":13,"Omega14":14,"Omega15":15,"Omega16":16,}

interfaces={"dOmega6nOmega7":24,"dOmega6nOmega5":22,"dOmega6nOmega10":23,"dOmega6nOmega2":21,"dOmega10nOmega11":40,"dOmega10nOmega9":38,"dOmega10nOmega14":39,"dOmega10nOmega6":37,"dOmega7nOmega8":28,"dOmega7nOmega6":26,"dOmega7nOmega11":27,"dOmega7nOmega3":25,"dOmega11nOmega12":44,"dOmega11nOmega10":42,"dOmega11nOmega15":43,"dOmega11nOmega7":41,"dOmega5nOmega6":20,"dOmega5nOmega9":19,"dOmega5nOmega1":17,"dOmega9nOmega10":36,"dOmega9nOmega13":35,"dOmega9nOmega5":33,"dOmega8nOmega7":30,"dOmega8nOmega12":31,"dOmega8nOmega4":29,"dOmega12nOmega11":46,"dOmega12nOmega16":47,"dOmega12nOmega8":45,"dOmega2nOmega3":8,"dOmega2nOmega1":6,"dOmega2nOmega6":7,"dOmega3nOmega4":12,"dOmega3nOmega2":10,"dOmega3nOmega7":11,"dOmega14nOmega15":56,"dOmega14nOmega13":54,"dOmega14nOmega10":53,"dOmega15nOmega16":60,"dOmega15nOmega14":58,"dOmega15nOmega11":57,"dOmega1nOmega2":4,"dOmega1nOmega5":3,"dOmega13nOmega14":52,"dOmega13nOmega9":49,"dOmega4nOmega3":14,"dOmega4nOmega8":15,"dOmega16nOmega15":62,"dOmega16nOmega12":61,}

Omega1=1
dOmega1nOmega2=4
dOmega1nOmega5=3
Omega2=2
dOmega2nOmega1=6
dOmega2nOmega3=8
dOmega2nOmega6=7
Omega3=3
dOmega3nOmega2=10
dOmega3nOmega4=12
dOmega3nOmega7=11
Omega4=4
dOmega4nOmega3=14
dOmega4nOmega8=15
Omega5=5
dOmega5nOmega1=17
dOmega5nOmega6=20
dOmega5nOmega9=19
Omega6=6
dOmega6nOmega2=21
dOmega6nOmega5=22
dOmega6nOmega7=24
dOmega6nOmega10=23
Omega7=7
dOmega7nOmega3=25
dOmega7nOmega6=26
dOmega7nOmega8=28
dOmega7nOmega11=27
Omega8=8
dOmega8nOmega4=29
dOmega8nOmega7=30
dOmega8nOmega12=31
Omega9=9
dOmega9nOmega5=33
dOmega9nOmega10=36
dOmega9nOmega13=35
Omega10=10
dOmega10nOmega6=37
dOmega10nOmega9=38
dOmega10nOmega11=40
dOmega10nOmega14=39
Omega11=11
dOmega11nOmega7=41
dOmega11nOmega10=42
dOmega11nOmega12=44
dOmega11nOmega15=43
Omega12=12
dOmega12nOmega8=45
dOmega12nOmega11=46
dOmega12nOmega16=47
Omega13=13
dOmega13nOmega9=49
dOmega13nOmega14=52
Omega14=14
dOmega14nOmega10=53
dOmega14nOmega13=54
dOmega14nOmega15=56
Omega15=15
dOmega15nOmega11=57
dOmega15nOmega14=58
dOmega15nOmega16=60
Omega16=16
dOmega16nOmega12=61
dOmega16nOmega15=62

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
hOmega5=FunctionSpace(mesh,'DG', 0)
I_Omega5 = Function(hOmega5)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega5), {'f': (I_Omega5, WRITE)} )
I_cg_Omega5 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega5, RW), 'B': (I_Omega5, READ)} )
hOmega6=FunctionSpace(mesh,'DG', 0)
I_Omega6 = Function(hOmega6)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega6), {'f': (I_Omega6, WRITE)} )
I_cg_Omega6 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega6, RW), 'B': (I_Omega6, READ)} )
hOmega7=FunctionSpace(mesh,'DG', 0)
I_Omega7 = Function(hOmega7)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega7), {'f': (I_Omega7, WRITE)} )
I_cg_Omega7 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega7, RW), 'B': (I_Omega7, READ)} )
hOmega8=FunctionSpace(mesh,'DG', 0)
I_Omega8 = Function(hOmega8)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega8), {'f': (I_Omega8, WRITE)} )
I_cg_Omega8 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega8, RW), 'B': (I_Omega8, READ)} )
hOmega9=FunctionSpace(mesh,'DG', 0)
I_Omega9 = Function(hOmega9)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega9), {'f': (I_Omega9, WRITE)} )
I_cg_Omega9 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega9, RW), 'B': (I_Omega9, READ)} )
hOmega10=FunctionSpace(mesh,'DG', 0)
I_Omega10 = Function(hOmega10)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega10), {'f': (I_Omega10, WRITE)} )
I_cg_Omega10 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega10, RW), 'B': (I_Omega10, READ)} )
hOmega11=FunctionSpace(mesh,'DG', 0)
I_Omega11 = Function(hOmega11)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega11), {'f': (I_Omega11, WRITE)} )
I_cg_Omega11 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega11, RW), 'B': (I_Omega11, READ)} )
hOmega12=FunctionSpace(mesh,'DG', 0)
I_Omega12 = Function(hOmega12)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega12), {'f': (I_Omega12, WRITE)} )
I_cg_Omega12 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega12, RW), 'B': (I_Omega12, READ)} )
hOmega13=FunctionSpace(mesh,'DG', 0)
I_Omega13 = Function(hOmega13)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega13), {'f': (I_Omega13, WRITE)} )
I_cg_Omega13 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega13, RW), 'B': (I_Omega13, READ)} )
hOmega14=FunctionSpace(mesh,'DG', 0)
I_Omega14 = Function(hOmega14)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega14), {'f': (I_Omega14, WRITE)} )
I_cg_Omega14 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega14, RW), 'B': (I_Omega14, READ)} )
hOmega15=FunctionSpace(mesh,'DG', 0)
I_Omega15 = Function(hOmega15)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega15), {'f': (I_Omega15, WRITE)} )
I_cg_Omega15 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega15, RW), 'B': (I_Omega15, READ)} )
hOmega16=FunctionSpace(mesh,'DG', 0)
I_Omega16 = Function(hOmega16)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega16), {'f': (I_Omega16, WRITE)} )
I_cg_Omega16 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega16, RW), 'B': (I_Omega16, READ)} )

bcs=[]
for surface, value in dirichletBCs.items():
    bcs.append(DirichletBC(V,value,surface))
    
BC_Omega1_only=MyBC(V,0,I_cg_Omega1)
BC_Omega2_only=MyBC(V,0,I_cg_Omega2)
BC_Omega3_only=MyBC(V,0,I_cg_Omega3)
BC_Omega4_only=MyBC(V,0,I_cg_Omega4)
BC_Omega5_only=MyBC(V,0,I_cg_Omega5)
BC_Omega6_only=MyBC(V,0,I_cg_Omega6)
BC_Omega7_only=MyBC(V,0,I_cg_Omega7)
BC_Omega8_only=MyBC(V,0,I_cg_Omega8)
BC_Omega9_only=MyBC(V,0,I_cg_Omega9)
BC_Omega10_only=MyBC(V,0,I_cg_Omega10)
BC_Omega11_only=MyBC(V,0,I_cg_Omega11)
BC_Omega12_only=MyBC(V,0,I_cg_Omega12)
BC_Omega13_only=MyBC(V,0,I_cg_Omega13)
BC_Omega14_only=MyBC(V,0,I_cg_Omega14)
BC_Omega15_only=MyBC(V,0,I_cg_Omega15)
BC_Omega16_only=MyBC(V,0,I_cg_Omega16)
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
uOmega5=Function(V)
uOmega5.interpolate(1*coord[0]*coord[1])
uOmega5_old=Function(V)
uOmega5_old.interpolate(1*coord[0]*coord[1])
uOmega6=Function(V)
uOmega6.interpolate(1*coord[0]*coord[1])
uOmega6_old=Function(V)
uOmega6_old.interpolate(1*coord[0]*coord[1])
uOmega7=Function(V)
uOmega7.interpolate(1*coord[0]*coord[1])
uOmega7_old=Function(V)
uOmega7_old.interpolate(1*coord[0]*coord[1])
uOmega8=Function(V)
uOmega8.interpolate(1*coord[0]*coord[1])
uOmega8_old=Function(V)
uOmega8_old.interpolate(1*coord[0]*coord[1])
uOmega9=Function(V)
uOmega9.interpolate(1*coord[0]*coord[1])
uOmega9_old=Function(V)
uOmega9_old.interpolate(1*coord[0]*coord[1])
uOmega10=Function(V)
uOmega10.interpolate(1*coord[0]*coord[1])
uOmega10_old=Function(V)
uOmega10_old.interpolate(1*coord[0]*coord[1])
uOmega11=Function(V)
uOmega11.interpolate(1*coord[0]*coord[1])
uOmega11_old=Function(V)
uOmega11_old.interpolate(1*coord[0]*coord[1])
uOmega12=Function(V)
uOmega12.interpolate(1*coord[0]*coord[1])
uOmega12_old=Function(V)
uOmega12_old.interpolate(1*coord[0]*coord[1])
uOmega13=Function(V)
uOmega13.interpolate(1*coord[0]*coord[1])
uOmega13_old=Function(V)
uOmega13_old.interpolate(1*coord[0]*coord[1])
uOmega14=Function(V)
uOmega14.interpolate(1*coord[0]*coord[1])
uOmega14_old=Function(V)
uOmega14_old.interpolate(1*coord[0]*coord[1])
uOmega15=Function(V)
uOmega15.interpolate(1*coord[0]*coord[1])
uOmega15_old=Function(V)
uOmega15_old.interpolate(1*coord[0]*coord[1])
uOmega16=Function(V)
uOmega16.interpolate(1*coord[0]*coord[1])
uOmega16_old=Function(V)
uOmega16_old.interpolate(1*coord[0]*coord[1])
aOmega1 = (dot(grad(u), grad(v))) * dx(Omega1)
LOmega1 = f * v * dx(Omega1)
aOmega2 = (dot(grad(u), grad(v))) * dx(Omega2)
LOmega2 = f * v * dx(Omega2)
aOmega3 = (dot(grad(u), grad(v))) * dx(Omega3)
LOmega3 = f * v * dx(Omega3)
aOmega4 = (dot(grad(u), grad(v))) * dx(Omega4)
LOmega4 = f * v * dx(Omega4)
aOmega5 = (dot(grad(u), grad(v))) * dx(Omega5)
LOmega5 = f * v * dx(Omega5)
aOmega6 = (dot(grad(u), grad(v))) * dx(Omega6)
LOmega6 = f * v * dx(Omega6)
aOmega7 = (dot(grad(u), grad(v))) * dx(Omega7)
LOmega7 = f * v * dx(Omega7)
aOmega8 = (dot(grad(u), grad(v))) * dx(Omega8)
LOmega8 = f * v * dx(Omega8)
aOmega9 = (dot(grad(u), grad(v))) * dx(Omega9)
LOmega9 = f * v * dx(Omega9)
aOmega10 = (dot(grad(u), grad(v))) * dx(Omega10)
LOmega10 = f * v * dx(Omega10)
aOmega11 = (dot(grad(u), grad(v))) * dx(Omega11)
LOmega11 = f * v * dx(Omega11)
aOmega12 = (dot(grad(u), grad(v))) * dx(Omega12)
LOmega12 = f * v * dx(Omega12)
aOmega13 = (dot(grad(u), grad(v))) * dx(Omega13)
LOmega13 = f * v * dx(Omega13)
aOmega14 = (dot(grad(u), grad(v))) * dx(Omega14)
LOmega14 = f * v * dx(Omega14)
aOmega15 = (dot(grad(u), grad(v))) * dx(Omega15)
LOmega15 = f * v * dx(Omega15)
aOmega16 = (dot(grad(u), grad(v))) * dx(Omega16)
LOmega16 = f * v * dx(Omega16)
bcOmega1=bcs.copy()
e1=uOmega2_old
bcOmega1.append(DirichletBC(V,e1,dOmega1nOmega2))
e2=uOmega5_old
bcOmega1.append(DirichletBC(V,e2,dOmega1nOmega5))
bcOmega1.append(BC_Omega1_only)
bcOmega2=bcs.copy()
e3=uOmega1_old
bcOmega2.append(DirichletBC(V,e3,dOmega2nOmega1))
e4=uOmega3_old
bcOmega2.append(DirichletBC(V,e4,dOmega2nOmega3))
e5=uOmega6_old
bcOmega2.append(DirichletBC(V,e5,dOmega2nOmega6))
bcOmega2.append(BC_Omega2_only)
bcOmega3=bcs.copy()
e6=uOmega2_old
bcOmega3.append(DirichletBC(V,e6,dOmega3nOmega2))
e7=uOmega4_old
bcOmega3.append(DirichletBC(V,e7,dOmega3nOmega4))
e8=uOmega7_old
bcOmega3.append(DirichletBC(V,e8,dOmega3nOmega7))
bcOmega3.append(BC_Omega3_only)
bcOmega4=bcs.copy()
e9=uOmega3_old
bcOmega4.append(DirichletBC(V,e9,dOmega4nOmega3))
e10=uOmega8_old
bcOmega4.append(DirichletBC(V,e10,dOmega4nOmega8))
bcOmega4.append(BC_Omega4_only)
bcOmega5=bcs.copy()
e11=uOmega1_old
bcOmega5.append(DirichletBC(V,e11,dOmega5nOmega1))
e12=uOmega6_old
bcOmega5.append(DirichletBC(V,e12,dOmega5nOmega6))
e13=uOmega9_old
bcOmega5.append(DirichletBC(V,e13,dOmega5nOmega9))
bcOmega5.append(BC_Omega5_only)
bcOmega6=bcs.copy()
e14=uOmega2_old
bcOmega6.append(DirichletBC(V,e14,dOmega6nOmega2))
e15=uOmega5_old
bcOmega6.append(DirichletBC(V,e15,dOmega6nOmega5))
e16=uOmega7_old
bcOmega6.append(DirichletBC(V,e16,dOmega6nOmega7))
e17=uOmega10_old
bcOmega6.append(DirichletBC(V,e17,dOmega6nOmega10))
bcOmega6.append(BC_Omega6_only)
bcOmega7=bcs.copy()
e18=uOmega3_old
bcOmega7.append(DirichletBC(V,e18,dOmega7nOmega3))
e19=uOmega6_old
bcOmega7.append(DirichletBC(V,e19,dOmega7nOmega6))
e20=uOmega8_old
bcOmega7.append(DirichletBC(V,e20,dOmega7nOmega8))
e21=uOmega11_old
bcOmega7.append(DirichletBC(V,e21,dOmega7nOmega11))
bcOmega7.append(BC_Omega7_only)
bcOmega8=bcs.copy()
e22=uOmega4_old
bcOmega8.append(DirichletBC(V,e22,dOmega8nOmega4))
e23=uOmega7_old
bcOmega8.append(DirichletBC(V,e23,dOmega8nOmega7))
e24=uOmega12_old
bcOmega8.append(DirichletBC(V,e24,dOmega8nOmega12))
bcOmega8.append(BC_Omega8_only)
bcOmega9=bcs.copy()
e25=uOmega5_old
bcOmega9.append(DirichletBC(V,e25,dOmega9nOmega5))
e26=uOmega10_old
bcOmega9.append(DirichletBC(V,e26,dOmega9nOmega10))
e27=uOmega13_old
bcOmega9.append(DirichletBC(V,e27,dOmega9nOmega13))
bcOmega9.append(BC_Omega9_only)
bcOmega10=bcs.copy()
e28=uOmega6_old
bcOmega10.append(DirichletBC(V,e28,dOmega10nOmega6))
e29=uOmega9_old
bcOmega10.append(DirichletBC(V,e29,dOmega10nOmega9))
e30=uOmega11_old
bcOmega10.append(DirichletBC(V,e30,dOmega10nOmega11))
e31=uOmega14_old
bcOmega10.append(DirichletBC(V,e31,dOmega10nOmega14))
bcOmega10.append(BC_Omega10_only)
bcOmega11=bcs.copy()
e32=uOmega7_old
bcOmega11.append(DirichletBC(V,e32,dOmega11nOmega7))
e33=uOmega10_old
bcOmega11.append(DirichletBC(V,e33,dOmega11nOmega10))
e34=uOmega12_old
bcOmega11.append(DirichletBC(V,e34,dOmega11nOmega12))
e35=uOmega15_old
bcOmega11.append(DirichletBC(V,e35,dOmega11nOmega15))
bcOmega11.append(BC_Omega11_only)
bcOmega12=bcs.copy()
e36=uOmega8_old
bcOmega12.append(DirichletBC(V,e36,dOmega12nOmega8))
e37=uOmega11_old
bcOmega12.append(DirichletBC(V,e37,dOmega12nOmega11))
e38=uOmega16_old
bcOmega12.append(DirichletBC(V,e38,dOmega12nOmega16))
bcOmega12.append(BC_Omega12_only)
bcOmega13=bcs.copy()
e39=uOmega9_old
bcOmega13.append(DirichletBC(V,e39,dOmega13nOmega9))
e40=uOmega14_old
bcOmega13.append(DirichletBC(V,e40,dOmega13nOmega14))
bcOmega13.append(BC_Omega13_only)
bcOmega14=bcs.copy()
e41=uOmega10_old
bcOmega14.append(DirichletBC(V,e41,dOmega14nOmega10))
e42=uOmega13_old
bcOmega14.append(DirichletBC(V,e42,dOmega14nOmega13))
e43=uOmega15_old
bcOmega14.append(DirichletBC(V,e43,dOmega14nOmega15))
bcOmega14.append(BC_Omega14_only)
bcOmega15=bcs.copy()
e44=uOmega11_old
bcOmega15.append(DirichletBC(V,e44,dOmega15nOmega11))
e45=uOmega14_old
bcOmega15.append(DirichletBC(V,e45,dOmega15nOmega14))
e46=uOmega16_old
bcOmega15.append(DirichletBC(V,e46,dOmega15nOmega16))
bcOmega15.append(BC_Omega15_only)
bcOmega16=bcs.copy()
e47=uOmega12_old
bcOmega16.append(DirichletBC(V,e47,dOmega16nOmega12))
e48=uOmega15_old
bcOmega16.append(DirichletBC(V,e48,dOmega16nOmega15))
bcOmega16.append(BC_Omega16_only)
nSchwarz=10
for iteration in range(nSchwarz):
	uOmega1_old=uOmega1
	uOmega2_old=uOmega2
	uOmega3_old=uOmega3
	uOmega4_old=uOmega4
	uOmega5_old=uOmega5
	uOmega6_old=uOmega6
	uOmega7_old=uOmega7
	uOmega8_old=uOmega8
	uOmega9_old=uOmega9
	uOmega10_old=uOmega10
	uOmega11_old=uOmega11
	uOmega12_old=uOmega12
	uOmega13_old=uOmega13
	uOmega14_old=uOmega14
	uOmega15_old=uOmega15
	uOmega16_old=uOmega16
	solve(aOmega1==LOmega1,uOmega1,bcs=bcOmega1,solver_parameters=params)
	solve(aOmega2==LOmega2,uOmega2,bcs=bcOmega2,solver_parameters=params)
	solve(aOmega3==LOmega3,uOmega3,bcs=bcOmega3,solver_parameters=params)
	solve(aOmega4==LOmega4,uOmega4,bcs=bcOmega4,solver_parameters=params)
	solve(aOmega5==LOmega5,uOmega5,bcs=bcOmega5,solver_parameters=params)
	solve(aOmega6==LOmega6,uOmega6,bcs=bcOmega6,solver_parameters=params)
	solve(aOmega7==LOmega7,uOmega7,bcs=bcOmega7,solver_parameters=params)
	solve(aOmega8==LOmega8,uOmega8,bcs=bcOmega8,solver_parameters=params)
	solve(aOmega9==LOmega9,uOmega9,bcs=bcOmega9,solver_parameters=params)
	solve(aOmega10==LOmega10,uOmega10,bcs=bcOmega10,solver_parameters=params)
	solve(aOmega11==LOmega11,uOmega11,bcs=bcOmega11,solver_parameters=params)
	solve(aOmega12==LOmega12,uOmega12,bcs=bcOmega12,solver_parameters=params)
	solve(aOmega13==LOmega13,uOmega13,bcs=bcOmega13,solver_parameters=params)
	solve(aOmega14==LOmega14,uOmega14,bcs=bcOmega14,solver_parameters=params)
	solve(aOmega15==LOmega15,uOmega15,bcs=bcOmega15,solver_parameters=params)
	solve(aOmega16==LOmega16,uOmega16,bcs=bcOmega16,solver_parameters=params)
end_time2=time.time()
print("{},{},{},{}".format(N,N_subdom,end_time1-start_time1,end_time2-start_time2))
# does this line exist?
# perhaps some other lines after the fact?
