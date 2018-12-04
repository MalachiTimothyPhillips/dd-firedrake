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
mesh = Mesh("TwoDomain.msh",dim=2)
V=FunctionSpace(mesh, "CG", 1) # piecewise linear elements

u=TrialFunction(V)
v=TestFunction(V)
f=Function(V)
x=SpatialCoordinate(mesh)
y=x[1]
x=x[0]
f.interpolate((8*pi*pi)*sin(2*pi*x)*sin(2*pi*y))
a = (dot(grad(v), grad(u))) * dx
L = f * v * dx
u=Function(V)

omega1_id=1
omega2_id=2

V_DG0_Omega1 = FunctionSpace(mesh, "DG", 0)
V_DG0_Omega2 = FunctionSpace(mesh, "DG", 0)

# Heaviside step function in fluid
I_W = Function( V_DG0_Omega1 )
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(omega1_id), {'f': (I_W, WRITE)} )
I_cg_W = Function(V_W)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',
                 dx, {'A' : (I_cg_W, RW), 'B': (I_W, READ)} )

# Heaviside step function in solid
I_B = Function( V_DG0_Omega2 )
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.;', dx(omega2_id), {'f': (I_B, WRITE)} )
I_cg_B = Function(V_B)
par_loop( ' for (int i=0; i<A.dofs; i++) for(int j=0; j<2; j++) A[i][j] = fmax(A[i][j], B[0][0]);',
                 dx, {'A' : (I_cg_B, RW), 'B': (I_B, READ)} )

class MyBC(DirichletBC):
    def __init__(self, V, value, markers):
        # Call superclass init
        # We provide a dummy subdomain id.
        super(MyBC, self).__init__(V, value, 0)
        # Override the "nodes" property which says where the boundary
        # condition is to be applied.
        self.nodes = np.unique(np.where(markers.dat.data_ro_with_halos == 0)[0])

def surface_BC():
    # This will set nodes on the top boundary to 1.
    bc = DirichletBC( V 1, top_id )
    # We will use this function to determine the new BC nodes (all those
    # that aren't on the boundary)
    f = Function( V, dtype=np.int32 )
    # f is now 0 everywhere, except on the boundary
    bc.apply(f)
    # Now I can use MyBC to create a "boundary condition" to zero out all
    # the nodes that are *not* on the top boundary:
    return MyBC( V, 0, f )

# same as above, but in the mixed space
#def surface_BC_mixed():
#    bc_mixed = DirichletBC( mixed_V.sub(0), 1, top_id )
#    f_mixed = Function( mixed_V.sub(0), dtype=np.int32 )
#    bc_mixed.apply(f_mixed)
#    return MyBC( mixed_V.sub(0), 0, f_mixed )

BC_exclude_beyond_surface = surface_BC()
BC_exclude_beyond_surface_mixed = surface_BC_mixed()
BC_exclude_beyond_solid = MyBC( V_B, 0, I_cg_B )
#bc1=DirichletBC(V,0,1)
#bc2=DirichletBC(V,0,4)
#bc3=DirichletBC(V,0,5)
#bc4=DirichletBC(V,0,6)
#bc5=DirichletBC(V,0,7)
#bc6=DirichletBC(V,0,8)


#solve(a == L, u, bcs=[bc1,bc2,bc3,bc4,bc5,bc6])
#solve(a == L, u, bcs=[bc1,bc2])
f.interpolate(sin(2*pi*x)*sin(2*pi*y))
print("Error in solution of: {}".format(sqrt(assemble(dot(u-f,u-f)*dx))))
