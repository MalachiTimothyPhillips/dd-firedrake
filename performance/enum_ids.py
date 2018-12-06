import numpy as np
"""
Small helper function for my particular case. Returns the interface id
numbers for my specific case. For more details, please consult the documentation.
"""
def five_point_stencil(N,i,j):
    ip1j=4*N*j+4*i+4
    im1j=4*N*j+4*i+2
    ijp1=4*N*j+4*i+3
    ijm1=4*N*j+4*i+1
    return [ip1j,im1j,ijp1,ijm1]
class EnumerateBoundaryConditions(object):
    """
    Extend this class for your particular use case
    For more information, please consult the documentation,
    especially under the SquareMeshBoundaryConditions class.
    """
    def __init__(self):
        """
        Since there is no real data members to this class,
        the constructor itself will just be a no-op
        """
        pass
    def enumerateDirichletBC(self):
        """
        Method that generates a python dictionary containing 
        mapping of subdomain id with the value to be applied
        at the boundary
        """
        pass
    def enumerateSubdomainMapping(self):
        """
        Method that generates a python dictionary
        mapping the subdomain name to the subdomain id
        """
        pass
    def enumerateSubdomainBoundaries(self):
        """
        Method that generates a python dictionary
        mapping subdomain boundaries by name of the type
        d<subdomain_i>n<subdomain_j> to the appropriate
        interface id. Please note that the strings
        must be of this particular form in order to
        generate the 
        """
        pass

class SquareMeshBoundaryConditions(EnumerateBoundaryConditions):
    def __init__(self,N):
        """
        Pass in the number of domains along a dimension (N=Nx=Ny)
        """
        self.N=N
        pass
    def enumerateDirichletBC(self):
        """
        Provide the dirichlet boundary conditions along
        surface x=0, y=0; x=1, y=0; x=0,y=1; x=1,y=1
        """
        N=self.N
        bcs={}
        bottom_id=1
        lhs_id=2
        rhs_id=4*N
        top_id=4*N*(N-1)+3
        for i in range(N):
            # add 4 along bottom
            bcs[bottom_id]=0 # apply homogeneous dirichlet
            bcs[lhs_id]=0 # apply homogeneous dirichlet
            bcs[rhs_id]=0 # apply homogeneous dirichlet
            bcs[top_id]=0 # apply homogeneous dirichlet
            bottom_id += 4
            lhs_id += 4*N
            rhs_id += 4*N
            top_id += 4
        return bcs
    def enumerateSubdomainMapping(self):
        """
        Create a simple mapping of the form Omega_i=i for i = 1 ... N_subdomains
        """
        N=self.N
        omegas={}
        omegaStr="Omega{}"
        for i in range(N):
            omegas[omegaStr.format(i+1)]=i+1
        return omegas
    def enumerateSubdomainBoundaries(self):
        """
        Create a mapping of the form d<subdomain_i>n<subdomain_j> for all
        the intersecting domains -- note that this corresponds to a
        5-point stencil for a NxN grid
        """
        N=N.self
        intefaces={}
        omegaStr="Omega{}"
        interfaceStr="d{}n{}"
        omegas=np.zeros((N,N)).astype(str)
        myDomain=1
        for i in range(N):
            for j in range(N):
                omegas[i][j]=omegaStr.format(myDomain)
                myDomain+=1
        for i in range(N):
            for j in range(N):
                # what connections are we allowed to make here?
                # at most, we have 4. Notice that the connections
                # are of the form (i,j) connects to (i+1,j),(i-1,j)
                # (i,j-1),(i,j+1) for the connection. In particular,
                # if i-1,i+1,j-1,j+1 are OUTSIDE of the regular domain,
                # then we know not to draw a connection there
                if(i-1 < 0):
                    # we are on the lhs boundary
                    if(j-1 < 0):
                        # we are on the lhs corner, only draw between i+1,j+1 connections
                        # my current domain
                        oij=omegas[i][j]
                        oip1j=omegas[i+1][j]
                        oijp1=omegas[i][j+1]
                        interface_ip1j=interfaceStr.format(oij,oip1j)
                        interfaces[interface_ip1j]=4
                        interface_ijp1=interfaceStr.format(oij,oijp1)
                        interfaces[interface_ijp1]=3 # gauranteed at this point
                        continue
                    if(j+1>N-1):
                        # we are on the top LHS corner
                        oij=omegas[i][j]
                        oip1j=omegas[i+1][j]
                        oijm1=omegas[i][j-1]
                        interface_ip1j=interfaceStr.format(oij,oip1j)
                        interfaces[interface_ip1j]=4*N*(N-1)+3
                        interface_ijm1=interfaceStr.format(oij,oijm1)
                        interfaces[interface_ijm1]=4*N*(N-1)+1
                        continue
                    # if it reaches here, then we are on the LHS but NOT on a corner
                    oij=omegas[i][j]
                    oip1j=omegas[i+1][j]
                    oijm1=omegas[i][j-1]
                    oijp1=omegas[i][j+1]
                    interface_ip1j=interfaceStr.format(oij,oip1j)
                    interfaces[interface_ip1j]=4*N*j+4
                    interface_ijp1=interfaceStr.format(oij,oijp1)
                    interfaces[interface_ijp1]=4*N*j+3
                    interface_ijm1=interfaceStr.format(oij,oijm1)
                    interfaces[interface_ijm1]=4*N*j+1
                    continue
                #TODO: code is not correct here -- need to ensure correctness!
                if(i+1 >N-1):
                    # we are on the rhs boundary
                    if(j-1 < 0):
                        # we are on the rhs bottom corner, only draw between i-1,j+1 connections
                        # my current domain
                        oij=omegas[i][j]
                        oim1j=omegas[i-1][j]
                        oijp1=omegas[i][j+1]
                        interface_im1j=interfaceStr.format(oij,oim1j)
                        interfaces[interface_im1j]=4*(N-1)+2
                        interface_ijp1=interfaceStr.format(oij,oijp1)
                        interfaces[interface_ijp1]=4*(N-1)+3 # gauranteed at this point
                        continue
                    if(j+1>N-1):
                        # we are on the top RHS corner
                        oij=omegas[i][j]
                        oim1j=omegas[i-1][j]
                        oijm1=omegas[i][j-1]
                        interface_im1j=interfaceStr.format(oij,oim1j)
                        interfaces[interface_im1j]=4*N*(N-1)+4*(N-1)+2
                        interface_ijm1=interfaceStr.format(oij,oijm1)
                        interfaces[interface_ijm1]=4*N*(N-1)+4*(N-1)+1
                        continue
                    # if it reaches here, then we are on the RHS but NOT on a corner
                    oij=omegas[i][j]
                    oim1j=omegas[i-1][j]
                    oijm1=omegas[i][j-1]
                    oijp1=omegas[i][j+1]
                    interface_im1j=interfaceStr.format(oij,oim1j)
                    interfaces[interface_im1j]=4*N*j+4*(N-1)+2
                    interface_ijp1=interfaceStr.format(oij,oijp1)
                    interfaces[interface_ijp1]=4*N*j+4*(N-1)+3
                    interface_ijm1=interfaceStr.format(oij,oijm1)
                    interfaces[interface_ijm1]=4*N*j+4*(N-1)+1
                    continue

