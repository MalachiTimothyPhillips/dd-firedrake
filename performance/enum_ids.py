import numpy as np
def five_point_stencil(N,i,j):
    """
    Small helper function for my particular case. Returns the interface id
    numbers for my specific case. For more details, please consult the documentation.
    """
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
        for i in range(N*N):
            omegas[omegaStr.format(i+1)]=i+1
        return omegas
    def enumerateSubdomainBoundaries(self):
        """
        Create a mapping of the form d<subdomain_i>n<subdomain_j> for all
        the intersecting domains -- note that this corresponds to a
        5-point stencil for a NxN grid
        """
        N=self.N
        interfaces={}
        omegaStr="Omega{}"
        interfaceStr="d{}n{}"
        omegas=np.zeros((N,N)).astype(str)
        myDomain=1
        for j in range(N):
            for i in range(N):
                omegas[i][j]=omegaStr.format(myDomain)
                myDomain+=1
        # interior points
        for i in range(1,N-1):
            for j in range(1,N-1):
                ids=five_point_stencil(N,i,j)
                interfaces[interfaceStr.format(omegas[i][j],omegas[i+1][j])]=ids[0]
                interfaces[interfaceStr.format(omegas[i][j],omegas[i-1][j])]=ids[1]
                interfaces[interfaceStr.format(omegas[i][j],omegas[i][j+1])]=ids[2]
                interfaces[interfaceStr.format(omegas[i][j],omegas[i][j-1])]=ids[3]
        # left hand side of the boundary
        i = 0
        for j in range(1,N-1):
            ids=five_point_stencil(N,i,j)
            interfaces[interfaceStr.format(omegas[i][j],omegas[i+1][j])]=ids[0]
            interfaces[interfaceStr.format(omegas[i][j],omegas[i][j+1])]=ids[2]
            interfaces[interfaceStr.format(omegas[i][j],omegas[i][j-1])]=ids[3]
        # right hand side of the boundary
        i = N-1
        for j in range(1,N-1):
            ids=five_point_stencil(N,i,j)
            interfaces[interfaceStr.format(omegas[i][j],omegas[i-1][j])]=ids[1]
            interfaces[interfaceStr.format(omegas[i][j],omegas[i][j+1])]=ids[2]
            interfaces[interfaceStr.format(omegas[i][j],omegas[i][j-1])]=ids[3]
        # bottom of the boundary
        j = 0
        for i in range(1,N-1):
            ids=five_point_stencil(N,i,j)
            interfaces[interfaceStr.format(omegas[i][j],omegas[i+1][j])]=ids[0]
            interfaces[interfaceStr.format(omegas[i][j],omegas[i-1][j])]=ids[1]
            interfaces[interfaceStr.format(omegas[i][j],omegas[i][j+1])]=ids[2]

        # top of the boundary
        j = N-1
        for i in range(1,N-1):
            ids=five_point_stencil(N,i,j)
            interfaces[interfaceStr.format(omegas[i][j],omegas[i+1][j])]=ids[0]
            interfaces[interfaceStr.format(omegas[i][j],omegas[i-1][j])]=ids[1]
            interfaces[interfaceStr.format(omegas[i][j],omegas[i][j-1])]=ids[3]

        # corners
        # bottom lhs, i=j=0
        i=0
        j=0
        ids=five_point_stencil(N,i,j)
        interfaces[interfaceStr.format(omegas[i][j],omegas[i+1][j])]=ids[0]
        interfaces[interfaceStr.format(omegas[i][j],omegas[i][j+1])]=ids[2]

        # top lhs
        i=0
        j=N-1
        ids=five_point_stencil(N,i,j)
        interfaces[interfaceStr.format(omegas[i][j],omegas[i+1][j])]=ids[0]
        interfaces[interfaceStr.format(omegas[i][j],omegas[i][j-1])]=ids[3]

        # bottom rhs
        i=N-1
        j=0
        ids=five_point_stencil(N,i,j)
        interfaces[interfaceStr.format(omegas[i][j],omegas[i-1][j])]=ids[1]
        interfaces[interfaceStr.format(omegas[i][j],omegas[i][j+1])]=ids[2]

        # top rhs
        i=N-1
        j=N-1
        ids=five_point_stencil(N,i,j)
        interfaces[interfaceStr.format(omegas[i][j],omegas[i-1][j])]=ids[1]
        interfaces[interfaceStr.format(omegas[i][j],omegas[i][j-1])]=ids[3]
        return interfaces

def stringify_dictionary(dictionary):
    """
    Output a python dictionary into a string representation that may be used 
    directly in another python program.
    """
    dictionary_str="{"
    for key, value in dictionary.items():
        dictionary_str += '"{}":{},'.format(key,value)
    dictionary_str += "}"
    return dictionary_str

def stringify_dictionary_as_literal(dictionary):
    """
    Output a python dictionary into a string representation that may be used 
    directly in another python program.
    """
    dictionary_str="{"
    for key, value in dictionary.items():
        dictionary_str += '{}:{},'.format(key,value)
    dictionary_str += "}"
    return dictionary_str

def inject_subdomain_information(input_file, output_file, bc_map, subdom_map, interface_map):
    """
    Given a particular input file, search through the the input file for the following key phrases:
        - BC_MAP: mapping over which boundaries to apply Dirichlet boundary condition (values associated with keys are the value to be applied)
        - SUBDOMAIN_MAP: mapping from names of subdomains to subdomain ids
        - INTERFACE_MAP: mapping from strings of the form d<subdomain_i>n<subdomain_j> which specify the correct interface id to access the surface
    Keyword arguments:
    input_file -- the input file to apply the switch with
    output_file -- the output file to write the changes to
    bc_map -- see above.
    subdom_map -- see above.
    interface_map -- see above.
    """
    pLines=[]
    with open(input_file) as f:
        for line in f:
            pLines.append(line)
    programStr="".join(pLines)
    # generate strings associated with the appropriate data structures
    bc_map_str="{"
    for key, value in bc_map.items():
        bc_map_str += "{}:{},".format(key,value)
    bc_map_str += "}"
    subdomain_map_str="{"
    for key, value in subdom_map.items():
        subdomain_map_str += "{}:{}".format(key,value)
    subdomain_map_str += "}"
    interface_map_str="{"
    for key,value in interface_map.items():
        interface_map_str += "{}:{}".format(key,value)
    interface_map_str+="}"

    programStr=programStr.replace("BC_MAP",bc_map_str)
    programStr=programStr.replace("SUBDOMAIN_MAP",subdomain_map_str)
    programStr=programStr.replace("INTERFACE_MAP",interface_map_str)
    output=open(output_file, "w")
    output.write(programStr)
    pass
