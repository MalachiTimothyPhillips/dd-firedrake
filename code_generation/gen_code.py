import sys
import ast
class MalformedProgramError(Exception):
    pass

def generate_program(input_file, output_file):
    if "BEGIN PROGRAM HERE".lower() not in open(input_file).read().lower():
        raise MalformedProgramError("Unable to find 'BEGIN PROGRAM HERE' label. Please check the program is correctly formed.")
    if "END PROGRAM HERE".lower() not in open(input_file).read().lower():
        raise MalformedProgramError("Unable to find 'END PROGRAM HERE' label. Please check the program is correctly formed.")
    
    doRead=False
    lines=[]
    plines=[]
    clines=[]
    domains={}
    interfaces={}
    dirichletBCs={}
    formStr=""
    rhsStr=""
    solutionStr=""
    solverStr=""
    mesh=""
    functionSpace=""
    algorithm_type=""
    spaceVar=""
    ## parsing
    
    with open(input_file) as f:
        for line in f:
            if "BEGIN PROGRAM HERE" in line.upper():
                doRead=True
                plines.append("PROGRAM_CODE\n")
                continue
            if "END PROGRAM HERE" in line.upper():
                doRead=False
                continue
            if doRead:
                if(line.startswith("domains")):
                    domains=ast.literal_eval(line.split("=")[1].strip())
                    clines.append(line)
                if(line.startswith("interfaces")):
                    interfaces=ast.literal_eval(line.split("=")[1].strip())
                    clines.append(line)
                if(line.startswith("form")):
                    formStr=line.split("~")[1].strip() + "({})"
                if(line.startswith("rhs")):
                    rhsStr=line.split("~")[1].strip() + "({})"
                if(line.startswith("solution")):
                    solutionStr=line.split("~")[1].strip() + "{}"
                if(line.startswith("solver_settings")):
                    solverStr=line.split("~")[1].strip()
                if(line.startswith("mesh_name")):
                    mesh=line.split("~")[1].strip()
                if(line.startswith("space_variable")):
                    spaceVar=line.split("~")[1].strip()
                if(line.startswith("functionSpace")):
                    functionSpace=line.split("~")[1].strip()
                if(line.startswith("dirichletBCs")):
                    dirichletBCs=ast.literal_eval(line.split("=")[1].strip())
                    clines.append(line)
                lines.append(line)
            if not doRead:
                plines.append(line)
    
    # generate index list from domains
    assmStr="{}={}"
    interfaceStr="d{}n{}"
    for subdomain_name, subdomain_id in domains.items():
        clines.append(assmStr.format(subdomain_name, subdomain_id))
        for other_subdomain_name in domains:
            if(subdomain_name == other_subdomain_name):
                continue
            my_interface=interfaceStr.format(subdomain_name,other_subdomain_name)
            if my_interface in interfaces:
                interface_id = interfaces[my_interface]
                clines.append(assmStr.format(my_interface,interface_id))
    
    customBC="""
class MyBC(DirichletBC):
    def __init__(self, V, value, markers):
        super(MyBC, self).__init__(V, value, 0)
        self.nodes = np.unique(np.where(markers.dat.data_ro_with_halos == 0)[0])
    """
    
    clines.append(customBC)
    
    space="h{}"
    heaviside_space="h{}=FunctionSpace({},'DG', 0)"
    heaviside="I_{} = Function({})"
    my_heaviside="I_{}"
    heaviside_V="I_cg_{} = Function({})"
    my_heaviside_V="I_cg_{}"
    parloop1="par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx({}), {{'f': ({}, WRITE)}} )"
    parloop2="par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {{'A' : ({}, RW), 'B': ({}, READ)}} )"
    
    mfuncs={}
    
    for subdomain_name in domains:
        func_space=space.format(subdomain_name)
        clines.append(heaviside_space.format(subdomain_name,mesh))
        heaviside_func=my_heaviside.format(subdomain_name)
        clines.append(heaviside.format(subdomain_name, func_space))
        clines.append(parloop1.format(subdomain_name, heaviside_func))
        heaviside_v_func=my_heaviside_V.format(subdomain_name)
        clines.append(heaviside_V.format(subdomain_name,functionSpace))
        clines.append(parloop2.format(heaviside_v_func,heaviside_func))
        mfuncs[subdomain_name]=heaviside_v_func
    
    
    # dirichlet boundaary conditions for homogeneous case (to begin with)
    dirichlet_bc_str="""
bcs=[]
for surface, value in dirichletBCs.items():
    bcs.append(DirichletBC({},value,surface))
    """
    clines.append(dirichlet_bc_str.format(functionSpace))
    
    bc_excludes={}
    bcStr="BC_{}_only=MyBC({},0,{})"
    for subdomain in domains:
        bc_excludes[subdomain] = "BC_{}_only".format(subdomain)
        clines.append(bcStr.format(subdomain,functionSpace, mfuncs[subdomain]))
    
    u_old={}
    u={}
    assmStr="{}=Function({})"
    clines.append("{}=SpatialCoordinate({})".format(spaceVar,mesh))
    for subdomain in domains:
        my_u=solutionStr.format(subdomain)
        u[subdomain]=my_u
        my_old_u=solutionStr.format(subdomain)+"_old"
        u_old[subdomain]=my_old_u
        clines.append(assmStr.format(my_u,functionSpace))
        interpolationStr="1*{}[0]*{}[1]".format(spaceVar,spaceVar)
        clines.append("{}.interpolate({})".format(my_u,interpolationStr))
        clines.append(assmStr.format(my_old_u,functionSpace))
        clines.append("{}.interpolate({})".format(my_old_u,interpolationStr))
    # form strings
    rhs={}
    lhs={}
    for subdomain in domains:
        my_lhs="a{}".format(subdomain)
        lhs[subdomain]=my_lhs
        clines.append(my_lhs + " = " + formStr.format(subdomain))
        my_rhs="L{}".format(subdomain)
        rhs[subdomain]=my_rhs
        clines.append(my_rhs + " = " + rhsStr.format(subdomain))
    
    # create all the relevant B.C.s
    bcs={}
    bcName="bc{}"
    copyStr="{}=bcs.copy()"
    interfaceStr="d{}n{}"
    uniqueExprCtr = 1
    exprStr="e{}={}+{}"
    for subdomain, subdomain_id in domains.items():
        name = bcName.format(subdomain)
        bcs[subdomain]=name
        clines.append(copyStr.format(name))
        for other_subdomain in domains:
            if(other_subdomain == subdomain):
                continue
            my_interface = interfaceStr.format(subdomain, other_subdomain)
            if my_interface in interfaces:
                expr="e{}".format(uniqueExprCtr)
                clines.append(exprStr.format(uniqueExprCtr,u_old[subdomain],u_old[other_subdomain]))
                clines.append("{}.append({})".format(name, "DirichletBC({},{},{})".format(functionSpace, expr, my_interface)))
                uniqueExprCtr+=1
        clines.append("{}.append({})".format(name,bc_excludes[subdomain]))
    
    clines.append("nSchwarz=10")
    clines.append("for iteration in range(nSchwarz):")
    solver="\tsolve({}=={},{},bcs={},{})"
    assmStr="\t{}={}"
    for subdomain in domains:
        clines.append(assmStr.format(u_old[subdomain],u[subdomain]))
        
    for subdomain in domains:
        clines.append(solver.format(lhs[subdomain],rhs[subdomain], u[subdomain], bcs[subdomain],solverStr))
    programStr="".join(plines)
    compiledStr="\n".join(clines)
    programStr=programStr.replace("PROGRAM_CODE",compiledStr)
    #print(programStr)
    output=open(output_file,"w")
    output.write(programStr)
