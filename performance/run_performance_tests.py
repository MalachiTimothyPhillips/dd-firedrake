import sys
sys.path.append('../')
import code_generation.gen_code as g
import enum_ids as e
isDebug=False
mesh_path="../meshing/"
template_file="template.py"
template_output="my_output_{}.py"
template_final_output="final_output_{}.py"
#Ns=[2,4,8,16,32,64]
Ns=[4]
meshName="{}x{}Overlap.msh"
programStr=""
with open(template_file,'r') as myfile:
    programStr=myfile.read()
for n in Ns:
    my_program=programStr
    mesh= '"' + mesh_path + meshName.format(n,n) + '"'
    my_program=my_program.replace("MY_NX",str(n))
    my_program=my_program.replace("MY_INPUT_MESH",mesh)
    sq = e.SquareMeshBoundaryConditions(n)
    bcMap = sq.enumerateDirichletBC()
    subdomainMap = sq.enumerateSubdomainMapping()
    interfaceMap = sq.enumerateSubdomainBoundaries()
    bcMapStr = e.stringify_dictionary_as_literal(bcMap)
    subdomainMapStr = e.stringify_dictionary(subdomainMap)
    interfaceMapStr = e.stringify_dictionary(interfaceMap)
    my_program=my_program.replace("BC_MAP",bcMapStr)
    my_program=my_program.replace("SUBDOMAIN_MAP",subdomainMapStr)
    my_program=my_program.replace("INTERFACE_MAP",interfaceMapStr)
    if(isDebug):
        print(my_program)
    # write program out
    output=template_output.format(n)
    f_out=open(output,"w")
    f_out.write(my_program)
    f_out.close()
    final_output=template_final_output.format(n)
    g.generate_program(output,final_output) 
