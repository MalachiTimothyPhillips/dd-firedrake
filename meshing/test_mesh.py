def generate_comparison_mesh_file(outputfile, Nx, Ny):
    template_file_name = "SquareMeshTemplate.geo"
    programStr=""
    with open(template_file_name) as f:
        programStr=f.read()
    programStr=programStr.replace("NX",str(Nx.astype(int)))
    programStr=programStr.replace("NY",str(Ny.astype(int)))
    out = open(outputfile, "w")
    out.write(programStr)

def generate_all_comparison_meshes():
    cases=[2,4,8,16,32,64]
    sizes=[128,192,768,1280,5120,20480]
    import numpy as np
    for idx in range(len(sizes)):
        size=sizes[idx]
        case=cases[idx]
        nx=np.ceil(np.sqrt(size))
        template="{}x{}Comparison.geo"
        generate_comparison_mesh_file(template.format(str(case),str(case)),nx,nx)
