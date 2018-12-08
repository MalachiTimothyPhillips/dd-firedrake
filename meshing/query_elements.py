def number_elements(mesh_file):
    f=open(mesh_file,"r")
    stringRead=False
    for line in f:
        if "$Elements" in line:
            stringRead=True
            continue
        if(stringRead):
            return int(line)
def generate_table():
    output_file="table.csv"
    input_file="{}x{}Overlap.msh"
    Ns=[2,4,8,16,32,64]
    myStr="nx,subdomains,elements\n"
    for n in Ns:
        inp=input_file.format(n,n)
        num_elems=number_elements(inp)
        myStr += "{},{},{}\n".format(n,n*n,num_elems)
    f = open(output_file,"w")
    f.write(myStr)
