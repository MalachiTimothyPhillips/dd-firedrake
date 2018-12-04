import generate_squaremesh as mesher
import subprocess
outputStr="{:d}x{:d}Overlap.geo"
for i in range(6):
    N=2**(i+1)
    Nx=N
    Ny=N
    dx=1/N
    d=dx/10
    output=outputStr.format(Nx,Ny)
    mesher.generate_mesh(Nx,Ny,d,output)
    subprocess.run(["gmsh","-2",output])
