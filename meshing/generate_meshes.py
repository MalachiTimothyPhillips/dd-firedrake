import generate_squaremesh as mesher
Nx=4
Ny=4
d=0.001
output="test.geo"
mesher.generate_mesh(Nx,Ny,d,output)
