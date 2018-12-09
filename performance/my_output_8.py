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
mesh=Mesh("../meshing/8x8Overlap.msh")
N=8
N_subdom=N*N
V=FunctionSpace(mesh, "CG", 1) # piecewise linear elements

ref_mesh=Mesh("../meshing/8x8Comparison.msh")
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
BEGIN_PROGRAM_HERE
mesh_name ~ mesh
form ~ (dot(grad(u), grad(v))) * dx
rhs ~ f * v * dx
space_variable ~ coord
functionSpace ~ V
dirichletBCs={1:0,2:0,32:0,227:0,5:0,34:0,64:0,231:0,9:0,66:0,96:0,235:0,13:0,98:0,128:0,239:0,17:0,130:0,160:0,243:0,21:0,162:0,192:0,247:0,25:0,194:0,224:0,251:0,29:0,226:0,256:0,255:0,}
# domain number to id mapping
domains={"Omega1":1,"Omega2":2,"Omega3":3,"Omega4":4,"Omega5":5,"Omega6":6,"Omega7":7,"Omega8":8,"Omega9":9,"Omega10":10,"Omega11":11,"Omega12":12,"Omega13":13,"Omega14":14,"Omega15":15,"Omega16":16,"Omega17":17,"Omega18":18,"Omega19":19,"Omega20":20,"Omega21":21,"Omega22":22,"Omega23":23,"Omega24":24,"Omega25":25,"Omega26":26,"Omega27":27,"Omega28":28,"Omega29":29,"Omega30":30,"Omega31":31,"Omega32":32,"Omega33":33,"Omega34":34,"Omega35":35,"Omega36":36,"Omega37":37,"Omega38":38,"Omega39":39,"Omega40":40,"Omega41":41,"Omega42":42,"Omega43":43,"Omega44":44,"Omega45":45,"Omega46":46,"Omega47":47,"Omega48":48,"Omega49":49,"Omega50":50,"Omega51":51,"Omega52":52,"Omega53":53,"Omega54":54,"Omega55":55,"Omega56":56,"Omega57":57,"Omega58":58,"Omega59":59,"Omega60":60,"Omega61":61,"Omega62":62,"Omega63":63,"Omega64":64,}
# [domain number, domain number]->interface id mapping
interfaces={"dOmega10nOmega11":40,"dOmega10nOmega9":38,"dOmega10nOmega18":39,"dOmega10nOmega2":37,"dOmega18nOmega19":72,"dOmega18nOmega17":70,"dOmega18nOmega26":71,"dOmega18nOmega10":69,"dOmega26nOmega27":104,"dOmega26nOmega25":102,"dOmega26nOmega34":103,"dOmega26nOmega18":101,"dOmega34nOmega35":136,"dOmega34nOmega33":134,"dOmega34nOmega42":135,"dOmega34nOmega26":133,"dOmega42nOmega43":168,"dOmega42nOmega41":166,"dOmega42nOmega50":167,"dOmega42nOmega34":165,"dOmega50nOmega51":200,"dOmega50nOmega49":198,"dOmega50nOmega58":199,"dOmega50nOmega42":197,"dOmega11nOmega12":44,"dOmega11nOmega10":42,"dOmega11nOmega19":43,"dOmega11nOmega3":41,"dOmega19nOmega20":76,"dOmega19nOmega18":74,"dOmega19nOmega27":75,"dOmega19nOmega11":73,"dOmega27nOmega28":108,"dOmega27nOmega26":106,"dOmega27nOmega35":107,"dOmega27nOmega19":105,"dOmega35nOmega36":140,"dOmega35nOmega34":138,"dOmega35nOmega43":139,"dOmega35nOmega27":137,"dOmega43nOmega44":172,"dOmega43nOmega42":170,"dOmega43nOmega51":171,"dOmega43nOmega35":169,"dOmega51nOmega52":204,"dOmega51nOmega50":202,"dOmega51nOmega59":203,"dOmega51nOmega43":201,"dOmega12nOmega13":48,"dOmega12nOmega11":46,"dOmega12nOmega20":47,"dOmega12nOmega4":45,"dOmega20nOmega21":80,"dOmega20nOmega19":78,"dOmega20nOmega28":79,"dOmega20nOmega12":77,"dOmega28nOmega29":112,"dOmega28nOmega27":110,"dOmega28nOmega36":111,"dOmega28nOmega20":109,"dOmega36nOmega37":144,"dOmega36nOmega35":142,"dOmega36nOmega44":143,"dOmega36nOmega28":141,"dOmega44nOmega45":176,"dOmega44nOmega43":174,"dOmega44nOmega52":175,"dOmega44nOmega36":173,"dOmega52nOmega53":208,"dOmega52nOmega51":206,"dOmega52nOmega60":207,"dOmega52nOmega44":205,"dOmega13nOmega14":52,"dOmega13nOmega12":50,"dOmega13nOmega21":51,"dOmega13nOmega5":49,"dOmega21nOmega22":84,"dOmega21nOmega20":82,"dOmega21nOmega29":83,"dOmega21nOmega13":81,"dOmega29nOmega30":116,"dOmega29nOmega28":114,"dOmega29nOmega37":115,"dOmega29nOmega21":113,"dOmega37nOmega38":148,"dOmega37nOmega36":146,"dOmega37nOmega45":147,"dOmega37nOmega29":145,"dOmega45nOmega46":180,"dOmega45nOmega44":178,"dOmega45nOmega53":179,"dOmega45nOmega37":177,"dOmega53nOmega54":212,"dOmega53nOmega52":210,"dOmega53nOmega61":211,"dOmega53nOmega45":209,"dOmega14nOmega15":56,"dOmega14nOmega13":54,"dOmega14nOmega22":55,"dOmega14nOmega6":53,"dOmega22nOmega23":88,"dOmega22nOmega21":86,"dOmega22nOmega30":87,"dOmega22nOmega14":85,"dOmega30nOmega31":120,"dOmega30nOmega29":118,"dOmega30nOmega38":119,"dOmega30nOmega22":117,"dOmega38nOmega39":152,"dOmega38nOmega37":150,"dOmega38nOmega46":151,"dOmega38nOmega30":149,"dOmega46nOmega47":184,"dOmega46nOmega45":182,"dOmega46nOmega54":183,"dOmega46nOmega38":181,"dOmega54nOmega55":216,"dOmega54nOmega53":214,"dOmega54nOmega62":215,"dOmega54nOmega46":213,"dOmega15nOmega16":60,"dOmega15nOmega14":58,"dOmega15nOmega23":59,"dOmega15nOmega7":57,"dOmega23nOmega24":92,"dOmega23nOmega22":90,"dOmega23nOmega31":91,"dOmega23nOmega15":89,"dOmega31nOmega32":124,"dOmega31nOmega30":122,"dOmega31nOmega39":123,"dOmega31nOmega23":121,"dOmega39nOmega40":156,"dOmega39nOmega38":154,"dOmega39nOmega47":155,"dOmega39nOmega31":153,"dOmega47nOmega48":188,"dOmega47nOmega46":186,"dOmega47nOmega55":187,"dOmega47nOmega39":185,"dOmega55nOmega56":220,"dOmega55nOmega54":218,"dOmega55nOmega63":219,"dOmega55nOmega47":217,"dOmega9nOmega10":36,"dOmega9nOmega17":35,"dOmega9nOmega1":33,"dOmega17nOmega18":68,"dOmega17nOmega25":67,"dOmega17nOmega9":65,"dOmega25nOmega26":100,"dOmega25nOmega33":99,"dOmega25nOmega17":97,"dOmega33nOmega34":132,"dOmega33nOmega41":131,"dOmega33nOmega25":129,"dOmega41nOmega42":164,"dOmega41nOmega49":163,"dOmega41nOmega33":161,"dOmega49nOmega50":196,"dOmega49nOmega57":195,"dOmega49nOmega41":193,"dOmega16nOmega15":62,"dOmega16nOmega24":63,"dOmega16nOmega8":61,"dOmega24nOmega23":94,"dOmega24nOmega32":95,"dOmega24nOmega16":93,"dOmega32nOmega31":126,"dOmega32nOmega40":127,"dOmega32nOmega24":125,"dOmega40nOmega39":158,"dOmega40nOmega48":159,"dOmega40nOmega32":157,"dOmega48nOmega47":190,"dOmega48nOmega56":191,"dOmega48nOmega40":189,"dOmega56nOmega55":222,"dOmega56nOmega64":223,"dOmega56nOmega48":221,"dOmega2nOmega3":8,"dOmega2nOmega1":6,"dOmega2nOmega10":7,"dOmega3nOmega4":12,"dOmega3nOmega2":10,"dOmega3nOmega11":11,"dOmega4nOmega5":16,"dOmega4nOmega3":14,"dOmega4nOmega12":15,"dOmega5nOmega6":20,"dOmega5nOmega4":18,"dOmega5nOmega13":19,"dOmega6nOmega7":24,"dOmega6nOmega5":22,"dOmega6nOmega14":23,"dOmega7nOmega8":28,"dOmega7nOmega6":26,"dOmega7nOmega15":27,"dOmega58nOmega59":232,"dOmega58nOmega57":230,"dOmega58nOmega50":229,"dOmega59nOmega60":236,"dOmega59nOmega58":234,"dOmega59nOmega51":233,"dOmega60nOmega61":240,"dOmega60nOmega59":238,"dOmega60nOmega52":237,"dOmega61nOmega62":244,"dOmega61nOmega60":242,"dOmega61nOmega53":241,"dOmega62nOmega63":248,"dOmega62nOmega61":246,"dOmega62nOmega54":245,"dOmega63nOmega64":252,"dOmega63nOmega62":250,"dOmega63nOmega55":249,"dOmega1nOmega2":4,"dOmega1nOmega9":3,"dOmega57nOmega58":228,"dOmega57nOmega49":225,"dOmega8nOmega7":30,"dOmega8nOmega16":31,"dOmega64nOmega63":254,"dOmega64nOmega56":253,}
# solver selection, parameters, etc
# respects everything past bcs (i.e., just keeps it the same)
solution~u

# always of the form: form == rhs, solution
solver_settings ~ solver_parameters=params

END_PROGRAM_HERE
end_time2=time.time()
print("{},{},{},{}".format(N,N_subdom,end_time1-start_time1,end_time2-start_time2))
# does this line exist?
# perhaps some other lines after the fact?
