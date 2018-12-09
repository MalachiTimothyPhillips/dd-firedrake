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
start_time1=time.time()
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
#start_time1=time.time()
solve(a==L,u_entire,bcs=[bc1,bc2,bc3,bc4],solver_parameters=params)
end_time1=time.time()

# for our purposes, we need to also generate timing data for the regular solve
u=TrialFunction(V)
v=TestFunction(V)

dirichletBCs={1:0,2:0,32:0,227:0,5:0,34:0,64:0,231:0,9:0,66:0,96:0,235:0,13:0,98:0,128:0,239:0,17:0,130:0,160:0,243:0,21:0,162:0,192:0,247:0,25:0,194:0,224:0,251:0,29:0,226:0,256:0,255:0,}

domains={"Omega1":1,"Omega2":2,"Omega3":3,"Omega4":4,"Omega5":5,"Omega6":6,"Omega7":7,"Omega8":8,"Omega9":9,"Omega10":10,"Omega11":11,"Omega12":12,"Omega13":13,"Omega14":14,"Omega15":15,"Omega16":16,"Omega17":17,"Omega18":18,"Omega19":19,"Omega20":20,"Omega21":21,"Omega22":22,"Omega23":23,"Omega24":24,"Omega25":25,"Omega26":26,"Omega27":27,"Omega28":28,"Omega29":29,"Omega30":30,"Omega31":31,"Omega32":32,"Omega33":33,"Omega34":34,"Omega35":35,"Omega36":36,"Omega37":37,"Omega38":38,"Omega39":39,"Omega40":40,"Omega41":41,"Omega42":42,"Omega43":43,"Omega44":44,"Omega45":45,"Omega46":46,"Omega47":47,"Omega48":48,"Omega49":49,"Omega50":50,"Omega51":51,"Omega52":52,"Omega53":53,"Omega54":54,"Omega55":55,"Omega56":56,"Omega57":57,"Omega58":58,"Omega59":59,"Omega60":60,"Omega61":61,"Omega62":62,"Omega63":63,"Omega64":64,}

interfaces={"dOmega10nOmega11":40,"dOmega10nOmega9":38,"dOmega10nOmega18":39,"dOmega10nOmega2":37,"dOmega18nOmega19":72,"dOmega18nOmega17":70,"dOmega18nOmega26":71,"dOmega18nOmega10":69,"dOmega26nOmega27":104,"dOmega26nOmega25":102,"dOmega26nOmega34":103,"dOmega26nOmega18":101,"dOmega34nOmega35":136,"dOmega34nOmega33":134,"dOmega34nOmega42":135,"dOmega34nOmega26":133,"dOmega42nOmega43":168,"dOmega42nOmega41":166,"dOmega42nOmega50":167,"dOmega42nOmega34":165,"dOmega50nOmega51":200,"dOmega50nOmega49":198,"dOmega50nOmega58":199,"dOmega50nOmega42":197,"dOmega11nOmega12":44,"dOmega11nOmega10":42,"dOmega11nOmega19":43,"dOmega11nOmega3":41,"dOmega19nOmega20":76,"dOmega19nOmega18":74,"dOmega19nOmega27":75,"dOmega19nOmega11":73,"dOmega27nOmega28":108,"dOmega27nOmega26":106,"dOmega27nOmega35":107,"dOmega27nOmega19":105,"dOmega35nOmega36":140,"dOmega35nOmega34":138,"dOmega35nOmega43":139,"dOmega35nOmega27":137,"dOmega43nOmega44":172,"dOmega43nOmega42":170,"dOmega43nOmega51":171,"dOmega43nOmega35":169,"dOmega51nOmega52":204,"dOmega51nOmega50":202,"dOmega51nOmega59":203,"dOmega51nOmega43":201,"dOmega12nOmega13":48,"dOmega12nOmega11":46,"dOmega12nOmega20":47,"dOmega12nOmega4":45,"dOmega20nOmega21":80,"dOmega20nOmega19":78,"dOmega20nOmega28":79,"dOmega20nOmega12":77,"dOmega28nOmega29":112,"dOmega28nOmega27":110,"dOmega28nOmega36":111,"dOmega28nOmega20":109,"dOmega36nOmega37":144,"dOmega36nOmega35":142,"dOmega36nOmega44":143,"dOmega36nOmega28":141,"dOmega44nOmega45":176,"dOmega44nOmega43":174,"dOmega44nOmega52":175,"dOmega44nOmega36":173,"dOmega52nOmega53":208,"dOmega52nOmega51":206,"dOmega52nOmega60":207,"dOmega52nOmega44":205,"dOmega13nOmega14":52,"dOmega13nOmega12":50,"dOmega13nOmega21":51,"dOmega13nOmega5":49,"dOmega21nOmega22":84,"dOmega21nOmega20":82,"dOmega21nOmega29":83,"dOmega21nOmega13":81,"dOmega29nOmega30":116,"dOmega29nOmega28":114,"dOmega29nOmega37":115,"dOmega29nOmega21":113,"dOmega37nOmega38":148,"dOmega37nOmega36":146,"dOmega37nOmega45":147,"dOmega37nOmega29":145,"dOmega45nOmega46":180,"dOmega45nOmega44":178,"dOmega45nOmega53":179,"dOmega45nOmega37":177,"dOmega53nOmega54":212,"dOmega53nOmega52":210,"dOmega53nOmega61":211,"dOmega53nOmega45":209,"dOmega14nOmega15":56,"dOmega14nOmega13":54,"dOmega14nOmega22":55,"dOmega14nOmega6":53,"dOmega22nOmega23":88,"dOmega22nOmega21":86,"dOmega22nOmega30":87,"dOmega22nOmega14":85,"dOmega30nOmega31":120,"dOmega30nOmega29":118,"dOmega30nOmega38":119,"dOmega30nOmega22":117,"dOmega38nOmega39":152,"dOmega38nOmega37":150,"dOmega38nOmega46":151,"dOmega38nOmega30":149,"dOmega46nOmega47":184,"dOmega46nOmega45":182,"dOmega46nOmega54":183,"dOmega46nOmega38":181,"dOmega54nOmega55":216,"dOmega54nOmega53":214,"dOmega54nOmega62":215,"dOmega54nOmega46":213,"dOmega15nOmega16":60,"dOmega15nOmega14":58,"dOmega15nOmega23":59,"dOmega15nOmega7":57,"dOmega23nOmega24":92,"dOmega23nOmega22":90,"dOmega23nOmega31":91,"dOmega23nOmega15":89,"dOmega31nOmega32":124,"dOmega31nOmega30":122,"dOmega31nOmega39":123,"dOmega31nOmega23":121,"dOmega39nOmega40":156,"dOmega39nOmega38":154,"dOmega39nOmega47":155,"dOmega39nOmega31":153,"dOmega47nOmega48":188,"dOmega47nOmega46":186,"dOmega47nOmega55":187,"dOmega47nOmega39":185,"dOmega55nOmega56":220,"dOmega55nOmega54":218,"dOmega55nOmega63":219,"dOmega55nOmega47":217,"dOmega9nOmega10":36,"dOmega9nOmega17":35,"dOmega9nOmega1":33,"dOmega17nOmega18":68,"dOmega17nOmega25":67,"dOmega17nOmega9":65,"dOmega25nOmega26":100,"dOmega25nOmega33":99,"dOmega25nOmega17":97,"dOmega33nOmega34":132,"dOmega33nOmega41":131,"dOmega33nOmega25":129,"dOmega41nOmega42":164,"dOmega41nOmega49":163,"dOmega41nOmega33":161,"dOmega49nOmega50":196,"dOmega49nOmega57":195,"dOmega49nOmega41":193,"dOmega16nOmega15":62,"dOmega16nOmega24":63,"dOmega16nOmega8":61,"dOmega24nOmega23":94,"dOmega24nOmega32":95,"dOmega24nOmega16":93,"dOmega32nOmega31":126,"dOmega32nOmega40":127,"dOmega32nOmega24":125,"dOmega40nOmega39":158,"dOmega40nOmega48":159,"dOmega40nOmega32":157,"dOmega48nOmega47":190,"dOmega48nOmega56":191,"dOmega48nOmega40":189,"dOmega56nOmega55":222,"dOmega56nOmega64":223,"dOmega56nOmega48":221,"dOmega2nOmega3":8,"dOmega2nOmega1":6,"dOmega2nOmega10":7,"dOmega3nOmega4":12,"dOmega3nOmega2":10,"dOmega3nOmega11":11,"dOmega4nOmega5":16,"dOmega4nOmega3":14,"dOmega4nOmega12":15,"dOmega5nOmega6":20,"dOmega5nOmega4":18,"dOmega5nOmega13":19,"dOmega6nOmega7":24,"dOmega6nOmega5":22,"dOmega6nOmega14":23,"dOmega7nOmega8":28,"dOmega7nOmega6":26,"dOmega7nOmega15":27,"dOmega58nOmega59":232,"dOmega58nOmega57":230,"dOmega58nOmega50":229,"dOmega59nOmega60":236,"dOmega59nOmega58":234,"dOmega59nOmega51":233,"dOmega60nOmega61":240,"dOmega60nOmega59":238,"dOmega60nOmega52":237,"dOmega61nOmega62":244,"dOmega61nOmega60":242,"dOmega61nOmega53":241,"dOmega62nOmega63":248,"dOmega62nOmega61":246,"dOmega62nOmega54":245,"dOmega63nOmega64":252,"dOmega63nOmega62":250,"dOmega63nOmega55":249,"dOmega1nOmega2":4,"dOmega1nOmega9":3,"dOmega57nOmega58":228,"dOmega57nOmega49":225,"dOmega8nOmega7":30,"dOmega8nOmega16":31,"dOmega64nOmega63":254,"dOmega64nOmega56":253,}

Omega1=1
dOmega1nOmega2=4
dOmega1nOmega9=3
Omega2=2
dOmega2nOmega1=6
dOmega2nOmega3=8
dOmega2nOmega10=7
Omega3=3
dOmega3nOmega2=10
dOmega3nOmega4=12
dOmega3nOmega11=11
Omega4=4
dOmega4nOmega3=14
dOmega4nOmega5=16
dOmega4nOmega12=15
Omega5=5
dOmega5nOmega4=18
dOmega5nOmega6=20
dOmega5nOmega13=19
Omega6=6
dOmega6nOmega5=22
dOmega6nOmega7=24
dOmega6nOmega14=23
Omega7=7
dOmega7nOmega6=26
dOmega7nOmega8=28
dOmega7nOmega15=27
Omega8=8
dOmega8nOmega7=30
dOmega8nOmega16=31
Omega9=9
dOmega9nOmega1=33
dOmega9nOmega10=36
dOmega9nOmega17=35
Omega10=10
dOmega10nOmega2=37
dOmega10nOmega9=38
dOmega10nOmega11=40
dOmega10nOmega18=39
Omega11=11
dOmega11nOmega3=41
dOmega11nOmega10=42
dOmega11nOmega12=44
dOmega11nOmega19=43
Omega12=12
dOmega12nOmega4=45
dOmega12nOmega11=46
dOmega12nOmega13=48
dOmega12nOmega20=47
Omega13=13
dOmega13nOmega5=49
dOmega13nOmega12=50
dOmega13nOmega14=52
dOmega13nOmega21=51
Omega14=14
dOmega14nOmega6=53
dOmega14nOmega13=54
dOmega14nOmega15=56
dOmega14nOmega22=55
Omega15=15
dOmega15nOmega7=57
dOmega15nOmega14=58
dOmega15nOmega16=60
dOmega15nOmega23=59
Omega16=16
dOmega16nOmega8=61
dOmega16nOmega15=62
dOmega16nOmega24=63
Omega17=17
dOmega17nOmega9=65
dOmega17nOmega18=68
dOmega17nOmega25=67
Omega18=18
dOmega18nOmega10=69
dOmega18nOmega17=70
dOmega18nOmega19=72
dOmega18nOmega26=71
Omega19=19
dOmega19nOmega11=73
dOmega19nOmega18=74
dOmega19nOmega20=76
dOmega19nOmega27=75
Omega20=20
dOmega20nOmega12=77
dOmega20nOmega19=78
dOmega20nOmega21=80
dOmega20nOmega28=79
Omega21=21
dOmega21nOmega13=81
dOmega21nOmega20=82
dOmega21nOmega22=84
dOmega21nOmega29=83
Omega22=22
dOmega22nOmega14=85
dOmega22nOmega21=86
dOmega22nOmega23=88
dOmega22nOmega30=87
Omega23=23
dOmega23nOmega15=89
dOmega23nOmega22=90
dOmega23nOmega24=92
dOmega23nOmega31=91
Omega24=24
dOmega24nOmega16=93
dOmega24nOmega23=94
dOmega24nOmega32=95
Omega25=25
dOmega25nOmega17=97
dOmega25nOmega26=100
dOmega25nOmega33=99
Omega26=26
dOmega26nOmega18=101
dOmega26nOmega25=102
dOmega26nOmega27=104
dOmega26nOmega34=103
Omega27=27
dOmega27nOmega19=105
dOmega27nOmega26=106
dOmega27nOmega28=108
dOmega27nOmega35=107
Omega28=28
dOmega28nOmega20=109
dOmega28nOmega27=110
dOmega28nOmega29=112
dOmega28nOmega36=111
Omega29=29
dOmega29nOmega21=113
dOmega29nOmega28=114
dOmega29nOmega30=116
dOmega29nOmega37=115
Omega30=30
dOmega30nOmega22=117
dOmega30nOmega29=118
dOmega30nOmega31=120
dOmega30nOmega38=119
Omega31=31
dOmega31nOmega23=121
dOmega31nOmega30=122
dOmega31nOmega32=124
dOmega31nOmega39=123
Omega32=32
dOmega32nOmega24=125
dOmega32nOmega31=126
dOmega32nOmega40=127
Omega33=33
dOmega33nOmega25=129
dOmega33nOmega34=132
dOmega33nOmega41=131
Omega34=34
dOmega34nOmega26=133
dOmega34nOmega33=134
dOmega34nOmega35=136
dOmega34nOmega42=135
Omega35=35
dOmega35nOmega27=137
dOmega35nOmega34=138
dOmega35nOmega36=140
dOmega35nOmega43=139
Omega36=36
dOmega36nOmega28=141
dOmega36nOmega35=142
dOmega36nOmega37=144
dOmega36nOmega44=143
Omega37=37
dOmega37nOmega29=145
dOmega37nOmega36=146
dOmega37nOmega38=148
dOmega37nOmega45=147
Omega38=38
dOmega38nOmega30=149
dOmega38nOmega37=150
dOmega38nOmega39=152
dOmega38nOmega46=151
Omega39=39
dOmega39nOmega31=153
dOmega39nOmega38=154
dOmega39nOmega40=156
dOmega39nOmega47=155
Omega40=40
dOmega40nOmega32=157
dOmega40nOmega39=158
dOmega40nOmega48=159
Omega41=41
dOmega41nOmega33=161
dOmega41nOmega42=164
dOmega41nOmega49=163
Omega42=42
dOmega42nOmega34=165
dOmega42nOmega41=166
dOmega42nOmega43=168
dOmega42nOmega50=167
Omega43=43
dOmega43nOmega35=169
dOmega43nOmega42=170
dOmega43nOmega44=172
dOmega43nOmega51=171
Omega44=44
dOmega44nOmega36=173
dOmega44nOmega43=174
dOmega44nOmega45=176
dOmega44nOmega52=175
Omega45=45
dOmega45nOmega37=177
dOmega45nOmega44=178
dOmega45nOmega46=180
dOmega45nOmega53=179
Omega46=46
dOmega46nOmega38=181
dOmega46nOmega45=182
dOmega46nOmega47=184
dOmega46nOmega54=183
Omega47=47
dOmega47nOmega39=185
dOmega47nOmega46=186
dOmega47nOmega48=188
dOmega47nOmega55=187
Omega48=48
dOmega48nOmega40=189
dOmega48nOmega47=190
dOmega48nOmega56=191
Omega49=49
dOmega49nOmega41=193
dOmega49nOmega50=196
dOmega49nOmega57=195
Omega50=50
dOmega50nOmega42=197
dOmega50nOmega49=198
dOmega50nOmega51=200
dOmega50nOmega58=199
Omega51=51
dOmega51nOmega43=201
dOmega51nOmega50=202
dOmega51nOmega52=204
dOmega51nOmega59=203
Omega52=52
dOmega52nOmega44=205
dOmega52nOmega51=206
dOmega52nOmega53=208
dOmega52nOmega60=207
Omega53=53
dOmega53nOmega45=209
dOmega53nOmega52=210
dOmega53nOmega54=212
dOmega53nOmega61=211
Omega54=54
dOmega54nOmega46=213
dOmega54nOmega53=214
dOmega54nOmega55=216
dOmega54nOmega62=215
Omega55=55
dOmega55nOmega47=217
dOmega55nOmega54=218
dOmega55nOmega56=220
dOmega55nOmega63=219
Omega56=56
dOmega56nOmega48=221
dOmega56nOmega55=222
dOmega56nOmega64=223
Omega57=57
dOmega57nOmega49=225
dOmega57nOmega58=228
Omega58=58
dOmega58nOmega50=229
dOmega58nOmega57=230
dOmega58nOmega59=232
Omega59=59
dOmega59nOmega51=233
dOmega59nOmega58=234
dOmega59nOmega60=236
Omega60=60
dOmega60nOmega52=237
dOmega60nOmega59=238
dOmega60nOmega61=240
Omega61=61
dOmega61nOmega53=241
dOmega61nOmega60=242
dOmega61nOmega62=244
Omega62=62
dOmega62nOmega54=245
dOmega62nOmega61=246
dOmega62nOmega63=248
Omega63=63
dOmega63nOmega55=249
dOmega63nOmega62=250
dOmega63nOmega64=252
Omega64=64
dOmega64nOmega56=253
dOmega64nOmega63=254

class MyBC(DirichletBC):
    def __init__(self, V, value, markers):
        super(MyBC, self).__init__(V, value, 0)
        self.nodes = np.unique(np.where(markers.dat.data_ro_with_halos == 0)[0])
    
hOmega1=FunctionSpace(mesh,'DG', 0)
I_Omega1 = Function(hOmega1)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega1), {'f': (I_Omega1, WRITE)} )
I_cg_Omega1 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega1, RW), 'B': (I_Omega1, READ)} )
hOmega2=FunctionSpace(mesh,'DG', 0)
I_Omega2 = Function(hOmega2)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega2), {'f': (I_Omega2, WRITE)} )
I_cg_Omega2 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega2, RW), 'B': (I_Omega2, READ)} )
hOmega3=FunctionSpace(mesh,'DG', 0)
I_Omega3 = Function(hOmega3)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega3), {'f': (I_Omega3, WRITE)} )
I_cg_Omega3 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega3, RW), 'B': (I_Omega3, READ)} )
hOmega4=FunctionSpace(mesh,'DG', 0)
I_Omega4 = Function(hOmega4)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega4), {'f': (I_Omega4, WRITE)} )
I_cg_Omega4 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega4, RW), 'B': (I_Omega4, READ)} )
hOmega5=FunctionSpace(mesh,'DG', 0)
I_Omega5 = Function(hOmega5)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega5), {'f': (I_Omega5, WRITE)} )
I_cg_Omega5 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega5, RW), 'B': (I_Omega5, READ)} )
hOmega6=FunctionSpace(mesh,'DG', 0)
I_Omega6 = Function(hOmega6)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega6), {'f': (I_Omega6, WRITE)} )
I_cg_Omega6 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega6, RW), 'B': (I_Omega6, READ)} )
hOmega7=FunctionSpace(mesh,'DG', 0)
I_Omega7 = Function(hOmega7)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega7), {'f': (I_Omega7, WRITE)} )
I_cg_Omega7 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega7, RW), 'B': (I_Omega7, READ)} )
hOmega8=FunctionSpace(mesh,'DG', 0)
I_Omega8 = Function(hOmega8)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega8), {'f': (I_Omega8, WRITE)} )
I_cg_Omega8 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega8, RW), 'B': (I_Omega8, READ)} )
hOmega9=FunctionSpace(mesh,'DG', 0)
I_Omega9 = Function(hOmega9)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega9), {'f': (I_Omega9, WRITE)} )
I_cg_Omega9 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega9, RW), 'B': (I_Omega9, READ)} )
hOmega10=FunctionSpace(mesh,'DG', 0)
I_Omega10 = Function(hOmega10)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega10), {'f': (I_Omega10, WRITE)} )
I_cg_Omega10 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega10, RW), 'B': (I_Omega10, READ)} )
hOmega11=FunctionSpace(mesh,'DG', 0)
I_Omega11 = Function(hOmega11)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega11), {'f': (I_Omega11, WRITE)} )
I_cg_Omega11 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega11, RW), 'B': (I_Omega11, READ)} )
hOmega12=FunctionSpace(mesh,'DG', 0)
I_Omega12 = Function(hOmega12)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega12), {'f': (I_Omega12, WRITE)} )
I_cg_Omega12 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega12, RW), 'B': (I_Omega12, READ)} )
hOmega13=FunctionSpace(mesh,'DG', 0)
I_Omega13 = Function(hOmega13)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega13), {'f': (I_Omega13, WRITE)} )
I_cg_Omega13 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega13, RW), 'B': (I_Omega13, READ)} )
hOmega14=FunctionSpace(mesh,'DG', 0)
I_Omega14 = Function(hOmega14)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega14), {'f': (I_Omega14, WRITE)} )
I_cg_Omega14 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega14, RW), 'B': (I_Omega14, READ)} )
hOmega15=FunctionSpace(mesh,'DG', 0)
I_Omega15 = Function(hOmega15)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega15), {'f': (I_Omega15, WRITE)} )
I_cg_Omega15 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega15, RW), 'B': (I_Omega15, READ)} )
hOmega16=FunctionSpace(mesh,'DG', 0)
I_Omega16 = Function(hOmega16)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega16), {'f': (I_Omega16, WRITE)} )
I_cg_Omega16 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega16, RW), 'B': (I_Omega16, READ)} )
hOmega17=FunctionSpace(mesh,'DG', 0)
I_Omega17 = Function(hOmega17)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega17), {'f': (I_Omega17, WRITE)} )
I_cg_Omega17 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega17, RW), 'B': (I_Omega17, READ)} )
hOmega18=FunctionSpace(mesh,'DG', 0)
I_Omega18 = Function(hOmega18)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega18), {'f': (I_Omega18, WRITE)} )
I_cg_Omega18 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega18, RW), 'B': (I_Omega18, READ)} )
hOmega19=FunctionSpace(mesh,'DG', 0)
I_Omega19 = Function(hOmega19)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega19), {'f': (I_Omega19, WRITE)} )
I_cg_Omega19 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega19, RW), 'B': (I_Omega19, READ)} )
hOmega20=FunctionSpace(mesh,'DG', 0)
I_Omega20 = Function(hOmega20)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega20), {'f': (I_Omega20, WRITE)} )
I_cg_Omega20 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega20, RW), 'B': (I_Omega20, READ)} )
hOmega21=FunctionSpace(mesh,'DG', 0)
I_Omega21 = Function(hOmega21)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega21), {'f': (I_Omega21, WRITE)} )
I_cg_Omega21 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega21, RW), 'B': (I_Omega21, READ)} )
hOmega22=FunctionSpace(mesh,'DG', 0)
I_Omega22 = Function(hOmega22)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega22), {'f': (I_Omega22, WRITE)} )
I_cg_Omega22 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega22, RW), 'B': (I_Omega22, READ)} )
hOmega23=FunctionSpace(mesh,'DG', 0)
I_Omega23 = Function(hOmega23)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega23), {'f': (I_Omega23, WRITE)} )
I_cg_Omega23 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega23, RW), 'B': (I_Omega23, READ)} )
hOmega24=FunctionSpace(mesh,'DG', 0)
I_Omega24 = Function(hOmega24)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega24), {'f': (I_Omega24, WRITE)} )
I_cg_Omega24 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega24, RW), 'B': (I_Omega24, READ)} )
hOmega25=FunctionSpace(mesh,'DG', 0)
I_Omega25 = Function(hOmega25)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega25), {'f': (I_Omega25, WRITE)} )
I_cg_Omega25 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega25, RW), 'B': (I_Omega25, READ)} )
hOmega26=FunctionSpace(mesh,'DG', 0)
I_Omega26 = Function(hOmega26)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega26), {'f': (I_Omega26, WRITE)} )
I_cg_Omega26 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega26, RW), 'B': (I_Omega26, READ)} )
hOmega27=FunctionSpace(mesh,'DG', 0)
I_Omega27 = Function(hOmega27)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega27), {'f': (I_Omega27, WRITE)} )
I_cg_Omega27 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega27, RW), 'B': (I_Omega27, READ)} )
hOmega28=FunctionSpace(mesh,'DG', 0)
I_Omega28 = Function(hOmega28)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega28), {'f': (I_Omega28, WRITE)} )
I_cg_Omega28 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega28, RW), 'B': (I_Omega28, READ)} )
hOmega29=FunctionSpace(mesh,'DG', 0)
I_Omega29 = Function(hOmega29)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega29), {'f': (I_Omega29, WRITE)} )
I_cg_Omega29 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega29, RW), 'B': (I_Omega29, READ)} )
hOmega30=FunctionSpace(mesh,'DG', 0)
I_Omega30 = Function(hOmega30)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega30), {'f': (I_Omega30, WRITE)} )
I_cg_Omega30 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega30, RW), 'B': (I_Omega30, READ)} )
hOmega31=FunctionSpace(mesh,'DG', 0)
I_Omega31 = Function(hOmega31)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega31), {'f': (I_Omega31, WRITE)} )
I_cg_Omega31 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega31, RW), 'B': (I_Omega31, READ)} )
hOmega32=FunctionSpace(mesh,'DG', 0)
I_Omega32 = Function(hOmega32)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega32), {'f': (I_Omega32, WRITE)} )
I_cg_Omega32 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega32, RW), 'B': (I_Omega32, READ)} )
hOmega33=FunctionSpace(mesh,'DG', 0)
I_Omega33 = Function(hOmega33)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega33), {'f': (I_Omega33, WRITE)} )
I_cg_Omega33 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega33, RW), 'B': (I_Omega33, READ)} )
hOmega34=FunctionSpace(mesh,'DG', 0)
I_Omega34 = Function(hOmega34)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega34), {'f': (I_Omega34, WRITE)} )
I_cg_Omega34 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega34, RW), 'B': (I_Omega34, READ)} )
hOmega35=FunctionSpace(mesh,'DG', 0)
I_Omega35 = Function(hOmega35)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega35), {'f': (I_Omega35, WRITE)} )
I_cg_Omega35 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega35, RW), 'B': (I_Omega35, READ)} )
hOmega36=FunctionSpace(mesh,'DG', 0)
I_Omega36 = Function(hOmega36)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega36), {'f': (I_Omega36, WRITE)} )
I_cg_Omega36 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega36, RW), 'B': (I_Omega36, READ)} )
hOmega37=FunctionSpace(mesh,'DG', 0)
I_Omega37 = Function(hOmega37)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega37), {'f': (I_Omega37, WRITE)} )
I_cg_Omega37 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega37, RW), 'B': (I_Omega37, READ)} )
hOmega38=FunctionSpace(mesh,'DG', 0)
I_Omega38 = Function(hOmega38)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega38), {'f': (I_Omega38, WRITE)} )
I_cg_Omega38 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega38, RW), 'B': (I_Omega38, READ)} )
hOmega39=FunctionSpace(mesh,'DG', 0)
I_Omega39 = Function(hOmega39)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega39), {'f': (I_Omega39, WRITE)} )
I_cg_Omega39 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega39, RW), 'B': (I_Omega39, READ)} )
hOmega40=FunctionSpace(mesh,'DG', 0)
I_Omega40 = Function(hOmega40)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega40), {'f': (I_Omega40, WRITE)} )
I_cg_Omega40 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega40, RW), 'B': (I_Omega40, READ)} )
hOmega41=FunctionSpace(mesh,'DG', 0)
I_Omega41 = Function(hOmega41)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega41), {'f': (I_Omega41, WRITE)} )
I_cg_Omega41 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega41, RW), 'B': (I_Omega41, READ)} )
hOmega42=FunctionSpace(mesh,'DG', 0)
I_Omega42 = Function(hOmega42)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega42), {'f': (I_Omega42, WRITE)} )
I_cg_Omega42 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega42, RW), 'B': (I_Omega42, READ)} )
hOmega43=FunctionSpace(mesh,'DG', 0)
I_Omega43 = Function(hOmega43)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega43), {'f': (I_Omega43, WRITE)} )
I_cg_Omega43 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega43, RW), 'B': (I_Omega43, READ)} )
hOmega44=FunctionSpace(mesh,'DG', 0)
I_Omega44 = Function(hOmega44)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega44), {'f': (I_Omega44, WRITE)} )
I_cg_Omega44 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega44, RW), 'B': (I_Omega44, READ)} )
hOmega45=FunctionSpace(mesh,'DG', 0)
I_Omega45 = Function(hOmega45)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega45), {'f': (I_Omega45, WRITE)} )
I_cg_Omega45 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega45, RW), 'B': (I_Omega45, READ)} )
hOmega46=FunctionSpace(mesh,'DG', 0)
I_Omega46 = Function(hOmega46)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega46), {'f': (I_Omega46, WRITE)} )
I_cg_Omega46 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega46, RW), 'B': (I_Omega46, READ)} )
hOmega47=FunctionSpace(mesh,'DG', 0)
I_Omega47 = Function(hOmega47)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega47), {'f': (I_Omega47, WRITE)} )
I_cg_Omega47 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega47, RW), 'B': (I_Omega47, READ)} )
hOmega48=FunctionSpace(mesh,'DG', 0)
I_Omega48 = Function(hOmega48)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega48), {'f': (I_Omega48, WRITE)} )
I_cg_Omega48 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega48, RW), 'B': (I_Omega48, READ)} )
hOmega49=FunctionSpace(mesh,'DG', 0)
I_Omega49 = Function(hOmega49)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega49), {'f': (I_Omega49, WRITE)} )
I_cg_Omega49 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega49, RW), 'B': (I_Omega49, READ)} )
hOmega50=FunctionSpace(mesh,'DG', 0)
I_Omega50 = Function(hOmega50)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega50), {'f': (I_Omega50, WRITE)} )
I_cg_Omega50 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega50, RW), 'B': (I_Omega50, READ)} )
hOmega51=FunctionSpace(mesh,'DG', 0)
I_Omega51 = Function(hOmega51)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega51), {'f': (I_Omega51, WRITE)} )
I_cg_Omega51 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega51, RW), 'B': (I_Omega51, READ)} )
hOmega52=FunctionSpace(mesh,'DG', 0)
I_Omega52 = Function(hOmega52)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega52), {'f': (I_Omega52, WRITE)} )
I_cg_Omega52 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega52, RW), 'B': (I_Omega52, READ)} )
hOmega53=FunctionSpace(mesh,'DG', 0)
I_Omega53 = Function(hOmega53)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega53), {'f': (I_Omega53, WRITE)} )
I_cg_Omega53 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega53, RW), 'B': (I_Omega53, READ)} )
hOmega54=FunctionSpace(mesh,'DG', 0)
I_Omega54 = Function(hOmega54)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega54), {'f': (I_Omega54, WRITE)} )
I_cg_Omega54 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega54, RW), 'B': (I_Omega54, READ)} )
hOmega55=FunctionSpace(mesh,'DG', 0)
I_Omega55 = Function(hOmega55)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega55), {'f': (I_Omega55, WRITE)} )
I_cg_Omega55 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega55, RW), 'B': (I_Omega55, READ)} )
hOmega56=FunctionSpace(mesh,'DG', 0)
I_Omega56 = Function(hOmega56)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega56), {'f': (I_Omega56, WRITE)} )
I_cg_Omega56 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega56, RW), 'B': (I_Omega56, READ)} )
hOmega57=FunctionSpace(mesh,'DG', 0)
I_Omega57 = Function(hOmega57)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega57), {'f': (I_Omega57, WRITE)} )
I_cg_Omega57 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega57, RW), 'B': (I_Omega57, READ)} )
hOmega58=FunctionSpace(mesh,'DG', 0)
I_Omega58 = Function(hOmega58)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega58), {'f': (I_Omega58, WRITE)} )
I_cg_Omega58 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega58, RW), 'B': (I_Omega58, READ)} )
hOmega59=FunctionSpace(mesh,'DG', 0)
I_Omega59 = Function(hOmega59)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega59), {'f': (I_Omega59, WRITE)} )
I_cg_Omega59 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega59, RW), 'B': (I_Omega59, READ)} )
hOmega60=FunctionSpace(mesh,'DG', 0)
I_Omega60 = Function(hOmega60)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega60), {'f': (I_Omega60, WRITE)} )
I_cg_Omega60 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega60, RW), 'B': (I_Omega60, READ)} )
hOmega61=FunctionSpace(mesh,'DG', 0)
I_Omega61 = Function(hOmega61)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega61), {'f': (I_Omega61, WRITE)} )
I_cg_Omega61 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega61, RW), 'B': (I_Omega61, READ)} )
hOmega62=FunctionSpace(mesh,'DG', 0)
I_Omega62 = Function(hOmega62)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega62), {'f': (I_Omega62, WRITE)} )
I_cg_Omega62 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega62, RW), 'B': (I_Omega62, READ)} )
hOmega63=FunctionSpace(mesh,'DG', 0)
I_Omega63 = Function(hOmega63)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega63), {'f': (I_Omega63, WRITE)} )
I_cg_Omega63 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega63, RW), 'B': (I_Omega63, READ)} )
hOmega64=FunctionSpace(mesh,'DG', 0)
I_Omega64 = Function(hOmega64)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega64), {'f': (I_Omega64, WRITE)} )
I_cg_Omega64 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega64, RW), 'B': (I_Omega64, READ)} )

bcs=[]
for surface, value in dirichletBCs.items():
    bcs.append(DirichletBC(V,value,surface))
    
BC_Omega1_only=MyBC(V,0,I_cg_Omega1)
BC_Omega2_only=MyBC(V,0,I_cg_Omega2)
BC_Omega3_only=MyBC(V,0,I_cg_Omega3)
BC_Omega4_only=MyBC(V,0,I_cg_Omega4)
BC_Omega5_only=MyBC(V,0,I_cg_Omega5)
BC_Omega6_only=MyBC(V,0,I_cg_Omega6)
BC_Omega7_only=MyBC(V,0,I_cg_Omega7)
BC_Omega8_only=MyBC(V,0,I_cg_Omega8)
BC_Omega9_only=MyBC(V,0,I_cg_Omega9)
BC_Omega10_only=MyBC(V,0,I_cg_Omega10)
BC_Omega11_only=MyBC(V,0,I_cg_Omega11)
BC_Omega12_only=MyBC(V,0,I_cg_Omega12)
BC_Omega13_only=MyBC(V,0,I_cg_Omega13)
BC_Omega14_only=MyBC(V,0,I_cg_Omega14)
BC_Omega15_only=MyBC(V,0,I_cg_Omega15)
BC_Omega16_only=MyBC(V,0,I_cg_Omega16)
BC_Omega17_only=MyBC(V,0,I_cg_Omega17)
BC_Omega18_only=MyBC(V,0,I_cg_Omega18)
BC_Omega19_only=MyBC(V,0,I_cg_Omega19)
BC_Omega20_only=MyBC(V,0,I_cg_Omega20)
BC_Omega21_only=MyBC(V,0,I_cg_Omega21)
BC_Omega22_only=MyBC(V,0,I_cg_Omega22)
BC_Omega23_only=MyBC(V,0,I_cg_Omega23)
BC_Omega24_only=MyBC(V,0,I_cg_Omega24)
BC_Omega25_only=MyBC(V,0,I_cg_Omega25)
BC_Omega26_only=MyBC(V,0,I_cg_Omega26)
BC_Omega27_only=MyBC(V,0,I_cg_Omega27)
BC_Omega28_only=MyBC(V,0,I_cg_Omega28)
BC_Omega29_only=MyBC(V,0,I_cg_Omega29)
BC_Omega30_only=MyBC(V,0,I_cg_Omega30)
BC_Omega31_only=MyBC(V,0,I_cg_Omega31)
BC_Omega32_only=MyBC(V,0,I_cg_Omega32)
BC_Omega33_only=MyBC(V,0,I_cg_Omega33)
BC_Omega34_only=MyBC(V,0,I_cg_Omega34)
BC_Omega35_only=MyBC(V,0,I_cg_Omega35)
BC_Omega36_only=MyBC(V,0,I_cg_Omega36)
BC_Omega37_only=MyBC(V,0,I_cg_Omega37)
BC_Omega38_only=MyBC(V,0,I_cg_Omega38)
BC_Omega39_only=MyBC(V,0,I_cg_Omega39)
BC_Omega40_only=MyBC(V,0,I_cg_Omega40)
BC_Omega41_only=MyBC(V,0,I_cg_Omega41)
BC_Omega42_only=MyBC(V,0,I_cg_Omega42)
BC_Omega43_only=MyBC(V,0,I_cg_Omega43)
BC_Omega44_only=MyBC(V,0,I_cg_Omega44)
BC_Omega45_only=MyBC(V,0,I_cg_Omega45)
BC_Omega46_only=MyBC(V,0,I_cg_Omega46)
BC_Omega47_only=MyBC(V,0,I_cg_Omega47)
BC_Omega48_only=MyBC(V,0,I_cg_Omega48)
BC_Omega49_only=MyBC(V,0,I_cg_Omega49)
BC_Omega50_only=MyBC(V,0,I_cg_Omega50)
BC_Omega51_only=MyBC(V,0,I_cg_Omega51)
BC_Omega52_only=MyBC(V,0,I_cg_Omega52)
BC_Omega53_only=MyBC(V,0,I_cg_Omega53)
BC_Omega54_only=MyBC(V,0,I_cg_Omega54)
BC_Omega55_only=MyBC(V,0,I_cg_Omega55)
BC_Omega56_only=MyBC(V,0,I_cg_Omega56)
BC_Omega57_only=MyBC(V,0,I_cg_Omega57)
BC_Omega58_only=MyBC(V,0,I_cg_Omega58)
BC_Omega59_only=MyBC(V,0,I_cg_Omega59)
BC_Omega60_only=MyBC(V,0,I_cg_Omega60)
BC_Omega61_only=MyBC(V,0,I_cg_Omega61)
BC_Omega62_only=MyBC(V,0,I_cg_Omega62)
BC_Omega63_only=MyBC(V,0,I_cg_Omega63)
BC_Omega64_only=MyBC(V,0,I_cg_Omega64)
coord=SpatialCoordinate(mesh)
uOmega1=Function(V)
uOmega1.interpolate(1*coord[0]*coord[1])
uOmega1_old=Function(V)
uOmega1_old.interpolate(1*coord[0]*coord[1])
uOmega2=Function(V)
uOmega2.interpolate(1*coord[0]*coord[1])
uOmega2_old=Function(V)
uOmega2_old.interpolate(1*coord[0]*coord[1])
uOmega3=Function(V)
uOmega3.interpolate(1*coord[0]*coord[1])
uOmega3_old=Function(V)
uOmega3_old.interpolate(1*coord[0]*coord[1])
uOmega4=Function(V)
uOmega4.interpolate(1*coord[0]*coord[1])
uOmega4_old=Function(V)
uOmega4_old.interpolate(1*coord[0]*coord[1])
uOmega5=Function(V)
uOmega5.interpolate(1*coord[0]*coord[1])
uOmega5_old=Function(V)
uOmega5_old.interpolate(1*coord[0]*coord[1])
uOmega6=Function(V)
uOmega6.interpolate(1*coord[0]*coord[1])
uOmega6_old=Function(V)
uOmega6_old.interpolate(1*coord[0]*coord[1])
uOmega7=Function(V)
uOmega7.interpolate(1*coord[0]*coord[1])
uOmega7_old=Function(V)
uOmega7_old.interpolate(1*coord[0]*coord[1])
uOmega8=Function(V)
uOmega8.interpolate(1*coord[0]*coord[1])
uOmega8_old=Function(V)
uOmega8_old.interpolate(1*coord[0]*coord[1])
uOmega9=Function(V)
uOmega9.interpolate(1*coord[0]*coord[1])
uOmega9_old=Function(V)
uOmega9_old.interpolate(1*coord[0]*coord[1])
uOmega10=Function(V)
uOmega10.interpolate(1*coord[0]*coord[1])
uOmega10_old=Function(V)
uOmega10_old.interpolate(1*coord[0]*coord[1])
uOmega11=Function(V)
uOmega11.interpolate(1*coord[0]*coord[1])
uOmega11_old=Function(V)
uOmega11_old.interpolate(1*coord[0]*coord[1])
uOmega12=Function(V)
uOmega12.interpolate(1*coord[0]*coord[1])
uOmega12_old=Function(V)
uOmega12_old.interpolate(1*coord[0]*coord[1])
uOmega13=Function(V)
uOmega13.interpolate(1*coord[0]*coord[1])
uOmega13_old=Function(V)
uOmega13_old.interpolate(1*coord[0]*coord[1])
uOmega14=Function(V)
uOmega14.interpolate(1*coord[0]*coord[1])
uOmega14_old=Function(V)
uOmega14_old.interpolate(1*coord[0]*coord[1])
uOmega15=Function(V)
uOmega15.interpolate(1*coord[0]*coord[1])
uOmega15_old=Function(V)
uOmega15_old.interpolate(1*coord[0]*coord[1])
uOmega16=Function(V)
uOmega16.interpolate(1*coord[0]*coord[1])
uOmega16_old=Function(V)
uOmega16_old.interpolate(1*coord[0]*coord[1])
uOmega17=Function(V)
uOmega17.interpolate(1*coord[0]*coord[1])
uOmega17_old=Function(V)
uOmega17_old.interpolate(1*coord[0]*coord[1])
uOmega18=Function(V)
uOmega18.interpolate(1*coord[0]*coord[1])
uOmega18_old=Function(V)
uOmega18_old.interpolate(1*coord[0]*coord[1])
uOmega19=Function(V)
uOmega19.interpolate(1*coord[0]*coord[1])
uOmega19_old=Function(V)
uOmega19_old.interpolate(1*coord[0]*coord[1])
uOmega20=Function(V)
uOmega20.interpolate(1*coord[0]*coord[1])
uOmega20_old=Function(V)
uOmega20_old.interpolate(1*coord[0]*coord[1])
uOmega21=Function(V)
uOmega21.interpolate(1*coord[0]*coord[1])
uOmega21_old=Function(V)
uOmega21_old.interpolate(1*coord[0]*coord[1])
uOmega22=Function(V)
uOmega22.interpolate(1*coord[0]*coord[1])
uOmega22_old=Function(V)
uOmega22_old.interpolate(1*coord[0]*coord[1])
uOmega23=Function(V)
uOmega23.interpolate(1*coord[0]*coord[1])
uOmega23_old=Function(V)
uOmega23_old.interpolate(1*coord[0]*coord[1])
uOmega24=Function(V)
uOmega24.interpolate(1*coord[0]*coord[1])
uOmega24_old=Function(V)
uOmega24_old.interpolate(1*coord[0]*coord[1])
uOmega25=Function(V)
uOmega25.interpolate(1*coord[0]*coord[1])
uOmega25_old=Function(V)
uOmega25_old.interpolate(1*coord[0]*coord[1])
uOmega26=Function(V)
uOmega26.interpolate(1*coord[0]*coord[1])
uOmega26_old=Function(V)
uOmega26_old.interpolate(1*coord[0]*coord[1])
uOmega27=Function(V)
uOmega27.interpolate(1*coord[0]*coord[1])
uOmega27_old=Function(V)
uOmega27_old.interpolate(1*coord[0]*coord[1])
uOmega28=Function(V)
uOmega28.interpolate(1*coord[0]*coord[1])
uOmega28_old=Function(V)
uOmega28_old.interpolate(1*coord[0]*coord[1])
uOmega29=Function(V)
uOmega29.interpolate(1*coord[0]*coord[1])
uOmega29_old=Function(V)
uOmega29_old.interpolate(1*coord[0]*coord[1])
uOmega30=Function(V)
uOmega30.interpolate(1*coord[0]*coord[1])
uOmega30_old=Function(V)
uOmega30_old.interpolate(1*coord[0]*coord[1])
uOmega31=Function(V)
uOmega31.interpolate(1*coord[0]*coord[1])
uOmega31_old=Function(V)
uOmega31_old.interpolate(1*coord[0]*coord[1])
uOmega32=Function(V)
uOmega32.interpolate(1*coord[0]*coord[1])
uOmega32_old=Function(V)
uOmega32_old.interpolate(1*coord[0]*coord[1])
uOmega33=Function(V)
uOmega33.interpolate(1*coord[0]*coord[1])
uOmega33_old=Function(V)
uOmega33_old.interpolate(1*coord[0]*coord[1])
uOmega34=Function(V)
uOmega34.interpolate(1*coord[0]*coord[1])
uOmega34_old=Function(V)
uOmega34_old.interpolate(1*coord[0]*coord[1])
uOmega35=Function(V)
uOmega35.interpolate(1*coord[0]*coord[1])
uOmega35_old=Function(V)
uOmega35_old.interpolate(1*coord[0]*coord[1])
uOmega36=Function(V)
uOmega36.interpolate(1*coord[0]*coord[1])
uOmega36_old=Function(V)
uOmega36_old.interpolate(1*coord[0]*coord[1])
uOmega37=Function(V)
uOmega37.interpolate(1*coord[0]*coord[1])
uOmega37_old=Function(V)
uOmega37_old.interpolate(1*coord[0]*coord[1])
uOmega38=Function(V)
uOmega38.interpolate(1*coord[0]*coord[1])
uOmega38_old=Function(V)
uOmega38_old.interpolate(1*coord[0]*coord[1])
uOmega39=Function(V)
uOmega39.interpolate(1*coord[0]*coord[1])
uOmega39_old=Function(V)
uOmega39_old.interpolate(1*coord[0]*coord[1])
uOmega40=Function(V)
uOmega40.interpolate(1*coord[0]*coord[1])
uOmega40_old=Function(V)
uOmega40_old.interpolate(1*coord[0]*coord[1])
uOmega41=Function(V)
uOmega41.interpolate(1*coord[0]*coord[1])
uOmega41_old=Function(V)
uOmega41_old.interpolate(1*coord[0]*coord[1])
uOmega42=Function(V)
uOmega42.interpolate(1*coord[0]*coord[1])
uOmega42_old=Function(V)
uOmega42_old.interpolate(1*coord[0]*coord[1])
uOmega43=Function(V)
uOmega43.interpolate(1*coord[0]*coord[1])
uOmega43_old=Function(V)
uOmega43_old.interpolate(1*coord[0]*coord[1])
uOmega44=Function(V)
uOmega44.interpolate(1*coord[0]*coord[1])
uOmega44_old=Function(V)
uOmega44_old.interpolate(1*coord[0]*coord[1])
uOmega45=Function(V)
uOmega45.interpolate(1*coord[0]*coord[1])
uOmega45_old=Function(V)
uOmega45_old.interpolate(1*coord[0]*coord[1])
uOmega46=Function(V)
uOmega46.interpolate(1*coord[0]*coord[1])
uOmega46_old=Function(V)
uOmega46_old.interpolate(1*coord[0]*coord[1])
uOmega47=Function(V)
uOmega47.interpolate(1*coord[0]*coord[1])
uOmega47_old=Function(V)
uOmega47_old.interpolate(1*coord[0]*coord[1])
uOmega48=Function(V)
uOmega48.interpolate(1*coord[0]*coord[1])
uOmega48_old=Function(V)
uOmega48_old.interpolate(1*coord[0]*coord[1])
uOmega49=Function(V)
uOmega49.interpolate(1*coord[0]*coord[1])
uOmega49_old=Function(V)
uOmega49_old.interpolate(1*coord[0]*coord[1])
uOmega50=Function(V)
uOmega50.interpolate(1*coord[0]*coord[1])
uOmega50_old=Function(V)
uOmega50_old.interpolate(1*coord[0]*coord[1])
uOmega51=Function(V)
uOmega51.interpolate(1*coord[0]*coord[1])
uOmega51_old=Function(V)
uOmega51_old.interpolate(1*coord[0]*coord[1])
uOmega52=Function(V)
uOmega52.interpolate(1*coord[0]*coord[1])
uOmega52_old=Function(V)
uOmega52_old.interpolate(1*coord[0]*coord[1])
uOmega53=Function(V)
uOmega53.interpolate(1*coord[0]*coord[1])
uOmega53_old=Function(V)
uOmega53_old.interpolate(1*coord[0]*coord[1])
uOmega54=Function(V)
uOmega54.interpolate(1*coord[0]*coord[1])
uOmega54_old=Function(V)
uOmega54_old.interpolate(1*coord[0]*coord[1])
uOmega55=Function(V)
uOmega55.interpolate(1*coord[0]*coord[1])
uOmega55_old=Function(V)
uOmega55_old.interpolate(1*coord[0]*coord[1])
uOmega56=Function(V)
uOmega56.interpolate(1*coord[0]*coord[1])
uOmega56_old=Function(V)
uOmega56_old.interpolate(1*coord[0]*coord[1])
uOmega57=Function(V)
uOmega57.interpolate(1*coord[0]*coord[1])
uOmega57_old=Function(V)
uOmega57_old.interpolate(1*coord[0]*coord[1])
uOmega58=Function(V)
uOmega58.interpolate(1*coord[0]*coord[1])
uOmega58_old=Function(V)
uOmega58_old.interpolate(1*coord[0]*coord[1])
uOmega59=Function(V)
uOmega59.interpolate(1*coord[0]*coord[1])
uOmega59_old=Function(V)
uOmega59_old.interpolate(1*coord[0]*coord[1])
uOmega60=Function(V)
uOmega60.interpolate(1*coord[0]*coord[1])
uOmega60_old=Function(V)
uOmega60_old.interpolate(1*coord[0]*coord[1])
uOmega61=Function(V)
uOmega61.interpolate(1*coord[0]*coord[1])
uOmega61_old=Function(V)
uOmega61_old.interpolate(1*coord[0]*coord[1])
uOmega62=Function(V)
uOmega62.interpolate(1*coord[0]*coord[1])
uOmega62_old=Function(V)
uOmega62_old.interpolate(1*coord[0]*coord[1])
uOmega63=Function(V)
uOmega63.interpolate(1*coord[0]*coord[1])
uOmega63_old=Function(V)
uOmega63_old.interpolate(1*coord[0]*coord[1])
uOmega64=Function(V)
uOmega64.interpolate(1*coord[0]*coord[1])
uOmega64_old=Function(V)
uOmega64_old.interpolate(1*coord[0]*coord[1])
aOmega1 = (dot(grad(u), grad(v))) * dx(Omega1)
LOmega1 = f * v * dx(Omega1)
aOmega2 = (dot(grad(u), grad(v))) * dx(Omega2)
LOmega2 = f * v * dx(Omega2)
aOmega3 = (dot(grad(u), grad(v))) * dx(Omega3)
LOmega3 = f * v * dx(Omega3)
aOmega4 = (dot(grad(u), grad(v))) * dx(Omega4)
LOmega4 = f * v * dx(Omega4)
aOmega5 = (dot(grad(u), grad(v))) * dx(Omega5)
LOmega5 = f * v * dx(Omega5)
aOmega6 = (dot(grad(u), grad(v))) * dx(Omega6)
LOmega6 = f * v * dx(Omega6)
aOmega7 = (dot(grad(u), grad(v))) * dx(Omega7)
LOmega7 = f * v * dx(Omega7)
aOmega8 = (dot(grad(u), grad(v))) * dx(Omega8)
LOmega8 = f * v * dx(Omega8)
aOmega9 = (dot(grad(u), grad(v))) * dx(Omega9)
LOmega9 = f * v * dx(Omega9)
aOmega10 = (dot(grad(u), grad(v))) * dx(Omega10)
LOmega10 = f * v * dx(Omega10)
aOmega11 = (dot(grad(u), grad(v))) * dx(Omega11)
LOmega11 = f * v * dx(Omega11)
aOmega12 = (dot(grad(u), grad(v))) * dx(Omega12)
LOmega12 = f * v * dx(Omega12)
aOmega13 = (dot(grad(u), grad(v))) * dx(Omega13)
LOmega13 = f * v * dx(Omega13)
aOmega14 = (dot(grad(u), grad(v))) * dx(Omega14)
LOmega14 = f * v * dx(Omega14)
aOmega15 = (dot(grad(u), grad(v))) * dx(Omega15)
LOmega15 = f * v * dx(Omega15)
aOmega16 = (dot(grad(u), grad(v))) * dx(Omega16)
LOmega16 = f * v * dx(Omega16)
aOmega17 = (dot(grad(u), grad(v))) * dx(Omega17)
LOmega17 = f * v * dx(Omega17)
aOmega18 = (dot(grad(u), grad(v))) * dx(Omega18)
LOmega18 = f * v * dx(Omega18)
aOmega19 = (dot(grad(u), grad(v))) * dx(Omega19)
LOmega19 = f * v * dx(Omega19)
aOmega20 = (dot(grad(u), grad(v))) * dx(Omega20)
LOmega20 = f * v * dx(Omega20)
aOmega21 = (dot(grad(u), grad(v))) * dx(Omega21)
LOmega21 = f * v * dx(Omega21)
aOmega22 = (dot(grad(u), grad(v))) * dx(Omega22)
LOmega22 = f * v * dx(Omega22)
aOmega23 = (dot(grad(u), grad(v))) * dx(Omega23)
LOmega23 = f * v * dx(Omega23)
aOmega24 = (dot(grad(u), grad(v))) * dx(Omega24)
LOmega24 = f * v * dx(Omega24)
aOmega25 = (dot(grad(u), grad(v))) * dx(Omega25)
LOmega25 = f * v * dx(Omega25)
aOmega26 = (dot(grad(u), grad(v))) * dx(Omega26)
LOmega26 = f * v * dx(Omega26)
aOmega27 = (dot(grad(u), grad(v))) * dx(Omega27)
LOmega27 = f * v * dx(Omega27)
aOmega28 = (dot(grad(u), grad(v))) * dx(Omega28)
LOmega28 = f * v * dx(Omega28)
aOmega29 = (dot(grad(u), grad(v))) * dx(Omega29)
LOmega29 = f * v * dx(Omega29)
aOmega30 = (dot(grad(u), grad(v))) * dx(Omega30)
LOmega30 = f * v * dx(Omega30)
aOmega31 = (dot(grad(u), grad(v))) * dx(Omega31)
LOmega31 = f * v * dx(Omega31)
aOmega32 = (dot(grad(u), grad(v))) * dx(Omega32)
LOmega32 = f * v * dx(Omega32)
aOmega33 = (dot(grad(u), grad(v))) * dx(Omega33)
LOmega33 = f * v * dx(Omega33)
aOmega34 = (dot(grad(u), grad(v))) * dx(Omega34)
LOmega34 = f * v * dx(Omega34)
aOmega35 = (dot(grad(u), grad(v))) * dx(Omega35)
LOmega35 = f * v * dx(Omega35)
aOmega36 = (dot(grad(u), grad(v))) * dx(Omega36)
LOmega36 = f * v * dx(Omega36)
aOmega37 = (dot(grad(u), grad(v))) * dx(Omega37)
LOmega37 = f * v * dx(Omega37)
aOmega38 = (dot(grad(u), grad(v))) * dx(Omega38)
LOmega38 = f * v * dx(Omega38)
aOmega39 = (dot(grad(u), grad(v))) * dx(Omega39)
LOmega39 = f * v * dx(Omega39)
aOmega40 = (dot(grad(u), grad(v))) * dx(Omega40)
LOmega40 = f * v * dx(Omega40)
aOmega41 = (dot(grad(u), grad(v))) * dx(Omega41)
LOmega41 = f * v * dx(Omega41)
aOmega42 = (dot(grad(u), grad(v))) * dx(Omega42)
LOmega42 = f * v * dx(Omega42)
aOmega43 = (dot(grad(u), grad(v))) * dx(Omega43)
LOmega43 = f * v * dx(Omega43)
aOmega44 = (dot(grad(u), grad(v))) * dx(Omega44)
LOmega44 = f * v * dx(Omega44)
aOmega45 = (dot(grad(u), grad(v))) * dx(Omega45)
LOmega45 = f * v * dx(Omega45)
aOmega46 = (dot(grad(u), grad(v))) * dx(Omega46)
LOmega46 = f * v * dx(Omega46)
aOmega47 = (dot(grad(u), grad(v))) * dx(Omega47)
LOmega47 = f * v * dx(Omega47)
aOmega48 = (dot(grad(u), grad(v))) * dx(Omega48)
LOmega48 = f * v * dx(Omega48)
aOmega49 = (dot(grad(u), grad(v))) * dx(Omega49)
LOmega49 = f * v * dx(Omega49)
aOmega50 = (dot(grad(u), grad(v))) * dx(Omega50)
LOmega50 = f * v * dx(Omega50)
aOmega51 = (dot(grad(u), grad(v))) * dx(Omega51)
LOmega51 = f * v * dx(Omega51)
aOmega52 = (dot(grad(u), grad(v))) * dx(Omega52)
LOmega52 = f * v * dx(Omega52)
aOmega53 = (dot(grad(u), grad(v))) * dx(Omega53)
LOmega53 = f * v * dx(Omega53)
aOmega54 = (dot(grad(u), grad(v))) * dx(Omega54)
LOmega54 = f * v * dx(Omega54)
aOmega55 = (dot(grad(u), grad(v))) * dx(Omega55)
LOmega55 = f * v * dx(Omega55)
aOmega56 = (dot(grad(u), grad(v))) * dx(Omega56)
LOmega56 = f * v * dx(Omega56)
aOmega57 = (dot(grad(u), grad(v))) * dx(Omega57)
LOmega57 = f * v * dx(Omega57)
aOmega58 = (dot(grad(u), grad(v))) * dx(Omega58)
LOmega58 = f * v * dx(Omega58)
aOmega59 = (dot(grad(u), grad(v))) * dx(Omega59)
LOmega59 = f * v * dx(Omega59)
aOmega60 = (dot(grad(u), grad(v))) * dx(Omega60)
LOmega60 = f * v * dx(Omega60)
aOmega61 = (dot(grad(u), grad(v))) * dx(Omega61)
LOmega61 = f * v * dx(Omega61)
aOmega62 = (dot(grad(u), grad(v))) * dx(Omega62)
LOmega62 = f * v * dx(Omega62)
aOmega63 = (dot(grad(u), grad(v))) * dx(Omega63)
LOmega63 = f * v * dx(Omega63)
aOmega64 = (dot(grad(u), grad(v))) * dx(Omega64)
LOmega64 = f * v * dx(Omega64)
bcOmega1=bcs.copy()
e1=uOmega1_old+uOmega2_old
bcOmega1.append(DirichletBC(V,e1,dOmega1nOmega2))
e2=uOmega1_old+uOmega9_old
bcOmega1.append(DirichletBC(V,e2,dOmega1nOmega9))
bcOmega1.append(BC_Omega1_only)
bcOmega2=bcs.copy()
e3=uOmega2_old+uOmega1_old
bcOmega2.append(DirichletBC(V,e3,dOmega2nOmega1))
e4=uOmega2_old+uOmega3_old
bcOmega2.append(DirichletBC(V,e4,dOmega2nOmega3))
e5=uOmega2_old+uOmega10_old
bcOmega2.append(DirichletBC(V,e5,dOmega2nOmega10))
bcOmega2.append(BC_Omega2_only)
bcOmega3=bcs.copy()
e6=uOmega3_old+uOmega2_old
bcOmega3.append(DirichletBC(V,e6,dOmega3nOmega2))
e7=uOmega3_old+uOmega4_old
bcOmega3.append(DirichletBC(V,e7,dOmega3nOmega4))
e8=uOmega3_old+uOmega11_old
bcOmega3.append(DirichletBC(V,e8,dOmega3nOmega11))
bcOmega3.append(BC_Omega3_only)
bcOmega4=bcs.copy()
e9=uOmega4_old+uOmega3_old
bcOmega4.append(DirichletBC(V,e9,dOmega4nOmega3))
e10=uOmega4_old+uOmega5_old
bcOmega4.append(DirichletBC(V,e10,dOmega4nOmega5))
e11=uOmega4_old+uOmega12_old
bcOmega4.append(DirichletBC(V,e11,dOmega4nOmega12))
bcOmega4.append(BC_Omega4_only)
bcOmega5=bcs.copy()
e12=uOmega5_old+uOmega4_old
bcOmega5.append(DirichletBC(V,e12,dOmega5nOmega4))
e13=uOmega5_old+uOmega6_old
bcOmega5.append(DirichletBC(V,e13,dOmega5nOmega6))
e14=uOmega5_old+uOmega13_old
bcOmega5.append(DirichletBC(V,e14,dOmega5nOmega13))
bcOmega5.append(BC_Omega5_only)
bcOmega6=bcs.copy()
e15=uOmega6_old+uOmega5_old
bcOmega6.append(DirichletBC(V,e15,dOmega6nOmega5))
e16=uOmega6_old+uOmega7_old
bcOmega6.append(DirichletBC(V,e16,dOmega6nOmega7))
e17=uOmega6_old+uOmega14_old
bcOmega6.append(DirichletBC(V,e17,dOmega6nOmega14))
bcOmega6.append(BC_Omega6_only)
bcOmega7=bcs.copy()
e18=uOmega7_old+uOmega6_old
bcOmega7.append(DirichletBC(V,e18,dOmega7nOmega6))
e19=uOmega7_old+uOmega8_old
bcOmega7.append(DirichletBC(V,e19,dOmega7nOmega8))
e20=uOmega7_old+uOmega15_old
bcOmega7.append(DirichletBC(V,e20,dOmega7nOmega15))
bcOmega7.append(BC_Omega7_only)
bcOmega8=bcs.copy()
e21=uOmega8_old+uOmega7_old
bcOmega8.append(DirichletBC(V,e21,dOmega8nOmega7))
e22=uOmega8_old+uOmega16_old
bcOmega8.append(DirichletBC(V,e22,dOmega8nOmega16))
bcOmega8.append(BC_Omega8_only)
bcOmega9=bcs.copy()
e23=uOmega9_old+uOmega1_old
bcOmega9.append(DirichletBC(V,e23,dOmega9nOmega1))
e24=uOmega9_old+uOmega10_old
bcOmega9.append(DirichletBC(V,e24,dOmega9nOmega10))
e25=uOmega9_old+uOmega17_old
bcOmega9.append(DirichletBC(V,e25,dOmega9nOmega17))
bcOmega9.append(BC_Omega9_only)
bcOmega10=bcs.copy()
e26=uOmega10_old+uOmega2_old
bcOmega10.append(DirichletBC(V,e26,dOmega10nOmega2))
e27=uOmega10_old+uOmega9_old
bcOmega10.append(DirichletBC(V,e27,dOmega10nOmega9))
e28=uOmega10_old+uOmega11_old
bcOmega10.append(DirichletBC(V,e28,dOmega10nOmega11))
e29=uOmega10_old+uOmega18_old
bcOmega10.append(DirichletBC(V,e29,dOmega10nOmega18))
bcOmega10.append(BC_Omega10_only)
bcOmega11=bcs.copy()
e30=uOmega11_old+uOmega3_old
bcOmega11.append(DirichletBC(V,e30,dOmega11nOmega3))
e31=uOmega11_old+uOmega10_old
bcOmega11.append(DirichletBC(V,e31,dOmega11nOmega10))
e32=uOmega11_old+uOmega12_old
bcOmega11.append(DirichletBC(V,e32,dOmega11nOmega12))
e33=uOmega11_old+uOmega19_old
bcOmega11.append(DirichletBC(V,e33,dOmega11nOmega19))
bcOmega11.append(BC_Omega11_only)
bcOmega12=bcs.copy()
e34=uOmega12_old+uOmega4_old
bcOmega12.append(DirichletBC(V,e34,dOmega12nOmega4))
e35=uOmega12_old+uOmega11_old
bcOmega12.append(DirichletBC(V,e35,dOmega12nOmega11))
e36=uOmega12_old+uOmega13_old
bcOmega12.append(DirichletBC(V,e36,dOmega12nOmega13))
e37=uOmega12_old+uOmega20_old
bcOmega12.append(DirichletBC(V,e37,dOmega12nOmega20))
bcOmega12.append(BC_Omega12_only)
bcOmega13=bcs.copy()
e38=uOmega13_old+uOmega5_old
bcOmega13.append(DirichletBC(V,e38,dOmega13nOmega5))
e39=uOmega13_old+uOmega12_old
bcOmega13.append(DirichletBC(V,e39,dOmega13nOmega12))
e40=uOmega13_old+uOmega14_old
bcOmega13.append(DirichletBC(V,e40,dOmega13nOmega14))
e41=uOmega13_old+uOmega21_old
bcOmega13.append(DirichletBC(V,e41,dOmega13nOmega21))
bcOmega13.append(BC_Omega13_only)
bcOmega14=bcs.copy()
e42=uOmega14_old+uOmega6_old
bcOmega14.append(DirichletBC(V,e42,dOmega14nOmega6))
e43=uOmega14_old+uOmega13_old
bcOmega14.append(DirichletBC(V,e43,dOmega14nOmega13))
e44=uOmega14_old+uOmega15_old
bcOmega14.append(DirichletBC(V,e44,dOmega14nOmega15))
e45=uOmega14_old+uOmega22_old
bcOmega14.append(DirichletBC(V,e45,dOmega14nOmega22))
bcOmega14.append(BC_Omega14_only)
bcOmega15=bcs.copy()
e46=uOmega15_old+uOmega7_old
bcOmega15.append(DirichletBC(V,e46,dOmega15nOmega7))
e47=uOmega15_old+uOmega14_old
bcOmega15.append(DirichletBC(V,e47,dOmega15nOmega14))
e48=uOmega15_old+uOmega16_old
bcOmega15.append(DirichletBC(V,e48,dOmega15nOmega16))
e49=uOmega15_old+uOmega23_old
bcOmega15.append(DirichletBC(V,e49,dOmega15nOmega23))
bcOmega15.append(BC_Omega15_only)
bcOmega16=bcs.copy()
e50=uOmega16_old+uOmega8_old
bcOmega16.append(DirichletBC(V,e50,dOmega16nOmega8))
e51=uOmega16_old+uOmega15_old
bcOmega16.append(DirichletBC(V,e51,dOmega16nOmega15))
e52=uOmega16_old+uOmega24_old
bcOmega16.append(DirichletBC(V,e52,dOmega16nOmega24))
bcOmega16.append(BC_Omega16_only)
bcOmega17=bcs.copy()
e53=uOmega17_old+uOmega9_old
bcOmega17.append(DirichletBC(V,e53,dOmega17nOmega9))
e54=uOmega17_old+uOmega18_old
bcOmega17.append(DirichletBC(V,e54,dOmega17nOmega18))
e55=uOmega17_old+uOmega25_old
bcOmega17.append(DirichletBC(V,e55,dOmega17nOmega25))
bcOmega17.append(BC_Omega17_only)
bcOmega18=bcs.copy()
e56=uOmega18_old+uOmega10_old
bcOmega18.append(DirichletBC(V,e56,dOmega18nOmega10))
e57=uOmega18_old+uOmega17_old
bcOmega18.append(DirichletBC(V,e57,dOmega18nOmega17))
e58=uOmega18_old+uOmega19_old
bcOmega18.append(DirichletBC(V,e58,dOmega18nOmega19))
e59=uOmega18_old+uOmega26_old
bcOmega18.append(DirichletBC(V,e59,dOmega18nOmega26))
bcOmega18.append(BC_Omega18_only)
bcOmega19=bcs.copy()
e60=uOmega19_old+uOmega11_old
bcOmega19.append(DirichletBC(V,e60,dOmega19nOmega11))
e61=uOmega19_old+uOmega18_old
bcOmega19.append(DirichletBC(V,e61,dOmega19nOmega18))
e62=uOmega19_old+uOmega20_old
bcOmega19.append(DirichletBC(V,e62,dOmega19nOmega20))
e63=uOmega19_old+uOmega27_old
bcOmega19.append(DirichletBC(V,e63,dOmega19nOmega27))
bcOmega19.append(BC_Omega19_only)
bcOmega20=bcs.copy()
e64=uOmega20_old+uOmega12_old
bcOmega20.append(DirichletBC(V,e64,dOmega20nOmega12))
e65=uOmega20_old+uOmega19_old
bcOmega20.append(DirichletBC(V,e65,dOmega20nOmega19))
e66=uOmega20_old+uOmega21_old
bcOmega20.append(DirichletBC(V,e66,dOmega20nOmega21))
e67=uOmega20_old+uOmega28_old
bcOmega20.append(DirichletBC(V,e67,dOmega20nOmega28))
bcOmega20.append(BC_Omega20_only)
bcOmega21=bcs.copy()
e68=uOmega21_old+uOmega13_old
bcOmega21.append(DirichletBC(V,e68,dOmega21nOmega13))
e69=uOmega21_old+uOmega20_old
bcOmega21.append(DirichletBC(V,e69,dOmega21nOmega20))
e70=uOmega21_old+uOmega22_old
bcOmega21.append(DirichletBC(V,e70,dOmega21nOmega22))
e71=uOmega21_old+uOmega29_old
bcOmega21.append(DirichletBC(V,e71,dOmega21nOmega29))
bcOmega21.append(BC_Omega21_only)
bcOmega22=bcs.copy()
e72=uOmega22_old+uOmega14_old
bcOmega22.append(DirichletBC(V,e72,dOmega22nOmega14))
e73=uOmega22_old+uOmega21_old
bcOmega22.append(DirichletBC(V,e73,dOmega22nOmega21))
e74=uOmega22_old+uOmega23_old
bcOmega22.append(DirichletBC(V,e74,dOmega22nOmega23))
e75=uOmega22_old+uOmega30_old
bcOmega22.append(DirichletBC(V,e75,dOmega22nOmega30))
bcOmega22.append(BC_Omega22_only)
bcOmega23=bcs.copy()
e76=uOmega23_old+uOmega15_old
bcOmega23.append(DirichletBC(V,e76,dOmega23nOmega15))
e77=uOmega23_old+uOmega22_old
bcOmega23.append(DirichletBC(V,e77,dOmega23nOmega22))
e78=uOmega23_old+uOmega24_old
bcOmega23.append(DirichletBC(V,e78,dOmega23nOmega24))
e79=uOmega23_old+uOmega31_old
bcOmega23.append(DirichletBC(V,e79,dOmega23nOmega31))
bcOmega23.append(BC_Omega23_only)
bcOmega24=bcs.copy()
e80=uOmega24_old+uOmega16_old
bcOmega24.append(DirichletBC(V,e80,dOmega24nOmega16))
e81=uOmega24_old+uOmega23_old
bcOmega24.append(DirichletBC(V,e81,dOmega24nOmega23))
e82=uOmega24_old+uOmega32_old
bcOmega24.append(DirichletBC(V,e82,dOmega24nOmega32))
bcOmega24.append(BC_Omega24_only)
bcOmega25=bcs.copy()
e83=uOmega25_old+uOmega17_old
bcOmega25.append(DirichletBC(V,e83,dOmega25nOmega17))
e84=uOmega25_old+uOmega26_old
bcOmega25.append(DirichletBC(V,e84,dOmega25nOmega26))
e85=uOmega25_old+uOmega33_old
bcOmega25.append(DirichletBC(V,e85,dOmega25nOmega33))
bcOmega25.append(BC_Omega25_only)
bcOmega26=bcs.copy()
e86=uOmega26_old+uOmega18_old
bcOmega26.append(DirichletBC(V,e86,dOmega26nOmega18))
e87=uOmega26_old+uOmega25_old
bcOmega26.append(DirichletBC(V,e87,dOmega26nOmega25))
e88=uOmega26_old+uOmega27_old
bcOmega26.append(DirichletBC(V,e88,dOmega26nOmega27))
e89=uOmega26_old+uOmega34_old
bcOmega26.append(DirichletBC(V,e89,dOmega26nOmega34))
bcOmega26.append(BC_Omega26_only)
bcOmega27=bcs.copy()
e90=uOmega27_old+uOmega19_old
bcOmega27.append(DirichletBC(V,e90,dOmega27nOmega19))
e91=uOmega27_old+uOmega26_old
bcOmega27.append(DirichletBC(V,e91,dOmega27nOmega26))
e92=uOmega27_old+uOmega28_old
bcOmega27.append(DirichletBC(V,e92,dOmega27nOmega28))
e93=uOmega27_old+uOmega35_old
bcOmega27.append(DirichletBC(V,e93,dOmega27nOmega35))
bcOmega27.append(BC_Omega27_only)
bcOmega28=bcs.copy()
e94=uOmega28_old+uOmega20_old
bcOmega28.append(DirichletBC(V,e94,dOmega28nOmega20))
e95=uOmega28_old+uOmega27_old
bcOmega28.append(DirichletBC(V,e95,dOmega28nOmega27))
e96=uOmega28_old+uOmega29_old
bcOmega28.append(DirichletBC(V,e96,dOmega28nOmega29))
e97=uOmega28_old+uOmega36_old
bcOmega28.append(DirichletBC(V,e97,dOmega28nOmega36))
bcOmega28.append(BC_Omega28_only)
bcOmega29=bcs.copy()
e98=uOmega29_old+uOmega21_old
bcOmega29.append(DirichletBC(V,e98,dOmega29nOmega21))
e99=uOmega29_old+uOmega28_old
bcOmega29.append(DirichletBC(V,e99,dOmega29nOmega28))
e100=uOmega29_old+uOmega30_old
bcOmega29.append(DirichletBC(V,e100,dOmega29nOmega30))
e101=uOmega29_old+uOmega37_old
bcOmega29.append(DirichletBC(V,e101,dOmega29nOmega37))
bcOmega29.append(BC_Omega29_only)
bcOmega30=bcs.copy()
e102=uOmega30_old+uOmega22_old
bcOmega30.append(DirichletBC(V,e102,dOmega30nOmega22))
e103=uOmega30_old+uOmega29_old
bcOmega30.append(DirichletBC(V,e103,dOmega30nOmega29))
e104=uOmega30_old+uOmega31_old
bcOmega30.append(DirichletBC(V,e104,dOmega30nOmega31))
e105=uOmega30_old+uOmega38_old
bcOmega30.append(DirichletBC(V,e105,dOmega30nOmega38))
bcOmega30.append(BC_Omega30_only)
bcOmega31=bcs.copy()
e106=uOmega31_old+uOmega23_old
bcOmega31.append(DirichletBC(V,e106,dOmega31nOmega23))
e107=uOmega31_old+uOmega30_old
bcOmega31.append(DirichletBC(V,e107,dOmega31nOmega30))
e108=uOmega31_old+uOmega32_old
bcOmega31.append(DirichletBC(V,e108,dOmega31nOmega32))
e109=uOmega31_old+uOmega39_old
bcOmega31.append(DirichletBC(V,e109,dOmega31nOmega39))
bcOmega31.append(BC_Omega31_only)
bcOmega32=bcs.copy()
e110=uOmega32_old+uOmega24_old
bcOmega32.append(DirichletBC(V,e110,dOmega32nOmega24))
e111=uOmega32_old+uOmega31_old
bcOmega32.append(DirichletBC(V,e111,dOmega32nOmega31))
e112=uOmega32_old+uOmega40_old
bcOmega32.append(DirichletBC(V,e112,dOmega32nOmega40))
bcOmega32.append(BC_Omega32_only)
bcOmega33=bcs.copy()
e113=uOmega33_old+uOmega25_old
bcOmega33.append(DirichletBC(V,e113,dOmega33nOmega25))
e114=uOmega33_old+uOmega34_old
bcOmega33.append(DirichletBC(V,e114,dOmega33nOmega34))
e115=uOmega33_old+uOmega41_old
bcOmega33.append(DirichletBC(V,e115,dOmega33nOmega41))
bcOmega33.append(BC_Omega33_only)
bcOmega34=bcs.copy()
e116=uOmega34_old+uOmega26_old
bcOmega34.append(DirichletBC(V,e116,dOmega34nOmega26))
e117=uOmega34_old+uOmega33_old
bcOmega34.append(DirichletBC(V,e117,dOmega34nOmega33))
e118=uOmega34_old+uOmega35_old
bcOmega34.append(DirichletBC(V,e118,dOmega34nOmega35))
e119=uOmega34_old+uOmega42_old
bcOmega34.append(DirichletBC(V,e119,dOmega34nOmega42))
bcOmega34.append(BC_Omega34_only)
bcOmega35=bcs.copy()
e120=uOmega35_old+uOmega27_old
bcOmega35.append(DirichletBC(V,e120,dOmega35nOmega27))
e121=uOmega35_old+uOmega34_old
bcOmega35.append(DirichletBC(V,e121,dOmega35nOmega34))
e122=uOmega35_old+uOmega36_old
bcOmega35.append(DirichletBC(V,e122,dOmega35nOmega36))
e123=uOmega35_old+uOmega43_old
bcOmega35.append(DirichletBC(V,e123,dOmega35nOmega43))
bcOmega35.append(BC_Omega35_only)
bcOmega36=bcs.copy()
e124=uOmega36_old+uOmega28_old
bcOmega36.append(DirichletBC(V,e124,dOmega36nOmega28))
e125=uOmega36_old+uOmega35_old
bcOmega36.append(DirichletBC(V,e125,dOmega36nOmega35))
e126=uOmega36_old+uOmega37_old
bcOmega36.append(DirichletBC(V,e126,dOmega36nOmega37))
e127=uOmega36_old+uOmega44_old
bcOmega36.append(DirichletBC(V,e127,dOmega36nOmega44))
bcOmega36.append(BC_Omega36_only)
bcOmega37=bcs.copy()
e128=uOmega37_old+uOmega29_old
bcOmega37.append(DirichletBC(V,e128,dOmega37nOmega29))
e129=uOmega37_old+uOmega36_old
bcOmega37.append(DirichletBC(V,e129,dOmega37nOmega36))
e130=uOmega37_old+uOmega38_old
bcOmega37.append(DirichletBC(V,e130,dOmega37nOmega38))
e131=uOmega37_old+uOmega45_old
bcOmega37.append(DirichletBC(V,e131,dOmega37nOmega45))
bcOmega37.append(BC_Omega37_only)
bcOmega38=bcs.copy()
e132=uOmega38_old+uOmega30_old
bcOmega38.append(DirichletBC(V,e132,dOmega38nOmega30))
e133=uOmega38_old+uOmega37_old
bcOmega38.append(DirichletBC(V,e133,dOmega38nOmega37))
e134=uOmega38_old+uOmega39_old
bcOmega38.append(DirichletBC(V,e134,dOmega38nOmega39))
e135=uOmega38_old+uOmega46_old
bcOmega38.append(DirichletBC(V,e135,dOmega38nOmega46))
bcOmega38.append(BC_Omega38_only)
bcOmega39=bcs.copy()
e136=uOmega39_old+uOmega31_old
bcOmega39.append(DirichletBC(V,e136,dOmega39nOmega31))
e137=uOmega39_old+uOmega38_old
bcOmega39.append(DirichletBC(V,e137,dOmega39nOmega38))
e138=uOmega39_old+uOmega40_old
bcOmega39.append(DirichletBC(V,e138,dOmega39nOmega40))
e139=uOmega39_old+uOmega47_old
bcOmega39.append(DirichletBC(V,e139,dOmega39nOmega47))
bcOmega39.append(BC_Omega39_only)
bcOmega40=bcs.copy()
e140=uOmega40_old+uOmega32_old
bcOmega40.append(DirichletBC(V,e140,dOmega40nOmega32))
e141=uOmega40_old+uOmega39_old
bcOmega40.append(DirichletBC(V,e141,dOmega40nOmega39))
e142=uOmega40_old+uOmega48_old
bcOmega40.append(DirichletBC(V,e142,dOmega40nOmega48))
bcOmega40.append(BC_Omega40_only)
bcOmega41=bcs.copy()
e143=uOmega41_old+uOmega33_old
bcOmega41.append(DirichletBC(V,e143,dOmega41nOmega33))
e144=uOmega41_old+uOmega42_old
bcOmega41.append(DirichletBC(V,e144,dOmega41nOmega42))
e145=uOmega41_old+uOmega49_old
bcOmega41.append(DirichletBC(V,e145,dOmega41nOmega49))
bcOmega41.append(BC_Omega41_only)
bcOmega42=bcs.copy()
e146=uOmega42_old+uOmega34_old
bcOmega42.append(DirichletBC(V,e146,dOmega42nOmega34))
e147=uOmega42_old+uOmega41_old
bcOmega42.append(DirichletBC(V,e147,dOmega42nOmega41))
e148=uOmega42_old+uOmega43_old
bcOmega42.append(DirichletBC(V,e148,dOmega42nOmega43))
e149=uOmega42_old+uOmega50_old
bcOmega42.append(DirichletBC(V,e149,dOmega42nOmega50))
bcOmega42.append(BC_Omega42_only)
bcOmega43=bcs.copy()
e150=uOmega43_old+uOmega35_old
bcOmega43.append(DirichletBC(V,e150,dOmega43nOmega35))
e151=uOmega43_old+uOmega42_old
bcOmega43.append(DirichletBC(V,e151,dOmega43nOmega42))
e152=uOmega43_old+uOmega44_old
bcOmega43.append(DirichletBC(V,e152,dOmega43nOmega44))
e153=uOmega43_old+uOmega51_old
bcOmega43.append(DirichletBC(V,e153,dOmega43nOmega51))
bcOmega43.append(BC_Omega43_only)
bcOmega44=bcs.copy()
e154=uOmega44_old+uOmega36_old
bcOmega44.append(DirichletBC(V,e154,dOmega44nOmega36))
e155=uOmega44_old+uOmega43_old
bcOmega44.append(DirichletBC(V,e155,dOmega44nOmega43))
e156=uOmega44_old+uOmega45_old
bcOmega44.append(DirichletBC(V,e156,dOmega44nOmega45))
e157=uOmega44_old+uOmega52_old
bcOmega44.append(DirichletBC(V,e157,dOmega44nOmega52))
bcOmega44.append(BC_Omega44_only)
bcOmega45=bcs.copy()
e158=uOmega45_old+uOmega37_old
bcOmega45.append(DirichletBC(V,e158,dOmega45nOmega37))
e159=uOmega45_old+uOmega44_old
bcOmega45.append(DirichletBC(V,e159,dOmega45nOmega44))
e160=uOmega45_old+uOmega46_old
bcOmega45.append(DirichletBC(V,e160,dOmega45nOmega46))
e161=uOmega45_old+uOmega53_old
bcOmega45.append(DirichletBC(V,e161,dOmega45nOmega53))
bcOmega45.append(BC_Omega45_only)
bcOmega46=bcs.copy()
e162=uOmega46_old+uOmega38_old
bcOmega46.append(DirichletBC(V,e162,dOmega46nOmega38))
e163=uOmega46_old+uOmega45_old
bcOmega46.append(DirichletBC(V,e163,dOmega46nOmega45))
e164=uOmega46_old+uOmega47_old
bcOmega46.append(DirichletBC(V,e164,dOmega46nOmega47))
e165=uOmega46_old+uOmega54_old
bcOmega46.append(DirichletBC(V,e165,dOmega46nOmega54))
bcOmega46.append(BC_Omega46_only)
bcOmega47=bcs.copy()
e166=uOmega47_old+uOmega39_old
bcOmega47.append(DirichletBC(V,e166,dOmega47nOmega39))
e167=uOmega47_old+uOmega46_old
bcOmega47.append(DirichletBC(V,e167,dOmega47nOmega46))
e168=uOmega47_old+uOmega48_old
bcOmega47.append(DirichletBC(V,e168,dOmega47nOmega48))
e169=uOmega47_old+uOmega55_old
bcOmega47.append(DirichletBC(V,e169,dOmega47nOmega55))
bcOmega47.append(BC_Omega47_only)
bcOmega48=bcs.copy()
e170=uOmega48_old+uOmega40_old
bcOmega48.append(DirichletBC(V,e170,dOmega48nOmega40))
e171=uOmega48_old+uOmega47_old
bcOmega48.append(DirichletBC(V,e171,dOmega48nOmega47))
e172=uOmega48_old+uOmega56_old
bcOmega48.append(DirichletBC(V,e172,dOmega48nOmega56))
bcOmega48.append(BC_Omega48_only)
bcOmega49=bcs.copy()
e173=uOmega49_old+uOmega41_old
bcOmega49.append(DirichletBC(V,e173,dOmega49nOmega41))
e174=uOmega49_old+uOmega50_old
bcOmega49.append(DirichletBC(V,e174,dOmega49nOmega50))
e175=uOmega49_old+uOmega57_old
bcOmega49.append(DirichletBC(V,e175,dOmega49nOmega57))
bcOmega49.append(BC_Omega49_only)
bcOmega50=bcs.copy()
e176=uOmega50_old+uOmega42_old
bcOmega50.append(DirichletBC(V,e176,dOmega50nOmega42))
e177=uOmega50_old+uOmega49_old
bcOmega50.append(DirichletBC(V,e177,dOmega50nOmega49))
e178=uOmega50_old+uOmega51_old
bcOmega50.append(DirichletBC(V,e178,dOmega50nOmega51))
e179=uOmega50_old+uOmega58_old
bcOmega50.append(DirichletBC(V,e179,dOmega50nOmega58))
bcOmega50.append(BC_Omega50_only)
bcOmega51=bcs.copy()
e180=uOmega51_old+uOmega43_old
bcOmega51.append(DirichletBC(V,e180,dOmega51nOmega43))
e181=uOmega51_old+uOmega50_old
bcOmega51.append(DirichletBC(V,e181,dOmega51nOmega50))
e182=uOmega51_old+uOmega52_old
bcOmega51.append(DirichletBC(V,e182,dOmega51nOmega52))
e183=uOmega51_old+uOmega59_old
bcOmega51.append(DirichletBC(V,e183,dOmega51nOmega59))
bcOmega51.append(BC_Omega51_only)
bcOmega52=bcs.copy()
e184=uOmega52_old+uOmega44_old
bcOmega52.append(DirichletBC(V,e184,dOmega52nOmega44))
e185=uOmega52_old+uOmega51_old
bcOmega52.append(DirichletBC(V,e185,dOmega52nOmega51))
e186=uOmega52_old+uOmega53_old
bcOmega52.append(DirichletBC(V,e186,dOmega52nOmega53))
e187=uOmega52_old+uOmega60_old
bcOmega52.append(DirichletBC(V,e187,dOmega52nOmega60))
bcOmega52.append(BC_Omega52_only)
bcOmega53=bcs.copy()
e188=uOmega53_old+uOmega45_old
bcOmega53.append(DirichletBC(V,e188,dOmega53nOmega45))
e189=uOmega53_old+uOmega52_old
bcOmega53.append(DirichletBC(V,e189,dOmega53nOmega52))
e190=uOmega53_old+uOmega54_old
bcOmega53.append(DirichletBC(V,e190,dOmega53nOmega54))
e191=uOmega53_old+uOmega61_old
bcOmega53.append(DirichletBC(V,e191,dOmega53nOmega61))
bcOmega53.append(BC_Omega53_only)
bcOmega54=bcs.copy()
e192=uOmega54_old+uOmega46_old
bcOmega54.append(DirichletBC(V,e192,dOmega54nOmega46))
e193=uOmega54_old+uOmega53_old
bcOmega54.append(DirichletBC(V,e193,dOmega54nOmega53))
e194=uOmega54_old+uOmega55_old
bcOmega54.append(DirichletBC(V,e194,dOmega54nOmega55))
e195=uOmega54_old+uOmega62_old
bcOmega54.append(DirichletBC(V,e195,dOmega54nOmega62))
bcOmega54.append(BC_Omega54_only)
bcOmega55=bcs.copy()
e196=uOmega55_old+uOmega47_old
bcOmega55.append(DirichletBC(V,e196,dOmega55nOmega47))
e197=uOmega55_old+uOmega54_old
bcOmega55.append(DirichletBC(V,e197,dOmega55nOmega54))
e198=uOmega55_old+uOmega56_old
bcOmega55.append(DirichletBC(V,e198,dOmega55nOmega56))
e199=uOmega55_old+uOmega63_old
bcOmega55.append(DirichletBC(V,e199,dOmega55nOmega63))
bcOmega55.append(BC_Omega55_only)
bcOmega56=bcs.copy()
e200=uOmega56_old+uOmega48_old
bcOmega56.append(DirichletBC(V,e200,dOmega56nOmega48))
e201=uOmega56_old+uOmega55_old
bcOmega56.append(DirichletBC(V,e201,dOmega56nOmega55))
e202=uOmega56_old+uOmega64_old
bcOmega56.append(DirichletBC(V,e202,dOmega56nOmega64))
bcOmega56.append(BC_Omega56_only)
bcOmega57=bcs.copy()
e203=uOmega57_old+uOmega49_old
bcOmega57.append(DirichletBC(V,e203,dOmega57nOmega49))
e204=uOmega57_old+uOmega58_old
bcOmega57.append(DirichletBC(V,e204,dOmega57nOmega58))
bcOmega57.append(BC_Omega57_only)
bcOmega58=bcs.copy()
e205=uOmega58_old+uOmega50_old
bcOmega58.append(DirichletBC(V,e205,dOmega58nOmega50))
e206=uOmega58_old+uOmega57_old
bcOmega58.append(DirichletBC(V,e206,dOmega58nOmega57))
e207=uOmega58_old+uOmega59_old
bcOmega58.append(DirichletBC(V,e207,dOmega58nOmega59))
bcOmega58.append(BC_Omega58_only)
bcOmega59=bcs.copy()
e208=uOmega59_old+uOmega51_old
bcOmega59.append(DirichletBC(V,e208,dOmega59nOmega51))
e209=uOmega59_old+uOmega58_old
bcOmega59.append(DirichletBC(V,e209,dOmega59nOmega58))
e210=uOmega59_old+uOmega60_old
bcOmega59.append(DirichletBC(V,e210,dOmega59nOmega60))
bcOmega59.append(BC_Omega59_only)
bcOmega60=bcs.copy()
e211=uOmega60_old+uOmega52_old
bcOmega60.append(DirichletBC(V,e211,dOmega60nOmega52))
e212=uOmega60_old+uOmega59_old
bcOmega60.append(DirichletBC(V,e212,dOmega60nOmega59))
e213=uOmega60_old+uOmega61_old
bcOmega60.append(DirichletBC(V,e213,dOmega60nOmega61))
bcOmega60.append(BC_Omega60_only)
bcOmega61=bcs.copy()
e214=uOmega61_old+uOmega53_old
bcOmega61.append(DirichletBC(V,e214,dOmega61nOmega53))
e215=uOmega61_old+uOmega60_old
bcOmega61.append(DirichletBC(V,e215,dOmega61nOmega60))
e216=uOmega61_old+uOmega62_old
bcOmega61.append(DirichletBC(V,e216,dOmega61nOmega62))
bcOmega61.append(BC_Omega61_only)
bcOmega62=bcs.copy()
e217=uOmega62_old+uOmega54_old
bcOmega62.append(DirichletBC(V,e217,dOmega62nOmega54))
e218=uOmega62_old+uOmega61_old
bcOmega62.append(DirichletBC(V,e218,dOmega62nOmega61))
e219=uOmega62_old+uOmega63_old
bcOmega62.append(DirichletBC(V,e219,dOmega62nOmega63))
bcOmega62.append(BC_Omega62_only)
bcOmega63=bcs.copy()
e220=uOmega63_old+uOmega55_old
bcOmega63.append(DirichletBC(V,e220,dOmega63nOmega55))
e221=uOmega63_old+uOmega62_old
bcOmega63.append(DirichletBC(V,e221,dOmega63nOmega62))
e222=uOmega63_old+uOmega64_old
bcOmega63.append(DirichletBC(V,e222,dOmega63nOmega64))
bcOmega63.append(BC_Omega63_only)
bcOmega64=bcs.copy()
e223=uOmega64_old+uOmega56_old
bcOmega64.append(DirichletBC(V,e223,dOmega64nOmega56))
e224=uOmega64_old+uOmega63_old
bcOmega64.append(DirichletBC(V,e224,dOmega64nOmega63))
bcOmega64.append(BC_Omega64_only)
nSchwarz=2
start_time2=time.time()
for iteration in range(nSchwarz):
	uOmega1_old=uOmega1
	uOmega2_old=uOmega2
	uOmega3_old=uOmega3
	uOmega4_old=uOmega4
	uOmega5_old=uOmega5
	uOmega6_old=uOmega6
	uOmega7_old=uOmega7
	uOmega8_old=uOmega8
	uOmega9_old=uOmega9
	uOmega10_old=uOmega10
	uOmega11_old=uOmega11
	uOmega12_old=uOmega12
	uOmega13_old=uOmega13
	uOmega14_old=uOmega14
	uOmega15_old=uOmega15
	uOmega16_old=uOmega16
	uOmega17_old=uOmega17
	uOmega18_old=uOmega18
	uOmega19_old=uOmega19
	uOmega20_old=uOmega20
	uOmega21_old=uOmega21
	uOmega22_old=uOmega22
	uOmega23_old=uOmega23
	uOmega24_old=uOmega24
	uOmega25_old=uOmega25
	uOmega26_old=uOmega26
	uOmega27_old=uOmega27
	uOmega28_old=uOmega28
	uOmega29_old=uOmega29
	uOmega30_old=uOmega30
	uOmega31_old=uOmega31
	uOmega32_old=uOmega32
	uOmega33_old=uOmega33
	uOmega34_old=uOmega34
	uOmega35_old=uOmega35
	uOmega36_old=uOmega36
	uOmega37_old=uOmega37
	uOmega38_old=uOmega38
	uOmega39_old=uOmega39
	uOmega40_old=uOmega40
	uOmega41_old=uOmega41
	uOmega42_old=uOmega42
	uOmega43_old=uOmega43
	uOmega44_old=uOmega44
	uOmega45_old=uOmega45
	uOmega46_old=uOmega46
	uOmega47_old=uOmega47
	uOmega48_old=uOmega48
	uOmega49_old=uOmega49
	uOmega50_old=uOmega50
	uOmega51_old=uOmega51
	uOmega52_old=uOmega52
	uOmega53_old=uOmega53
	uOmega54_old=uOmega54
	uOmega55_old=uOmega55
	uOmega56_old=uOmega56
	uOmega57_old=uOmega57
	uOmega58_old=uOmega58
	uOmega59_old=uOmega59
	uOmega60_old=uOmega60
	uOmega61_old=uOmega61
	uOmega62_old=uOmega62
	uOmega63_old=uOmega63
	uOmega64_old=uOmega64
	solve(aOmega1==LOmega1,uOmega1,bcs=bcOmega1,solver_parameters=params)
	solve(aOmega2==LOmega2,uOmega2,bcs=bcOmega2,solver_parameters=params)
	solve(aOmega3==LOmega3,uOmega3,bcs=bcOmega3,solver_parameters=params)
	solve(aOmega4==LOmega4,uOmega4,bcs=bcOmega4,solver_parameters=params)
	solve(aOmega5==LOmega5,uOmega5,bcs=bcOmega5,solver_parameters=params)
	solve(aOmega6==LOmega6,uOmega6,bcs=bcOmega6,solver_parameters=params)
	solve(aOmega7==LOmega7,uOmega7,bcs=bcOmega7,solver_parameters=params)
	solve(aOmega8==LOmega8,uOmega8,bcs=bcOmega8,solver_parameters=params)
	solve(aOmega9==LOmega9,uOmega9,bcs=bcOmega9,solver_parameters=params)
	solve(aOmega10==LOmega10,uOmega10,bcs=bcOmega10,solver_parameters=params)
	solve(aOmega11==LOmega11,uOmega11,bcs=bcOmega11,solver_parameters=params)
	solve(aOmega12==LOmega12,uOmega12,bcs=bcOmega12,solver_parameters=params)
	solve(aOmega13==LOmega13,uOmega13,bcs=bcOmega13,solver_parameters=params)
	solve(aOmega14==LOmega14,uOmega14,bcs=bcOmega14,solver_parameters=params)
	solve(aOmega15==LOmega15,uOmega15,bcs=bcOmega15,solver_parameters=params)
	solve(aOmega16==LOmega16,uOmega16,bcs=bcOmega16,solver_parameters=params)
	solve(aOmega17==LOmega17,uOmega17,bcs=bcOmega17,solver_parameters=params)
	solve(aOmega18==LOmega18,uOmega18,bcs=bcOmega18,solver_parameters=params)
	solve(aOmega19==LOmega19,uOmega19,bcs=bcOmega19,solver_parameters=params)
	solve(aOmega20==LOmega20,uOmega20,bcs=bcOmega20,solver_parameters=params)
	solve(aOmega21==LOmega21,uOmega21,bcs=bcOmega21,solver_parameters=params)
	solve(aOmega22==LOmega22,uOmega22,bcs=bcOmega22,solver_parameters=params)
	solve(aOmega23==LOmega23,uOmega23,bcs=bcOmega23,solver_parameters=params)
	solve(aOmega24==LOmega24,uOmega24,bcs=bcOmega24,solver_parameters=params)
	solve(aOmega25==LOmega25,uOmega25,bcs=bcOmega25,solver_parameters=params)
	solve(aOmega26==LOmega26,uOmega26,bcs=bcOmega26,solver_parameters=params)
	solve(aOmega27==LOmega27,uOmega27,bcs=bcOmega27,solver_parameters=params)
	solve(aOmega28==LOmega28,uOmega28,bcs=bcOmega28,solver_parameters=params)
	solve(aOmega29==LOmega29,uOmega29,bcs=bcOmega29,solver_parameters=params)
	solve(aOmega30==LOmega30,uOmega30,bcs=bcOmega30,solver_parameters=params)
	solve(aOmega31==LOmega31,uOmega31,bcs=bcOmega31,solver_parameters=params)
	solve(aOmega32==LOmega32,uOmega32,bcs=bcOmega32,solver_parameters=params)
	solve(aOmega33==LOmega33,uOmega33,bcs=bcOmega33,solver_parameters=params)
	solve(aOmega34==LOmega34,uOmega34,bcs=bcOmega34,solver_parameters=params)
	solve(aOmega35==LOmega35,uOmega35,bcs=bcOmega35,solver_parameters=params)
	solve(aOmega36==LOmega36,uOmega36,bcs=bcOmega36,solver_parameters=params)
	solve(aOmega37==LOmega37,uOmega37,bcs=bcOmega37,solver_parameters=params)
	solve(aOmega38==LOmega38,uOmega38,bcs=bcOmega38,solver_parameters=params)
	solve(aOmega39==LOmega39,uOmega39,bcs=bcOmega39,solver_parameters=params)
	solve(aOmega40==LOmega40,uOmega40,bcs=bcOmega40,solver_parameters=params)
	solve(aOmega41==LOmega41,uOmega41,bcs=bcOmega41,solver_parameters=params)
	solve(aOmega42==LOmega42,uOmega42,bcs=bcOmega42,solver_parameters=params)
	solve(aOmega43==LOmega43,uOmega43,bcs=bcOmega43,solver_parameters=params)
	solve(aOmega44==LOmega44,uOmega44,bcs=bcOmega44,solver_parameters=params)
	solve(aOmega45==LOmega45,uOmega45,bcs=bcOmega45,solver_parameters=params)
	solve(aOmega46==LOmega46,uOmega46,bcs=bcOmega46,solver_parameters=params)
	solve(aOmega47==LOmega47,uOmega47,bcs=bcOmega47,solver_parameters=params)
	solve(aOmega48==LOmega48,uOmega48,bcs=bcOmega48,solver_parameters=params)
	solve(aOmega49==LOmega49,uOmega49,bcs=bcOmega49,solver_parameters=params)
	solve(aOmega50==LOmega50,uOmega50,bcs=bcOmega50,solver_parameters=params)
	solve(aOmega51==LOmega51,uOmega51,bcs=bcOmega51,solver_parameters=params)
	solve(aOmega52==LOmega52,uOmega52,bcs=bcOmega52,solver_parameters=params)
	solve(aOmega53==LOmega53,uOmega53,bcs=bcOmega53,solver_parameters=params)
	solve(aOmega54==LOmega54,uOmega54,bcs=bcOmega54,solver_parameters=params)
	solve(aOmega55==LOmega55,uOmega55,bcs=bcOmega55,solver_parameters=params)
	solve(aOmega56==LOmega56,uOmega56,bcs=bcOmega56,solver_parameters=params)
	solve(aOmega57==LOmega57,uOmega57,bcs=bcOmega57,solver_parameters=params)
	solve(aOmega58==LOmega58,uOmega58,bcs=bcOmega58,solver_parameters=params)
	solve(aOmega59==LOmega59,uOmega59,bcs=bcOmega59,solver_parameters=params)
	solve(aOmega60==LOmega60,uOmega60,bcs=bcOmega60,solver_parameters=params)
	solve(aOmega61==LOmega61,uOmega61,bcs=bcOmega61,solver_parameters=params)
	solve(aOmega62==LOmega62,uOmega62,bcs=bcOmega62,solver_parameters=params)
	solve(aOmega63==LOmega63,uOmega63,bcs=bcOmega63,solver_parameters=params)
	solve(aOmega64==LOmega64,uOmega64,bcs=bcOmega64,solver_parameters=params)
end_time2=time.time()
print("{},{},{},{}".format(N,N_subdom,end_time1-start_time1,end_time2-start_time2))
# does this line exist?
# perhaps some other lines after the fact?
