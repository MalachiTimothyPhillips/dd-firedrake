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
mesh=Mesh("../meshing/16x16Overlap.msh")
N=16
N_subdom=N*N
V=FunctionSpace(mesh, "CG", 1) # piecewise linear elements

ref_mesh=Mesh("../meshing/16x16Comparison.msh")
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
dirichletBCs={1:0,2:0,64:0,963:0,5:0,66:0,128:0,967:0,9:0,130:0,192:0,971:0,13:0,194:0,256:0,975:0,17:0,258:0,320:0,979:0,21:0,322:0,384:0,983:0,25:0,386:0,448:0,987:0,29:0,450:0,512:0,991:0,33:0,514:0,576:0,995:0,37:0,578:0,640:0,999:0,41:0,642:0,704:0,1003:0,45:0,706:0,768:0,1007:0,49:0,770:0,832:0,1011:0,53:0,834:0,896:0,1015:0,57:0,898:0,960:0,1019:0,61:0,962:0,1024:0,1023:0,}

domains={"Omega1":1,"Omega2":2,"Omega3":3,"Omega4":4,"Omega5":5,"Omega6":6,"Omega7":7,"Omega8":8,"Omega9":9,"Omega10":10,"Omega11":11,"Omega12":12,"Omega13":13,"Omega14":14,"Omega15":15,"Omega16":16,"Omega17":17,"Omega18":18,"Omega19":19,"Omega20":20,"Omega21":21,"Omega22":22,"Omega23":23,"Omega24":24,"Omega25":25,"Omega26":26,"Omega27":27,"Omega28":28,"Omega29":29,"Omega30":30,"Omega31":31,"Omega32":32,"Omega33":33,"Omega34":34,"Omega35":35,"Omega36":36,"Omega37":37,"Omega38":38,"Omega39":39,"Omega40":40,"Omega41":41,"Omega42":42,"Omega43":43,"Omega44":44,"Omega45":45,"Omega46":46,"Omega47":47,"Omega48":48,"Omega49":49,"Omega50":50,"Omega51":51,"Omega52":52,"Omega53":53,"Omega54":54,"Omega55":55,"Omega56":56,"Omega57":57,"Omega58":58,"Omega59":59,"Omega60":60,"Omega61":61,"Omega62":62,"Omega63":63,"Omega64":64,"Omega65":65,"Omega66":66,"Omega67":67,"Omega68":68,"Omega69":69,"Omega70":70,"Omega71":71,"Omega72":72,"Omega73":73,"Omega74":74,"Omega75":75,"Omega76":76,"Omega77":77,"Omega78":78,"Omega79":79,"Omega80":80,"Omega81":81,"Omega82":82,"Omega83":83,"Omega84":84,"Omega85":85,"Omega86":86,"Omega87":87,"Omega88":88,"Omega89":89,"Omega90":90,"Omega91":91,"Omega92":92,"Omega93":93,"Omega94":94,"Omega95":95,"Omega96":96,"Omega97":97,"Omega98":98,"Omega99":99,"Omega100":100,"Omega101":101,"Omega102":102,"Omega103":103,"Omega104":104,"Omega105":105,"Omega106":106,"Omega107":107,"Omega108":108,"Omega109":109,"Omega110":110,"Omega111":111,"Omega112":112,"Omega113":113,"Omega114":114,"Omega115":115,"Omega116":116,"Omega117":117,"Omega118":118,"Omega119":119,"Omega120":120,"Omega121":121,"Omega122":122,"Omega123":123,"Omega124":124,"Omega125":125,"Omega126":126,"Omega127":127,"Omega128":128,"Omega129":129,"Omega130":130,"Omega131":131,"Omega132":132,"Omega133":133,"Omega134":134,"Omega135":135,"Omega136":136,"Omega137":137,"Omega138":138,"Omega139":139,"Omega140":140,"Omega141":141,"Omega142":142,"Omega143":143,"Omega144":144,"Omega145":145,"Omega146":146,"Omega147":147,"Omega148":148,"Omega149":149,"Omega150":150,"Omega151":151,"Omega152":152,"Omega153":153,"Omega154":154,"Omega155":155,"Omega156":156,"Omega157":157,"Omega158":158,"Omega159":159,"Omega160":160,"Omega161":161,"Omega162":162,"Omega163":163,"Omega164":164,"Omega165":165,"Omega166":166,"Omega167":167,"Omega168":168,"Omega169":169,"Omega170":170,"Omega171":171,"Omega172":172,"Omega173":173,"Omega174":174,"Omega175":175,"Omega176":176,"Omega177":177,"Omega178":178,"Omega179":179,"Omega180":180,"Omega181":181,"Omega182":182,"Omega183":183,"Omega184":184,"Omega185":185,"Omega186":186,"Omega187":187,"Omega188":188,"Omega189":189,"Omega190":190,"Omega191":191,"Omega192":192,"Omega193":193,"Omega194":194,"Omega195":195,"Omega196":196,"Omega197":197,"Omega198":198,"Omega199":199,"Omega200":200,"Omega201":201,"Omega202":202,"Omega203":203,"Omega204":204,"Omega205":205,"Omega206":206,"Omega207":207,"Omega208":208,"Omega209":209,"Omega210":210,"Omega211":211,"Omega212":212,"Omega213":213,"Omega214":214,"Omega215":215,"Omega216":216,"Omega217":217,"Omega218":218,"Omega219":219,"Omega220":220,"Omega221":221,"Omega222":222,"Omega223":223,"Omega224":224,"Omega225":225,"Omega226":226,"Omega227":227,"Omega228":228,"Omega229":229,"Omega230":230,"Omega231":231,"Omega232":232,"Omega233":233,"Omega234":234,"Omega235":235,"Omega236":236,"Omega237":237,"Omega238":238,"Omega239":239,"Omega240":240,"Omega241":241,"Omega242":242,"Omega243":243,"Omega244":244,"Omega245":245,"Omega246":246,"Omega247":247,"Omega248":248,"Omega249":249,"Omega250":250,"Omega251":251,"Omega252":252,"Omega253":253,"Omega254":254,"Omega255":255,"Omega256":256,}

interfaces={"dOmega18nOmega19":72,"dOmega18nOmega17":70,"dOmega18nOmega34":71,"dOmega18nOmega2":69,"dOmega34nOmega35":136,"dOmega34nOmega33":134,"dOmega34nOmega50":135,"dOmega34nOmega18":133,"dOmega50nOmega51":200,"dOmega50nOmega49":198,"dOmega50nOmega66":199,"dOmega50nOmega34":197,"dOmega66nOmega67":264,"dOmega66nOmega65":262,"dOmega66nOmega82":263,"dOmega66nOmega50":261,"dOmega82nOmega83":328,"dOmega82nOmega81":326,"dOmega82nOmega98":327,"dOmega82nOmega66":325,"dOmega98nOmega99":392,"dOmega98nOmega97":390,"dOmega98nOmega114":391,"dOmega98nOmega82":389,"dOmega114nOmega115":456,"dOmega114nOmega113":454,"dOmega114nOmega130":455,"dOmega114nOmega98":453,"dOmega130nOmega131":520,"dOmega130nOmega129":518,"dOmega130nOmega146":519,"dOmega130nOmega114":517,"dOmega146nOmega147":584,"dOmega146nOmega145":582,"dOmega146nOmega162":583,"dOmega146nOmega130":581,"dOmega162nOmega163":648,"dOmega162nOmega161":646,"dOmega162nOmega178":647,"dOmega162nOmega146":645,"dOmega178nOmega179":712,"dOmega178nOmega177":710,"dOmega178nOmega194":711,"dOmega178nOmega162":709,"dOmega194nOmega195":776,"dOmega194nOmega193":774,"dOmega194nOmega210":775,"dOmega194nOmega178":773,"dOmega210nOmega211":840,"dOmega210nOmega209":838,"dOmega210nOmega226":839,"dOmega210nOmega194":837,"dOmega226nOmega227":904,"dOmega226nOmega225":902,"dOmega226nOmega242":903,"dOmega226nOmega210":901,"dOmega19nOmega20":76,"dOmega19nOmega18":74,"dOmega19nOmega35":75,"dOmega19nOmega3":73,"dOmega35nOmega36":140,"dOmega35nOmega34":138,"dOmega35nOmega51":139,"dOmega35nOmega19":137,"dOmega51nOmega52":204,"dOmega51nOmega50":202,"dOmega51nOmega67":203,"dOmega51nOmega35":201,"dOmega67nOmega68":268,"dOmega67nOmega66":266,"dOmega67nOmega83":267,"dOmega67nOmega51":265,"dOmega83nOmega84":332,"dOmega83nOmega82":330,"dOmega83nOmega99":331,"dOmega83nOmega67":329,"dOmega99nOmega100":396,"dOmega99nOmega98":394,"dOmega99nOmega115":395,"dOmega99nOmega83":393,"dOmega115nOmega116":460,"dOmega115nOmega114":458,"dOmega115nOmega131":459,"dOmega115nOmega99":457,"dOmega131nOmega132":524,"dOmega131nOmega130":522,"dOmega131nOmega147":523,"dOmega131nOmega115":521,"dOmega147nOmega148":588,"dOmega147nOmega146":586,"dOmega147nOmega163":587,"dOmega147nOmega131":585,"dOmega163nOmega164":652,"dOmega163nOmega162":650,"dOmega163nOmega179":651,"dOmega163nOmega147":649,"dOmega179nOmega180":716,"dOmega179nOmega178":714,"dOmega179nOmega195":715,"dOmega179nOmega163":713,"dOmega195nOmega196":780,"dOmega195nOmega194":778,"dOmega195nOmega211":779,"dOmega195nOmega179":777,"dOmega211nOmega212":844,"dOmega211nOmega210":842,"dOmega211nOmega227":843,"dOmega211nOmega195":841,"dOmega227nOmega228":908,"dOmega227nOmega226":906,"dOmega227nOmega243":907,"dOmega227nOmega211":905,"dOmega20nOmega21":80,"dOmega20nOmega19":78,"dOmega20nOmega36":79,"dOmega20nOmega4":77,"dOmega36nOmega37":144,"dOmega36nOmega35":142,"dOmega36nOmega52":143,"dOmega36nOmega20":141,"dOmega52nOmega53":208,"dOmega52nOmega51":206,"dOmega52nOmega68":207,"dOmega52nOmega36":205,"dOmega68nOmega69":272,"dOmega68nOmega67":270,"dOmega68nOmega84":271,"dOmega68nOmega52":269,"dOmega84nOmega85":336,"dOmega84nOmega83":334,"dOmega84nOmega100":335,"dOmega84nOmega68":333,"dOmega100nOmega101":400,"dOmega100nOmega99":398,"dOmega100nOmega116":399,"dOmega100nOmega84":397,"dOmega116nOmega117":464,"dOmega116nOmega115":462,"dOmega116nOmega132":463,"dOmega116nOmega100":461,"dOmega132nOmega133":528,"dOmega132nOmega131":526,"dOmega132nOmega148":527,"dOmega132nOmega116":525,"dOmega148nOmega149":592,"dOmega148nOmega147":590,"dOmega148nOmega164":591,"dOmega148nOmega132":589,"dOmega164nOmega165":656,"dOmega164nOmega163":654,"dOmega164nOmega180":655,"dOmega164nOmega148":653,"dOmega180nOmega181":720,"dOmega180nOmega179":718,"dOmega180nOmega196":719,"dOmega180nOmega164":717,"dOmega196nOmega197":784,"dOmega196nOmega195":782,"dOmega196nOmega212":783,"dOmega196nOmega180":781,"dOmega212nOmega213":848,"dOmega212nOmega211":846,"dOmega212nOmega228":847,"dOmega212nOmega196":845,"dOmega228nOmega229":912,"dOmega228nOmega227":910,"dOmega228nOmega244":911,"dOmega228nOmega212":909,"dOmega21nOmega22":84,"dOmega21nOmega20":82,"dOmega21nOmega37":83,"dOmega21nOmega5":81,"dOmega37nOmega38":148,"dOmega37nOmega36":146,"dOmega37nOmega53":147,"dOmega37nOmega21":145,"dOmega53nOmega54":212,"dOmega53nOmega52":210,"dOmega53nOmega69":211,"dOmega53nOmega37":209,"dOmega69nOmega70":276,"dOmega69nOmega68":274,"dOmega69nOmega85":275,"dOmega69nOmega53":273,"dOmega85nOmega86":340,"dOmega85nOmega84":338,"dOmega85nOmega101":339,"dOmega85nOmega69":337,"dOmega101nOmega102":404,"dOmega101nOmega100":402,"dOmega101nOmega117":403,"dOmega101nOmega85":401,"dOmega117nOmega118":468,"dOmega117nOmega116":466,"dOmega117nOmega133":467,"dOmega117nOmega101":465,"dOmega133nOmega134":532,"dOmega133nOmega132":530,"dOmega133nOmega149":531,"dOmega133nOmega117":529,"dOmega149nOmega150":596,"dOmega149nOmega148":594,"dOmega149nOmega165":595,"dOmega149nOmega133":593,"dOmega165nOmega166":660,"dOmega165nOmega164":658,"dOmega165nOmega181":659,"dOmega165nOmega149":657,"dOmega181nOmega182":724,"dOmega181nOmega180":722,"dOmega181nOmega197":723,"dOmega181nOmega165":721,"dOmega197nOmega198":788,"dOmega197nOmega196":786,"dOmega197nOmega213":787,"dOmega197nOmega181":785,"dOmega213nOmega214":852,"dOmega213nOmega212":850,"dOmega213nOmega229":851,"dOmega213nOmega197":849,"dOmega229nOmega230":916,"dOmega229nOmega228":914,"dOmega229nOmega245":915,"dOmega229nOmega213":913,"dOmega22nOmega23":88,"dOmega22nOmega21":86,"dOmega22nOmega38":87,"dOmega22nOmega6":85,"dOmega38nOmega39":152,"dOmega38nOmega37":150,"dOmega38nOmega54":151,"dOmega38nOmega22":149,"dOmega54nOmega55":216,"dOmega54nOmega53":214,"dOmega54nOmega70":215,"dOmega54nOmega38":213,"dOmega70nOmega71":280,"dOmega70nOmega69":278,"dOmega70nOmega86":279,"dOmega70nOmega54":277,"dOmega86nOmega87":344,"dOmega86nOmega85":342,"dOmega86nOmega102":343,"dOmega86nOmega70":341,"dOmega102nOmega103":408,"dOmega102nOmega101":406,"dOmega102nOmega118":407,"dOmega102nOmega86":405,"dOmega118nOmega119":472,"dOmega118nOmega117":470,"dOmega118nOmega134":471,"dOmega118nOmega102":469,"dOmega134nOmega135":536,"dOmega134nOmega133":534,"dOmega134nOmega150":535,"dOmega134nOmega118":533,"dOmega150nOmega151":600,"dOmega150nOmega149":598,"dOmega150nOmega166":599,"dOmega150nOmega134":597,"dOmega166nOmega167":664,"dOmega166nOmega165":662,"dOmega166nOmega182":663,"dOmega166nOmega150":661,"dOmega182nOmega183":728,"dOmega182nOmega181":726,"dOmega182nOmega198":727,"dOmega182nOmega166":725,"dOmega198nOmega199":792,"dOmega198nOmega197":790,"dOmega198nOmega214":791,"dOmega198nOmega182":789,"dOmega214nOmega215":856,"dOmega214nOmega213":854,"dOmega214nOmega230":855,"dOmega214nOmega198":853,"dOmega230nOmega231":920,"dOmega230nOmega229":918,"dOmega230nOmega246":919,"dOmega230nOmega214":917,"dOmega23nOmega24":92,"dOmega23nOmega22":90,"dOmega23nOmega39":91,"dOmega23nOmega7":89,"dOmega39nOmega40":156,"dOmega39nOmega38":154,"dOmega39nOmega55":155,"dOmega39nOmega23":153,"dOmega55nOmega56":220,"dOmega55nOmega54":218,"dOmega55nOmega71":219,"dOmega55nOmega39":217,"dOmega71nOmega72":284,"dOmega71nOmega70":282,"dOmega71nOmega87":283,"dOmega71nOmega55":281,"dOmega87nOmega88":348,"dOmega87nOmega86":346,"dOmega87nOmega103":347,"dOmega87nOmega71":345,"dOmega103nOmega104":412,"dOmega103nOmega102":410,"dOmega103nOmega119":411,"dOmega103nOmega87":409,"dOmega119nOmega120":476,"dOmega119nOmega118":474,"dOmega119nOmega135":475,"dOmega119nOmega103":473,"dOmega135nOmega136":540,"dOmega135nOmega134":538,"dOmega135nOmega151":539,"dOmega135nOmega119":537,"dOmega151nOmega152":604,"dOmega151nOmega150":602,"dOmega151nOmega167":603,"dOmega151nOmega135":601,"dOmega167nOmega168":668,"dOmega167nOmega166":666,"dOmega167nOmega183":667,"dOmega167nOmega151":665,"dOmega183nOmega184":732,"dOmega183nOmega182":730,"dOmega183nOmega199":731,"dOmega183nOmega167":729,"dOmega199nOmega200":796,"dOmega199nOmega198":794,"dOmega199nOmega215":795,"dOmega199nOmega183":793,"dOmega215nOmega216":860,"dOmega215nOmega214":858,"dOmega215nOmega231":859,"dOmega215nOmega199":857,"dOmega231nOmega232":924,"dOmega231nOmega230":922,"dOmega231nOmega247":923,"dOmega231nOmega215":921,"dOmega24nOmega25":96,"dOmega24nOmega23":94,"dOmega24nOmega40":95,"dOmega24nOmega8":93,"dOmega40nOmega41":160,"dOmega40nOmega39":158,"dOmega40nOmega56":159,"dOmega40nOmega24":157,"dOmega56nOmega57":224,"dOmega56nOmega55":222,"dOmega56nOmega72":223,"dOmega56nOmega40":221,"dOmega72nOmega73":288,"dOmega72nOmega71":286,"dOmega72nOmega88":287,"dOmega72nOmega56":285,"dOmega88nOmega89":352,"dOmega88nOmega87":350,"dOmega88nOmega104":351,"dOmega88nOmega72":349,"dOmega104nOmega105":416,"dOmega104nOmega103":414,"dOmega104nOmega120":415,"dOmega104nOmega88":413,"dOmega120nOmega121":480,"dOmega120nOmega119":478,"dOmega120nOmega136":479,"dOmega120nOmega104":477,"dOmega136nOmega137":544,"dOmega136nOmega135":542,"dOmega136nOmega152":543,"dOmega136nOmega120":541,"dOmega152nOmega153":608,"dOmega152nOmega151":606,"dOmega152nOmega168":607,"dOmega152nOmega136":605,"dOmega168nOmega169":672,"dOmega168nOmega167":670,"dOmega168nOmega184":671,"dOmega168nOmega152":669,"dOmega184nOmega185":736,"dOmega184nOmega183":734,"dOmega184nOmega200":735,"dOmega184nOmega168":733,"dOmega200nOmega201":800,"dOmega200nOmega199":798,"dOmega200nOmega216":799,"dOmega200nOmega184":797,"dOmega216nOmega217":864,"dOmega216nOmega215":862,"dOmega216nOmega232":863,"dOmega216nOmega200":861,"dOmega232nOmega233":928,"dOmega232nOmega231":926,"dOmega232nOmega248":927,"dOmega232nOmega216":925,"dOmega25nOmega26":100,"dOmega25nOmega24":98,"dOmega25nOmega41":99,"dOmega25nOmega9":97,"dOmega41nOmega42":164,"dOmega41nOmega40":162,"dOmega41nOmega57":163,"dOmega41nOmega25":161,"dOmega57nOmega58":228,"dOmega57nOmega56":226,"dOmega57nOmega73":227,"dOmega57nOmega41":225,"dOmega73nOmega74":292,"dOmega73nOmega72":290,"dOmega73nOmega89":291,"dOmega73nOmega57":289,"dOmega89nOmega90":356,"dOmega89nOmega88":354,"dOmega89nOmega105":355,"dOmega89nOmega73":353,"dOmega105nOmega106":420,"dOmega105nOmega104":418,"dOmega105nOmega121":419,"dOmega105nOmega89":417,"dOmega121nOmega122":484,"dOmega121nOmega120":482,"dOmega121nOmega137":483,"dOmega121nOmega105":481,"dOmega137nOmega138":548,"dOmega137nOmega136":546,"dOmega137nOmega153":547,"dOmega137nOmega121":545,"dOmega153nOmega154":612,"dOmega153nOmega152":610,"dOmega153nOmega169":611,"dOmega153nOmega137":609,"dOmega169nOmega170":676,"dOmega169nOmega168":674,"dOmega169nOmega185":675,"dOmega169nOmega153":673,"dOmega185nOmega186":740,"dOmega185nOmega184":738,"dOmega185nOmega201":739,"dOmega185nOmega169":737,"dOmega201nOmega202":804,"dOmega201nOmega200":802,"dOmega201nOmega217":803,"dOmega201nOmega185":801,"dOmega217nOmega218":868,"dOmega217nOmega216":866,"dOmega217nOmega233":867,"dOmega217nOmega201":865,"dOmega233nOmega234":932,"dOmega233nOmega232":930,"dOmega233nOmega249":931,"dOmega233nOmega217":929,"dOmega26nOmega27":104,"dOmega26nOmega25":102,"dOmega26nOmega42":103,"dOmega26nOmega10":101,"dOmega42nOmega43":168,"dOmega42nOmega41":166,"dOmega42nOmega58":167,"dOmega42nOmega26":165,"dOmega58nOmega59":232,"dOmega58nOmega57":230,"dOmega58nOmega74":231,"dOmega58nOmega42":229,"dOmega74nOmega75":296,"dOmega74nOmega73":294,"dOmega74nOmega90":295,"dOmega74nOmega58":293,"dOmega90nOmega91":360,"dOmega90nOmega89":358,"dOmega90nOmega106":359,"dOmega90nOmega74":357,"dOmega106nOmega107":424,"dOmega106nOmega105":422,"dOmega106nOmega122":423,"dOmega106nOmega90":421,"dOmega122nOmega123":488,"dOmega122nOmega121":486,"dOmega122nOmega138":487,"dOmega122nOmega106":485,"dOmega138nOmega139":552,"dOmega138nOmega137":550,"dOmega138nOmega154":551,"dOmega138nOmega122":549,"dOmega154nOmega155":616,"dOmega154nOmega153":614,"dOmega154nOmega170":615,"dOmega154nOmega138":613,"dOmega170nOmega171":680,"dOmega170nOmega169":678,"dOmega170nOmega186":679,"dOmega170nOmega154":677,"dOmega186nOmega187":744,"dOmega186nOmega185":742,"dOmega186nOmega202":743,"dOmega186nOmega170":741,"dOmega202nOmega203":808,"dOmega202nOmega201":806,"dOmega202nOmega218":807,"dOmega202nOmega186":805,"dOmega218nOmega219":872,"dOmega218nOmega217":870,"dOmega218nOmega234":871,"dOmega218nOmega202":869,"dOmega234nOmega235":936,"dOmega234nOmega233":934,"dOmega234nOmega250":935,"dOmega234nOmega218":933,"dOmega27nOmega28":108,"dOmega27nOmega26":106,"dOmega27nOmega43":107,"dOmega27nOmega11":105,"dOmega43nOmega44":172,"dOmega43nOmega42":170,"dOmega43nOmega59":171,"dOmega43nOmega27":169,"dOmega59nOmega60":236,"dOmega59nOmega58":234,"dOmega59nOmega75":235,"dOmega59nOmega43":233,"dOmega75nOmega76":300,"dOmega75nOmega74":298,"dOmega75nOmega91":299,"dOmega75nOmega59":297,"dOmega91nOmega92":364,"dOmega91nOmega90":362,"dOmega91nOmega107":363,"dOmega91nOmega75":361,"dOmega107nOmega108":428,"dOmega107nOmega106":426,"dOmega107nOmega123":427,"dOmega107nOmega91":425,"dOmega123nOmega124":492,"dOmega123nOmega122":490,"dOmega123nOmega139":491,"dOmega123nOmega107":489,"dOmega139nOmega140":556,"dOmega139nOmega138":554,"dOmega139nOmega155":555,"dOmega139nOmega123":553,"dOmega155nOmega156":620,"dOmega155nOmega154":618,"dOmega155nOmega171":619,"dOmega155nOmega139":617,"dOmega171nOmega172":684,"dOmega171nOmega170":682,"dOmega171nOmega187":683,"dOmega171nOmega155":681,"dOmega187nOmega188":748,"dOmega187nOmega186":746,"dOmega187nOmega203":747,"dOmega187nOmega171":745,"dOmega203nOmega204":812,"dOmega203nOmega202":810,"dOmega203nOmega219":811,"dOmega203nOmega187":809,"dOmega219nOmega220":876,"dOmega219nOmega218":874,"dOmega219nOmega235":875,"dOmega219nOmega203":873,"dOmega235nOmega236":940,"dOmega235nOmega234":938,"dOmega235nOmega251":939,"dOmega235nOmega219":937,"dOmega28nOmega29":112,"dOmega28nOmega27":110,"dOmega28nOmega44":111,"dOmega28nOmega12":109,"dOmega44nOmega45":176,"dOmega44nOmega43":174,"dOmega44nOmega60":175,"dOmega44nOmega28":173,"dOmega60nOmega61":240,"dOmega60nOmega59":238,"dOmega60nOmega76":239,"dOmega60nOmega44":237,"dOmega76nOmega77":304,"dOmega76nOmega75":302,"dOmega76nOmega92":303,"dOmega76nOmega60":301,"dOmega92nOmega93":368,"dOmega92nOmega91":366,"dOmega92nOmega108":367,"dOmega92nOmega76":365,"dOmega108nOmega109":432,"dOmega108nOmega107":430,"dOmega108nOmega124":431,"dOmega108nOmega92":429,"dOmega124nOmega125":496,"dOmega124nOmega123":494,"dOmega124nOmega140":495,"dOmega124nOmega108":493,"dOmega140nOmega141":560,"dOmega140nOmega139":558,"dOmega140nOmega156":559,"dOmega140nOmega124":557,"dOmega156nOmega157":624,"dOmega156nOmega155":622,"dOmega156nOmega172":623,"dOmega156nOmega140":621,"dOmega172nOmega173":688,"dOmega172nOmega171":686,"dOmega172nOmega188":687,"dOmega172nOmega156":685,"dOmega188nOmega189":752,"dOmega188nOmega187":750,"dOmega188nOmega204":751,"dOmega188nOmega172":749,"dOmega204nOmega205":816,"dOmega204nOmega203":814,"dOmega204nOmega220":815,"dOmega204nOmega188":813,"dOmega220nOmega221":880,"dOmega220nOmega219":878,"dOmega220nOmega236":879,"dOmega220nOmega204":877,"dOmega236nOmega237":944,"dOmega236nOmega235":942,"dOmega236nOmega252":943,"dOmega236nOmega220":941,"dOmega29nOmega30":116,"dOmega29nOmega28":114,"dOmega29nOmega45":115,"dOmega29nOmega13":113,"dOmega45nOmega46":180,"dOmega45nOmega44":178,"dOmega45nOmega61":179,"dOmega45nOmega29":177,"dOmega61nOmega62":244,"dOmega61nOmega60":242,"dOmega61nOmega77":243,"dOmega61nOmega45":241,"dOmega77nOmega78":308,"dOmega77nOmega76":306,"dOmega77nOmega93":307,"dOmega77nOmega61":305,"dOmega93nOmega94":372,"dOmega93nOmega92":370,"dOmega93nOmega109":371,"dOmega93nOmega77":369,"dOmega109nOmega110":436,"dOmega109nOmega108":434,"dOmega109nOmega125":435,"dOmega109nOmega93":433,"dOmega125nOmega126":500,"dOmega125nOmega124":498,"dOmega125nOmega141":499,"dOmega125nOmega109":497,"dOmega141nOmega142":564,"dOmega141nOmega140":562,"dOmega141nOmega157":563,"dOmega141nOmega125":561,"dOmega157nOmega158":628,"dOmega157nOmega156":626,"dOmega157nOmega173":627,"dOmega157nOmega141":625,"dOmega173nOmega174":692,"dOmega173nOmega172":690,"dOmega173nOmega189":691,"dOmega173nOmega157":689,"dOmega189nOmega190":756,"dOmega189nOmega188":754,"dOmega189nOmega205":755,"dOmega189nOmega173":753,"dOmega205nOmega206":820,"dOmega205nOmega204":818,"dOmega205nOmega221":819,"dOmega205nOmega189":817,"dOmega221nOmega222":884,"dOmega221nOmega220":882,"dOmega221nOmega237":883,"dOmega221nOmega205":881,"dOmega237nOmega238":948,"dOmega237nOmega236":946,"dOmega237nOmega253":947,"dOmega237nOmega221":945,"dOmega30nOmega31":120,"dOmega30nOmega29":118,"dOmega30nOmega46":119,"dOmega30nOmega14":117,"dOmega46nOmega47":184,"dOmega46nOmega45":182,"dOmega46nOmega62":183,"dOmega46nOmega30":181,"dOmega62nOmega63":248,"dOmega62nOmega61":246,"dOmega62nOmega78":247,"dOmega62nOmega46":245,"dOmega78nOmega79":312,"dOmega78nOmega77":310,"dOmega78nOmega94":311,"dOmega78nOmega62":309,"dOmega94nOmega95":376,"dOmega94nOmega93":374,"dOmega94nOmega110":375,"dOmega94nOmega78":373,"dOmega110nOmega111":440,"dOmega110nOmega109":438,"dOmega110nOmega126":439,"dOmega110nOmega94":437,"dOmega126nOmega127":504,"dOmega126nOmega125":502,"dOmega126nOmega142":503,"dOmega126nOmega110":501,"dOmega142nOmega143":568,"dOmega142nOmega141":566,"dOmega142nOmega158":567,"dOmega142nOmega126":565,"dOmega158nOmega159":632,"dOmega158nOmega157":630,"dOmega158nOmega174":631,"dOmega158nOmega142":629,"dOmega174nOmega175":696,"dOmega174nOmega173":694,"dOmega174nOmega190":695,"dOmega174nOmega158":693,"dOmega190nOmega191":760,"dOmega190nOmega189":758,"dOmega190nOmega206":759,"dOmega190nOmega174":757,"dOmega206nOmega207":824,"dOmega206nOmega205":822,"dOmega206nOmega222":823,"dOmega206nOmega190":821,"dOmega222nOmega223":888,"dOmega222nOmega221":886,"dOmega222nOmega238":887,"dOmega222nOmega206":885,"dOmega238nOmega239":952,"dOmega238nOmega237":950,"dOmega238nOmega254":951,"dOmega238nOmega222":949,"dOmega31nOmega32":124,"dOmega31nOmega30":122,"dOmega31nOmega47":123,"dOmega31nOmega15":121,"dOmega47nOmega48":188,"dOmega47nOmega46":186,"dOmega47nOmega63":187,"dOmega47nOmega31":185,"dOmega63nOmega64":252,"dOmega63nOmega62":250,"dOmega63nOmega79":251,"dOmega63nOmega47":249,"dOmega79nOmega80":316,"dOmega79nOmega78":314,"dOmega79nOmega95":315,"dOmega79nOmega63":313,"dOmega95nOmega96":380,"dOmega95nOmega94":378,"dOmega95nOmega111":379,"dOmega95nOmega79":377,"dOmega111nOmega112":444,"dOmega111nOmega110":442,"dOmega111nOmega127":443,"dOmega111nOmega95":441,"dOmega127nOmega128":508,"dOmega127nOmega126":506,"dOmega127nOmega143":507,"dOmega127nOmega111":505,"dOmega143nOmega144":572,"dOmega143nOmega142":570,"dOmega143nOmega159":571,"dOmega143nOmega127":569,"dOmega159nOmega160":636,"dOmega159nOmega158":634,"dOmega159nOmega175":635,"dOmega159nOmega143":633,"dOmega175nOmega176":700,"dOmega175nOmega174":698,"dOmega175nOmega191":699,"dOmega175nOmega159":697,"dOmega191nOmega192":764,"dOmega191nOmega190":762,"dOmega191nOmega207":763,"dOmega191nOmega175":761,"dOmega207nOmega208":828,"dOmega207nOmega206":826,"dOmega207nOmega223":827,"dOmega207nOmega191":825,"dOmega223nOmega224":892,"dOmega223nOmega222":890,"dOmega223nOmega239":891,"dOmega223nOmega207":889,"dOmega239nOmega240":956,"dOmega239nOmega238":954,"dOmega239nOmega255":955,"dOmega239nOmega223":953,"dOmega17nOmega18":68,"dOmega17nOmega33":67,"dOmega17nOmega1":65,"dOmega33nOmega34":132,"dOmega33nOmega49":131,"dOmega33nOmega17":129,"dOmega49nOmega50":196,"dOmega49nOmega65":195,"dOmega49nOmega33":193,"dOmega65nOmega66":260,"dOmega65nOmega81":259,"dOmega65nOmega49":257,"dOmega81nOmega82":324,"dOmega81nOmega97":323,"dOmega81nOmega65":321,"dOmega97nOmega98":388,"dOmega97nOmega113":387,"dOmega97nOmega81":385,"dOmega113nOmega114":452,"dOmega113nOmega129":451,"dOmega113nOmega97":449,"dOmega129nOmega130":516,"dOmega129nOmega145":515,"dOmega129nOmega113":513,"dOmega145nOmega146":580,"dOmega145nOmega161":579,"dOmega145nOmega129":577,"dOmega161nOmega162":644,"dOmega161nOmega177":643,"dOmega161nOmega145":641,"dOmega177nOmega178":708,"dOmega177nOmega193":707,"dOmega177nOmega161":705,"dOmega193nOmega194":772,"dOmega193nOmega209":771,"dOmega193nOmega177":769,"dOmega209nOmega210":836,"dOmega209nOmega225":835,"dOmega209nOmega193":833,"dOmega225nOmega226":900,"dOmega225nOmega241":899,"dOmega225nOmega209":897,"dOmega32nOmega31":126,"dOmega32nOmega48":127,"dOmega32nOmega16":125,"dOmega48nOmega47":190,"dOmega48nOmega64":191,"dOmega48nOmega32":189,"dOmega64nOmega63":254,"dOmega64nOmega80":255,"dOmega64nOmega48":253,"dOmega80nOmega79":318,"dOmega80nOmega96":319,"dOmega80nOmega64":317,"dOmega96nOmega95":382,"dOmega96nOmega112":383,"dOmega96nOmega80":381,"dOmega112nOmega111":446,"dOmega112nOmega128":447,"dOmega112nOmega96":445,"dOmega128nOmega127":510,"dOmega128nOmega144":511,"dOmega128nOmega112":509,"dOmega144nOmega143":574,"dOmega144nOmega160":575,"dOmega144nOmega128":573,"dOmega160nOmega159":638,"dOmega160nOmega176":639,"dOmega160nOmega144":637,"dOmega176nOmega175":702,"dOmega176nOmega192":703,"dOmega176nOmega160":701,"dOmega192nOmega191":766,"dOmega192nOmega208":767,"dOmega192nOmega176":765,"dOmega208nOmega207":830,"dOmega208nOmega224":831,"dOmega208nOmega192":829,"dOmega224nOmega223":894,"dOmega224nOmega240":895,"dOmega224nOmega208":893,"dOmega240nOmega239":958,"dOmega240nOmega256":959,"dOmega240nOmega224":957,"dOmega2nOmega3":8,"dOmega2nOmega1":6,"dOmega2nOmega18":7,"dOmega3nOmega4":12,"dOmega3nOmega2":10,"dOmega3nOmega19":11,"dOmega4nOmega5":16,"dOmega4nOmega3":14,"dOmega4nOmega20":15,"dOmega5nOmega6":20,"dOmega5nOmega4":18,"dOmega5nOmega21":19,"dOmega6nOmega7":24,"dOmega6nOmega5":22,"dOmega6nOmega22":23,"dOmega7nOmega8":28,"dOmega7nOmega6":26,"dOmega7nOmega23":27,"dOmega8nOmega9":32,"dOmega8nOmega7":30,"dOmega8nOmega24":31,"dOmega9nOmega10":36,"dOmega9nOmega8":34,"dOmega9nOmega25":35,"dOmega10nOmega11":40,"dOmega10nOmega9":38,"dOmega10nOmega26":39,"dOmega11nOmega12":44,"dOmega11nOmega10":42,"dOmega11nOmega27":43,"dOmega12nOmega13":48,"dOmega12nOmega11":46,"dOmega12nOmega28":47,"dOmega13nOmega14":52,"dOmega13nOmega12":50,"dOmega13nOmega29":51,"dOmega14nOmega15":56,"dOmega14nOmega13":54,"dOmega14nOmega30":55,"dOmega15nOmega16":60,"dOmega15nOmega14":58,"dOmega15nOmega31":59,"dOmega242nOmega243":968,"dOmega242nOmega241":966,"dOmega242nOmega226":965,"dOmega243nOmega244":972,"dOmega243nOmega242":970,"dOmega243nOmega227":969,"dOmega244nOmega245":976,"dOmega244nOmega243":974,"dOmega244nOmega228":973,"dOmega245nOmega246":980,"dOmega245nOmega244":978,"dOmega245nOmega229":977,"dOmega246nOmega247":984,"dOmega246nOmega245":982,"dOmega246nOmega230":981,"dOmega247nOmega248":988,"dOmega247nOmega246":986,"dOmega247nOmega231":985,"dOmega248nOmega249":992,"dOmega248nOmega247":990,"dOmega248nOmega232":989,"dOmega249nOmega250":996,"dOmega249nOmega248":994,"dOmega249nOmega233":993,"dOmega250nOmega251":1000,"dOmega250nOmega249":998,"dOmega250nOmega234":997,"dOmega251nOmega252":1004,"dOmega251nOmega250":1002,"dOmega251nOmega235":1001,"dOmega252nOmega253":1008,"dOmega252nOmega251":1006,"dOmega252nOmega236":1005,"dOmega253nOmega254":1012,"dOmega253nOmega252":1010,"dOmega253nOmega237":1009,"dOmega254nOmega255":1016,"dOmega254nOmega253":1014,"dOmega254nOmega238":1013,"dOmega255nOmega256":1020,"dOmega255nOmega254":1018,"dOmega255nOmega239":1017,"dOmega1nOmega2":4,"dOmega1nOmega17":3,"dOmega241nOmega242":964,"dOmega241nOmega225":961,"dOmega16nOmega15":62,"dOmega16nOmega32":63,"dOmega256nOmega255":1022,"dOmega256nOmega240":1021,}

Omega1=1
dOmega1nOmega2=4
dOmega1nOmega17=3
Omega2=2
dOmega2nOmega1=6
dOmega2nOmega3=8
dOmega2nOmega18=7
Omega3=3
dOmega3nOmega2=10
dOmega3nOmega4=12
dOmega3nOmega19=11
Omega4=4
dOmega4nOmega3=14
dOmega4nOmega5=16
dOmega4nOmega20=15
Omega5=5
dOmega5nOmega4=18
dOmega5nOmega6=20
dOmega5nOmega21=19
Omega6=6
dOmega6nOmega5=22
dOmega6nOmega7=24
dOmega6nOmega22=23
Omega7=7
dOmega7nOmega6=26
dOmega7nOmega8=28
dOmega7nOmega23=27
Omega8=8
dOmega8nOmega7=30
dOmega8nOmega9=32
dOmega8nOmega24=31
Omega9=9
dOmega9nOmega8=34
dOmega9nOmega10=36
dOmega9nOmega25=35
Omega10=10
dOmega10nOmega9=38
dOmega10nOmega11=40
dOmega10nOmega26=39
Omega11=11
dOmega11nOmega10=42
dOmega11nOmega12=44
dOmega11nOmega27=43
Omega12=12
dOmega12nOmega11=46
dOmega12nOmega13=48
dOmega12nOmega28=47
Omega13=13
dOmega13nOmega12=50
dOmega13nOmega14=52
dOmega13nOmega29=51
Omega14=14
dOmega14nOmega13=54
dOmega14nOmega15=56
dOmega14nOmega30=55
Omega15=15
dOmega15nOmega14=58
dOmega15nOmega16=60
dOmega15nOmega31=59
Omega16=16
dOmega16nOmega15=62
dOmega16nOmega32=63
Omega17=17
dOmega17nOmega1=65
dOmega17nOmega18=68
dOmega17nOmega33=67
Omega18=18
dOmega18nOmega2=69
dOmega18nOmega17=70
dOmega18nOmega19=72
dOmega18nOmega34=71
Omega19=19
dOmega19nOmega3=73
dOmega19nOmega18=74
dOmega19nOmega20=76
dOmega19nOmega35=75
Omega20=20
dOmega20nOmega4=77
dOmega20nOmega19=78
dOmega20nOmega21=80
dOmega20nOmega36=79
Omega21=21
dOmega21nOmega5=81
dOmega21nOmega20=82
dOmega21nOmega22=84
dOmega21nOmega37=83
Omega22=22
dOmega22nOmega6=85
dOmega22nOmega21=86
dOmega22nOmega23=88
dOmega22nOmega38=87
Omega23=23
dOmega23nOmega7=89
dOmega23nOmega22=90
dOmega23nOmega24=92
dOmega23nOmega39=91
Omega24=24
dOmega24nOmega8=93
dOmega24nOmega23=94
dOmega24nOmega25=96
dOmega24nOmega40=95
Omega25=25
dOmega25nOmega9=97
dOmega25nOmega24=98
dOmega25nOmega26=100
dOmega25nOmega41=99
Omega26=26
dOmega26nOmega10=101
dOmega26nOmega25=102
dOmega26nOmega27=104
dOmega26nOmega42=103
Omega27=27
dOmega27nOmega11=105
dOmega27nOmega26=106
dOmega27nOmega28=108
dOmega27nOmega43=107
Omega28=28
dOmega28nOmega12=109
dOmega28nOmega27=110
dOmega28nOmega29=112
dOmega28nOmega44=111
Omega29=29
dOmega29nOmega13=113
dOmega29nOmega28=114
dOmega29nOmega30=116
dOmega29nOmega45=115
Omega30=30
dOmega30nOmega14=117
dOmega30nOmega29=118
dOmega30nOmega31=120
dOmega30nOmega46=119
Omega31=31
dOmega31nOmega15=121
dOmega31nOmega30=122
dOmega31nOmega32=124
dOmega31nOmega47=123
Omega32=32
dOmega32nOmega16=125
dOmega32nOmega31=126
dOmega32nOmega48=127
Omega33=33
dOmega33nOmega17=129
dOmega33nOmega34=132
dOmega33nOmega49=131
Omega34=34
dOmega34nOmega18=133
dOmega34nOmega33=134
dOmega34nOmega35=136
dOmega34nOmega50=135
Omega35=35
dOmega35nOmega19=137
dOmega35nOmega34=138
dOmega35nOmega36=140
dOmega35nOmega51=139
Omega36=36
dOmega36nOmega20=141
dOmega36nOmega35=142
dOmega36nOmega37=144
dOmega36nOmega52=143
Omega37=37
dOmega37nOmega21=145
dOmega37nOmega36=146
dOmega37nOmega38=148
dOmega37nOmega53=147
Omega38=38
dOmega38nOmega22=149
dOmega38nOmega37=150
dOmega38nOmega39=152
dOmega38nOmega54=151
Omega39=39
dOmega39nOmega23=153
dOmega39nOmega38=154
dOmega39nOmega40=156
dOmega39nOmega55=155
Omega40=40
dOmega40nOmega24=157
dOmega40nOmega39=158
dOmega40nOmega41=160
dOmega40nOmega56=159
Omega41=41
dOmega41nOmega25=161
dOmega41nOmega40=162
dOmega41nOmega42=164
dOmega41nOmega57=163
Omega42=42
dOmega42nOmega26=165
dOmega42nOmega41=166
dOmega42nOmega43=168
dOmega42nOmega58=167
Omega43=43
dOmega43nOmega27=169
dOmega43nOmega42=170
dOmega43nOmega44=172
dOmega43nOmega59=171
Omega44=44
dOmega44nOmega28=173
dOmega44nOmega43=174
dOmega44nOmega45=176
dOmega44nOmega60=175
Omega45=45
dOmega45nOmega29=177
dOmega45nOmega44=178
dOmega45nOmega46=180
dOmega45nOmega61=179
Omega46=46
dOmega46nOmega30=181
dOmega46nOmega45=182
dOmega46nOmega47=184
dOmega46nOmega62=183
Omega47=47
dOmega47nOmega31=185
dOmega47nOmega46=186
dOmega47nOmega48=188
dOmega47nOmega63=187
Omega48=48
dOmega48nOmega32=189
dOmega48nOmega47=190
dOmega48nOmega64=191
Omega49=49
dOmega49nOmega33=193
dOmega49nOmega50=196
dOmega49nOmega65=195
Omega50=50
dOmega50nOmega34=197
dOmega50nOmega49=198
dOmega50nOmega51=200
dOmega50nOmega66=199
Omega51=51
dOmega51nOmega35=201
dOmega51nOmega50=202
dOmega51nOmega52=204
dOmega51nOmega67=203
Omega52=52
dOmega52nOmega36=205
dOmega52nOmega51=206
dOmega52nOmega53=208
dOmega52nOmega68=207
Omega53=53
dOmega53nOmega37=209
dOmega53nOmega52=210
dOmega53nOmega54=212
dOmega53nOmega69=211
Omega54=54
dOmega54nOmega38=213
dOmega54nOmega53=214
dOmega54nOmega55=216
dOmega54nOmega70=215
Omega55=55
dOmega55nOmega39=217
dOmega55nOmega54=218
dOmega55nOmega56=220
dOmega55nOmega71=219
Omega56=56
dOmega56nOmega40=221
dOmega56nOmega55=222
dOmega56nOmega57=224
dOmega56nOmega72=223
Omega57=57
dOmega57nOmega41=225
dOmega57nOmega56=226
dOmega57nOmega58=228
dOmega57nOmega73=227
Omega58=58
dOmega58nOmega42=229
dOmega58nOmega57=230
dOmega58nOmega59=232
dOmega58nOmega74=231
Omega59=59
dOmega59nOmega43=233
dOmega59nOmega58=234
dOmega59nOmega60=236
dOmega59nOmega75=235
Omega60=60
dOmega60nOmega44=237
dOmega60nOmega59=238
dOmega60nOmega61=240
dOmega60nOmega76=239
Omega61=61
dOmega61nOmega45=241
dOmega61nOmega60=242
dOmega61nOmega62=244
dOmega61nOmega77=243
Omega62=62
dOmega62nOmega46=245
dOmega62nOmega61=246
dOmega62nOmega63=248
dOmega62nOmega78=247
Omega63=63
dOmega63nOmega47=249
dOmega63nOmega62=250
dOmega63nOmega64=252
dOmega63nOmega79=251
Omega64=64
dOmega64nOmega48=253
dOmega64nOmega63=254
dOmega64nOmega80=255
Omega65=65
dOmega65nOmega49=257
dOmega65nOmega66=260
dOmega65nOmega81=259
Omega66=66
dOmega66nOmega50=261
dOmega66nOmega65=262
dOmega66nOmega67=264
dOmega66nOmega82=263
Omega67=67
dOmega67nOmega51=265
dOmega67nOmega66=266
dOmega67nOmega68=268
dOmega67nOmega83=267
Omega68=68
dOmega68nOmega52=269
dOmega68nOmega67=270
dOmega68nOmega69=272
dOmega68nOmega84=271
Omega69=69
dOmega69nOmega53=273
dOmega69nOmega68=274
dOmega69nOmega70=276
dOmega69nOmega85=275
Omega70=70
dOmega70nOmega54=277
dOmega70nOmega69=278
dOmega70nOmega71=280
dOmega70nOmega86=279
Omega71=71
dOmega71nOmega55=281
dOmega71nOmega70=282
dOmega71nOmega72=284
dOmega71nOmega87=283
Omega72=72
dOmega72nOmega56=285
dOmega72nOmega71=286
dOmega72nOmega73=288
dOmega72nOmega88=287
Omega73=73
dOmega73nOmega57=289
dOmega73nOmega72=290
dOmega73nOmega74=292
dOmega73nOmega89=291
Omega74=74
dOmega74nOmega58=293
dOmega74nOmega73=294
dOmega74nOmega75=296
dOmega74nOmega90=295
Omega75=75
dOmega75nOmega59=297
dOmega75nOmega74=298
dOmega75nOmega76=300
dOmega75nOmega91=299
Omega76=76
dOmega76nOmega60=301
dOmega76nOmega75=302
dOmega76nOmega77=304
dOmega76nOmega92=303
Omega77=77
dOmega77nOmega61=305
dOmega77nOmega76=306
dOmega77nOmega78=308
dOmega77nOmega93=307
Omega78=78
dOmega78nOmega62=309
dOmega78nOmega77=310
dOmega78nOmega79=312
dOmega78nOmega94=311
Omega79=79
dOmega79nOmega63=313
dOmega79nOmega78=314
dOmega79nOmega80=316
dOmega79nOmega95=315
Omega80=80
dOmega80nOmega64=317
dOmega80nOmega79=318
dOmega80nOmega96=319
Omega81=81
dOmega81nOmega65=321
dOmega81nOmega82=324
dOmega81nOmega97=323
Omega82=82
dOmega82nOmega66=325
dOmega82nOmega81=326
dOmega82nOmega83=328
dOmega82nOmega98=327
Omega83=83
dOmega83nOmega67=329
dOmega83nOmega82=330
dOmega83nOmega84=332
dOmega83nOmega99=331
Omega84=84
dOmega84nOmega68=333
dOmega84nOmega83=334
dOmega84nOmega85=336
dOmega84nOmega100=335
Omega85=85
dOmega85nOmega69=337
dOmega85nOmega84=338
dOmega85nOmega86=340
dOmega85nOmega101=339
Omega86=86
dOmega86nOmega70=341
dOmega86nOmega85=342
dOmega86nOmega87=344
dOmega86nOmega102=343
Omega87=87
dOmega87nOmega71=345
dOmega87nOmega86=346
dOmega87nOmega88=348
dOmega87nOmega103=347
Omega88=88
dOmega88nOmega72=349
dOmega88nOmega87=350
dOmega88nOmega89=352
dOmega88nOmega104=351
Omega89=89
dOmega89nOmega73=353
dOmega89nOmega88=354
dOmega89nOmega90=356
dOmega89nOmega105=355
Omega90=90
dOmega90nOmega74=357
dOmega90nOmega89=358
dOmega90nOmega91=360
dOmega90nOmega106=359
Omega91=91
dOmega91nOmega75=361
dOmega91nOmega90=362
dOmega91nOmega92=364
dOmega91nOmega107=363
Omega92=92
dOmega92nOmega76=365
dOmega92nOmega91=366
dOmega92nOmega93=368
dOmega92nOmega108=367
Omega93=93
dOmega93nOmega77=369
dOmega93nOmega92=370
dOmega93nOmega94=372
dOmega93nOmega109=371
Omega94=94
dOmega94nOmega78=373
dOmega94nOmega93=374
dOmega94nOmega95=376
dOmega94nOmega110=375
Omega95=95
dOmega95nOmega79=377
dOmega95nOmega94=378
dOmega95nOmega96=380
dOmega95nOmega111=379
Omega96=96
dOmega96nOmega80=381
dOmega96nOmega95=382
dOmega96nOmega112=383
Omega97=97
dOmega97nOmega81=385
dOmega97nOmega98=388
dOmega97nOmega113=387
Omega98=98
dOmega98nOmega82=389
dOmega98nOmega97=390
dOmega98nOmega99=392
dOmega98nOmega114=391
Omega99=99
dOmega99nOmega83=393
dOmega99nOmega98=394
dOmega99nOmega100=396
dOmega99nOmega115=395
Omega100=100
dOmega100nOmega84=397
dOmega100nOmega99=398
dOmega100nOmega101=400
dOmega100nOmega116=399
Omega101=101
dOmega101nOmega85=401
dOmega101nOmega100=402
dOmega101nOmega102=404
dOmega101nOmega117=403
Omega102=102
dOmega102nOmega86=405
dOmega102nOmega101=406
dOmega102nOmega103=408
dOmega102nOmega118=407
Omega103=103
dOmega103nOmega87=409
dOmega103nOmega102=410
dOmega103nOmega104=412
dOmega103nOmega119=411
Omega104=104
dOmega104nOmega88=413
dOmega104nOmega103=414
dOmega104nOmega105=416
dOmega104nOmega120=415
Omega105=105
dOmega105nOmega89=417
dOmega105nOmega104=418
dOmega105nOmega106=420
dOmega105nOmega121=419
Omega106=106
dOmega106nOmega90=421
dOmega106nOmega105=422
dOmega106nOmega107=424
dOmega106nOmega122=423
Omega107=107
dOmega107nOmega91=425
dOmega107nOmega106=426
dOmega107nOmega108=428
dOmega107nOmega123=427
Omega108=108
dOmega108nOmega92=429
dOmega108nOmega107=430
dOmega108nOmega109=432
dOmega108nOmega124=431
Omega109=109
dOmega109nOmega93=433
dOmega109nOmega108=434
dOmega109nOmega110=436
dOmega109nOmega125=435
Omega110=110
dOmega110nOmega94=437
dOmega110nOmega109=438
dOmega110nOmega111=440
dOmega110nOmega126=439
Omega111=111
dOmega111nOmega95=441
dOmega111nOmega110=442
dOmega111nOmega112=444
dOmega111nOmega127=443
Omega112=112
dOmega112nOmega96=445
dOmega112nOmega111=446
dOmega112nOmega128=447
Omega113=113
dOmega113nOmega97=449
dOmega113nOmega114=452
dOmega113nOmega129=451
Omega114=114
dOmega114nOmega98=453
dOmega114nOmega113=454
dOmega114nOmega115=456
dOmega114nOmega130=455
Omega115=115
dOmega115nOmega99=457
dOmega115nOmega114=458
dOmega115nOmega116=460
dOmega115nOmega131=459
Omega116=116
dOmega116nOmega100=461
dOmega116nOmega115=462
dOmega116nOmega117=464
dOmega116nOmega132=463
Omega117=117
dOmega117nOmega101=465
dOmega117nOmega116=466
dOmega117nOmega118=468
dOmega117nOmega133=467
Omega118=118
dOmega118nOmega102=469
dOmega118nOmega117=470
dOmega118nOmega119=472
dOmega118nOmega134=471
Omega119=119
dOmega119nOmega103=473
dOmega119nOmega118=474
dOmega119nOmega120=476
dOmega119nOmega135=475
Omega120=120
dOmega120nOmega104=477
dOmega120nOmega119=478
dOmega120nOmega121=480
dOmega120nOmega136=479
Omega121=121
dOmega121nOmega105=481
dOmega121nOmega120=482
dOmega121nOmega122=484
dOmega121nOmega137=483
Omega122=122
dOmega122nOmega106=485
dOmega122nOmega121=486
dOmega122nOmega123=488
dOmega122nOmega138=487
Omega123=123
dOmega123nOmega107=489
dOmega123nOmega122=490
dOmega123nOmega124=492
dOmega123nOmega139=491
Omega124=124
dOmega124nOmega108=493
dOmega124nOmega123=494
dOmega124nOmega125=496
dOmega124nOmega140=495
Omega125=125
dOmega125nOmega109=497
dOmega125nOmega124=498
dOmega125nOmega126=500
dOmega125nOmega141=499
Omega126=126
dOmega126nOmega110=501
dOmega126nOmega125=502
dOmega126nOmega127=504
dOmega126nOmega142=503
Omega127=127
dOmega127nOmega111=505
dOmega127nOmega126=506
dOmega127nOmega128=508
dOmega127nOmega143=507
Omega128=128
dOmega128nOmega112=509
dOmega128nOmega127=510
dOmega128nOmega144=511
Omega129=129
dOmega129nOmega113=513
dOmega129nOmega130=516
dOmega129nOmega145=515
Omega130=130
dOmega130nOmega114=517
dOmega130nOmega129=518
dOmega130nOmega131=520
dOmega130nOmega146=519
Omega131=131
dOmega131nOmega115=521
dOmega131nOmega130=522
dOmega131nOmega132=524
dOmega131nOmega147=523
Omega132=132
dOmega132nOmega116=525
dOmega132nOmega131=526
dOmega132nOmega133=528
dOmega132nOmega148=527
Omega133=133
dOmega133nOmega117=529
dOmega133nOmega132=530
dOmega133nOmega134=532
dOmega133nOmega149=531
Omega134=134
dOmega134nOmega118=533
dOmega134nOmega133=534
dOmega134nOmega135=536
dOmega134nOmega150=535
Omega135=135
dOmega135nOmega119=537
dOmega135nOmega134=538
dOmega135nOmega136=540
dOmega135nOmega151=539
Omega136=136
dOmega136nOmega120=541
dOmega136nOmega135=542
dOmega136nOmega137=544
dOmega136nOmega152=543
Omega137=137
dOmega137nOmega121=545
dOmega137nOmega136=546
dOmega137nOmega138=548
dOmega137nOmega153=547
Omega138=138
dOmega138nOmega122=549
dOmega138nOmega137=550
dOmega138nOmega139=552
dOmega138nOmega154=551
Omega139=139
dOmega139nOmega123=553
dOmega139nOmega138=554
dOmega139nOmega140=556
dOmega139nOmega155=555
Omega140=140
dOmega140nOmega124=557
dOmega140nOmega139=558
dOmega140nOmega141=560
dOmega140nOmega156=559
Omega141=141
dOmega141nOmega125=561
dOmega141nOmega140=562
dOmega141nOmega142=564
dOmega141nOmega157=563
Omega142=142
dOmega142nOmega126=565
dOmega142nOmega141=566
dOmega142nOmega143=568
dOmega142nOmega158=567
Omega143=143
dOmega143nOmega127=569
dOmega143nOmega142=570
dOmega143nOmega144=572
dOmega143nOmega159=571
Omega144=144
dOmega144nOmega128=573
dOmega144nOmega143=574
dOmega144nOmega160=575
Omega145=145
dOmega145nOmega129=577
dOmega145nOmega146=580
dOmega145nOmega161=579
Omega146=146
dOmega146nOmega130=581
dOmega146nOmega145=582
dOmega146nOmega147=584
dOmega146nOmega162=583
Omega147=147
dOmega147nOmega131=585
dOmega147nOmega146=586
dOmega147nOmega148=588
dOmega147nOmega163=587
Omega148=148
dOmega148nOmega132=589
dOmega148nOmega147=590
dOmega148nOmega149=592
dOmega148nOmega164=591
Omega149=149
dOmega149nOmega133=593
dOmega149nOmega148=594
dOmega149nOmega150=596
dOmega149nOmega165=595
Omega150=150
dOmega150nOmega134=597
dOmega150nOmega149=598
dOmega150nOmega151=600
dOmega150nOmega166=599
Omega151=151
dOmega151nOmega135=601
dOmega151nOmega150=602
dOmega151nOmega152=604
dOmega151nOmega167=603
Omega152=152
dOmega152nOmega136=605
dOmega152nOmega151=606
dOmega152nOmega153=608
dOmega152nOmega168=607
Omega153=153
dOmega153nOmega137=609
dOmega153nOmega152=610
dOmega153nOmega154=612
dOmega153nOmega169=611
Omega154=154
dOmega154nOmega138=613
dOmega154nOmega153=614
dOmega154nOmega155=616
dOmega154nOmega170=615
Omega155=155
dOmega155nOmega139=617
dOmega155nOmega154=618
dOmega155nOmega156=620
dOmega155nOmega171=619
Omega156=156
dOmega156nOmega140=621
dOmega156nOmega155=622
dOmega156nOmega157=624
dOmega156nOmega172=623
Omega157=157
dOmega157nOmega141=625
dOmega157nOmega156=626
dOmega157nOmega158=628
dOmega157nOmega173=627
Omega158=158
dOmega158nOmega142=629
dOmega158nOmega157=630
dOmega158nOmega159=632
dOmega158nOmega174=631
Omega159=159
dOmega159nOmega143=633
dOmega159nOmega158=634
dOmega159nOmega160=636
dOmega159nOmega175=635
Omega160=160
dOmega160nOmega144=637
dOmega160nOmega159=638
dOmega160nOmega176=639
Omega161=161
dOmega161nOmega145=641
dOmega161nOmega162=644
dOmega161nOmega177=643
Omega162=162
dOmega162nOmega146=645
dOmega162nOmega161=646
dOmega162nOmega163=648
dOmega162nOmega178=647
Omega163=163
dOmega163nOmega147=649
dOmega163nOmega162=650
dOmega163nOmega164=652
dOmega163nOmega179=651
Omega164=164
dOmega164nOmega148=653
dOmega164nOmega163=654
dOmega164nOmega165=656
dOmega164nOmega180=655
Omega165=165
dOmega165nOmega149=657
dOmega165nOmega164=658
dOmega165nOmega166=660
dOmega165nOmega181=659
Omega166=166
dOmega166nOmega150=661
dOmega166nOmega165=662
dOmega166nOmega167=664
dOmega166nOmega182=663
Omega167=167
dOmega167nOmega151=665
dOmega167nOmega166=666
dOmega167nOmega168=668
dOmega167nOmega183=667
Omega168=168
dOmega168nOmega152=669
dOmega168nOmega167=670
dOmega168nOmega169=672
dOmega168nOmega184=671
Omega169=169
dOmega169nOmega153=673
dOmega169nOmega168=674
dOmega169nOmega170=676
dOmega169nOmega185=675
Omega170=170
dOmega170nOmega154=677
dOmega170nOmega169=678
dOmega170nOmega171=680
dOmega170nOmega186=679
Omega171=171
dOmega171nOmega155=681
dOmega171nOmega170=682
dOmega171nOmega172=684
dOmega171nOmega187=683
Omega172=172
dOmega172nOmega156=685
dOmega172nOmega171=686
dOmega172nOmega173=688
dOmega172nOmega188=687
Omega173=173
dOmega173nOmega157=689
dOmega173nOmega172=690
dOmega173nOmega174=692
dOmega173nOmega189=691
Omega174=174
dOmega174nOmega158=693
dOmega174nOmega173=694
dOmega174nOmega175=696
dOmega174nOmega190=695
Omega175=175
dOmega175nOmega159=697
dOmega175nOmega174=698
dOmega175nOmega176=700
dOmega175nOmega191=699
Omega176=176
dOmega176nOmega160=701
dOmega176nOmega175=702
dOmega176nOmega192=703
Omega177=177
dOmega177nOmega161=705
dOmega177nOmega178=708
dOmega177nOmega193=707
Omega178=178
dOmega178nOmega162=709
dOmega178nOmega177=710
dOmega178nOmega179=712
dOmega178nOmega194=711
Omega179=179
dOmega179nOmega163=713
dOmega179nOmega178=714
dOmega179nOmega180=716
dOmega179nOmega195=715
Omega180=180
dOmega180nOmega164=717
dOmega180nOmega179=718
dOmega180nOmega181=720
dOmega180nOmega196=719
Omega181=181
dOmega181nOmega165=721
dOmega181nOmega180=722
dOmega181nOmega182=724
dOmega181nOmega197=723
Omega182=182
dOmega182nOmega166=725
dOmega182nOmega181=726
dOmega182nOmega183=728
dOmega182nOmega198=727
Omega183=183
dOmega183nOmega167=729
dOmega183nOmega182=730
dOmega183nOmega184=732
dOmega183nOmega199=731
Omega184=184
dOmega184nOmega168=733
dOmega184nOmega183=734
dOmega184nOmega185=736
dOmega184nOmega200=735
Omega185=185
dOmega185nOmega169=737
dOmega185nOmega184=738
dOmega185nOmega186=740
dOmega185nOmega201=739
Omega186=186
dOmega186nOmega170=741
dOmega186nOmega185=742
dOmega186nOmega187=744
dOmega186nOmega202=743
Omega187=187
dOmega187nOmega171=745
dOmega187nOmega186=746
dOmega187nOmega188=748
dOmega187nOmega203=747
Omega188=188
dOmega188nOmega172=749
dOmega188nOmega187=750
dOmega188nOmega189=752
dOmega188nOmega204=751
Omega189=189
dOmega189nOmega173=753
dOmega189nOmega188=754
dOmega189nOmega190=756
dOmega189nOmega205=755
Omega190=190
dOmega190nOmega174=757
dOmega190nOmega189=758
dOmega190nOmega191=760
dOmega190nOmega206=759
Omega191=191
dOmega191nOmega175=761
dOmega191nOmega190=762
dOmega191nOmega192=764
dOmega191nOmega207=763
Omega192=192
dOmega192nOmega176=765
dOmega192nOmega191=766
dOmega192nOmega208=767
Omega193=193
dOmega193nOmega177=769
dOmega193nOmega194=772
dOmega193nOmega209=771
Omega194=194
dOmega194nOmega178=773
dOmega194nOmega193=774
dOmega194nOmega195=776
dOmega194nOmega210=775
Omega195=195
dOmega195nOmega179=777
dOmega195nOmega194=778
dOmega195nOmega196=780
dOmega195nOmega211=779
Omega196=196
dOmega196nOmega180=781
dOmega196nOmega195=782
dOmega196nOmega197=784
dOmega196nOmega212=783
Omega197=197
dOmega197nOmega181=785
dOmega197nOmega196=786
dOmega197nOmega198=788
dOmega197nOmega213=787
Omega198=198
dOmega198nOmega182=789
dOmega198nOmega197=790
dOmega198nOmega199=792
dOmega198nOmega214=791
Omega199=199
dOmega199nOmega183=793
dOmega199nOmega198=794
dOmega199nOmega200=796
dOmega199nOmega215=795
Omega200=200
dOmega200nOmega184=797
dOmega200nOmega199=798
dOmega200nOmega201=800
dOmega200nOmega216=799
Omega201=201
dOmega201nOmega185=801
dOmega201nOmega200=802
dOmega201nOmega202=804
dOmega201nOmega217=803
Omega202=202
dOmega202nOmega186=805
dOmega202nOmega201=806
dOmega202nOmega203=808
dOmega202nOmega218=807
Omega203=203
dOmega203nOmega187=809
dOmega203nOmega202=810
dOmega203nOmega204=812
dOmega203nOmega219=811
Omega204=204
dOmega204nOmega188=813
dOmega204nOmega203=814
dOmega204nOmega205=816
dOmega204nOmega220=815
Omega205=205
dOmega205nOmega189=817
dOmega205nOmega204=818
dOmega205nOmega206=820
dOmega205nOmega221=819
Omega206=206
dOmega206nOmega190=821
dOmega206nOmega205=822
dOmega206nOmega207=824
dOmega206nOmega222=823
Omega207=207
dOmega207nOmega191=825
dOmega207nOmega206=826
dOmega207nOmega208=828
dOmega207nOmega223=827
Omega208=208
dOmega208nOmega192=829
dOmega208nOmega207=830
dOmega208nOmega224=831
Omega209=209
dOmega209nOmega193=833
dOmega209nOmega210=836
dOmega209nOmega225=835
Omega210=210
dOmega210nOmega194=837
dOmega210nOmega209=838
dOmega210nOmega211=840
dOmega210nOmega226=839
Omega211=211
dOmega211nOmega195=841
dOmega211nOmega210=842
dOmega211nOmega212=844
dOmega211nOmega227=843
Omega212=212
dOmega212nOmega196=845
dOmega212nOmega211=846
dOmega212nOmega213=848
dOmega212nOmega228=847
Omega213=213
dOmega213nOmega197=849
dOmega213nOmega212=850
dOmega213nOmega214=852
dOmega213nOmega229=851
Omega214=214
dOmega214nOmega198=853
dOmega214nOmega213=854
dOmega214nOmega215=856
dOmega214nOmega230=855
Omega215=215
dOmega215nOmega199=857
dOmega215nOmega214=858
dOmega215nOmega216=860
dOmega215nOmega231=859
Omega216=216
dOmega216nOmega200=861
dOmega216nOmega215=862
dOmega216nOmega217=864
dOmega216nOmega232=863
Omega217=217
dOmega217nOmega201=865
dOmega217nOmega216=866
dOmega217nOmega218=868
dOmega217nOmega233=867
Omega218=218
dOmega218nOmega202=869
dOmega218nOmega217=870
dOmega218nOmega219=872
dOmega218nOmega234=871
Omega219=219
dOmega219nOmega203=873
dOmega219nOmega218=874
dOmega219nOmega220=876
dOmega219nOmega235=875
Omega220=220
dOmega220nOmega204=877
dOmega220nOmega219=878
dOmega220nOmega221=880
dOmega220nOmega236=879
Omega221=221
dOmega221nOmega205=881
dOmega221nOmega220=882
dOmega221nOmega222=884
dOmega221nOmega237=883
Omega222=222
dOmega222nOmega206=885
dOmega222nOmega221=886
dOmega222nOmega223=888
dOmega222nOmega238=887
Omega223=223
dOmega223nOmega207=889
dOmega223nOmega222=890
dOmega223nOmega224=892
dOmega223nOmega239=891
Omega224=224
dOmega224nOmega208=893
dOmega224nOmega223=894
dOmega224nOmega240=895
Omega225=225
dOmega225nOmega209=897
dOmega225nOmega226=900
dOmega225nOmega241=899
Omega226=226
dOmega226nOmega210=901
dOmega226nOmega225=902
dOmega226nOmega227=904
dOmega226nOmega242=903
Omega227=227
dOmega227nOmega211=905
dOmega227nOmega226=906
dOmega227nOmega228=908
dOmega227nOmega243=907
Omega228=228
dOmega228nOmega212=909
dOmega228nOmega227=910
dOmega228nOmega229=912
dOmega228nOmega244=911
Omega229=229
dOmega229nOmega213=913
dOmega229nOmega228=914
dOmega229nOmega230=916
dOmega229nOmega245=915
Omega230=230
dOmega230nOmega214=917
dOmega230nOmega229=918
dOmega230nOmega231=920
dOmega230nOmega246=919
Omega231=231
dOmega231nOmega215=921
dOmega231nOmega230=922
dOmega231nOmega232=924
dOmega231nOmega247=923
Omega232=232
dOmega232nOmega216=925
dOmega232nOmega231=926
dOmega232nOmega233=928
dOmega232nOmega248=927
Omega233=233
dOmega233nOmega217=929
dOmega233nOmega232=930
dOmega233nOmega234=932
dOmega233nOmega249=931
Omega234=234
dOmega234nOmega218=933
dOmega234nOmega233=934
dOmega234nOmega235=936
dOmega234nOmega250=935
Omega235=235
dOmega235nOmega219=937
dOmega235nOmega234=938
dOmega235nOmega236=940
dOmega235nOmega251=939
Omega236=236
dOmega236nOmega220=941
dOmega236nOmega235=942
dOmega236nOmega237=944
dOmega236nOmega252=943
Omega237=237
dOmega237nOmega221=945
dOmega237nOmega236=946
dOmega237nOmega238=948
dOmega237nOmega253=947
Omega238=238
dOmega238nOmega222=949
dOmega238nOmega237=950
dOmega238nOmega239=952
dOmega238nOmega254=951
Omega239=239
dOmega239nOmega223=953
dOmega239nOmega238=954
dOmega239nOmega240=956
dOmega239nOmega255=955
Omega240=240
dOmega240nOmega224=957
dOmega240nOmega239=958
dOmega240nOmega256=959
Omega241=241
dOmega241nOmega225=961
dOmega241nOmega242=964
Omega242=242
dOmega242nOmega226=965
dOmega242nOmega241=966
dOmega242nOmega243=968
Omega243=243
dOmega243nOmega227=969
dOmega243nOmega242=970
dOmega243nOmega244=972
Omega244=244
dOmega244nOmega228=973
dOmega244nOmega243=974
dOmega244nOmega245=976
Omega245=245
dOmega245nOmega229=977
dOmega245nOmega244=978
dOmega245nOmega246=980
Omega246=246
dOmega246nOmega230=981
dOmega246nOmega245=982
dOmega246nOmega247=984
Omega247=247
dOmega247nOmega231=985
dOmega247nOmega246=986
dOmega247nOmega248=988
Omega248=248
dOmega248nOmega232=989
dOmega248nOmega247=990
dOmega248nOmega249=992
Omega249=249
dOmega249nOmega233=993
dOmega249nOmega248=994
dOmega249nOmega250=996
Omega250=250
dOmega250nOmega234=997
dOmega250nOmega249=998
dOmega250nOmega251=1000
Omega251=251
dOmega251nOmega235=1001
dOmega251nOmega250=1002
dOmega251nOmega252=1004
Omega252=252
dOmega252nOmega236=1005
dOmega252nOmega251=1006
dOmega252nOmega253=1008
Omega253=253
dOmega253nOmega237=1009
dOmega253nOmega252=1010
dOmega253nOmega254=1012
Omega254=254
dOmega254nOmega238=1013
dOmega254nOmega253=1014
dOmega254nOmega255=1016
Omega255=255
dOmega255nOmega239=1017
dOmega255nOmega254=1018
dOmega255nOmega256=1020
Omega256=256
dOmega256nOmega240=1021
dOmega256nOmega255=1022

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
hOmega65=FunctionSpace(mesh,'DG', 0)
I_Omega65 = Function(hOmega65)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega65), {'f': (I_Omega65, WRITE)} )
I_cg_Omega65 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega65, RW), 'B': (I_Omega65, READ)} )
hOmega66=FunctionSpace(mesh,'DG', 0)
I_Omega66 = Function(hOmega66)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega66), {'f': (I_Omega66, WRITE)} )
I_cg_Omega66 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega66, RW), 'B': (I_Omega66, READ)} )
hOmega67=FunctionSpace(mesh,'DG', 0)
I_Omega67 = Function(hOmega67)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega67), {'f': (I_Omega67, WRITE)} )
I_cg_Omega67 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega67, RW), 'B': (I_Omega67, READ)} )
hOmega68=FunctionSpace(mesh,'DG', 0)
I_Omega68 = Function(hOmega68)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega68), {'f': (I_Omega68, WRITE)} )
I_cg_Omega68 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega68, RW), 'B': (I_Omega68, READ)} )
hOmega69=FunctionSpace(mesh,'DG', 0)
I_Omega69 = Function(hOmega69)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega69), {'f': (I_Omega69, WRITE)} )
I_cg_Omega69 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega69, RW), 'B': (I_Omega69, READ)} )
hOmega70=FunctionSpace(mesh,'DG', 0)
I_Omega70 = Function(hOmega70)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega70), {'f': (I_Omega70, WRITE)} )
I_cg_Omega70 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega70, RW), 'B': (I_Omega70, READ)} )
hOmega71=FunctionSpace(mesh,'DG', 0)
I_Omega71 = Function(hOmega71)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega71), {'f': (I_Omega71, WRITE)} )
I_cg_Omega71 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega71, RW), 'B': (I_Omega71, READ)} )
hOmega72=FunctionSpace(mesh,'DG', 0)
I_Omega72 = Function(hOmega72)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega72), {'f': (I_Omega72, WRITE)} )
I_cg_Omega72 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega72, RW), 'B': (I_Omega72, READ)} )
hOmega73=FunctionSpace(mesh,'DG', 0)
I_Omega73 = Function(hOmega73)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega73), {'f': (I_Omega73, WRITE)} )
I_cg_Omega73 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega73, RW), 'B': (I_Omega73, READ)} )
hOmega74=FunctionSpace(mesh,'DG', 0)
I_Omega74 = Function(hOmega74)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega74), {'f': (I_Omega74, WRITE)} )
I_cg_Omega74 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega74, RW), 'B': (I_Omega74, READ)} )
hOmega75=FunctionSpace(mesh,'DG', 0)
I_Omega75 = Function(hOmega75)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega75), {'f': (I_Omega75, WRITE)} )
I_cg_Omega75 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega75, RW), 'B': (I_Omega75, READ)} )
hOmega76=FunctionSpace(mesh,'DG', 0)
I_Omega76 = Function(hOmega76)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega76), {'f': (I_Omega76, WRITE)} )
I_cg_Omega76 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega76, RW), 'B': (I_Omega76, READ)} )
hOmega77=FunctionSpace(mesh,'DG', 0)
I_Omega77 = Function(hOmega77)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega77), {'f': (I_Omega77, WRITE)} )
I_cg_Omega77 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega77, RW), 'B': (I_Omega77, READ)} )
hOmega78=FunctionSpace(mesh,'DG', 0)
I_Omega78 = Function(hOmega78)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega78), {'f': (I_Omega78, WRITE)} )
I_cg_Omega78 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega78, RW), 'B': (I_Omega78, READ)} )
hOmega79=FunctionSpace(mesh,'DG', 0)
I_Omega79 = Function(hOmega79)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega79), {'f': (I_Omega79, WRITE)} )
I_cg_Omega79 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega79, RW), 'B': (I_Omega79, READ)} )
hOmega80=FunctionSpace(mesh,'DG', 0)
I_Omega80 = Function(hOmega80)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega80), {'f': (I_Omega80, WRITE)} )
I_cg_Omega80 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega80, RW), 'B': (I_Omega80, READ)} )
hOmega81=FunctionSpace(mesh,'DG', 0)
I_Omega81 = Function(hOmega81)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega81), {'f': (I_Omega81, WRITE)} )
I_cg_Omega81 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega81, RW), 'B': (I_Omega81, READ)} )
hOmega82=FunctionSpace(mesh,'DG', 0)
I_Omega82 = Function(hOmega82)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega82), {'f': (I_Omega82, WRITE)} )
I_cg_Omega82 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega82, RW), 'B': (I_Omega82, READ)} )
hOmega83=FunctionSpace(mesh,'DG', 0)
I_Omega83 = Function(hOmega83)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega83), {'f': (I_Omega83, WRITE)} )
I_cg_Omega83 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega83, RW), 'B': (I_Omega83, READ)} )
hOmega84=FunctionSpace(mesh,'DG', 0)
I_Omega84 = Function(hOmega84)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega84), {'f': (I_Omega84, WRITE)} )
I_cg_Omega84 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega84, RW), 'B': (I_Omega84, READ)} )
hOmega85=FunctionSpace(mesh,'DG', 0)
I_Omega85 = Function(hOmega85)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega85), {'f': (I_Omega85, WRITE)} )
I_cg_Omega85 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega85, RW), 'B': (I_Omega85, READ)} )
hOmega86=FunctionSpace(mesh,'DG', 0)
I_Omega86 = Function(hOmega86)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega86), {'f': (I_Omega86, WRITE)} )
I_cg_Omega86 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega86, RW), 'B': (I_Omega86, READ)} )
hOmega87=FunctionSpace(mesh,'DG', 0)
I_Omega87 = Function(hOmega87)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega87), {'f': (I_Omega87, WRITE)} )
I_cg_Omega87 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega87, RW), 'B': (I_Omega87, READ)} )
hOmega88=FunctionSpace(mesh,'DG', 0)
I_Omega88 = Function(hOmega88)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega88), {'f': (I_Omega88, WRITE)} )
I_cg_Omega88 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega88, RW), 'B': (I_Omega88, READ)} )
hOmega89=FunctionSpace(mesh,'DG', 0)
I_Omega89 = Function(hOmega89)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega89), {'f': (I_Omega89, WRITE)} )
I_cg_Omega89 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega89, RW), 'B': (I_Omega89, READ)} )
hOmega90=FunctionSpace(mesh,'DG', 0)
I_Omega90 = Function(hOmega90)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega90), {'f': (I_Omega90, WRITE)} )
I_cg_Omega90 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega90, RW), 'B': (I_Omega90, READ)} )
hOmega91=FunctionSpace(mesh,'DG', 0)
I_Omega91 = Function(hOmega91)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega91), {'f': (I_Omega91, WRITE)} )
I_cg_Omega91 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega91, RW), 'B': (I_Omega91, READ)} )
hOmega92=FunctionSpace(mesh,'DG', 0)
I_Omega92 = Function(hOmega92)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega92), {'f': (I_Omega92, WRITE)} )
I_cg_Omega92 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega92, RW), 'B': (I_Omega92, READ)} )
hOmega93=FunctionSpace(mesh,'DG', 0)
I_Omega93 = Function(hOmega93)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega93), {'f': (I_Omega93, WRITE)} )
I_cg_Omega93 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega93, RW), 'B': (I_Omega93, READ)} )
hOmega94=FunctionSpace(mesh,'DG', 0)
I_Omega94 = Function(hOmega94)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega94), {'f': (I_Omega94, WRITE)} )
I_cg_Omega94 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega94, RW), 'B': (I_Omega94, READ)} )
hOmega95=FunctionSpace(mesh,'DG', 0)
I_Omega95 = Function(hOmega95)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega95), {'f': (I_Omega95, WRITE)} )
I_cg_Omega95 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega95, RW), 'B': (I_Omega95, READ)} )
hOmega96=FunctionSpace(mesh,'DG', 0)
I_Omega96 = Function(hOmega96)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega96), {'f': (I_Omega96, WRITE)} )
I_cg_Omega96 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega96, RW), 'B': (I_Omega96, READ)} )
hOmega97=FunctionSpace(mesh,'DG', 0)
I_Omega97 = Function(hOmega97)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega97), {'f': (I_Omega97, WRITE)} )
I_cg_Omega97 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega97, RW), 'B': (I_Omega97, READ)} )
hOmega98=FunctionSpace(mesh,'DG', 0)
I_Omega98 = Function(hOmega98)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega98), {'f': (I_Omega98, WRITE)} )
I_cg_Omega98 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega98, RW), 'B': (I_Omega98, READ)} )
hOmega99=FunctionSpace(mesh,'DG', 0)
I_Omega99 = Function(hOmega99)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega99), {'f': (I_Omega99, WRITE)} )
I_cg_Omega99 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega99, RW), 'B': (I_Omega99, READ)} )
hOmega100=FunctionSpace(mesh,'DG', 0)
I_Omega100 = Function(hOmega100)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega100), {'f': (I_Omega100, WRITE)} )
I_cg_Omega100 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega100, RW), 'B': (I_Omega100, READ)} )
hOmega101=FunctionSpace(mesh,'DG', 0)
I_Omega101 = Function(hOmega101)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega101), {'f': (I_Omega101, WRITE)} )
I_cg_Omega101 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega101, RW), 'B': (I_Omega101, READ)} )
hOmega102=FunctionSpace(mesh,'DG', 0)
I_Omega102 = Function(hOmega102)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega102), {'f': (I_Omega102, WRITE)} )
I_cg_Omega102 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega102, RW), 'B': (I_Omega102, READ)} )
hOmega103=FunctionSpace(mesh,'DG', 0)
I_Omega103 = Function(hOmega103)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega103), {'f': (I_Omega103, WRITE)} )
I_cg_Omega103 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega103, RW), 'B': (I_Omega103, READ)} )
hOmega104=FunctionSpace(mesh,'DG', 0)
I_Omega104 = Function(hOmega104)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega104), {'f': (I_Omega104, WRITE)} )
I_cg_Omega104 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega104, RW), 'B': (I_Omega104, READ)} )
hOmega105=FunctionSpace(mesh,'DG', 0)
I_Omega105 = Function(hOmega105)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega105), {'f': (I_Omega105, WRITE)} )
I_cg_Omega105 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega105, RW), 'B': (I_Omega105, READ)} )
hOmega106=FunctionSpace(mesh,'DG', 0)
I_Omega106 = Function(hOmega106)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega106), {'f': (I_Omega106, WRITE)} )
I_cg_Omega106 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega106, RW), 'B': (I_Omega106, READ)} )
hOmega107=FunctionSpace(mesh,'DG', 0)
I_Omega107 = Function(hOmega107)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega107), {'f': (I_Omega107, WRITE)} )
I_cg_Omega107 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega107, RW), 'B': (I_Omega107, READ)} )
hOmega108=FunctionSpace(mesh,'DG', 0)
I_Omega108 = Function(hOmega108)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega108), {'f': (I_Omega108, WRITE)} )
I_cg_Omega108 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega108, RW), 'B': (I_Omega108, READ)} )
hOmega109=FunctionSpace(mesh,'DG', 0)
I_Omega109 = Function(hOmega109)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega109), {'f': (I_Omega109, WRITE)} )
I_cg_Omega109 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega109, RW), 'B': (I_Omega109, READ)} )
hOmega110=FunctionSpace(mesh,'DG', 0)
I_Omega110 = Function(hOmega110)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega110), {'f': (I_Omega110, WRITE)} )
I_cg_Omega110 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega110, RW), 'B': (I_Omega110, READ)} )
hOmega111=FunctionSpace(mesh,'DG', 0)
I_Omega111 = Function(hOmega111)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega111), {'f': (I_Omega111, WRITE)} )
I_cg_Omega111 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega111, RW), 'B': (I_Omega111, READ)} )
hOmega112=FunctionSpace(mesh,'DG', 0)
I_Omega112 = Function(hOmega112)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega112), {'f': (I_Omega112, WRITE)} )
I_cg_Omega112 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega112, RW), 'B': (I_Omega112, READ)} )
hOmega113=FunctionSpace(mesh,'DG', 0)
I_Omega113 = Function(hOmega113)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega113), {'f': (I_Omega113, WRITE)} )
I_cg_Omega113 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega113, RW), 'B': (I_Omega113, READ)} )
hOmega114=FunctionSpace(mesh,'DG', 0)
I_Omega114 = Function(hOmega114)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega114), {'f': (I_Omega114, WRITE)} )
I_cg_Omega114 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega114, RW), 'B': (I_Omega114, READ)} )
hOmega115=FunctionSpace(mesh,'DG', 0)
I_Omega115 = Function(hOmega115)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega115), {'f': (I_Omega115, WRITE)} )
I_cg_Omega115 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega115, RW), 'B': (I_Omega115, READ)} )
hOmega116=FunctionSpace(mesh,'DG', 0)
I_Omega116 = Function(hOmega116)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega116), {'f': (I_Omega116, WRITE)} )
I_cg_Omega116 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega116, RW), 'B': (I_Omega116, READ)} )
hOmega117=FunctionSpace(mesh,'DG', 0)
I_Omega117 = Function(hOmega117)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega117), {'f': (I_Omega117, WRITE)} )
I_cg_Omega117 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega117, RW), 'B': (I_Omega117, READ)} )
hOmega118=FunctionSpace(mesh,'DG', 0)
I_Omega118 = Function(hOmega118)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega118), {'f': (I_Omega118, WRITE)} )
I_cg_Omega118 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega118, RW), 'B': (I_Omega118, READ)} )
hOmega119=FunctionSpace(mesh,'DG', 0)
I_Omega119 = Function(hOmega119)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega119), {'f': (I_Omega119, WRITE)} )
I_cg_Omega119 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega119, RW), 'B': (I_Omega119, READ)} )
hOmega120=FunctionSpace(mesh,'DG', 0)
I_Omega120 = Function(hOmega120)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega120), {'f': (I_Omega120, WRITE)} )
I_cg_Omega120 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega120, RW), 'B': (I_Omega120, READ)} )
hOmega121=FunctionSpace(mesh,'DG', 0)
I_Omega121 = Function(hOmega121)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega121), {'f': (I_Omega121, WRITE)} )
I_cg_Omega121 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega121, RW), 'B': (I_Omega121, READ)} )
hOmega122=FunctionSpace(mesh,'DG', 0)
I_Omega122 = Function(hOmega122)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega122), {'f': (I_Omega122, WRITE)} )
I_cg_Omega122 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega122, RW), 'B': (I_Omega122, READ)} )
hOmega123=FunctionSpace(mesh,'DG', 0)
I_Omega123 = Function(hOmega123)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega123), {'f': (I_Omega123, WRITE)} )
I_cg_Omega123 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega123, RW), 'B': (I_Omega123, READ)} )
hOmega124=FunctionSpace(mesh,'DG', 0)
I_Omega124 = Function(hOmega124)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega124), {'f': (I_Omega124, WRITE)} )
I_cg_Omega124 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega124, RW), 'B': (I_Omega124, READ)} )
hOmega125=FunctionSpace(mesh,'DG', 0)
I_Omega125 = Function(hOmega125)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega125), {'f': (I_Omega125, WRITE)} )
I_cg_Omega125 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega125, RW), 'B': (I_Omega125, READ)} )
hOmega126=FunctionSpace(mesh,'DG', 0)
I_Omega126 = Function(hOmega126)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega126), {'f': (I_Omega126, WRITE)} )
I_cg_Omega126 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega126, RW), 'B': (I_Omega126, READ)} )
hOmega127=FunctionSpace(mesh,'DG', 0)
I_Omega127 = Function(hOmega127)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega127), {'f': (I_Omega127, WRITE)} )
I_cg_Omega127 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega127, RW), 'B': (I_Omega127, READ)} )
hOmega128=FunctionSpace(mesh,'DG', 0)
I_Omega128 = Function(hOmega128)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega128), {'f': (I_Omega128, WRITE)} )
I_cg_Omega128 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega128, RW), 'B': (I_Omega128, READ)} )
hOmega129=FunctionSpace(mesh,'DG', 0)
I_Omega129 = Function(hOmega129)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega129), {'f': (I_Omega129, WRITE)} )
I_cg_Omega129 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega129, RW), 'B': (I_Omega129, READ)} )
hOmega130=FunctionSpace(mesh,'DG', 0)
I_Omega130 = Function(hOmega130)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega130), {'f': (I_Omega130, WRITE)} )
I_cg_Omega130 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega130, RW), 'B': (I_Omega130, READ)} )
hOmega131=FunctionSpace(mesh,'DG', 0)
I_Omega131 = Function(hOmega131)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega131), {'f': (I_Omega131, WRITE)} )
I_cg_Omega131 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega131, RW), 'B': (I_Omega131, READ)} )
hOmega132=FunctionSpace(mesh,'DG', 0)
I_Omega132 = Function(hOmega132)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega132), {'f': (I_Omega132, WRITE)} )
I_cg_Omega132 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega132, RW), 'B': (I_Omega132, READ)} )
hOmega133=FunctionSpace(mesh,'DG', 0)
I_Omega133 = Function(hOmega133)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega133), {'f': (I_Omega133, WRITE)} )
I_cg_Omega133 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega133, RW), 'B': (I_Omega133, READ)} )
hOmega134=FunctionSpace(mesh,'DG', 0)
I_Omega134 = Function(hOmega134)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega134), {'f': (I_Omega134, WRITE)} )
I_cg_Omega134 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega134, RW), 'B': (I_Omega134, READ)} )
hOmega135=FunctionSpace(mesh,'DG', 0)
I_Omega135 = Function(hOmega135)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega135), {'f': (I_Omega135, WRITE)} )
I_cg_Omega135 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega135, RW), 'B': (I_Omega135, READ)} )
hOmega136=FunctionSpace(mesh,'DG', 0)
I_Omega136 = Function(hOmega136)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega136), {'f': (I_Omega136, WRITE)} )
I_cg_Omega136 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega136, RW), 'B': (I_Omega136, READ)} )
hOmega137=FunctionSpace(mesh,'DG', 0)
I_Omega137 = Function(hOmega137)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega137), {'f': (I_Omega137, WRITE)} )
I_cg_Omega137 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega137, RW), 'B': (I_Omega137, READ)} )
hOmega138=FunctionSpace(mesh,'DG', 0)
I_Omega138 = Function(hOmega138)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega138), {'f': (I_Omega138, WRITE)} )
I_cg_Omega138 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega138, RW), 'B': (I_Omega138, READ)} )
hOmega139=FunctionSpace(mesh,'DG', 0)
I_Omega139 = Function(hOmega139)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega139), {'f': (I_Omega139, WRITE)} )
I_cg_Omega139 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega139, RW), 'B': (I_Omega139, READ)} )
hOmega140=FunctionSpace(mesh,'DG', 0)
I_Omega140 = Function(hOmega140)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega140), {'f': (I_Omega140, WRITE)} )
I_cg_Omega140 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega140, RW), 'B': (I_Omega140, READ)} )
hOmega141=FunctionSpace(mesh,'DG', 0)
I_Omega141 = Function(hOmega141)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega141), {'f': (I_Omega141, WRITE)} )
I_cg_Omega141 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega141, RW), 'B': (I_Omega141, READ)} )
hOmega142=FunctionSpace(mesh,'DG', 0)
I_Omega142 = Function(hOmega142)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega142), {'f': (I_Omega142, WRITE)} )
I_cg_Omega142 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega142, RW), 'B': (I_Omega142, READ)} )
hOmega143=FunctionSpace(mesh,'DG', 0)
I_Omega143 = Function(hOmega143)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega143), {'f': (I_Omega143, WRITE)} )
I_cg_Omega143 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega143, RW), 'B': (I_Omega143, READ)} )
hOmega144=FunctionSpace(mesh,'DG', 0)
I_Omega144 = Function(hOmega144)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega144), {'f': (I_Omega144, WRITE)} )
I_cg_Omega144 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega144, RW), 'B': (I_Omega144, READ)} )
hOmega145=FunctionSpace(mesh,'DG', 0)
I_Omega145 = Function(hOmega145)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega145), {'f': (I_Omega145, WRITE)} )
I_cg_Omega145 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega145, RW), 'B': (I_Omega145, READ)} )
hOmega146=FunctionSpace(mesh,'DG', 0)
I_Omega146 = Function(hOmega146)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega146), {'f': (I_Omega146, WRITE)} )
I_cg_Omega146 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega146, RW), 'B': (I_Omega146, READ)} )
hOmega147=FunctionSpace(mesh,'DG', 0)
I_Omega147 = Function(hOmega147)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega147), {'f': (I_Omega147, WRITE)} )
I_cg_Omega147 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega147, RW), 'B': (I_Omega147, READ)} )
hOmega148=FunctionSpace(mesh,'DG', 0)
I_Omega148 = Function(hOmega148)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega148), {'f': (I_Omega148, WRITE)} )
I_cg_Omega148 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega148, RW), 'B': (I_Omega148, READ)} )
hOmega149=FunctionSpace(mesh,'DG', 0)
I_Omega149 = Function(hOmega149)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega149), {'f': (I_Omega149, WRITE)} )
I_cg_Omega149 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega149, RW), 'B': (I_Omega149, READ)} )
hOmega150=FunctionSpace(mesh,'DG', 0)
I_Omega150 = Function(hOmega150)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega150), {'f': (I_Omega150, WRITE)} )
I_cg_Omega150 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega150, RW), 'B': (I_Omega150, READ)} )
hOmega151=FunctionSpace(mesh,'DG', 0)
I_Omega151 = Function(hOmega151)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega151), {'f': (I_Omega151, WRITE)} )
I_cg_Omega151 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega151, RW), 'B': (I_Omega151, READ)} )
hOmega152=FunctionSpace(mesh,'DG', 0)
I_Omega152 = Function(hOmega152)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega152), {'f': (I_Omega152, WRITE)} )
I_cg_Omega152 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega152, RW), 'B': (I_Omega152, READ)} )
hOmega153=FunctionSpace(mesh,'DG', 0)
I_Omega153 = Function(hOmega153)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega153), {'f': (I_Omega153, WRITE)} )
I_cg_Omega153 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega153, RW), 'B': (I_Omega153, READ)} )
hOmega154=FunctionSpace(mesh,'DG', 0)
I_Omega154 = Function(hOmega154)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega154), {'f': (I_Omega154, WRITE)} )
I_cg_Omega154 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega154, RW), 'B': (I_Omega154, READ)} )
hOmega155=FunctionSpace(mesh,'DG', 0)
I_Omega155 = Function(hOmega155)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega155), {'f': (I_Omega155, WRITE)} )
I_cg_Omega155 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega155, RW), 'B': (I_Omega155, READ)} )
hOmega156=FunctionSpace(mesh,'DG', 0)
I_Omega156 = Function(hOmega156)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega156), {'f': (I_Omega156, WRITE)} )
I_cg_Omega156 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega156, RW), 'B': (I_Omega156, READ)} )
hOmega157=FunctionSpace(mesh,'DG', 0)
I_Omega157 = Function(hOmega157)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega157), {'f': (I_Omega157, WRITE)} )
I_cg_Omega157 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega157, RW), 'B': (I_Omega157, READ)} )
hOmega158=FunctionSpace(mesh,'DG', 0)
I_Omega158 = Function(hOmega158)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega158), {'f': (I_Omega158, WRITE)} )
I_cg_Omega158 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega158, RW), 'B': (I_Omega158, READ)} )
hOmega159=FunctionSpace(mesh,'DG', 0)
I_Omega159 = Function(hOmega159)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega159), {'f': (I_Omega159, WRITE)} )
I_cg_Omega159 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega159, RW), 'B': (I_Omega159, READ)} )
hOmega160=FunctionSpace(mesh,'DG', 0)
I_Omega160 = Function(hOmega160)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega160), {'f': (I_Omega160, WRITE)} )
I_cg_Omega160 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega160, RW), 'B': (I_Omega160, READ)} )
hOmega161=FunctionSpace(mesh,'DG', 0)
I_Omega161 = Function(hOmega161)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega161), {'f': (I_Omega161, WRITE)} )
I_cg_Omega161 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega161, RW), 'B': (I_Omega161, READ)} )
hOmega162=FunctionSpace(mesh,'DG', 0)
I_Omega162 = Function(hOmega162)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega162), {'f': (I_Omega162, WRITE)} )
I_cg_Omega162 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega162, RW), 'B': (I_Omega162, READ)} )
hOmega163=FunctionSpace(mesh,'DG', 0)
I_Omega163 = Function(hOmega163)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega163), {'f': (I_Omega163, WRITE)} )
I_cg_Omega163 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega163, RW), 'B': (I_Omega163, READ)} )
hOmega164=FunctionSpace(mesh,'DG', 0)
I_Omega164 = Function(hOmega164)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega164), {'f': (I_Omega164, WRITE)} )
I_cg_Omega164 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega164, RW), 'B': (I_Omega164, READ)} )
hOmega165=FunctionSpace(mesh,'DG', 0)
I_Omega165 = Function(hOmega165)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega165), {'f': (I_Omega165, WRITE)} )
I_cg_Omega165 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega165, RW), 'B': (I_Omega165, READ)} )
hOmega166=FunctionSpace(mesh,'DG', 0)
I_Omega166 = Function(hOmega166)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega166), {'f': (I_Omega166, WRITE)} )
I_cg_Omega166 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega166, RW), 'B': (I_Omega166, READ)} )
hOmega167=FunctionSpace(mesh,'DG', 0)
I_Omega167 = Function(hOmega167)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega167), {'f': (I_Omega167, WRITE)} )
I_cg_Omega167 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega167, RW), 'B': (I_Omega167, READ)} )
hOmega168=FunctionSpace(mesh,'DG', 0)
I_Omega168 = Function(hOmega168)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega168), {'f': (I_Omega168, WRITE)} )
I_cg_Omega168 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega168, RW), 'B': (I_Omega168, READ)} )
hOmega169=FunctionSpace(mesh,'DG', 0)
I_Omega169 = Function(hOmega169)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega169), {'f': (I_Omega169, WRITE)} )
I_cg_Omega169 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega169, RW), 'B': (I_Omega169, READ)} )
hOmega170=FunctionSpace(mesh,'DG', 0)
I_Omega170 = Function(hOmega170)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega170), {'f': (I_Omega170, WRITE)} )
I_cg_Omega170 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega170, RW), 'B': (I_Omega170, READ)} )
hOmega171=FunctionSpace(mesh,'DG', 0)
I_Omega171 = Function(hOmega171)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega171), {'f': (I_Omega171, WRITE)} )
I_cg_Omega171 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega171, RW), 'B': (I_Omega171, READ)} )
hOmega172=FunctionSpace(mesh,'DG', 0)
I_Omega172 = Function(hOmega172)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega172), {'f': (I_Omega172, WRITE)} )
I_cg_Omega172 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega172, RW), 'B': (I_Omega172, READ)} )
hOmega173=FunctionSpace(mesh,'DG', 0)
I_Omega173 = Function(hOmega173)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega173), {'f': (I_Omega173, WRITE)} )
I_cg_Omega173 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega173, RW), 'B': (I_Omega173, READ)} )
hOmega174=FunctionSpace(mesh,'DG', 0)
I_Omega174 = Function(hOmega174)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega174), {'f': (I_Omega174, WRITE)} )
I_cg_Omega174 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega174, RW), 'B': (I_Omega174, READ)} )
hOmega175=FunctionSpace(mesh,'DG', 0)
I_Omega175 = Function(hOmega175)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega175), {'f': (I_Omega175, WRITE)} )
I_cg_Omega175 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega175, RW), 'B': (I_Omega175, READ)} )
hOmega176=FunctionSpace(mesh,'DG', 0)
I_Omega176 = Function(hOmega176)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega176), {'f': (I_Omega176, WRITE)} )
I_cg_Omega176 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega176, RW), 'B': (I_Omega176, READ)} )
hOmega177=FunctionSpace(mesh,'DG', 0)
I_Omega177 = Function(hOmega177)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega177), {'f': (I_Omega177, WRITE)} )
I_cg_Omega177 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega177, RW), 'B': (I_Omega177, READ)} )
hOmega178=FunctionSpace(mesh,'DG', 0)
I_Omega178 = Function(hOmega178)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega178), {'f': (I_Omega178, WRITE)} )
I_cg_Omega178 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega178, RW), 'B': (I_Omega178, READ)} )
hOmega179=FunctionSpace(mesh,'DG', 0)
I_Omega179 = Function(hOmega179)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega179), {'f': (I_Omega179, WRITE)} )
I_cg_Omega179 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega179, RW), 'B': (I_Omega179, READ)} )
hOmega180=FunctionSpace(mesh,'DG', 0)
I_Omega180 = Function(hOmega180)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega180), {'f': (I_Omega180, WRITE)} )
I_cg_Omega180 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega180, RW), 'B': (I_Omega180, READ)} )
hOmega181=FunctionSpace(mesh,'DG', 0)
I_Omega181 = Function(hOmega181)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega181), {'f': (I_Omega181, WRITE)} )
I_cg_Omega181 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega181, RW), 'B': (I_Omega181, READ)} )
hOmega182=FunctionSpace(mesh,'DG', 0)
I_Omega182 = Function(hOmega182)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega182), {'f': (I_Omega182, WRITE)} )
I_cg_Omega182 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega182, RW), 'B': (I_Omega182, READ)} )
hOmega183=FunctionSpace(mesh,'DG', 0)
I_Omega183 = Function(hOmega183)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega183), {'f': (I_Omega183, WRITE)} )
I_cg_Omega183 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega183, RW), 'B': (I_Omega183, READ)} )
hOmega184=FunctionSpace(mesh,'DG', 0)
I_Omega184 = Function(hOmega184)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega184), {'f': (I_Omega184, WRITE)} )
I_cg_Omega184 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega184, RW), 'B': (I_Omega184, READ)} )
hOmega185=FunctionSpace(mesh,'DG', 0)
I_Omega185 = Function(hOmega185)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega185), {'f': (I_Omega185, WRITE)} )
I_cg_Omega185 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega185, RW), 'B': (I_Omega185, READ)} )
hOmega186=FunctionSpace(mesh,'DG', 0)
I_Omega186 = Function(hOmega186)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega186), {'f': (I_Omega186, WRITE)} )
I_cg_Omega186 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega186, RW), 'B': (I_Omega186, READ)} )
hOmega187=FunctionSpace(mesh,'DG', 0)
I_Omega187 = Function(hOmega187)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega187), {'f': (I_Omega187, WRITE)} )
I_cg_Omega187 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega187, RW), 'B': (I_Omega187, READ)} )
hOmega188=FunctionSpace(mesh,'DG', 0)
I_Omega188 = Function(hOmega188)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega188), {'f': (I_Omega188, WRITE)} )
I_cg_Omega188 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega188, RW), 'B': (I_Omega188, READ)} )
hOmega189=FunctionSpace(mesh,'DG', 0)
I_Omega189 = Function(hOmega189)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega189), {'f': (I_Omega189, WRITE)} )
I_cg_Omega189 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega189, RW), 'B': (I_Omega189, READ)} )
hOmega190=FunctionSpace(mesh,'DG', 0)
I_Omega190 = Function(hOmega190)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega190), {'f': (I_Omega190, WRITE)} )
I_cg_Omega190 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega190, RW), 'B': (I_Omega190, READ)} )
hOmega191=FunctionSpace(mesh,'DG', 0)
I_Omega191 = Function(hOmega191)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega191), {'f': (I_Omega191, WRITE)} )
I_cg_Omega191 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega191, RW), 'B': (I_Omega191, READ)} )
hOmega192=FunctionSpace(mesh,'DG', 0)
I_Omega192 = Function(hOmega192)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega192), {'f': (I_Omega192, WRITE)} )
I_cg_Omega192 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega192, RW), 'B': (I_Omega192, READ)} )
hOmega193=FunctionSpace(mesh,'DG', 0)
I_Omega193 = Function(hOmega193)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega193), {'f': (I_Omega193, WRITE)} )
I_cg_Omega193 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega193, RW), 'B': (I_Omega193, READ)} )
hOmega194=FunctionSpace(mesh,'DG', 0)
I_Omega194 = Function(hOmega194)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega194), {'f': (I_Omega194, WRITE)} )
I_cg_Omega194 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega194, RW), 'B': (I_Omega194, READ)} )
hOmega195=FunctionSpace(mesh,'DG', 0)
I_Omega195 = Function(hOmega195)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega195), {'f': (I_Omega195, WRITE)} )
I_cg_Omega195 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega195, RW), 'B': (I_Omega195, READ)} )
hOmega196=FunctionSpace(mesh,'DG', 0)
I_Omega196 = Function(hOmega196)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega196), {'f': (I_Omega196, WRITE)} )
I_cg_Omega196 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega196, RW), 'B': (I_Omega196, READ)} )
hOmega197=FunctionSpace(mesh,'DG', 0)
I_Omega197 = Function(hOmega197)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega197), {'f': (I_Omega197, WRITE)} )
I_cg_Omega197 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega197, RW), 'B': (I_Omega197, READ)} )
hOmega198=FunctionSpace(mesh,'DG', 0)
I_Omega198 = Function(hOmega198)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega198), {'f': (I_Omega198, WRITE)} )
I_cg_Omega198 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega198, RW), 'B': (I_Omega198, READ)} )
hOmega199=FunctionSpace(mesh,'DG', 0)
I_Omega199 = Function(hOmega199)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega199), {'f': (I_Omega199, WRITE)} )
I_cg_Omega199 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega199, RW), 'B': (I_Omega199, READ)} )
hOmega200=FunctionSpace(mesh,'DG', 0)
I_Omega200 = Function(hOmega200)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega200), {'f': (I_Omega200, WRITE)} )
I_cg_Omega200 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega200, RW), 'B': (I_Omega200, READ)} )
hOmega201=FunctionSpace(mesh,'DG', 0)
I_Omega201 = Function(hOmega201)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega201), {'f': (I_Omega201, WRITE)} )
I_cg_Omega201 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega201, RW), 'B': (I_Omega201, READ)} )
hOmega202=FunctionSpace(mesh,'DG', 0)
I_Omega202 = Function(hOmega202)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega202), {'f': (I_Omega202, WRITE)} )
I_cg_Omega202 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega202, RW), 'B': (I_Omega202, READ)} )
hOmega203=FunctionSpace(mesh,'DG', 0)
I_Omega203 = Function(hOmega203)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega203), {'f': (I_Omega203, WRITE)} )
I_cg_Omega203 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega203, RW), 'B': (I_Omega203, READ)} )
hOmega204=FunctionSpace(mesh,'DG', 0)
I_Omega204 = Function(hOmega204)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega204), {'f': (I_Omega204, WRITE)} )
I_cg_Omega204 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega204, RW), 'B': (I_Omega204, READ)} )
hOmega205=FunctionSpace(mesh,'DG', 0)
I_Omega205 = Function(hOmega205)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega205), {'f': (I_Omega205, WRITE)} )
I_cg_Omega205 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega205, RW), 'B': (I_Omega205, READ)} )
hOmega206=FunctionSpace(mesh,'DG', 0)
I_Omega206 = Function(hOmega206)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega206), {'f': (I_Omega206, WRITE)} )
I_cg_Omega206 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega206, RW), 'B': (I_Omega206, READ)} )
hOmega207=FunctionSpace(mesh,'DG', 0)
I_Omega207 = Function(hOmega207)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega207), {'f': (I_Omega207, WRITE)} )
I_cg_Omega207 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega207, RW), 'B': (I_Omega207, READ)} )
hOmega208=FunctionSpace(mesh,'DG', 0)
I_Omega208 = Function(hOmega208)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega208), {'f': (I_Omega208, WRITE)} )
I_cg_Omega208 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega208, RW), 'B': (I_Omega208, READ)} )
hOmega209=FunctionSpace(mesh,'DG', 0)
I_Omega209 = Function(hOmega209)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega209), {'f': (I_Omega209, WRITE)} )
I_cg_Omega209 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega209, RW), 'B': (I_Omega209, READ)} )
hOmega210=FunctionSpace(mesh,'DG', 0)
I_Omega210 = Function(hOmega210)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega210), {'f': (I_Omega210, WRITE)} )
I_cg_Omega210 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega210, RW), 'B': (I_Omega210, READ)} )
hOmega211=FunctionSpace(mesh,'DG', 0)
I_Omega211 = Function(hOmega211)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega211), {'f': (I_Omega211, WRITE)} )
I_cg_Omega211 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega211, RW), 'B': (I_Omega211, READ)} )
hOmega212=FunctionSpace(mesh,'DG', 0)
I_Omega212 = Function(hOmega212)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega212), {'f': (I_Omega212, WRITE)} )
I_cg_Omega212 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega212, RW), 'B': (I_Omega212, READ)} )
hOmega213=FunctionSpace(mesh,'DG', 0)
I_Omega213 = Function(hOmega213)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega213), {'f': (I_Omega213, WRITE)} )
I_cg_Omega213 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega213, RW), 'B': (I_Omega213, READ)} )
hOmega214=FunctionSpace(mesh,'DG', 0)
I_Omega214 = Function(hOmega214)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega214), {'f': (I_Omega214, WRITE)} )
I_cg_Omega214 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega214, RW), 'B': (I_Omega214, READ)} )
hOmega215=FunctionSpace(mesh,'DG', 0)
I_Omega215 = Function(hOmega215)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega215), {'f': (I_Omega215, WRITE)} )
I_cg_Omega215 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega215, RW), 'B': (I_Omega215, READ)} )
hOmega216=FunctionSpace(mesh,'DG', 0)
I_Omega216 = Function(hOmega216)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega216), {'f': (I_Omega216, WRITE)} )
I_cg_Omega216 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega216, RW), 'B': (I_Omega216, READ)} )
hOmega217=FunctionSpace(mesh,'DG', 0)
I_Omega217 = Function(hOmega217)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega217), {'f': (I_Omega217, WRITE)} )
I_cg_Omega217 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega217, RW), 'B': (I_Omega217, READ)} )
hOmega218=FunctionSpace(mesh,'DG', 0)
I_Omega218 = Function(hOmega218)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega218), {'f': (I_Omega218, WRITE)} )
I_cg_Omega218 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega218, RW), 'B': (I_Omega218, READ)} )
hOmega219=FunctionSpace(mesh,'DG', 0)
I_Omega219 = Function(hOmega219)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega219), {'f': (I_Omega219, WRITE)} )
I_cg_Omega219 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega219, RW), 'B': (I_Omega219, READ)} )
hOmega220=FunctionSpace(mesh,'DG', 0)
I_Omega220 = Function(hOmega220)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega220), {'f': (I_Omega220, WRITE)} )
I_cg_Omega220 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega220, RW), 'B': (I_Omega220, READ)} )
hOmega221=FunctionSpace(mesh,'DG', 0)
I_Omega221 = Function(hOmega221)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega221), {'f': (I_Omega221, WRITE)} )
I_cg_Omega221 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega221, RW), 'B': (I_Omega221, READ)} )
hOmega222=FunctionSpace(mesh,'DG', 0)
I_Omega222 = Function(hOmega222)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega222), {'f': (I_Omega222, WRITE)} )
I_cg_Omega222 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega222, RW), 'B': (I_Omega222, READ)} )
hOmega223=FunctionSpace(mesh,'DG', 0)
I_Omega223 = Function(hOmega223)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega223), {'f': (I_Omega223, WRITE)} )
I_cg_Omega223 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega223, RW), 'B': (I_Omega223, READ)} )
hOmega224=FunctionSpace(mesh,'DG', 0)
I_Omega224 = Function(hOmega224)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega224), {'f': (I_Omega224, WRITE)} )
I_cg_Omega224 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega224, RW), 'B': (I_Omega224, READ)} )
hOmega225=FunctionSpace(mesh,'DG', 0)
I_Omega225 = Function(hOmega225)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega225), {'f': (I_Omega225, WRITE)} )
I_cg_Omega225 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega225, RW), 'B': (I_Omega225, READ)} )
hOmega226=FunctionSpace(mesh,'DG', 0)
I_Omega226 = Function(hOmega226)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega226), {'f': (I_Omega226, WRITE)} )
I_cg_Omega226 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega226, RW), 'B': (I_Omega226, READ)} )
hOmega227=FunctionSpace(mesh,'DG', 0)
I_Omega227 = Function(hOmega227)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega227), {'f': (I_Omega227, WRITE)} )
I_cg_Omega227 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega227, RW), 'B': (I_Omega227, READ)} )
hOmega228=FunctionSpace(mesh,'DG', 0)
I_Omega228 = Function(hOmega228)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega228), {'f': (I_Omega228, WRITE)} )
I_cg_Omega228 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega228, RW), 'B': (I_Omega228, READ)} )
hOmega229=FunctionSpace(mesh,'DG', 0)
I_Omega229 = Function(hOmega229)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega229), {'f': (I_Omega229, WRITE)} )
I_cg_Omega229 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega229, RW), 'B': (I_Omega229, READ)} )
hOmega230=FunctionSpace(mesh,'DG', 0)
I_Omega230 = Function(hOmega230)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega230), {'f': (I_Omega230, WRITE)} )
I_cg_Omega230 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega230, RW), 'B': (I_Omega230, READ)} )
hOmega231=FunctionSpace(mesh,'DG', 0)
I_Omega231 = Function(hOmega231)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega231), {'f': (I_Omega231, WRITE)} )
I_cg_Omega231 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega231, RW), 'B': (I_Omega231, READ)} )
hOmega232=FunctionSpace(mesh,'DG', 0)
I_Omega232 = Function(hOmega232)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega232), {'f': (I_Omega232, WRITE)} )
I_cg_Omega232 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega232, RW), 'B': (I_Omega232, READ)} )
hOmega233=FunctionSpace(mesh,'DG', 0)
I_Omega233 = Function(hOmega233)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega233), {'f': (I_Omega233, WRITE)} )
I_cg_Omega233 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega233, RW), 'B': (I_Omega233, READ)} )
hOmega234=FunctionSpace(mesh,'DG', 0)
I_Omega234 = Function(hOmega234)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega234), {'f': (I_Omega234, WRITE)} )
I_cg_Omega234 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega234, RW), 'B': (I_Omega234, READ)} )
hOmega235=FunctionSpace(mesh,'DG', 0)
I_Omega235 = Function(hOmega235)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega235), {'f': (I_Omega235, WRITE)} )
I_cg_Omega235 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega235, RW), 'B': (I_Omega235, READ)} )
hOmega236=FunctionSpace(mesh,'DG', 0)
I_Omega236 = Function(hOmega236)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega236), {'f': (I_Omega236, WRITE)} )
I_cg_Omega236 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega236, RW), 'B': (I_Omega236, READ)} )
hOmega237=FunctionSpace(mesh,'DG', 0)
I_Omega237 = Function(hOmega237)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega237), {'f': (I_Omega237, WRITE)} )
I_cg_Omega237 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega237, RW), 'B': (I_Omega237, READ)} )
hOmega238=FunctionSpace(mesh,'DG', 0)
I_Omega238 = Function(hOmega238)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega238), {'f': (I_Omega238, WRITE)} )
I_cg_Omega238 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega238, RW), 'B': (I_Omega238, READ)} )
hOmega239=FunctionSpace(mesh,'DG', 0)
I_Omega239 = Function(hOmega239)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega239), {'f': (I_Omega239, WRITE)} )
I_cg_Omega239 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega239, RW), 'B': (I_Omega239, READ)} )
hOmega240=FunctionSpace(mesh,'DG', 0)
I_Omega240 = Function(hOmega240)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega240), {'f': (I_Omega240, WRITE)} )
I_cg_Omega240 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega240, RW), 'B': (I_Omega240, READ)} )
hOmega241=FunctionSpace(mesh,'DG', 0)
I_Omega241 = Function(hOmega241)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega241), {'f': (I_Omega241, WRITE)} )
I_cg_Omega241 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega241, RW), 'B': (I_Omega241, READ)} )
hOmega242=FunctionSpace(mesh,'DG', 0)
I_Omega242 = Function(hOmega242)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega242), {'f': (I_Omega242, WRITE)} )
I_cg_Omega242 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega242, RW), 'B': (I_Omega242, READ)} )
hOmega243=FunctionSpace(mesh,'DG', 0)
I_Omega243 = Function(hOmega243)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega243), {'f': (I_Omega243, WRITE)} )
I_cg_Omega243 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega243, RW), 'B': (I_Omega243, READ)} )
hOmega244=FunctionSpace(mesh,'DG', 0)
I_Omega244 = Function(hOmega244)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega244), {'f': (I_Omega244, WRITE)} )
I_cg_Omega244 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega244, RW), 'B': (I_Omega244, READ)} )
hOmega245=FunctionSpace(mesh,'DG', 0)
I_Omega245 = Function(hOmega245)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega245), {'f': (I_Omega245, WRITE)} )
I_cg_Omega245 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega245, RW), 'B': (I_Omega245, READ)} )
hOmega246=FunctionSpace(mesh,'DG', 0)
I_Omega246 = Function(hOmega246)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega246), {'f': (I_Omega246, WRITE)} )
I_cg_Omega246 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega246, RW), 'B': (I_Omega246, READ)} )
hOmega247=FunctionSpace(mesh,'DG', 0)
I_Omega247 = Function(hOmega247)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega247), {'f': (I_Omega247, WRITE)} )
I_cg_Omega247 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega247, RW), 'B': (I_Omega247, READ)} )
hOmega248=FunctionSpace(mesh,'DG', 0)
I_Omega248 = Function(hOmega248)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega248), {'f': (I_Omega248, WRITE)} )
I_cg_Omega248 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega248, RW), 'B': (I_Omega248, READ)} )
hOmega249=FunctionSpace(mesh,'DG', 0)
I_Omega249 = Function(hOmega249)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega249), {'f': (I_Omega249, WRITE)} )
I_cg_Omega249 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega249, RW), 'B': (I_Omega249, READ)} )
hOmega250=FunctionSpace(mesh,'DG', 0)
I_Omega250 = Function(hOmega250)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega250), {'f': (I_Omega250, WRITE)} )
I_cg_Omega250 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega250, RW), 'B': (I_Omega250, READ)} )
hOmega251=FunctionSpace(mesh,'DG', 0)
I_Omega251 = Function(hOmega251)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega251), {'f': (I_Omega251, WRITE)} )
I_cg_Omega251 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega251, RW), 'B': (I_Omega251, READ)} )
hOmega252=FunctionSpace(mesh,'DG', 0)
I_Omega252 = Function(hOmega252)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega252), {'f': (I_Omega252, WRITE)} )
I_cg_Omega252 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega252, RW), 'B': (I_Omega252, READ)} )
hOmega253=FunctionSpace(mesh,'DG', 0)
I_Omega253 = Function(hOmega253)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega253), {'f': (I_Omega253, WRITE)} )
I_cg_Omega253 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega253, RW), 'B': (I_Omega253, READ)} )
hOmega254=FunctionSpace(mesh,'DG', 0)
I_Omega254 = Function(hOmega254)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega254), {'f': (I_Omega254, WRITE)} )
I_cg_Omega254 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega254, RW), 'B': (I_Omega254, READ)} )
hOmega255=FunctionSpace(mesh,'DG', 0)
I_Omega255 = Function(hOmega255)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega255), {'f': (I_Omega255, WRITE)} )
I_cg_Omega255 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega255, RW), 'B': (I_Omega255, READ)} )
hOmega256=FunctionSpace(mesh,'DG', 0)
I_Omega256 = Function(hOmega256)
par_loop( 'for ( int i=0; i < f.dofs; i++ ) f[i][0] = 1.0;', dx(Omega256), {'f': (I_Omega256, WRITE)} )
I_cg_Omega256 = Function(V)
par_loop( 'for (int i=0; i<A.dofs; i++) A[i][0] = fmax(A[i][0], B[0][0]);',dx, {'A' : (I_cg_Omega256, RW), 'B': (I_Omega256, READ)} )

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
BC_Omega65_only=MyBC(V,0,I_cg_Omega65)
BC_Omega66_only=MyBC(V,0,I_cg_Omega66)
BC_Omega67_only=MyBC(V,0,I_cg_Omega67)
BC_Omega68_only=MyBC(V,0,I_cg_Omega68)
BC_Omega69_only=MyBC(V,0,I_cg_Omega69)
BC_Omega70_only=MyBC(V,0,I_cg_Omega70)
BC_Omega71_only=MyBC(V,0,I_cg_Omega71)
BC_Omega72_only=MyBC(V,0,I_cg_Omega72)
BC_Omega73_only=MyBC(V,0,I_cg_Omega73)
BC_Omega74_only=MyBC(V,0,I_cg_Omega74)
BC_Omega75_only=MyBC(V,0,I_cg_Omega75)
BC_Omega76_only=MyBC(V,0,I_cg_Omega76)
BC_Omega77_only=MyBC(V,0,I_cg_Omega77)
BC_Omega78_only=MyBC(V,0,I_cg_Omega78)
BC_Omega79_only=MyBC(V,0,I_cg_Omega79)
BC_Omega80_only=MyBC(V,0,I_cg_Omega80)
BC_Omega81_only=MyBC(V,0,I_cg_Omega81)
BC_Omega82_only=MyBC(V,0,I_cg_Omega82)
BC_Omega83_only=MyBC(V,0,I_cg_Omega83)
BC_Omega84_only=MyBC(V,0,I_cg_Omega84)
BC_Omega85_only=MyBC(V,0,I_cg_Omega85)
BC_Omega86_only=MyBC(V,0,I_cg_Omega86)
BC_Omega87_only=MyBC(V,0,I_cg_Omega87)
BC_Omega88_only=MyBC(V,0,I_cg_Omega88)
BC_Omega89_only=MyBC(V,0,I_cg_Omega89)
BC_Omega90_only=MyBC(V,0,I_cg_Omega90)
BC_Omega91_only=MyBC(V,0,I_cg_Omega91)
BC_Omega92_only=MyBC(V,0,I_cg_Omega92)
BC_Omega93_only=MyBC(V,0,I_cg_Omega93)
BC_Omega94_only=MyBC(V,0,I_cg_Omega94)
BC_Omega95_only=MyBC(V,0,I_cg_Omega95)
BC_Omega96_only=MyBC(V,0,I_cg_Omega96)
BC_Omega97_only=MyBC(V,0,I_cg_Omega97)
BC_Omega98_only=MyBC(V,0,I_cg_Omega98)
BC_Omega99_only=MyBC(V,0,I_cg_Omega99)
BC_Omega100_only=MyBC(V,0,I_cg_Omega100)
BC_Omega101_only=MyBC(V,0,I_cg_Omega101)
BC_Omega102_only=MyBC(V,0,I_cg_Omega102)
BC_Omega103_only=MyBC(V,0,I_cg_Omega103)
BC_Omega104_only=MyBC(V,0,I_cg_Omega104)
BC_Omega105_only=MyBC(V,0,I_cg_Omega105)
BC_Omega106_only=MyBC(V,0,I_cg_Omega106)
BC_Omega107_only=MyBC(V,0,I_cg_Omega107)
BC_Omega108_only=MyBC(V,0,I_cg_Omega108)
BC_Omega109_only=MyBC(V,0,I_cg_Omega109)
BC_Omega110_only=MyBC(V,0,I_cg_Omega110)
BC_Omega111_only=MyBC(V,0,I_cg_Omega111)
BC_Omega112_only=MyBC(V,0,I_cg_Omega112)
BC_Omega113_only=MyBC(V,0,I_cg_Omega113)
BC_Omega114_only=MyBC(V,0,I_cg_Omega114)
BC_Omega115_only=MyBC(V,0,I_cg_Omega115)
BC_Omega116_only=MyBC(V,0,I_cg_Omega116)
BC_Omega117_only=MyBC(V,0,I_cg_Omega117)
BC_Omega118_only=MyBC(V,0,I_cg_Omega118)
BC_Omega119_only=MyBC(V,0,I_cg_Omega119)
BC_Omega120_only=MyBC(V,0,I_cg_Omega120)
BC_Omega121_only=MyBC(V,0,I_cg_Omega121)
BC_Omega122_only=MyBC(V,0,I_cg_Omega122)
BC_Omega123_only=MyBC(V,0,I_cg_Omega123)
BC_Omega124_only=MyBC(V,0,I_cg_Omega124)
BC_Omega125_only=MyBC(V,0,I_cg_Omega125)
BC_Omega126_only=MyBC(V,0,I_cg_Omega126)
BC_Omega127_only=MyBC(V,0,I_cg_Omega127)
BC_Omega128_only=MyBC(V,0,I_cg_Omega128)
BC_Omega129_only=MyBC(V,0,I_cg_Omega129)
BC_Omega130_only=MyBC(V,0,I_cg_Omega130)
BC_Omega131_only=MyBC(V,0,I_cg_Omega131)
BC_Omega132_only=MyBC(V,0,I_cg_Omega132)
BC_Omega133_only=MyBC(V,0,I_cg_Omega133)
BC_Omega134_only=MyBC(V,0,I_cg_Omega134)
BC_Omega135_only=MyBC(V,0,I_cg_Omega135)
BC_Omega136_only=MyBC(V,0,I_cg_Omega136)
BC_Omega137_only=MyBC(V,0,I_cg_Omega137)
BC_Omega138_only=MyBC(V,0,I_cg_Omega138)
BC_Omega139_only=MyBC(V,0,I_cg_Omega139)
BC_Omega140_only=MyBC(V,0,I_cg_Omega140)
BC_Omega141_only=MyBC(V,0,I_cg_Omega141)
BC_Omega142_only=MyBC(V,0,I_cg_Omega142)
BC_Omega143_only=MyBC(V,0,I_cg_Omega143)
BC_Omega144_only=MyBC(V,0,I_cg_Omega144)
BC_Omega145_only=MyBC(V,0,I_cg_Omega145)
BC_Omega146_only=MyBC(V,0,I_cg_Omega146)
BC_Omega147_only=MyBC(V,0,I_cg_Omega147)
BC_Omega148_only=MyBC(V,0,I_cg_Omega148)
BC_Omega149_only=MyBC(V,0,I_cg_Omega149)
BC_Omega150_only=MyBC(V,0,I_cg_Omega150)
BC_Omega151_only=MyBC(V,0,I_cg_Omega151)
BC_Omega152_only=MyBC(V,0,I_cg_Omega152)
BC_Omega153_only=MyBC(V,0,I_cg_Omega153)
BC_Omega154_only=MyBC(V,0,I_cg_Omega154)
BC_Omega155_only=MyBC(V,0,I_cg_Omega155)
BC_Omega156_only=MyBC(V,0,I_cg_Omega156)
BC_Omega157_only=MyBC(V,0,I_cg_Omega157)
BC_Omega158_only=MyBC(V,0,I_cg_Omega158)
BC_Omega159_only=MyBC(V,0,I_cg_Omega159)
BC_Omega160_only=MyBC(V,0,I_cg_Omega160)
BC_Omega161_only=MyBC(V,0,I_cg_Omega161)
BC_Omega162_only=MyBC(V,0,I_cg_Omega162)
BC_Omega163_only=MyBC(V,0,I_cg_Omega163)
BC_Omega164_only=MyBC(V,0,I_cg_Omega164)
BC_Omega165_only=MyBC(V,0,I_cg_Omega165)
BC_Omega166_only=MyBC(V,0,I_cg_Omega166)
BC_Omega167_only=MyBC(V,0,I_cg_Omega167)
BC_Omega168_only=MyBC(V,0,I_cg_Omega168)
BC_Omega169_only=MyBC(V,0,I_cg_Omega169)
BC_Omega170_only=MyBC(V,0,I_cg_Omega170)
BC_Omega171_only=MyBC(V,0,I_cg_Omega171)
BC_Omega172_only=MyBC(V,0,I_cg_Omega172)
BC_Omega173_only=MyBC(V,0,I_cg_Omega173)
BC_Omega174_only=MyBC(V,0,I_cg_Omega174)
BC_Omega175_only=MyBC(V,0,I_cg_Omega175)
BC_Omega176_only=MyBC(V,0,I_cg_Omega176)
BC_Omega177_only=MyBC(V,0,I_cg_Omega177)
BC_Omega178_only=MyBC(V,0,I_cg_Omega178)
BC_Omega179_only=MyBC(V,0,I_cg_Omega179)
BC_Omega180_only=MyBC(V,0,I_cg_Omega180)
BC_Omega181_only=MyBC(V,0,I_cg_Omega181)
BC_Omega182_only=MyBC(V,0,I_cg_Omega182)
BC_Omega183_only=MyBC(V,0,I_cg_Omega183)
BC_Omega184_only=MyBC(V,0,I_cg_Omega184)
BC_Omega185_only=MyBC(V,0,I_cg_Omega185)
BC_Omega186_only=MyBC(V,0,I_cg_Omega186)
BC_Omega187_only=MyBC(V,0,I_cg_Omega187)
BC_Omega188_only=MyBC(V,0,I_cg_Omega188)
BC_Omega189_only=MyBC(V,0,I_cg_Omega189)
BC_Omega190_only=MyBC(V,0,I_cg_Omega190)
BC_Omega191_only=MyBC(V,0,I_cg_Omega191)
BC_Omega192_only=MyBC(V,0,I_cg_Omega192)
BC_Omega193_only=MyBC(V,0,I_cg_Omega193)
BC_Omega194_only=MyBC(V,0,I_cg_Omega194)
BC_Omega195_only=MyBC(V,0,I_cg_Omega195)
BC_Omega196_only=MyBC(V,0,I_cg_Omega196)
BC_Omega197_only=MyBC(V,0,I_cg_Omega197)
BC_Omega198_only=MyBC(V,0,I_cg_Omega198)
BC_Omega199_only=MyBC(V,0,I_cg_Omega199)
BC_Omega200_only=MyBC(V,0,I_cg_Omega200)
BC_Omega201_only=MyBC(V,0,I_cg_Omega201)
BC_Omega202_only=MyBC(V,0,I_cg_Omega202)
BC_Omega203_only=MyBC(V,0,I_cg_Omega203)
BC_Omega204_only=MyBC(V,0,I_cg_Omega204)
BC_Omega205_only=MyBC(V,0,I_cg_Omega205)
BC_Omega206_only=MyBC(V,0,I_cg_Omega206)
BC_Omega207_only=MyBC(V,0,I_cg_Omega207)
BC_Omega208_only=MyBC(V,0,I_cg_Omega208)
BC_Omega209_only=MyBC(V,0,I_cg_Omega209)
BC_Omega210_only=MyBC(V,0,I_cg_Omega210)
BC_Omega211_only=MyBC(V,0,I_cg_Omega211)
BC_Omega212_only=MyBC(V,0,I_cg_Omega212)
BC_Omega213_only=MyBC(V,0,I_cg_Omega213)
BC_Omega214_only=MyBC(V,0,I_cg_Omega214)
BC_Omega215_only=MyBC(V,0,I_cg_Omega215)
BC_Omega216_only=MyBC(V,0,I_cg_Omega216)
BC_Omega217_only=MyBC(V,0,I_cg_Omega217)
BC_Omega218_only=MyBC(V,0,I_cg_Omega218)
BC_Omega219_only=MyBC(V,0,I_cg_Omega219)
BC_Omega220_only=MyBC(V,0,I_cg_Omega220)
BC_Omega221_only=MyBC(V,0,I_cg_Omega221)
BC_Omega222_only=MyBC(V,0,I_cg_Omega222)
BC_Omega223_only=MyBC(V,0,I_cg_Omega223)
BC_Omega224_only=MyBC(V,0,I_cg_Omega224)
BC_Omega225_only=MyBC(V,0,I_cg_Omega225)
BC_Omega226_only=MyBC(V,0,I_cg_Omega226)
BC_Omega227_only=MyBC(V,0,I_cg_Omega227)
BC_Omega228_only=MyBC(V,0,I_cg_Omega228)
BC_Omega229_only=MyBC(V,0,I_cg_Omega229)
BC_Omega230_only=MyBC(V,0,I_cg_Omega230)
BC_Omega231_only=MyBC(V,0,I_cg_Omega231)
BC_Omega232_only=MyBC(V,0,I_cg_Omega232)
BC_Omega233_only=MyBC(V,0,I_cg_Omega233)
BC_Omega234_only=MyBC(V,0,I_cg_Omega234)
BC_Omega235_only=MyBC(V,0,I_cg_Omega235)
BC_Omega236_only=MyBC(V,0,I_cg_Omega236)
BC_Omega237_only=MyBC(V,0,I_cg_Omega237)
BC_Omega238_only=MyBC(V,0,I_cg_Omega238)
BC_Omega239_only=MyBC(V,0,I_cg_Omega239)
BC_Omega240_only=MyBC(V,0,I_cg_Omega240)
BC_Omega241_only=MyBC(V,0,I_cg_Omega241)
BC_Omega242_only=MyBC(V,0,I_cg_Omega242)
BC_Omega243_only=MyBC(V,0,I_cg_Omega243)
BC_Omega244_only=MyBC(V,0,I_cg_Omega244)
BC_Omega245_only=MyBC(V,0,I_cg_Omega245)
BC_Omega246_only=MyBC(V,0,I_cg_Omega246)
BC_Omega247_only=MyBC(V,0,I_cg_Omega247)
BC_Omega248_only=MyBC(V,0,I_cg_Omega248)
BC_Omega249_only=MyBC(V,0,I_cg_Omega249)
BC_Omega250_only=MyBC(V,0,I_cg_Omega250)
BC_Omega251_only=MyBC(V,0,I_cg_Omega251)
BC_Omega252_only=MyBC(V,0,I_cg_Omega252)
BC_Omega253_only=MyBC(V,0,I_cg_Omega253)
BC_Omega254_only=MyBC(V,0,I_cg_Omega254)
BC_Omega255_only=MyBC(V,0,I_cg_Omega255)
BC_Omega256_only=MyBC(V,0,I_cg_Omega256)
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
uOmega65=Function(V)
uOmega65.interpolate(1*coord[0]*coord[1])
uOmega65_old=Function(V)
uOmega65_old.interpolate(1*coord[0]*coord[1])
uOmega66=Function(V)
uOmega66.interpolate(1*coord[0]*coord[1])
uOmega66_old=Function(V)
uOmega66_old.interpolate(1*coord[0]*coord[1])
uOmega67=Function(V)
uOmega67.interpolate(1*coord[0]*coord[1])
uOmega67_old=Function(V)
uOmega67_old.interpolate(1*coord[0]*coord[1])
uOmega68=Function(V)
uOmega68.interpolate(1*coord[0]*coord[1])
uOmega68_old=Function(V)
uOmega68_old.interpolate(1*coord[0]*coord[1])
uOmega69=Function(V)
uOmega69.interpolate(1*coord[0]*coord[1])
uOmega69_old=Function(V)
uOmega69_old.interpolate(1*coord[0]*coord[1])
uOmega70=Function(V)
uOmega70.interpolate(1*coord[0]*coord[1])
uOmega70_old=Function(V)
uOmega70_old.interpolate(1*coord[0]*coord[1])
uOmega71=Function(V)
uOmega71.interpolate(1*coord[0]*coord[1])
uOmega71_old=Function(V)
uOmega71_old.interpolate(1*coord[0]*coord[1])
uOmega72=Function(V)
uOmega72.interpolate(1*coord[0]*coord[1])
uOmega72_old=Function(V)
uOmega72_old.interpolate(1*coord[0]*coord[1])
uOmega73=Function(V)
uOmega73.interpolate(1*coord[0]*coord[1])
uOmega73_old=Function(V)
uOmega73_old.interpolate(1*coord[0]*coord[1])
uOmega74=Function(V)
uOmega74.interpolate(1*coord[0]*coord[1])
uOmega74_old=Function(V)
uOmega74_old.interpolate(1*coord[0]*coord[1])
uOmega75=Function(V)
uOmega75.interpolate(1*coord[0]*coord[1])
uOmega75_old=Function(V)
uOmega75_old.interpolate(1*coord[0]*coord[1])
uOmega76=Function(V)
uOmega76.interpolate(1*coord[0]*coord[1])
uOmega76_old=Function(V)
uOmega76_old.interpolate(1*coord[0]*coord[1])
uOmega77=Function(V)
uOmega77.interpolate(1*coord[0]*coord[1])
uOmega77_old=Function(V)
uOmega77_old.interpolate(1*coord[0]*coord[1])
uOmega78=Function(V)
uOmega78.interpolate(1*coord[0]*coord[1])
uOmega78_old=Function(V)
uOmega78_old.interpolate(1*coord[0]*coord[1])
uOmega79=Function(V)
uOmega79.interpolate(1*coord[0]*coord[1])
uOmega79_old=Function(V)
uOmega79_old.interpolate(1*coord[0]*coord[1])
uOmega80=Function(V)
uOmega80.interpolate(1*coord[0]*coord[1])
uOmega80_old=Function(V)
uOmega80_old.interpolate(1*coord[0]*coord[1])
uOmega81=Function(V)
uOmega81.interpolate(1*coord[0]*coord[1])
uOmega81_old=Function(V)
uOmega81_old.interpolate(1*coord[0]*coord[1])
uOmega82=Function(V)
uOmega82.interpolate(1*coord[0]*coord[1])
uOmega82_old=Function(V)
uOmega82_old.interpolate(1*coord[0]*coord[1])
uOmega83=Function(V)
uOmega83.interpolate(1*coord[0]*coord[1])
uOmega83_old=Function(V)
uOmega83_old.interpolate(1*coord[0]*coord[1])
uOmega84=Function(V)
uOmega84.interpolate(1*coord[0]*coord[1])
uOmega84_old=Function(V)
uOmega84_old.interpolate(1*coord[0]*coord[1])
uOmega85=Function(V)
uOmega85.interpolate(1*coord[0]*coord[1])
uOmega85_old=Function(V)
uOmega85_old.interpolate(1*coord[0]*coord[1])
uOmega86=Function(V)
uOmega86.interpolate(1*coord[0]*coord[1])
uOmega86_old=Function(V)
uOmega86_old.interpolate(1*coord[0]*coord[1])
uOmega87=Function(V)
uOmega87.interpolate(1*coord[0]*coord[1])
uOmega87_old=Function(V)
uOmega87_old.interpolate(1*coord[0]*coord[1])
uOmega88=Function(V)
uOmega88.interpolate(1*coord[0]*coord[1])
uOmega88_old=Function(V)
uOmega88_old.interpolate(1*coord[0]*coord[1])
uOmega89=Function(V)
uOmega89.interpolate(1*coord[0]*coord[1])
uOmega89_old=Function(V)
uOmega89_old.interpolate(1*coord[0]*coord[1])
uOmega90=Function(V)
uOmega90.interpolate(1*coord[0]*coord[1])
uOmega90_old=Function(V)
uOmega90_old.interpolate(1*coord[0]*coord[1])
uOmega91=Function(V)
uOmega91.interpolate(1*coord[0]*coord[1])
uOmega91_old=Function(V)
uOmega91_old.interpolate(1*coord[0]*coord[1])
uOmega92=Function(V)
uOmega92.interpolate(1*coord[0]*coord[1])
uOmega92_old=Function(V)
uOmega92_old.interpolate(1*coord[0]*coord[1])
uOmega93=Function(V)
uOmega93.interpolate(1*coord[0]*coord[1])
uOmega93_old=Function(V)
uOmega93_old.interpolate(1*coord[0]*coord[1])
uOmega94=Function(V)
uOmega94.interpolate(1*coord[0]*coord[1])
uOmega94_old=Function(V)
uOmega94_old.interpolate(1*coord[0]*coord[1])
uOmega95=Function(V)
uOmega95.interpolate(1*coord[0]*coord[1])
uOmega95_old=Function(V)
uOmega95_old.interpolate(1*coord[0]*coord[1])
uOmega96=Function(V)
uOmega96.interpolate(1*coord[0]*coord[1])
uOmega96_old=Function(V)
uOmega96_old.interpolate(1*coord[0]*coord[1])
uOmega97=Function(V)
uOmega97.interpolate(1*coord[0]*coord[1])
uOmega97_old=Function(V)
uOmega97_old.interpolate(1*coord[0]*coord[1])
uOmega98=Function(V)
uOmega98.interpolate(1*coord[0]*coord[1])
uOmega98_old=Function(V)
uOmega98_old.interpolate(1*coord[0]*coord[1])
uOmega99=Function(V)
uOmega99.interpolate(1*coord[0]*coord[1])
uOmega99_old=Function(V)
uOmega99_old.interpolate(1*coord[0]*coord[1])
uOmega100=Function(V)
uOmega100.interpolate(1*coord[0]*coord[1])
uOmega100_old=Function(V)
uOmega100_old.interpolate(1*coord[0]*coord[1])
uOmega101=Function(V)
uOmega101.interpolate(1*coord[0]*coord[1])
uOmega101_old=Function(V)
uOmega101_old.interpolate(1*coord[0]*coord[1])
uOmega102=Function(V)
uOmega102.interpolate(1*coord[0]*coord[1])
uOmega102_old=Function(V)
uOmega102_old.interpolate(1*coord[0]*coord[1])
uOmega103=Function(V)
uOmega103.interpolate(1*coord[0]*coord[1])
uOmega103_old=Function(V)
uOmega103_old.interpolate(1*coord[0]*coord[1])
uOmega104=Function(V)
uOmega104.interpolate(1*coord[0]*coord[1])
uOmega104_old=Function(V)
uOmega104_old.interpolate(1*coord[0]*coord[1])
uOmega105=Function(V)
uOmega105.interpolate(1*coord[0]*coord[1])
uOmega105_old=Function(V)
uOmega105_old.interpolate(1*coord[0]*coord[1])
uOmega106=Function(V)
uOmega106.interpolate(1*coord[0]*coord[1])
uOmega106_old=Function(V)
uOmega106_old.interpolate(1*coord[0]*coord[1])
uOmega107=Function(V)
uOmega107.interpolate(1*coord[0]*coord[1])
uOmega107_old=Function(V)
uOmega107_old.interpolate(1*coord[0]*coord[1])
uOmega108=Function(V)
uOmega108.interpolate(1*coord[0]*coord[1])
uOmega108_old=Function(V)
uOmega108_old.interpolate(1*coord[0]*coord[1])
uOmega109=Function(V)
uOmega109.interpolate(1*coord[0]*coord[1])
uOmega109_old=Function(V)
uOmega109_old.interpolate(1*coord[0]*coord[1])
uOmega110=Function(V)
uOmega110.interpolate(1*coord[0]*coord[1])
uOmega110_old=Function(V)
uOmega110_old.interpolate(1*coord[0]*coord[1])
uOmega111=Function(V)
uOmega111.interpolate(1*coord[0]*coord[1])
uOmega111_old=Function(V)
uOmega111_old.interpolate(1*coord[0]*coord[1])
uOmega112=Function(V)
uOmega112.interpolate(1*coord[0]*coord[1])
uOmega112_old=Function(V)
uOmega112_old.interpolate(1*coord[0]*coord[1])
uOmega113=Function(V)
uOmega113.interpolate(1*coord[0]*coord[1])
uOmega113_old=Function(V)
uOmega113_old.interpolate(1*coord[0]*coord[1])
uOmega114=Function(V)
uOmega114.interpolate(1*coord[0]*coord[1])
uOmega114_old=Function(V)
uOmega114_old.interpolate(1*coord[0]*coord[1])
uOmega115=Function(V)
uOmega115.interpolate(1*coord[0]*coord[1])
uOmega115_old=Function(V)
uOmega115_old.interpolate(1*coord[0]*coord[1])
uOmega116=Function(V)
uOmega116.interpolate(1*coord[0]*coord[1])
uOmega116_old=Function(V)
uOmega116_old.interpolate(1*coord[0]*coord[1])
uOmega117=Function(V)
uOmega117.interpolate(1*coord[0]*coord[1])
uOmega117_old=Function(V)
uOmega117_old.interpolate(1*coord[0]*coord[1])
uOmega118=Function(V)
uOmega118.interpolate(1*coord[0]*coord[1])
uOmega118_old=Function(V)
uOmega118_old.interpolate(1*coord[0]*coord[1])
uOmega119=Function(V)
uOmega119.interpolate(1*coord[0]*coord[1])
uOmega119_old=Function(V)
uOmega119_old.interpolate(1*coord[0]*coord[1])
uOmega120=Function(V)
uOmega120.interpolate(1*coord[0]*coord[1])
uOmega120_old=Function(V)
uOmega120_old.interpolate(1*coord[0]*coord[1])
uOmega121=Function(V)
uOmega121.interpolate(1*coord[0]*coord[1])
uOmega121_old=Function(V)
uOmega121_old.interpolate(1*coord[0]*coord[1])
uOmega122=Function(V)
uOmega122.interpolate(1*coord[0]*coord[1])
uOmega122_old=Function(V)
uOmega122_old.interpolate(1*coord[0]*coord[1])
uOmega123=Function(V)
uOmega123.interpolate(1*coord[0]*coord[1])
uOmega123_old=Function(V)
uOmega123_old.interpolate(1*coord[0]*coord[1])
uOmega124=Function(V)
uOmega124.interpolate(1*coord[0]*coord[1])
uOmega124_old=Function(V)
uOmega124_old.interpolate(1*coord[0]*coord[1])
uOmega125=Function(V)
uOmega125.interpolate(1*coord[0]*coord[1])
uOmega125_old=Function(V)
uOmega125_old.interpolate(1*coord[0]*coord[1])
uOmega126=Function(V)
uOmega126.interpolate(1*coord[0]*coord[1])
uOmega126_old=Function(V)
uOmega126_old.interpolate(1*coord[0]*coord[1])
uOmega127=Function(V)
uOmega127.interpolate(1*coord[0]*coord[1])
uOmega127_old=Function(V)
uOmega127_old.interpolate(1*coord[0]*coord[1])
uOmega128=Function(V)
uOmega128.interpolate(1*coord[0]*coord[1])
uOmega128_old=Function(V)
uOmega128_old.interpolate(1*coord[0]*coord[1])
uOmega129=Function(V)
uOmega129.interpolate(1*coord[0]*coord[1])
uOmega129_old=Function(V)
uOmega129_old.interpolate(1*coord[0]*coord[1])
uOmega130=Function(V)
uOmega130.interpolate(1*coord[0]*coord[1])
uOmega130_old=Function(V)
uOmega130_old.interpolate(1*coord[0]*coord[1])
uOmega131=Function(V)
uOmega131.interpolate(1*coord[0]*coord[1])
uOmega131_old=Function(V)
uOmega131_old.interpolate(1*coord[0]*coord[1])
uOmega132=Function(V)
uOmega132.interpolate(1*coord[0]*coord[1])
uOmega132_old=Function(V)
uOmega132_old.interpolate(1*coord[0]*coord[1])
uOmega133=Function(V)
uOmega133.interpolate(1*coord[0]*coord[1])
uOmega133_old=Function(V)
uOmega133_old.interpolate(1*coord[0]*coord[1])
uOmega134=Function(V)
uOmega134.interpolate(1*coord[0]*coord[1])
uOmega134_old=Function(V)
uOmega134_old.interpolate(1*coord[0]*coord[1])
uOmega135=Function(V)
uOmega135.interpolate(1*coord[0]*coord[1])
uOmega135_old=Function(V)
uOmega135_old.interpolate(1*coord[0]*coord[1])
uOmega136=Function(V)
uOmega136.interpolate(1*coord[0]*coord[1])
uOmega136_old=Function(V)
uOmega136_old.interpolate(1*coord[0]*coord[1])
uOmega137=Function(V)
uOmega137.interpolate(1*coord[0]*coord[1])
uOmega137_old=Function(V)
uOmega137_old.interpolate(1*coord[0]*coord[1])
uOmega138=Function(V)
uOmega138.interpolate(1*coord[0]*coord[1])
uOmega138_old=Function(V)
uOmega138_old.interpolate(1*coord[0]*coord[1])
uOmega139=Function(V)
uOmega139.interpolate(1*coord[0]*coord[1])
uOmega139_old=Function(V)
uOmega139_old.interpolate(1*coord[0]*coord[1])
uOmega140=Function(V)
uOmega140.interpolate(1*coord[0]*coord[1])
uOmega140_old=Function(V)
uOmega140_old.interpolate(1*coord[0]*coord[1])
uOmega141=Function(V)
uOmega141.interpolate(1*coord[0]*coord[1])
uOmega141_old=Function(V)
uOmega141_old.interpolate(1*coord[0]*coord[1])
uOmega142=Function(V)
uOmega142.interpolate(1*coord[0]*coord[1])
uOmega142_old=Function(V)
uOmega142_old.interpolate(1*coord[0]*coord[1])
uOmega143=Function(V)
uOmega143.interpolate(1*coord[0]*coord[1])
uOmega143_old=Function(V)
uOmega143_old.interpolate(1*coord[0]*coord[1])
uOmega144=Function(V)
uOmega144.interpolate(1*coord[0]*coord[1])
uOmega144_old=Function(V)
uOmega144_old.interpolate(1*coord[0]*coord[1])
uOmega145=Function(V)
uOmega145.interpolate(1*coord[0]*coord[1])
uOmega145_old=Function(V)
uOmega145_old.interpolate(1*coord[0]*coord[1])
uOmega146=Function(V)
uOmega146.interpolate(1*coord[0]*coord[1])
uOmega146_old=Function(V)
uOmega146_old.interpolate(1*coord[0]*coord[1])
uOmega147=Function(V)
uOmega147.interpolate(1*coord[0]*coord[1])
uOmega147_old=Function(V)
uOmega147_old.interpolate(1*coord[0]*coord[1])
uOmega148=Function(V)
uOmega148.interpolate(1*coord[0]*coord[1])
uOmega148_old=Function(V)
uOmega148_old.interpolate(1*coord[0]*coord[1])
uOmega149=Function(V)
uOmega149.interpolate(1*coord[0]*coord[1])
uOmega149_old=Function(V)
uOmega149_old.interpolate(1*coord[0]*coord[1])
uOmega150=Function(V)
uOmega150.interpolate(1*coord[0]*coord[1])
uOmega150_old=Function(V)
uOmega150_old.interpolate(1*coord[0]*coord[1])
uOmega151=Function(V)
uOmega151.interpolate(1*coord[0]*coord[1])
uOmega151_old=Function(V)
uOmega151_old.interpolate(1*coord[0]*coord[1])
uOmega152=Function(V)
uOmega152.interpolate(1*coord[0]*coord[1])
uOmega152_old=Function(V)
uOmega152_old.interpolate(1*coord[0]*coord[1])
uOmega153=Function(V)
uOmega153.interpolate(1*coord[0]*coord[1])
uOmega153_old=Function(V)
uOmega153_old.interpolate(1*coord[0]*coord[1])
uOmega154=Function(V)
uOmega154.interpolate(1*coord[0]*coord[1])
uOmega154_old=Function(V)
uOmega154_old.interpolate(1*coord[0]*coord[1])
uOmega155=Function(V)
uOmega155.interpolate(1*coord[0]*coord[1])
uOmega155_old=Function(V)
uOmega155_old.interpolate(1*coord[0]*coord[1])
uOmega156=Function(V)
uOmega156.interpolate(1*coord[0]*coord[1])
uOmega156_old=Function(V)
uOmega156_old.interpolate(1*coord[0]*coord[1])
uOmega157=Function(V)
uOmega157.interpolate(1*coord[0]*coord[1])
uOmega157_old=Function(V)
uOmega157_old.interpolate(1*coord[0]*coord[1])
uOmega158=Function(V)
uOmega158.interpolate(1*coord[0]*coord[1])
uOmega158_old=Function(V)
uOmega158_old.interpolate(1*coord[0]*coord[1])
uOmega159=Function(V)
uOmega159.interpolate(1*coord[0]*coord[1])
uOmega159_old=Function(V)
uOmega159_old.interpolate(1*coord[0]*coord[1])
uOmega160=Function(V)
uOmega160.interpolate(1*coord[0]*coord[1])
uOmega160_old=Function(V)
uOmega160_old.interpolate(1*coord[0]*coord[1])
uOmega161=Function(V)
uOmega161.interpolate(1*coord[0]*coord[1])
uOmega161_old=Function(V)
uOmega161_old.interpolate(1*coord[0]*coord[1])
uOmega162=Function(V)
uOmega162.interpolate(1*coord[0]*coord[1])
uOmega162_old=Function(V)
uOmega162_old.interpolate(1*coord[0]*coord[1])
uOmega163=Function(V)
uOmega163.interpolate(1*coord[0]*coord[1])
uOmega163_old=Function(V)
uOmega163_old.interpolate(1*coord[0]*coord[1])
uOmega164=Function(V)
uOmega164.interpolate(1*coord[0]*coord[1])
uOmega164_old=Function(V)
uOmega164_old.interpolate(1*coord[0]*coord[1])
uOmega165=Function(V)
uOmega165.interpolate(1*coord[0]*coord[1])
uOmega165_old=Function(V)
uOmega165_old.interpolate(1*coord[0]*coord[1])
uOmega166=Function(V)
uOmega166.interpolate(1*coord[0]*coord[1])
uOmega166_old=Function(V)
uOmega166_old.interpolate(1*coord[0]*coord[1])
uOmega167=Function(V)
uOmega167.interpolate(1*coord[0]*coord[1])
uOmega167_old=Function(V)
uOmega167_old.interpolate(1*coord[0]*coord[1])
uOmega168=Function(V)
uOmega168.interpolate(1*coord[0]*coord[1])
uOmega168_old=Function(V)
uOmega168_old.interpolate(1*coord[0]*coord[1])
uOmega169=Function(V)
uOmega169.interpolate(1*coord[0]*coord[1])
uOmega169_old=Function(V)
uOmega169_old.interpolate(1*coord[0]*coord[1])
uOmega170=Function(V)
uOmega170.interpolate(1*coord[0]*coord[1])
uOmega170_old=Function(V)
uOmega170_old.interpolate(1*coord[0]*coord[1])
uOmega171=Function(V)
uOmega171.interpolate(1*coord[0]*coord[1])
uOmega171_old=Function(V)
uOmega171_old.interpolate(1*coord[0]*coord[1])
uOmega172=Function(V)
uOmega172.interpolate(1*coord[0]*coord[1])
uOmega172_old=Function(V)
uOmega172_old.interpolate(1*coord[0]*coord[1])
uOmega173=Function(V)
uOmega173.interpolate(1*coord[0]*coord[1])
uOmega173_old=Function(V)
uOmega173_old.interpolate(1*coord[0]*coord[1])
uOmega174=Function(V)
uOmega174.interpolate(1*coord[0]*coord[1])
uOmega174_old=Function(V)
uOmega174_old.interpolate(1*coord[0]*coord[1])
uOmega175=Function(V)
uOmega175.interpolate(1*coord[0]*coord[1])
uOmega175_old=Function(V)
uOmega175_old.interpolate(1*coord[0]*coord[1])
uOmega176=Function(V)
uOmega176.interpolate(1*coord[0]*coord[1])
uOmega176_old=Function(V)
uOmega176_old.interpolate(1*coord[0]*coord[1])
uOmega177=Function(V)
uOmega177.interpolate(1*coord[0]*coord[1])
uOmega177_old=Function(V)
uOmega177_old.interpolate(1*coord[0]*coord[1])
uOmega178=Function(V)
uOmega178.interpolate(1*coord[0]*coord[1])
uOmega178_old=Function(V)
uOmega178_old.interpolate(1*coord[0]*coord[1])
uOmega179=Function(V)
uOmega179.interpolate(1*coord[0]*coord[1])
uOmega179_old=Function(V)
uOmega179_old.interpolate(1*coord[0]*coord[1])
uOmega180=Function(V)
uOmega180.interpolate(1*coord[0]*coord[1])
uOmega180_old=Function(V)
uOmega180_old.interpolate(1*coord[0]*coord[1])
uOmega181=Function(V)
uOmega181.interpolate(1*coord[0]*coord[1])
uOmega181_old=Function(V)
uOmega181_old.interpolate(1*coord[0]*coord[1])
uOmega182=Function(V)
uOmega182.interpolate(1*coord[0]*coord[1])
uOmega182_old=Function(V)
uOmega182_old.interpolate(1*coord[0]*coord[1])
uOmega183=Function(V)
uOmega183.interpolate(1*coord[0]*coord[1])
uOmega183_old=Function(V)
uOmega183_old.interpolate(1*coord[0]*coord[1])
uOmega184=Function(V)
uOmega184.interpolate(1*coord[0]*coord[1])
uOmega184_old=Function(V)
uOmega184_old.interpolate(1*coord[0]*coord[1])
uOmega185=Function(V)
uOmega185.interpolate(1*coord[0]*coord[1])
uOmega185_old=Function(V)
uOmega185_old.interpolate(1*coord[0]*coord[1])
uOmega186=Function(V)
uOmega186.interpolate(1*coord[0]*coord[1])
uOmega186_old=Function(V)
uOmega186_old.interpolate(1*coord[0]*coord[1])
uOmega187=Function(V)
uOmega187.interpolate(1*coord[0]*coord[1])
uOmega187_old=Function(V)
uOmega187_old.interpolate(1*coord[0]*coord[1])
uOmega188=Function(V)
uOmega188.interpolate(1*coord[0]*coord[1])
uOmega188_old=Function(V)
uOmega188_old.interpolate(1*coord[0]*coord[1])
uOmega189=Function(V)
uOmega189.interpolate(1*coord[0]*coord[1])
uOmega189_old=Function(V)
uOmega189_old.interpolate(1*coord[0]*coord[1])
uOmega190=Function(V)
uOmega190.interpolate(1*coord[0]*coord[1])
uOmega190_old=Function(V)
uOmega190_old.interpolate(1*coord[0]*coord[1])
uOmega191=Function(V)
uOmega191.interpolate(1*coord[0]*coord[1])
uOmega191_old=Function(V)
uOmega191_old.interpolate(1*coord[0]*coord[1])
uOmega192=Function(V)
uOmega192.interpolate(1*coord[0]*coord[1])
uOmega192_old=Function(V)
uOmega192_old.interpolate(1*coord[0]*coord[1])
uOmega193=Function(V)
uOmega193.interpolate(1*coord[0]*coord[1])
uOmega193_old=Function(V)
uOmega193_old.interpolate(1*coord[0]*coord[1])
uOmega194=Function(V)
uOmega194.interpolate(1*coord[0]*coord[1])
uOmega194_old=Function(V)
uOmega194_old.interpolate(1*coord[0]*coord[1])
uOmega195=Function(V)
uOmega195.interpolate(1*coord[0]*coord[1])
uOmega195_old=Function(V)
uOmega195_old.interpolate(1*coord[0]*coord[1])
uOmega196=Function(V)
uOmega196.interpolate(1*coord[0]*coord[1])
uOmega196_old=Function(V)
uOmega196_old.interpolate(1*coord[0]*coord[1])
uOmega197=Function(V)
uOmega197.interpolate(1*coord[0]*coord[1])
uOmega197_old=Function(V)
uOmega197_old.interpolate(1*coord[0]*coord[1])
uOmega198=Function(V)
uOmega198.interpolate(1*coord[0]*coord[1])
uOmega198_old=Function(V)
uOmega198_old.interpolate(1*coord[0]*coord[1])
uOmega199=Function(V)
uOmega199.interpolate(1*coord[0]*coord[1])
uOmega199_old=Function(V)
uOmega199_old.interpolate(1*coord[0]*coord[1])
uOmega200=Function(V)
uOmega200.interpolate(1*coord[0]*coord[1])
uOmega200_old=Function(V)
uOmega200_old.interpolate(1*coord[0]*coord[1])
uOmega201=Function(V)
uOmega201.interpolate(1*coord[0]*coord[1])
uOmega201_old=Function(V)
uOmega201_old.interpolate(1*coord[0]*coord[1])
uOmega202=Function(V)
uOmega202.interpolate(1*coord[0]*coord[1])
uOmega202_old=Function(V)
uOmega202_old.interpolate(1*coord[0]*coord[1])
uOmega203=Function(V)
uOmega203.interpolate(1*coord[0]*coord[1])
uOmega203_old=Function(V)
uOmega203_old.interpolate(1*coord[0]*coord[1])
uOmega204=Function(V)
uOmega204.interpolate(1*coord[0]*coord[1])
uOmega204_old=Function(V)
uOmega204_old.interpolate(1*coord[0]*coord[1])
uOmega205=Function(V)
uOmega205.interpolate(1*coord[0]*coord[1])
uOmega205_old=Function(V)
uOmega205_old.interpolate(1*coord[0]*coord[1])
uOmega206=Function(V)
uOmega206.interpolate(1*coord[0]*coord[1])
uOmega206_old=Function(V)
uOmega206_old.interpolate(1*coord[0]*coord[1])
uOmega207=Function(V)
uOmega207.interpolate(1*coord[0]*coord[1])
uOmega207_old=Function(V)
uOmega207_old.interpolate(1*coord[0]*coord[1])
uOmega208=Function(V)
uOmega208.interpolate(1*coord[0]*coord[1])
uOmega208_old=Function(V)
uOmega208_old.interpolate(1*coord[0]*coord[1])
uOmega209=Function(V)
uOmega209.interpolate(1*coord[0]*coord[1])
uOmega209_old=Function(V)
uOmega209_old.interpolate(1*coord[0]*coord[1])
uOmega210=Function(V)
uOmega210.interpolate(1*coord[0]*coord[1])
uOmega210_old=Function(V)
uOmega210_old.interpolate(1*coord[0]*coord[1])
uOmega211=Function(V)
uOmega211.interpolate(1*coord[0]*coord[1])
uOmega211_old=Function(V)
uOmega211_old.interpolate(1*coord[0]*coord[1])
uOmega212=Function(V)
uOmega212.interpolate(1*coord[0]*coord[1])
uOmega212_old=Function(V)
uOmega212_old.interpolate(1*coord[0]*coord[1])
uOmega213=Function(V)
uOmega213.interpolate(1*coord[0]*coord[1])
uOmega213_old=Function(V)
uOmega213_old.interpolate(1*coord[0]*coord[1])
uOmega214=Function(V)
uOmega214.interpolate(1*coord[0]*coord[1])
uOmega214_old=Function(V)
uOmega214_old.interpolate(1*coord[0]*coord[1])
uOmega215=Function(V)
uOmega215.interpolate(1*coord[0]*coord[1])
uOmega215_old=Function(V)
uOmega215_old.interpolate(1*coord[0]*coord[1])
uOmega216=Function(V)
uOmega216.interpolate(1*coord[0]*coord[1])
uOmega216_old=Function(V)
uOmega216_old.interpolate(1*coord[0]*coord[1])
uOmega217=Function(V)
uOmega217.interpolate(1*coord[0]*coord[1])
uOmega217_old=Function(V)
uOmega217_old.interpolate(1*coord[0]*coord[1])
uOmega218=Function(V)
uOmega218.interpolate(1*coord[0]*coord[1])
uOmega218_old=Function(V)
uOmega218_old.interpolate(1*coord[0]*coord[1])
uOmega219=Function(V)
uOmega219.interpolate(1*coord[0]*coord[1])
uOmega219_old=Function(V)
uOmega219_old.interpolate(1*coord[0]*coord[1])
uOmega220=Function(V)
uOmega220.interpolate(1*coord[0]*coord[1])
uOmega220_old=Function(V)
uOmega220_old.interpolate(1*coord[0]*coord[1])
uOmega221=Function(V)
uOmega221.interpolate(1*coord[0]*coord[1])
uOmega221_old=Function(V)
uOmega221_old.interpolate(1*coord[0]*coord[1])
uOmega222=Function(V)
uOmega222.interpolate(1*coord[0]*coord[1])
uOmega222_old=Function(V)
uOmega222_old.interpolate(1*coord[0]*coord[1])
uOmega223=Function(V)
uOmega223.interpolate(1*coord[0]*coord[1])
uOmega223_old=Function(V)
uOmega223_old.interpolate(1*coord[0]*coord[1])
uOmega224=Function(V)
uOmega224.interpolate(1*coord[0]*coord[1])
uOmega224_old=Function(V)
uOmega224_old.interpolate(1*coord[0]*coord[1])
uOmega225=Function(V)
uOmega225.interpolate(1*coord[0]*coord[1])
uOmega225_old=Function(V)
uOmega225_old.interpolate(1*coord[0]*coord[1])
uOmega226=Function(V)
uOmega226.interpolate(1*coord[0]*coord[1])
uOmega226_old=Function(V)
uOmega226_old.interpolate(1*coord[0]*coord[1])
uOmega227=Function(V)
uOmega227.interpolate(1*coord[0]*coord[1])
uOmega227_old=Function(V)
uOmega227_old.interpolate(1*coord[0]*coord[1])
uOmega228=Function(V)
uOmega228.interpolate(1*coord[0]*coord[1])
uOmega228_old=Function(V)
uOmega228_old.interpolate(1*coord[0]*coord[1])
uOmega229=Function(V)
uOmega229.interpolate(1*coord[0]*coord[1])
uOmega229_old=Function(V)
uOmega229_old.interpolate(1*coord[0]*coord[1])
uOmega230=Function(V)
uOmega230.interpolate(1*coord[0]*coord[1])
uOmega230_old=Function(V)
uOmega230_old.interpolate(1*coord[0]*coord[1])
uOmega231=Function(V)
uOmega231.interpolate(1*coord[0]*coord[1])
uOmega231_old=Function(V)
uOmega231_old.interpolate(1*coord[0]*coord[1])
uOmega232=Function(V)
uOmega232.interpolate(1*coord[0]*coord[1])
uOmega232_old=Function(V)
uOmega232_old.interpolate(1*coord[0]*coord[1])
uOmega233=Function(V)
uOmega233.interpolate(1*coord[0]*coord[1])
uOmega233_old=Function(V)
uOmega233_old.interpolate(1*coord[0]*coord[1])
uOmega234=Function(V)
uOmega234.interpolate(1*coord[0]*coord[1])
uOmega234_old=Function(V)
uOmega234_old.interpolate(1*coord[0]*coord[1])
uOmega235=Function(V)
uOmega235.interpolate(1*coord[0]*coord[1])
uOmega235_old=Function(V)
uOmega235_old.interpolate(1*coord[0]*coord[1])
uOmega236=Function(V)
uOmega236.interpolate(1*coord[0]*coord[1])
uOmega236_old=Function(V)
uOmega236_old.interpolate(1*coord[0]*coord[1])
uOmega237=Function(V)
uOmega237.interpolate(1*coord[0]*coord[1])
uOmega237_old=Function(V)
uOmega237_old.interpolate(1*coord[0]*coord[1])
uOmega238=Function(V)
uOmega238.interpolate(1*coord[0]*coord[1])
uOmega238_old=Function(V)
uOmega238_old.interpolate(1*coord[0]*coord[1])
uOmega239=Function(V)
uOmega239.interpolate(1*coord[0]*coord[1])
uOmega239_old=Function(V)
uOmega239_old.interpolate(1*coord[0]*coord[1])
uOmega240=Function(V)
uOmega240.interpolate(1*coord[0]*coord[1])
uOmega240_old=Function(V)
uOmega240_old.interpolate(1*coord[0]*coord[1])
uOmega241=Function(V)
uOmega241.interpolate(1*coord[0]*coord[1])
uOmega241_old=Function(V)
uOmega241_old.interpolate(1*coord[0]*coord[1])
uOmega242=Function(V)
uOmega242.interpolate(1*coord[0]*coord[1])
uOmega242_old=Function(V)
uOmega242_old.interpolate(1*coord[0]*coord[1])
uOmega243=Function(V)
uOmega243.interpolate(1*coord[0]*coord[1])
uOmega243_old=Function(V)
uOmega243_old.interpolate(1*coord[0]*coord[1])
uOmega244=Function(V)
uOmega244.interpolate(1*coord[0]*coord[1])
uOmega244_old=Function(V)
uOmega244_old.interpolate(1*coord[0]*coord[1])
uOmega245=Function(V)
uOmega245.interpolate(1*coord[0]*coord[1])
uOmega245_old=Function(V)
uOmega245_old.interpolate(1*coord[0]*coord[1])
uOmega246=Function(V)
uOmega246.interpolate(1*coord[0]*coord[1])
uOmega246_old=Function(V)
uOmega246_old.interpolate(1*coord[0]*coord[1])
uOmega247=Function(V)
uOmega247.interpolate(1*coord[0]*coord[1])
uOmega247_old=Function(V)
uOmega247_old.interpolate(1*coord[0]*coord[1])
uOmega248=Function(V)
uOmega248.interpolate(1*coord[0]*coord[1])
uOmega248_old=Function(V)
uOmega248_old.interpolate(1*coord[0]*coord[1])
uOmega249=Function(V)
uOmega249.interpolate(1*coord[0]*coord[1])
uOmega249_old=Function(V)
uOmega249_old.interpolate(1*coord[0]*coord[1])
uOmega250=Function(V)
uOmega250.interpolate(1*coord[0]*coord[1])
uOmega250_old=Function(V)
uOmega250_old.interpolate(1*coord[0]*coord[1])
uOmega251=Function(V)
uOmega251.interpolate(1*coord[0]*coord[1])
uOmega251_old=Function(V)
uOmega251_old.interpolate(1*coord[0]*coord[1])
uOmega252=Function(V)
uOmega252.interpolate(1*coord[0]*coord[1])
uOmega252_old=Function(V)
uOmega252_old.interpolate(1*coord[0]*coord[1])
uOmega253=Function(V)
uOmega253.interpolate(1*coord[0]*coord[1])
uOmega253_old=Function(V)
uOmega253_old.interpolate(1*coord[0]*coord[1])
uOmega254=Function(V)
uOmega254.interpolate(1*coord[0]*coord[1])
uOmega254_old=Function(V)
uOmega254_old.interpolate(1*coord[0]*coord[1])
uOmega255=Function(V)
uOmega255.interpolate(1*coord[0]*coord[1])
uOmega255_old=Function(V)
uOmega255_old.interpolate(1*coord[0]*coord[1])
uOmega256=Function(V)
uOmega256.interpolate(1*coord[0]*coord[1])
uOmega256_old=Function(V)
uOmega256_old.interpolate(1*coord[0]*coord[1])
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
aOmega65 = (dot(grad(u), grad(v))) * dx(Omega65)
LOmega65 = f * v * dx(Omega65)
aOmega66 = (dot(grad(u), grad(v))) * dx(Omega66)
LOmega66 = f * v * dx(Omega66)
aOmega67 = (dot(grad(u), grad(v))) * dx(Omega67)
LOmega67 = f * v * dx(Omega67)
aOmega68 = (dot(grad(u), grad(v))) * dx(Omega68)
LOmega68 = f * v * dx(Omega68)
aOmega69 = (dot(grad(u), grad(v))) * dx(Omega69)
LOmega69 = f * v * dx(Omega69)
aOmega70 = (dot(grad(u), grad(v))) * dx(Omega70)
LOmega70 = f * v * dx(Omega70)
aOmega71 = (dot(grad(u), grad(v))) * dx(Omega71)
LOmega71 = f * v * dx(Omega71)
aOmega72 = (dot(grad(u), grad(v))) * dx(Omega72)
LOmega72 = f * v * dx(Omega72)
aOmega73 = (dot(grad(u), grad(v))) * dx(Omega73)
LOmega73 = f * v * dx(Omega73)
aOmega74 = (dot(grad(u), grad(v))) * dx(Omega74)
LOmega74 = f * v * dx(Omega74)
aOmega75 = (dot(grad(u), grad(v))) * dx(Omega75)
LOmega75 = f * v * dx(Omega75)
aOmega76 = (dot(grad(u), grad(v))) * dx(Omega76)
LOmega76 = f * v * dx(Omega76)
aOmega77 = (dot(grad(u), grad(v))) * dx(Omega77)
LOmega77 = f * v * dx(Omega77)
aOmega78 = (dot(grad(u), grad(v))) * dx(Omega78)
LOmega78 = f * v * dx(Omega78)
aOmega79 = (dot(grad(u), grad(v))) * dx(Omega79)
LOmega79 = f * v * dx(Omega79)
aOmega80 = (dot(grad(u), grad(v))) * dx(Omega80)
LOmega80 = f * v * dx(Omega80)
aOmega81 = (dot(grad(u), grad(v))) * dx(Omega81)
LOmega81 = f * v * dx(Omega81)
aOmega82 = (dot(grad(u), grad(v))) * dx(Omega82)
LOmega82 = f * v * dx(Omega82)
aOmega83 = (dot(grad(u), grad(v))) * dx(Omega83)
LOmega83 = f * v * dx(Omega83)
aOmega84 = (dot(grad(u), grad(v))) * dx(Omega84)
LOmega84 = f * v * dx(Omega84)
aOmega85 = (dot(grad(u), grad(v))) * dx(Omega85)
LOmega85 = f * v * dx(Omega85)
aOmega86 = (dot(grad(u), grad(v))) * dx(Omega86)
LOmega86 = f * v * dx(Omega86)
aOmega87 = (dot(grad(u), grad(v))) * dx(Omega87)
LOmega87 = f * v * dx(Omega87)
aOmega88 = (dot(grad(u), grad(v))) * dx(Omega88)
LOmega88 = f * v * dx(Omega88)
aOmega89 = (dot(grad(u), grad(v))) * dx(Omega89)
LOmega89 = f * v * dx(Omega89)
aOmega90 = (dot(grad(u), grad(v))) * dx(Omega90)
LOmega90 = f * v * dx(Omega90)
aOmega91 = (dot(grad(u), grad(v))) * dx(Omega91)
LOmega91 = f * v * dx(Omega91)
aOmega92 = (dot(grad(u), grad(v))) * dx(Omega92)
LOmega92 = f * v * dx(Omega92)
aOmega93 = (dot(grad(u), grad(v))) * dx(Omega93)
LOmega93 = f * v * dx(Omega93)
aOmega94 = (dot(grad(u), grad(v))) * dx(Omega94)
LOmega94 = f * v * dx(Omega94)
aOmega95 = (dot(grad(u), grad(v))) * dx(Omega95)
LOmega95 = f * v * dx(Omega95)
aOmega96 = (dot(grad(u), grad(v))) * dx(Omega96)
LOmega96 = f * v * dx(Omega96)
aOmega97 = (dot(grad(u), grad(v))) * dx(Omega97)
LOmega97 = f * v * dx(Omega97)
aOmega98 = (dot(grad(u), grad(v))) * dx(Omega98)
LOmega98 = f * v * dx(Omega98)
aOmega99 = (dot(grad(u), grad(v))) * dx(Omega99)
LOmega99 = f * v * dx(Omega99)
aOmega100 = (dot(grad(u), grad(v))) * dx(Omega100)
LOmega100 = f * v * dx(Omega100)
aOmega101 = (dot(grad(u), grad(v))) * dx(Omega101)
LOmega101 = f * v * dx(Omega101)
aOmega102 = (dot(grad(u), grad(v))) * dx(Omega102)
LOmega102 = f * v * dx(Omega102)
aOmega103 = (dot(grad(u), grad(v))) * dx(Omega103)
LOmega103 = f * v * dx(Omega103)
aOmega104 = (dot(grad(u), grad(v))) * dx(Omega104)
LOmega104 = f * v * dx(Omega104)
aOmega105 = (dot(grad(u), grad(v))) * dx(Omega105)
LOmega105 = f * v * dx(Omega105)
aOmega106 = (dot(grad(u), grad(v))) * dx(Omega106)
LOmega106 = f * v * dx(Omega106)
aOmega107 = (dot(grad(u), grad(v))) * dx(Omega107)
LOmega107 = f * v * dx(Omega107)
aOmega108 = (dot(grad(u), grad(v))) * dx(Omega108)
LOmega108 = f * v * dx(Omega108)
aOmega109 = (dot(grad(u), grad(v))) * dx(Omega109)
LOmega109 = f * v * dx(Omega109)
aOmega110 = (dot(grad(u), grad(v))) * dx(Omega110)
LOmega110 = f * v * dx(Omega110)
aOmega111 = (dot(grad(u), grad(v))) * dx(Omega111)
LOmega111 = f * v * dx(Omega111)
aOmega112 = (dot(grad(u), grad(v))) * dx(Omega112)
LOmega112 = f * v * dx(Omega112)
aOmega113 = (dot(grad(u), grad(v))) * dx(Omega113)
LOmega113 = f * v * dx(Omega113)
aOmega114 = (dot(grad(u), grad(v))) * dx(Omega114)
LOmega114 = f * v * dx(Omega114)
aOmega115 = (dot(grad(u), grad(v))) * dx(Omega115)
LOmega115 = f * v * dx(Omega115)
aOmega116 = (dot(grad(u), grad(v))) * dx(Omega116)
LOmega116 = f * v * dx(Omega116)
aOmega117 = (dot(grad(u), grad(v))) * dx(Omega117)
LOmega117 = f * v * dx(Omega117)
aOmega118 = (dot(grad(u), grad(v))) * dx(Omega118)
LOmega118 = f * v * dx(Omega118)
aOmega119 = (dot(grad(u), grad(v))) * dx(Omega119)
LOmega119 = f * v * dx(Omega119)
aOmega120 = (dot(grad(u), grad(v))) * dx(Omega120)
LOmega120 = f * v * dx(Omega120)
aOmega121 = (dot(grad(u), grad(v))) * dx(Omega121)
LOmega121 = f * v * dx(Omega121)
aOmega122 = (dot(grad(u), grad(v))) * dx(Omega122)
LOmega122 = f * v * dx(Omega122)
aOmega123 = (dot(grad(u), grad(v))) * dx(Omega123)
LOmega123 = f * v * dx(Omega123)
aOmega124 = (dot(grad(u), grad(v))) * dx(Omega124)
LOmega124 = f * v * dx(Omega124)
aOmega125 = (dot(grad(u), grad(v))) * dx(Omega125)
LOmega125 = f * v * dx(Omega125)
aOmega126 = (dot(grad(u), grad(v))) * dx(Omega126)
LOmega126 = f * v * dx(Omega126)
aOmega127 = (dot(grad(u), grad(v))) * dx(Omega127)
LOmega127 = f * v * dx(Omega127)
aOmega128 = (dot(grad(u), grad(v))) * dx(Omega128)
LOmega128 = f * v * dx(Omega128)
aOmega129 = (dot(grad(u), grad(v))) * dx(Omega129)
LOmega129 = f * v * dx(Omega129)
aOmega130 = (dot(grad(u), grad(v))) * dx(Omega130)
LOmega130 = f * v * dx(Omega130)
aOmega131 = (dot(grad(u), grad(v))) * dx(Omega131)
LOmega131 = f * v * dx(Omega131)
aOmega132 = (dot(grad(u), grad(v))) * dx(Omega132)
LOmega132 = f * v * dx(Omega132)
aOmega133 = (dot(grad(u), grad(v))) * dx(Omega133)
LOmega133 = f * v * dx(Omega133)
aOmega134 = (dot(grad(u), grad(v))) * dx(Omega134)
LOmega134 = f * v * dx(Omega134)
aOmega135 = (dot(grad(u), grad(v))) * dx(Omega135)
LOmega135 = f * v * dx(Omega135)
aOmega136 = (dot(grad(u), grad(v))) * dx(Omega136)
LOmega136 = f * v * dx(Omega136)
aOmega137 = (dot(grad(u), grad(v))) * dx(Omega137)
LOmega137 = f * v * dx(Omega137)
aOmega138 = (dot(grad(u), grad(v))) * dx(Omega138)
LOmega138 = f * v * dx(Omega138)
aOmega139 = (dot(grad(u), grad(v))) * dx(Omega139)
LOmega139 = f * v * dx(Omega139)
aOmega140 = (dot(grad(u), grad(v))) * dx(Omega140)
LOmega140 = f * v * dx(Omega140)
aOmega141 = (dot(grad(u), grad(v))) * dx(Omega141)
LOmega141 = f * v * dx(Omega141)
aOmega142 = (dot(grad(u), grad(v))) * dx(Omega142)
LOmega142 = f * v * dx(Omega142)
aOmega143 = (dot(grad(u), grad(v))) * dx(Omega143)
LOmega143 = f * v * dx(Omega143)
aOmega144 = (dot(grad(u), grad(v))) * dx(Omega144)
LOmega144 = f * v * dx(Omega144)
aOmega145 = (dot(grad(u), grad(v))) * dx(Omega145)
LOmega145 = f * v * dx(Omega145)
aOmega146 = (dot(grad(u), grad(v))) * dx(Omega146)
LOmega146 = f * v * dx(Omega146)
aOmega147 = (dot(grad(u), grad(v))) * dx(Omega147)
LOmega147 = f * v * dx(Omega147)
aOmega148 = (dot(grad(u), grad(v))) * dx(Omega148)
LOmega148 = f * v * dx(Omega148)
aOmega149 = (dot(grad(u), grad(v))) * dx(Omega149)
LOmega149 = f * v * dx(Omega149)
aOmega150 = (dot(grad(u), grad(v))) * dx(Omega150)
LOmega150 = f * v * dx(Omega150)
aOmega151 = (dot(grad(u), grad(v))) * dx(Omega151)
LOmega151 = f * v * dx(Omega151)
aOmega152 = (dot(grad(u), grad(v))) * dx(Omega152)
LOmega152 = f * v * dx(Omega152)
aOmega153 = (dot(grad(u), grad(v))) * dx(Omega153)
LOmega153 = f * v * dx(Omega153)
aOmega154 = (dot(grad(u), grad(v))) * dx(Omega154)
LOmega154 = f * v * dx(Omega154)
aOmega155 = (dot(grad(u), grad(v))) * dx(Omega155)
LOmega155 = f * v * dx(Omega155)
aOmega156 = (dot(grad(u), grad(v))) * dx(Omega156)
LOmega156 = f * v * dx(Omega156)
aOmega157 = (dot(grad(u), grad(v))) * dx(Omega157)
LOmega157 = f * v * dx(Omega157)
aOmega158 = (dot(grad(u), grad(v))) * dx(Omega158)
LOmega158 = f * v * dx(Omega158)
aOmega159 = (dot(grad(u), grad(v))) * dx(Omega159)
LOmega159 = f * v * dx(Omega159)
aOmega160 = (dot(grad(u), grad(v))) * dx(Omega160)
LOmega160 = f * v * dx(Omega160)
aOmega161 = (dot(grad(u), grad(v))) * dx(Omega161)
LOmega161 = f * v * dx(Omega161)
aOmega162 = (dot(grad(u), grad(v))) * dx(Omega162)
LOmega162 = f * v * dx(Omega162)
aOmega163 = (dot(grad(u), grad(v))) * dx(Omega163)
LOmega163 = f * v * dx(Omega163)
aOmega164 = (dot(grad(u), grad(v))) * dx(Omega164)
LOmega164 = f * v * dx(Omega164)
aOmega165 = (dot(grad(u), grad(v))) * dx(Omega165)
LOmega165 = f * v * dx(Omega165)
aOmega166 = (dot(grad(u), grad(v))) * dx(Omega166)
LOmega166 = f * v * dx(Omega166)
aOmega167 = (dot(grad(u), grad(v))) * dx(Omega167)
LOmega167 = f * v * dx(Omega167)
aOmega168 = (dot(grad(u), grad(v))) * dx(Omega168)
LOmega168 = f * v * dx(Omega168)
aOmega169 = (dot(grad(u), grad(v))) * dx(Omega169)
LOmega169 = f * v * dx(Omega169)
aOmega170 = (dot(grad(u), grad(v))) * dx(Omega170)
LOmega170 = f * v * dx(Omega170)
aOmega171 = (dot(grad(u), grad(v))) * dx(Omega171)
LOmega171 = f * v * dx(Omega171)
aOmega172 = (dot(grad(u), grad(v))) * dx(Omega172)
LOmega172 = f * v * dx(Omega172)
aOmega173 = (dot(grad(u), grad(v))) * dx(Omega173)
LOmega173 = f * v * dx(Omega173)
aOmega174 = (dot(grad(u), grad(v))) * dx(Omega174)
LOmega174 = f * v * dx(Omega174)
aOmega175 = (dot(grad(u), grad(v))) * dx(Omega175)
LOmega175 = f * v * dx(Omega175)
aOmega176 = (dot(grad(u), grad(v))) * dx(Omega176)
LOmega176 = f * v * dx(Omega176)
aOmega177 = (dot(grad(u), grad(v))) * dx(Omega177)
LOmega177 = f * v * dx(Omega177)
aOmega178 = (dot(grad(u), grad(v))) * dx(Omega178)
LOmega178 = f * v * dx(Omega178)
aOmega179 = (dot(grad(u), grad(v))) * dx(Omega179)
LOmega179 = f * v * dx(Omega179)
aOmega180 = (dot(grad(u), grad(v))) * dx(Omega180)
LOmega180 = f * v * dx(Omega180)
aOmega181 = (dot(grad(u), grad(v))) * dx(Omega181)
LOmega181 = f * v * dx(Omega181)
aOmega182 = (dot(grad(u), grad(v))) * dx(Omega182)
LOmega182 = f * v * dx(Omega182)
aOmega183 = (dot(grad(u), grad(v))) * dx(Omega183)
LOmega183 = f * v * dx(Omega183)
aOmega184 = (dot(grad(u), grad(v))) * dx(Omega184)
LOmega184 = f * v * dx(Omega184)
aOmega185 = (dot(grad(u), grad(v))) * dx(Omega185)
LOmega185 = f * v * dx(Omega185)
aOmega186 = (dot(grad(u), grad(v))) * dx(Omega186)
LOmega186 = f * v * dx(Omega186)
aOmega187 = (dot(grad(u), grad(v))) * dx(Omega187)
LOmega187 = f * v * dx(Omega187)
aOmega188 = (dot(grad(u), grad(v))) * dx(Omega188)
LOmega188 = f * v * dx(Omega188)
aOmega189 = (dot(grad(u), grad(v))) * dx(Omega189)
LOmega189 = f * v * dx(Omega189)
aOmega190 = (dot(grad(u), grad(v))) * dx(Omega190)
LOmega190 = f * v * dx(Omega190)
aOmega191 = (dot(grad(u), grad(v))) * dx(Omega191)
LOmega191 = f * v * dx(Omega191)
aOmega192 = (dot(grad(u), grad(v))) * dx(Omega192)
LOmega192 = f * v * dx(Omega192)
aOmega193 = (dot(grad(u), grad(v))) * dx(Omega193)
LOmega193 = f * v * dx(Omega193)
aOmega194 = (dot(grad(u), grad(v))) * dx(Omega194)
LOmega194 = f * v * dx(Omega194)
aOmega195 = (dot(grad(u), grad(v))) * dx(Omega195)
LOmega195 = f * v * dx(Omega195)
aOmega196 = (dot(grad(u), grad(v))) * dx(Omega196)
LOmega196 = f * v * dx(Omega196)
aOmega197 = (dot(grad(u), grad(v))) * dx(Omega197)
LOmega197 = f * v * dx(Omega197)
aOmega198 = (dot(grad(u), grad(v))) * dx(Omega198)
LOmega198 = f * v * dx(Omega198)
aOmega199 = (dot(grad(u), grad(v))) * dx(Omega199)
LOmega199 = f * v * dx(Omega199)
aOmega200 = (dot(grad(u), grad(v))) * dx(Omega200)
LOmega200 = f * v * dx(Omega200)
aOmega201 = (dot(grad(u), grad(v))) * dx(Omega201)
LOmega201 = f * v * dx(Omega201)
aOmega202 = (dot(grad(u), grad(v))) * dx(Omega202)
LOmega202 = f * v * dx(Omega202)
aOmega203 = (dot(grad(u), grad(v))) * dx(Omega203)
LOmega203 = f * v * dx(Omega203)
aOmega204 = (dot(grad(u), grad(v))) * dx(Omega204)
LOmega204 = f * v * dx(Omega204)
aOmega205 = (dot(grad(u), grad(v))) * dx(Omega205)
LOmega205 = f * v * dx(Omega205)
aOmega206 = (dot(grad(u), grad(v))) * dx(Omega206)
LOmega206 = f * v * dx(Omega206)
aOmega207 = (dot(grad(u), grad(v))) * dx(Omega207)
LOmega207 = f * v * dx(Omega207)
aOmega208 = (dot(grad(u), grad(v))) * dx(Omega208)
LOmega208 = f * v * dx(Omega208)
aOmega209 = (dot(grad(u), grad(v))) * dx(Omega209)
LOmega209 = f * v * dx(Omega209)
aOmega210 = (dot(grad(u), grad(v))) * dx(Omega210)
LOmega210 = f * v * dx(Omega210)
aOmega211 = (dot(grad(u), grad(v))) * dx(Omega211)
LOmega211 = f * v * dx(Omega211)
aOmega212 = (dot(grad(u), grad(v))) * dx(Omega212)
LOmega212 = f * v * dx(Omega212)
aOmega213 = (dot(grad(u), grad(v))) * dx(Omega213)
LOmega213 = f * v * dx(Omega213)
aOmega214 = (dot(grad(u), grad(v))) * dx(Omega214)
LOmega214 = f * v * dx(Omega214)
aOmega215 = (dot(grad(u), grad(v))) * dx(Omega215)
LOmega215 = f * v * dx(Omega215)
aOmega216 = (dot(grad(u), grad(v))) * dx(Omega216)
LOmega216 = f * v * dx(Omega216)
aOmega217 = (dot(grad(u), grad(v))) * dx(Omega217)
LOmega217 = f * v * dx(Omega217)
aOmega218 = (dot(grad(u), grad(v))) * dx(Omega218)
LOmega218 = f * v * dx(Omega218)
aOmega219 = (dot(grad(u), grad(v))) * dx(Omega219)
LOmega219 = f * v * dx(Omega219)
aOmega220 = (dot(grad(u), grad(v))) * dx(Omega220)
LOmega220 = f * v * dx(Omega220)
aOmega221 = (dot(grad(u), grad(v))) * dx(Omega221)
LOmega221 = f * v * dx(Omega221)
aOmega222 = (dot(grad(u), grad(v))) * dx(Omega222)
LOmega222 = f * v * dx(Omega222)
aOmega223 = (dot(grad(u), grad(v))) * dx(Omega223)
LOmega223 = f * v * dx(Omega223)
aOmega224 = (dot(grad(u), grad(v))) * dx(Omega224)
LOmega224 = f * v * dx(Omega224)
aOmega225 = (dot(grad(u), grad(v))) * dx(Omega225)
LOmega225 = f * v * dx(Omega225)
aOmega226 = (dot(grad(u), grad(v))) * dx(Omega226)
LOmega226 = f * v * dx(Omega226)
aOmega227 = (dot(grad(u), grad(v))) * dx(Omega227)
LOmega227 = f * v * dx(Omega227)
aOmega228 = (dot(grad(u), grad(v))) * dx(Omega228)
LOmega228 = f * v * dx(Omega228)
aOmega229 = (dot(grad(u), grad(v))) * dx(Omega229)
LOmega229 = f * v * dx(Omega229)
aOmega230 = (dot(grad(u), grad(v))) * dx(Omega230)
LOmega230 = f * v * dx(Omega230)
aOmega231 = (dot(grad(u), grad(v))) * dx(Omega231)
LOmega231 = f * v * dx(Omega231)
aOmega232 = (dot(grad(u), grad(v))) * dx(Omega232)
LOmega232 = f * v * dx(Omega232)
aOmega233 = (dot(grad(u), grad(v))) * dx(Omega233)
LOmega233 = f * v * dx(Omega233)
aOmega234 = (dot(grad(u), grad(v))) * dx(Omega234)
LOmega234 = f * v * dx(Omega234)
aOmega235 = (dot(grad(u), grad(v))) * dx(Omega235)
LOmega235 = f * v * dx(Omega235)
aOmega236 = (dot(grad(u), grad(v))) * dx(Omega236)
LOmega236 = f * v * dx(Omega236)
aOmega237 = (dot(grad(u), grad(v))) * dx(Omega237)
LOmega237 = f * v * dx(Omega237)
aOmega238 = (dot(grad(u), grad(v))) * dx(Omega238)
LOmega238 = f * v * dx(Omega238)
aOmega239 = (dot(grad(u), grad(v))) * dx(Omega239)
LOmega239 = f * v * dx(Omega239)
aOmega240 = (dot(grad(u), grad(v))) * dx(Omega240)
LOmega240 = f * v * dx(Omega240)
aOmega241 = (dot(grad(u), grad(v))) * dx(Omega241)
LOmega241 = f * v * dx(Omega241)
aOmega242 = (dot(grad(u), grad(v))) * dx(Omega242)
LOmega242 = f * v * dx(Omega242)
aOmega243 = (dot(grad(u), grad(v))) * dx(Omega243)
LOmega243 = f * v * dx(Omega243)
aOmega244 = (dot(grad(u), grad(v))) * dx(Omega244)
LOmega244 = f * v * dx(Omega244)
aOmega245 = (dot(grad(u), grad(v))) * dx(Omega245)
LOmega245 = f * v * dx(Omega245)
aOmega246 = (dot(grad(u), grad(v))) * dx(Omega246)
LOmega246 = f * v * dx(Omega246)
aOmega247 = (dot(grad(u), grad(v))) * dx(Omega247)
LOmega247 = f * v * dx(Omega247)
aOmega248 = (dot(grad(u), grad(v))) * dx(Omega248)
LOmega248 = f * v * dx(Omega248)
aOmega249 = (dot(grad(u), grad(v))) * dx(Omega249)
LOmega249 = f * v * dx(Omega249)
aOmega250 = (dot(grad(u), grad(v))) * dx(Omega250)
LOmega250 = f * v * dx(Omega250)
aOmega251 = (dot(grad(u), grad(v))) * dx(Omega251)
LOmega251 = f * v * dx(Omega251)
aOmega252 = (dot(grad(u), grad(v))) * dx(Omega252)
LOmega252 = f * v * dx(Omega252)
aOmega253 = (dot(grad(u), grad(v))) * dx(Omega253)
LOmega253 = f * v * dx(Omega253)
aOmega254 = (dot(grad(u), grad(v))) * dx(Omega254)
LOmega254 = f * v * dx(Omega254)
aOmega255 = (dot(grad(u), grad(v))) * dx(Omega255)
LOmega255 = f * v * dx(Omega255)
aOmega256 = (dot(grad(u), grad(v))) * dx(Omega256)
LOmega256 = f * v * dx(Omega256)
bcOmega1=bcs.copy()
e1=uOmega1_old+uOmega2_old
bcOmega1.append(DirichletBC(V,e1,dOmega1nOmega2))
e2=uOmega1_old+uOmega17_old
bcOmega1.append(DirichletBC(V,e2,dOmega1nOmega17))
bcOmega1.append(BC_Omega1_only)
bcOmega2=bcs.copy()
e3=uOmega2_old+uOmega1_old
bcOmega2.append(DirichletBC(V,e3,dOmega2nOmega1))
e4=uOmega2_old+uOmega3_old
bcOmega2.append(DirichletBC(V,e4,dOmega2nOmega3))
e5=uOmega2_old+uOmega18_old
bcOmega2.append(DirichletBC(V,e5,dOmega2nOmega18))
bcOmega2.append(BC_Omega2_only)
bcOmega3=bcs.copy()
e6=uOmega3_old+uOmega2_old
bcOmega3.append(DirichletBC(V,e6,dOmega3nOmega2))
e7=uOmega3_old+uOmega4_old
bcOmega3.append(DirichletBC(V,e7,dOmega3nOmega4))
e8=uOmega3_old+uOmega19_old
bcOmega3.append(DirichletBC(V,e8,dOmega3nOmega19))
bcOmega3.append(BC_Omega3_only)
bcOmega4=bcs.copy()
e9=uOmega4_old+uOmega3_old
bcOmega4.append(DirichletBC(V,e9,dOmega4nOmega3))
e10=uOmega4_old+uOmega5_old
bcOmega4.append(DirichletBC(V,e10,dOmega4nOmega5))
e11=uOmega4_old+uOmega20_old
bcOmega4.append(DirichletBC(V,e11,dOmega4nOmega20))
bcOmega4.append(BC_Omega4_only)
bcOmega5=bcs.copy()
e12=uOmega5_old+uOmega4_old
bcOmega5.append(DirichletBC(V,e12,dOmega5nOmega4))
e13=uOmega5_old+uOmega6_old
bcOmega5.append(DirichletBC(V,e13,dOmega5nOmega6))
e14=uOmega5_old+uOmega21_old
bcOmega5.append(DirichletBC(V,e14,dOmega5nOmega21))
bcOmega5.append(BC_Omega5_only)
bcOmega6=bcs.copy()
e15=uOmega6_old+uOmega5_old
bcOmega6.append(DirichletBC(V,e15,dOmega6nOmega5))
e16=uOmega6_old+uOmega7_old
bcOmega6.append(DirichletBC(V,e16,dOmega6nOmega7))
e17=uOmega6_old+uOmega22_old
bcOmega6.append(DirichletBC(V,e17,dOmega6nOmega22))
bcOmega6.append(BC_Omega6_only)
bcOmega7=bcs.copy()
e18=uOmega7_old+uOmega6_old
bcOmega7.append(DirichletBC(V,e18,dOmega7nOmega6))
e19=uOmega7_old+uOmega8_old
bcOmega7.append(DirichletBC(V,e19,dOmega7nOmega8))
e20=uOmega7_old+uOmega23_old
bcOmega7.append(DirichletBC(V,e20,dOmega7nOmega23))
bcOmega7.append(BC_Omega7_only)
bcOmega8=bcs.copy()
e21=uOmega8_old+uOmega7_old
bcOmega8.append(DirichletBC(V,e21,dOmega8nOmega7))
e22=uOmega8_old+uOmega9_old
bcOmega8.append(DirichletBC(V,e22,dOmega8nOmega9))
e23=uOmega8_old+uOmega24_old
bcOmega8.append(DirichletBC(V,e23,dOmega8nOmega24))
bcOmega8.append(BC_Omega8_only)
bcOmega9=bcs.copy()
e24=uOmega9_old+uOmega8_old
bcOmega9.append(DirichletBC(V,e24,dOmega9nOmega8))
e25=uOmega9_old+uOmega10_old
bcOmega9.append(DirichletBC(V,e25,dOmega9nOmega10))
e26=uOmega9_old+uOmega25_old
bcOmega9.append(DirichletBC(V,e26,dOmega9nOmega25))
bcOmega9.append(BC_Omega9_only)
bcOmega10=bcs.copy()
e27=uOmega10_old+uOmega9_old
bcOmega10.append(DirichletBC(V,e27,dOmega10nOmega9))
e28=uOmega10_old+uOmega11_old
bcOmega10.append(DirichletBC(V,e28,dOmega10nOmega11))
e29=uOmega10_old+uOmega26_old
bcOmega10.append(DirichletBC(V,e29,dOmega10nOmega26))
bcOmega10.append(BC_Omega10_only)
bcOmega11=bcs.copy()
e30=uOmega11_old+uOmega10_old
bcOmega11.append(DirichletBC(V,e30,dOmega11nOmega10))
e31=uOmega11_old+uOmega12_old
bcOmega11.append(DirichletBC(V,e31,dOmega11nOmega12))
e32=uOmega11_old+uOmega27_old
bcOmega11.append(DirichletBC(V,e32,dOmega11nOmega27))
bcOmega11.append(BC_Omega11_only)
bcOmega12=bcs.copy()
e33=uOmega12_old+uOmega11_old
bcOmega12.append(DirichletBC(V,e33,dOmega12nOmega11))
e34=uOmega12_old+uOmega13_old
bcOmega12.append(DirichletBC(V,e34,dOmega12nOmega13))
e35=uOmega12_old+uOmega28_old
bcOmega12.append(DirichletBC(V,e35,dOmega12nOmega28))
bcOmega12.append(BC_Omega12_only)
bcOmega13=bcs.copy()
e36=uOmega13_old+uOmega12_old
bcOmega13.append(DirichletBC(V,e36,dOmega13nOmega12))
e37=uOmega13_old+uOmega14_old
bcOmega13.append(DirichletBC(V,e37,dOmega13nOmega14))
e38=uOmega13_old+uOmega29_old
bcOmega13.append(DirichletBC(V,e38,dOmega13nOmega29))
bcOmega13.append(BC_Omega13_only)
bcOmega14=bcs.copy()
e39=uOmega14_old+uOmega13_old
bcOmega14.append(DirichletBC(V,e39,dOmega14nOmega13))
e40=uOmega14_old+uOmega15_old
bcOmega14.append(DirichletBC(V,e40,dOmega14nOmega15))
e41=uOmega14_old+uOmega30_old
bcOmega14.append(DirichletBC(V,e41,dOmega14nOmega30))
bcOmega14.append(BC_Omega14_only)
bcOmega15=bcs.copy()
e42=uOmega15_old+uOmega14_old
bcOmega15.append(DirichletBC(V,e42,dOmega15nOmega14))
e43=uOmega15_old+uOmega16_old
bcOmega15.append(DirichletBC(V,e43,dOmega15nOmega16))
e44=uOmega15_old+uOmega31_old
bcOmega15.append(DirichletBC(V,e44,dOmega15nOmega31))
bcOmega15.append(BC_Omega15_only)
bcOmega16=bcs.copy()
e45=uOmega16_old+uOmega15_old
bcOmega16.append(DirichletBC(V,e45,dOmega16nOmega15))
e46=uOmega16_old+uOmega32_old
bcOmega16.append(DirichletBC(V,e46,dOmega16nOmega32))
bcOmega16.append(BC_Omega16_only)
bcOmega17=bcs.copy()
e47=uOmega17_old+uOmega1_old
bcOmega17.append(DirichletBC(V,e47,dOmega17nOmega1))
e48=uOmega17_old+uOmega18_old
bcOmega17.append(DirichletBC(V,e48,dOmega17nOmega18))
e49=uOmega17_old+uOmega33_old
bcOmega17.append(DirichletBC(V,e49,dOmega17nOmega33))
bcOmega17.append(BC_Omega17_only)
bcOmega18=bcs.copy()
e50=uOmega18_old+uOmega2_old
bcOmega18.append(DirichletBC(V,e50,dOmega18nOmega2))
e51=uOmega18_old+uOmega17_old
bcOmega18.append(DirichletBC(V,e51,dOmega18nOmega17))
e52=uOmega18_old+uOmega19_old
bcOmega18.append(DirichletBC(V,e52,dOmega18nOmega19))
e53=uOmega18_old+uOmega34_old
bcOmega18.append(DirichletBC(V,e53,dOmega18nOmega34))
bcOmega18.append(BC_Omega18_only)
bcOmega19=bcs.copy()
e54=uOmega19_old+uOmega3_old
bcOmega19.append(DirichletBC(V,e54,dOmega19nOmega3))
e55=uOmega19_old+uOmega18_old
bcOmega19.append(DirichletBC(V,e55,dOmega19nOmega18))
e56=uOmega19_old+uOmega20_old
bcOmega19.append(DirichletBC(V,e56,dOmega19nOmega20))
e57=uOmega19_old+uOmega35_old
bcOmega19.append(DirichletBC(V,e57,dOmega19nOmega35))
bcOmega19.append(BC_Omega19_only)
bcOmega20=bcs.copy()
e58=uOmega20_old+uOmega4_old
bcOmega20.append(DirichletBC(V,e58,dOmega20nOmega4))
e59=uOmega20_old+uOmega19_old
bcOmega20.append(DirichletBC(V,e59,dOmega20nOmega19))
e60=uOmega20_old+uOmega21_old
bcOmega20.append(DirichletBC(V,e60,dOmega20nOmega21))
e61=uOmega20_old+uOmega36_old
bcOmega20.append(DirichletBC(V,e61,dOmega20nOmega36))
bcOmega20.append(BC_Omega20_only)
bcOmega21=bcs.copy()
e62=uOmega21_old+uOmega5_old
bcOmega21.append(DirichletBC(V,e62,dOmega21nOmega5))
e63=uOmega21_old+uOmega20_old
bcOmega21.append(DirichletBC(V,e63,dOmega21nOmega20))
e64=uOmega21_old+uOmega22_old
bcOmega21.append(DirichletBC(V,e64,dOmega21nOmega22))
e65=uOmega21_old+uOmega37_old
bcOmega21.append(DirichletBC(V,e65,dOmega21nOmega37))
bcOmega21.append(BC_Omega21_only)
bcOmega22=bcs.copy()
e66=uOmega22_old+uOmega6_old
bcOmega22.append(DirichletBC(V,e66,dOmega22nOmega6))
e67=uOmega22_old+uOmega21_old
bcOmega22.append(DirichletBC(V,e67,dOmega22nOmega21))
e68=uOmega22_old+uOmega23_old
bcOmega22.append(DirichletBC(V,e68,dOmega22nOmega23))
e69=uOmega22_old+uOmega38_old
bcOmega22.append(DirichletBC(V,e69,dOmega22nOmega38))
bcOmega22.append(BC_Omega22_only)
bcOmega23=bcs.copy()
e70=uOmega23_old+uOmega7_old
bcOmega23.append(DirichletBC(V,e70,dOmega23nOmega7))
e71=uOmega23_old+uOmega22_old
bcOmega23.append(DirichletBC(V,e71,dOmega23nOmega22))
e72=uOmega23_old+uOmega24_old
bcOmega23.append(DirichletBC(V,e72,dOmega23nOmega24))
e73=uOmega23_old+uOmega39_old
bcOmega23.append(DirichletBC(V,e73,dOmega23nOmega39))
bcOmega23.append(BC_Omega23_only)
bcOmega24=bcs.copy()
e74=uOmega24_old+uOmega8_old
bcOmega24.append(DirichletBC(V,e74,dOmega24nOmega8))
e75=uOmega24_old+uOmega23_old
bcOmega24.append(DirichletBC(V,e75,dOmega24nOmega23))
e76=uOmega24_old+uOmega25_old
bcOmega24.append(DirichletBC(V,e76,dOmega24nOmega25))
e77=uOmega24_old+uOmega40_old
bcOmega24.append(DirichletBC(V,e77,dOmega24nOmega40))
bcOmega24.append(BC_Omega24_only)
bcOmega25=bcs.copy()
e78=uOmega25_old+uOmega9_old
bcOmega25.append(DirichletBC(V,e78,dOmega25nOmega9))
e79=uOmega25_old+uOmega24_old
bcOmega25.append(DirichletBC(V,e79,dOmega25nOmega24))
e80=uOmega25_old+uOmega26_old
bcOmega25.append(DirichletBC(V,e80,dOmega25nOmega26))
e81=uOmega25_old+uOmega41_old
bcOmega25.append(DirichletBC(V,e81,dOmega25nOmega41))
bcOmega25.append(BC_Omega25_only)
bcOmega26=bcs.copy()
e82=uOmega26_old+uOmega10_old
bcOmega26.append(DirichletBC(V,e82,dOmega26nOmega10))
e83=uOmega26_old+uOmega25_old
bcOmega26.append(DirichletBC(V,e83,dOmega26nOmega25))
e84=uOmega26_old+uOmega27_old
bcOmega26.append(DirichletBC(V,e84,dOmega26nOmega27))
e85=uOmega26_old+uOmega42_old
bcOmega26.append(DirichletBC(V,e85,dOmega26nOmega42))
bcOmega26.append(BC_Omega26_only)
bcOmega27=bcs.copy()
e86=uOmega27_old+uOmega11_old
bcOmega27.append(DirichletBC(V,e86,dOmega27nOmega11))
e87=uOmega27_old+uOmega26_old
bcOmega27.append(DirichletBC(V,e87,dOmega27nOmega26))
e88=uOmega27_old+uOmega28_old
bcOmega27.append(DirichletBC(V,e88,dOmega27nOmega28))
e89=uOmega27_old+uOmega43_old
bcOmega27.append(DirichletBC(V,e89,dOmega27nOmega43))
bcOmega27.append(BC_Omega27_only)
bcOmega28=bcs.copy()
e90=uOmega28_old+uOmega12_old
bcOmega28.append(DirichletBC(V,e90,dOmega28nOmega12))
e91=uOmega28_old+uOmega27_old
bcOmega28.append(DirichletBC(V,e91,dOmega28nOmega27))
e92=uOmega28_old+uOmega29_old
bcOmega28.append(DirichletBC(V,e92,dOmega28nOmega29))
e93=uOmega28_old+uOmega44_old
bcOmega28.append(DirichletBC(V,e93,dOmega28nOmega44))
bcOmega28.append(BC_Omega28_only)
bcOmega29=bcs.copy()
e94=uOmega29_old+uOmega13_old
bcOmega29.append(DirichletBC(V,e94,dOmega29nOmega13))
e95=uOmega29_old+uOmega28_old
bcOmega29.append(DirichletBC(V,e95,dOmega29nOmega28))
e96=uOmega29_old+uOmega30_old
bcOmega29.append(DirichletBC(V,e96,dOmega29nOmega30))
e97=uOmega29_old+uOmega45_old
bcOmega29.append(DirichletBC(V,e97,dOmega29nOmega45))
bcOmega29.append(BC_Omega29_only)
bcOmega30=bcs.copy()
e98=uOmega30_old+uOmega14_old
bcOmega30.append(DirichletBC(V,e98,dOmega30nOmega14))
e99=uOmega30_old+uOmega29_old
bcOmega30.append(DirichletBC(V,e99,dOmega30nOmega29))
e100=uOmega30_old+uOmega31_old
bcOmega30.append(DirichletBC(V,e100,dOmega30nOmega31))
e101=uOmega30_old+uOmega46_old
bcOmega30.append(DirichletBC(V,e101,dOmega30nOmega46))
bcOmega30.append(BC_Omega30_only)
bcOmega31=bcs.copy()
e102=uOmega31_old+uOmega15_old
bcOmega31.append(DirichletBC(V,e102,dOmega31nOmega15))
e103=uOmega31_old+uOmega30_old
bcOmega31.append(DirichletBC(V,e103,dOmega31nOmega30))
e104=uOmega31_old+uOmega32_old
bcOmega31.append(DirichletBC(V,e104,dOmega31nOmega32))
e105=uOmega31_old+uOmega47_old
bcOmega31.append(DirichletBC(V,e105,dOmega31nOmega47))
bcOmega31.append(BC_Omega31_only)
bcOmega32=bcs.copy()
e106=uOmega32_old+uOmega16_old
bcOmega32.append(DirichletBC(V,e106,dOmega32nOmega16))
e107=uOmega32_old+uOmega31_old
bcOmega32.append(DirichletBC(V,e107,dOmega32nOmega31))
e108=uOmega32_old+uOmega48_old
bcOmega32.append(DirichletBC(V,e108,dOmega32nOmega48))
bcOmega32.append(BC_Omega32_only)
bcOmega33=bcs.copy()
e109=uOmega33_old+uOmega17_old
bcOmega33.append(DirichletBC(V,e109,dOmega33nOmega17))
e110=uOmega33_old+uOmega34_old
bcOmega33.append(DirichletBC(V,e110,dOmega33nOmega34))
e111=uOmega33_old+uOmega49_old
bcOmega33.append(DirichletBC(V,e111,dOmega33nOmega49))
bcOmega33.append(BC_Omega33_only)
bcOmega34=bcs.copy()
e112=uOmega34_old+uOmega18_old
bcOmega34.append(DirichletBC(V,e112,dOmega34nOmega18))
e113=uOmega34_old+uOmega33_old
bcOmega34.append(DirichletBC(V,e113,dOmega34nOmega33))
e114=uOmega34_old+uOmega35_old
bcOmega34.append(DirichletBC(V,e114,dOmega34nOmega35))
e115=uOmega34_old+uOmega50_old
bcOmega34.append(DirichletBC(V,e115,dOmega34nOmega50))
bcOmega34.append(BC_Omega34_only)
bcOmega35=bcs.copy()
e116=uOmega35_old+uOmega19_old
bcOmega35.append(DirichletBC(V,e116,dOmega35nOmega19))
e117=uOmega35_old+uOmega34_old
bcOmega35.append(DirichletBC(V,e117,dOmega35nOmega34))
e118=uOmega35_old+uOmega36_old
bcOmega35.append(DirichletBC(V,e118,dOmega35nOmega36))
e119=uOmega35_old+uOmega51_old
bcOmega35.append(DirichletBC(V,e119,dOmega35nOmega51))
bcOmega35.append(BC_Omega35_only)
bcOmega36=bcs.copy()
e120=uOmega36_old+uOmega20_old
bcOmega36.append(DirichletBC(V,e120,dOmega36nOmega20))
e121=uOmega36_old+uOmega35_old
bcOmega36.append(DirichletBC(V,e121,dOmega36nOmega35))
e122=uOmega36_old+uOmega37_old
bcOmega36.append(DirichletBC(V,e122,dOmega36nOmega37))
e123=uOmega36_old+uOmega52_old
bcOmega36.append(DirichletBC(V,e123,dOmega36nOmega52))
bcOmega36.append(BC_Omega36_only)
bcOmega37=bcs.copy()
e124=uOmega37_old+uOmega21_old
bcOmega37.append(DirichletBC(V,e124,dOmega37nOmega21))
e125=uOmega37_old+uOmega36_old
bcOmega37.append(DirichletBC(V,e125,dOmega37nOmega36))
e126=uOmega37_old+uOmega38_old
bcOmega37.append(DirichletBC(V,e126,dOmega37nOmega38))
e127=uOmega37_old+uOmega53_old
bcOmega37.append(DirichletBC(V,e127,dOmega37nOmega53))
bcOmega37.append(BC_Omega37_only)
bcOmega38=bcs.copy()
e128=uOmega38_old+uOmega22_old
bcOmega38.append(DirichletBC(V,e128,dOmega38nOmega22))
e129=uOmega38_old+uOmega37_old
bcOmega38.append(DirichletBC(V,e129,dOmega38nOmega37))
e130=uOmega38_old+uOmega39_old
bcOmega38.append(DirichletBC(V,e130,dOmega38nOmega39))
e131=uOmega38_old+uOmega54_old
bcOmega38.append(DirichletBC(V,e131,dOmega38nOmega54))
bcOmega38.append(BC_Omega38_only)
bcOmega39=bcs.copy()
e132=uOmega39_old+uOmega23_old
bcOmega39.append(DirichletBC(V,e132,dOmega39nOmega23))
e133=uOmega39_old+uOmega38_old
bcOmega39.append(DirichletBC(V,e133,dOmega39nOmega38))
e134=uOmega39_old+uOmega40_old
bcOmega39.append(DirichletBC(V,e134,dOmega39nOmega40))
e135=uOmega39_old+uOmega55_old
bcOmega39.append(DirichletBC(V,e135,dOmega39nOmega55))
bcOmega39.append(BC_Omega39_only)
bcOmega40=bcs.copy()
e136=uOmega40_old+uOmega24_old
bcOmega40.append(DirichletBC(V,e136,dOmega40nOmega24))
e137=uOmega40_old+uOmega39_old
bcOmega40.append(DirichletBC(V,e137,dOmega40nOmega39))
e138=uOmega40_old+uOmega41_old
bcOmega40.append(DirichletBC(V,e138,dOmega40nOmega41))
e139=uOmega40_old+uOmega56_old
bcOmega40.append(DirichletBC(V,e139,dOmega40nOmega56))
bcOmega40.append(BC_Omega40_only)
bcOmega41=bcs.copy()
e140=uOmega41_old+uOmega25_old
bcOmega41.append(DirichletBC(V,e140,dOmega41nOmega25))
e141=uOmega41_old+uOmega40_old
bcOmega41.append(DirichletBC(V,e141,dOmega41nOmega40))
e142=uOmega41_old+uOmega42_old
bcOmega41.append(DirichletBC(V,e142,dOmega41nOmega42))
e143=uOmega41_old+uOmega57_old
bcOmega41.append(DirichletBC(V,e143,dOmega41nOmega57))
bcOmega41.append(BC_Omega41_only)
bcOmega42=bcs.copy()
e144=uOmega42_old+uOmega26_old
bcOmega42.append(DirichletBC(V,e144,dOmega42nOmega26))
e145=uOmega42_old+uOmega41_old
bcOmega42.append(DirichletBC(V,e145,dOmega42nOmega41))
e146=uOmega42_old+uOmega43_old
bcOmega42.append(DirichletBC(V,e146,dOmega42nOmega43))
e147=uOmega42_old+uOmega58_old
bcOmega42.append(DirichletBC(V,e147,dOmega42nOmega58))
bcOmega42.append(BC_Omega42_only)
bcOmega43=bcs.copy()
e148=uOmega43_old+uOmega27_old
bcOmega43.append(DirichletBC(V,e148,dOmega43nOmega27))
e149=uOmega43_old+uOmega42_old
bcOmega43.append(DirichletBC(V,e149,dOmega43nOmega42))
e150=uOmega43_old+uOmega44_old
bcOmega43.append(DirichletBC(V,e150,dOmega43nOmega44))
e151=uOmega43_old+uOmega59_old
bcOmega43.append(DirichletBC(V,e151,dOmega43nOmega59))
bcOmega43.append(BC_Omega43_only)
bcOmega44=bcs.copy()
e152=uOmega44_old+uOmega28_old
bcOmega44.append(DirichletBC(V,e152,dOmega44nOmega28))
e153=uOmega44_old+uOmega43_old
bcOmega44.append(DirichletBC(V,e153,dOmega44nOmega43))
e154=uOmega44_old+uOmega45_old
bcOmega44.append(DirichletBC(V,e154,dOmega44nOmega45))
e155=uOmega44_old+uOmega60_old
bcOmega44.append(DirichletBC(V,e155,dOmega44nOmega60))
bcOmega44.append(BC_Omega44_only)
bcOmega45=bcs.copy()
e156=uOmega45_old+uOmega29_old
bcOmega45.append(DirichletBC(V,e156,dOmega45nOmega29))
e157=uOmega45_old+uOmega44_old
bcOmega45.append(DirichletBC(V,e157,dOmega45nOmega44))
e158=uOmega45_old+uOmega46_old
bcOmega45.append(DirichletBC(V,e158,dOmega45nOmega46))
e159=uOmega45_old+uOmega61_old
bcOmega45.append(DirichletBC(V,e159,dOmega45nOmega61))
bcOmega45.append(BC_Omega45_only)
bcOmega46=bcs.copy()
e160=uOmega46_old+uOmega30_old
bcOmega46.append(DirichletBC(V,e160,dOmega46nOmega30))
e161=uOmega46_old+uOmega45_old
bcOmega46.append(DirichletBC(V,e161,dOmega46nOmega45))
e162=uOmega46_old+uOmega47_old
bcOmega46.append(DirichletBC(V,e162,dOmega46nOmega47))
e163=uOmega46_old+uOmega62_old
bcOmega46.append(DirichletBC(V,e163,dOmega46nOmega62))
bcOmega46.append(BC_Omega46_only)
bcOmega47=bcs.copy()
e164=uOmega47_old+uOmega31_old
bcOmega47.append(DirichletBC(V,e164,dOmega47nOmega31))
e165=uOmega47_old+uOmega46_old
bcOmega47.append(DirichletBC(V,e165,dOmega47nOmega46))
e166=uOmega47_old+uOmega48_old
bcOmega47.append(DirichletBC(V,e166,dOmega47nOmega48))
e167=uOmega47_old+uOmega63_old
bcOmega47.append(DirichletBC(V,e167,dOmega47nOmega63))
bcOmega47.append(BC_Omega47_only)
bcOmega48=bcs.copy()
e168=uOmega48_old+uOmega32_old
bcOmega48.append(DirichletBC(V,e168,dOmega48nOmega32))
e169=uOmega48_old+uOmega47_old
bcOmega48.append(DirichletBC(V,e169,dOmega48nOmega47))
e170=uOmega48_old+uOmega64_old
bcOmega48.append(DirichletBC(V,e170,dOmega48nOmega64))
bcOmega48.append(BC_Omega48_only)
bcOmega49=bcs.copy()
e171=uOmega49_old+uOmega33_old
bcOmega49.append(DirichletBC(V,e171,dOmega49nOmega33))
e172=uOmega49_old+uOmega50_old
bcOmega49.append(DirichletBC(V,e172,dOmega49nOmega50))
e173=uOmega49_old+uOmega65_old
bcOmega49.append(DirichletBC(V,e173,dOmega49nOmega65))
bcOmega49.append(BC_Omega49_only)
bcOmega50=bcs.copy()
e174=uOmega50_old+uOmega34_old
bcOmega50.append(DirichletBC(V,e174,dOmega50nOmega34))
e175=uOmega50_old+uOmega49_old
bcOmega50.append(DirichletBC(V,e175,dOmega50nOmega49))
e176=uOmega50_old+uOmega51_old
bcOmega50.append(DirichletBC(V,e176,dOmega50nOmega51))
e177=uOmega50_old+uOmega66_old
bcOmega50.append(DirichletBC(V,e177,dOmega50nOmega66))
bcOmega50.append(BC_Omega50_only)
bcOmega51=bcs.copy()
e178=uOmega51_old+uOmega35_old
bcOmega51.append(DirichletBC(V,e178,dOmega51nOmega35))
e179=uOmega51_old+uOmega50_old
bcOmega51.append(DirichletBC(V,e179,dOmega51nOmega50))
e180=uOmega51_old+uOmega52_old
bcOmega51.append(DirichletBC(V,e180,dOmega51nOmega52))
e181=uOmega51_old+uOmega67_old
bcOmega51.append(DirichletBC(V,e181,dOmega51nOmega67))
bcOmega51.append(BC_Omega51_only)
bcOmega52=bcs.copy()
e182=uOmega52_old+uOmega36_old
bcOmega52.append(DirichletBC(V,e182,dOmega52nOmega36))
e183=uOmega52_old+uOmega51_old
bcOmega52.append(DirichletBC(V,e183,dOmega52nOmega51))
e184=uOmega52_old+uOmega53_old
bcOmega52.append(DirichletBC(V,e184,dOmega52nOmega53))
e185=uOmega52_old+uOmega68_old
bcOmega52.append(DirichletBC(V,e185,dOmega52nOmega68))
bcOmega52.append(BC_Omega52_only)
bcOmega53=bcs.copy()
e186=uOmega53_old+uOmega37_old
bcOmega53.append(DirichletBC(V,e186,dOmega53nOmega37))
e187=uOmega53_old+uOmega52_old
bcOmega53.append(DirichletBC(V,e187,dOmega53nOmega52))
e188=uOmega53_old+uOmega54_old
bcOmega53.append(DirichletBC(V,e188,dOmega53nOmega54))
e189=uOmega53_old+uOmega69_old
bcOmega53.append(DirichletBC(V,e189,dOmega53nOmega69))
bcOmega53.append(BC_Omega53_only)
bcOmega54=bcs.copy()
e190=uOmega54_old+uOmega38_old
bcOmega54.append(DirichletBC(V,e190,dOmega54nOmega38))
e191=uOmega54_old+uOmega53_old
bcOmega54.append(DirichletBC(V,e191,dOmega54nOmega53))
e192=uOmega54_old+uOmega55_old
bcOmega54.append(DirichletBC(V,e192,dOmega54nOmega55))
e193=uOmega54_old+uOmega70_old
bcOmega54.append(DirichletBC(V,e193,dOmega54nOmega70))
bcOmega54.append(BC_Omega54_only)
bcOmega55=bcs.copy()
e194=uOmega55_old+uOmega39_old
bcOmega55.append(DirichletBC(V,e194,dOmega55nOmega39))
e195=uOmega55_old+uOmega54_old
bcOmega55.append(DirichletBC(V,e195,dOmega55nOmega54))
e196=uOmega55_old+uOmega56_old
bcOmega55.append(DirichletBC(V,e196,dOmega55nOmega56))
e197=uOmega55_old+uOmega71_old
bcOmega55.append(DirichletBC(V,e197,dOmega55nOmega71))
bcOmega55.append(BC_Omega55_only)
bcOmega56=bcs.copy()
e198=uOmega56_old+uOmega40_old
bcOmega56.append(DirichletBC(V,e198,dOmega56nOmega40))
e199=uOmega56_old+uOmega55_old
bcOmega56.append(DirichletBC(V,e199,dOmega56nOmega55))
e200=uOmega56_old+uOmega57_old
bcOmega56.append(DirichletBC(V,e200,dOmega56nOmega57))
e201=uOmega56_old+uOmega72_old
bcOmega56.append(DirichletBC(V,e201,dOmega56nOmega72))
bcOmega56.append(BC_Omega56_only)
bcOmega57=bcs.copy()
e202=uOmega57_old+uOmega41_old
bcOmega57.append(DirichletBC(V,e202,dOmega57nOmega41))
e203=uOmega57_old+uOmega56_old
bcOmega57.append(DirichletBC(V,e203,dOmega57nOmega56))
e204=uOmega57_old+uOmega58_old
bcOmega57.append(DirichletBC(V,e204,dOmega57nOmega58))
e205=uOmega57_old+uOmega73_old
bcOmega57.append(DirichletBC(V,e205,dOmega57nOmega73))
bcOmega57.append(BC_Omega57_only)
bcOmega58=bcs.copy()
e206=uOmega58_old+uOmega42_old
bcOmega58.append(DirichletBC(V,e206,dOmega58nOmega42))
e207=uOmega58_old+uOmega57_old
bcOmega58.append(DirichletBC(V,e207,dOmega58nOmega57))
e208=uOmega58_old+uOmega59_old
bcOmega58.append(DirichletBC(V,e208,dOmega58nOmega59))
e209=uOmega58_old+uOmega74_old
bcOmega58.append(DirichletBC(V,e209,dOmega58nOmega74))
bcOmega58.append(BC_Omega58_only)
bcOmega59=bcs.copy()
e210=uOmega59_old+uOmega43_old
bcOmega59.append(DirichletBC(V,e210,dOmega59nOmega43))
e211=uOmega59_old+uOmega58_old
bcOmega59.append(DirichletBC(V,e211,dOmega59nOmega58))
e212=uOmega59_old+uOmega60_old
bcOmega59.append(DirichletBC(V,e212,dOmega59nOmega60))
e213=uOmega59_old+uOmega75_old
bcOmega59.append(DirichletBC(V,e213,dOmega59nOmega75))
bcOmega59.append(BC_Omega59_only)
bcOmega60=bcs.copy()
e214=uOmega60_old+uOmega44_old
bcOmega60.append(DirichletBC(V,e214,dOmega60nOmega44))
e215=uOmega60_old+uOmega59_old
bcOmega60.append(DirichletBC(V,e215,dOmega60nOmega59))
e216=uOmega60_old+uOmega61_old
bcOmega60.append(DirichletBC(V,e216,dOmega60nOmega61))
e217=uOmega60_old+uOmega76_old
bcOmega60.append(DirichletBC(V,e217,dOmega60nOmega76))
bcOmega60.append(BC_Omega60_only)
bcOmega61=bcs.copy()
e218=uOmega61_old+uOmega45_old
bcOmega61.append(DirichletBC(V,e218,dOmega61nOmega45))
e219=uOmega61_old+uOmega60_old
bcOmega61.append(DirichletBC(V,e219,dOmega61nOmega60))
e220=uOmega61_old+uOmega62_old
bcOmega61.append(DirichletBC(V,e220,dOmega61nOmega62))
e221=uOmega61_old+uOmega77_old
bcOmega61.append(DirichletBC(V,e221,dOmega61nOmega77))
bcOmega61.append(BC_Omega61_only)
bcOmega62=bcs.copy()
e222=uOmega62_old+uOmega46_old
bcOmega62.append(DirichletBC(V,e222,dOmega62nOmega46))
e223=uOmega62_old+uOmega61_old
bcOmega62.append(DirichletBC(V,e223,dOmega62nOmega61))
e224=uOmega62_old+uOmega63_old
bcOmega62.append(DirichletBC(V,e224,dOmega62nOmega63))
e225=uOmega62_old+uOmega78_old
bcOmega62.append(DirichletBC(V,e225,dOmega62nOmega78))
bcOmega62.append(BC_Omega62_only)
bcOmega63=bcs.copy()
e226=uOmega63_old+uOmega47_old
bcOmega63.append(DirichletBC(V,e226,dOmega63nOmega47))
e227=uOmega63_old+uOmega62_old
bcOmega63.append(DirichletBC(V,e227,dOmega63nOmega62))
e228=uOmega63_old+uOmega64_old
bcOmega63.append(DirichletBC(V,e228,dOmega63nOmega64))
e229=uOmega63_old+uOmega79_old
bcOmega63.append(DirichletBC(V,e229,dOmega63nOmega79))
bcOmega63.append(BC_Omega63_only)
bcOmega64=bcs.copy()
e230=uOmega64_old+uOmega48_old
bcOmega64.append(DirichletBC(V,e230,dOmega64nOmega48))
e231=uOmega64_old+uOmega63_old
bcOmega64.append(DirichletBC(V,e231,dOmega64nOmega63))
e232=uOmega64_old+uOmega80_old
bcOmega64.append(DirichletBC(V,e232,dOmega64nOmega80))
bcOmega64.append(BC_Omega64_only)
bcOmega65=bcs.copy()
e233=uOmega65_old+uOmega49_old
bcOmega65.append(DirichletBC(V,e233,dOmega65nOmega49))
e234=uOmega65_old+uOmega66_old
bcOmega65.append(DirichletBC(V,e234,dOmega65nOmega66))
e235=uOmega65_old+uOmega81_old
bcOmega65.append(DirichletBC(V,e235,dOmega65nOmega81))
bcOmega65.append(BC_Omega65_only)
bcOmega66=bcs.copy()
e236=uOmega66_old+uOmega50_old
bcOmega66.append(DirichletBC(V,e236,dOmega66nOmega50))
e237=uOmega66_old+uOmega65_old
bcOmega66.append(DirichletBC(V,e237,dOmega66nOmega65))
e238=uOmega66_old+uOmega67_old
bcOmega66.append(DirichletBC(V,e238,dOmega66nOmega67))
e239=uOmega66_old+uOmega82_old
bcOmega66.append(DirichletBC(V,e239,dOmega66nOmega82))
bcOmega66.append(BC_Omega66_only)
bcOmega67=bcs.copy()
e240=uOmega67_old+uOmega51_old
bcOmega67.append(DirichletBC(V,e240,dOmega67nOmega51))
e241=uOmega67_old+uOmega66_old
bcOmega67.append(DirichletBC(V,e241,dOmega67nOmega66))
e242=uOmega67_old+uOmega68_old
bcOmega67.append(DirichletBC(V,e242,dOmega67nOmega68))
e243=uOmega67_old+uOmega83_old
bcOmega67.append(DirichletBC(V,e243,dOmega67nOmega83))
bcOmega67.append(BC_Omega67_only)
bcOmega68=bcs.copy()
e244=uOmega68_old+uOmega52_old
bcOmega68.append(DirichletBC(V,e244,dOmega68nOmega52))
e245=uOmega68_old+uOmega67_old
bcOmega68.append(DirichletBC(V,e245,dOmega68nOmega67))
e246=uOmega68_old+uOmega69_old
bcOmega68.append(DirichletBC(V,e246,dOmega68nOmega69))
e247=uOmega68_old+uOmega84_old
bcOmega68.append(DirichletBC(V,e247,dOmega68nOmega84))
bcOmega68.append(BC_Omega68_only)
bcOmega69=bcs.copy()
e248=uOmega69_old+uOmega53_old
bcOmega69.append(DirichletBC(V,e248,dOmega69nOmega53))
e249=uOmega69_old+uOmega68_old
bcOmega69.append(DirichletBC(V,e249,dOmega69nOmega68))
e250=uOmega69_old+uOmega70_old
bcOmega69.append(DirichletBC(V,e250,dOmega69nOmega70))
e251=uOmega69_old+uOmega85_old
bcOmega69.append(DirichletBC(V,e251,dOmega69nOmega85))
bcOmega69.append(BC_Omega69_only)
bcOmega70=bcs.copy()
e252=uOmega70_old+uOmega54_old
bcOmega70.append(DirichletBC(V,e252,dOmega70nOmega54))
e253=uOmega70_old+uOmega69_old
bcOmega70.append(DirichletBC(V,e253,dOmega70nOmega69))
e254=uOmega70_old+uOmega71_old
bcOmega70.append(DirichletBC(V,e254,dOmega70nOmega71))
e255=uOmega70_old+uOmega86_old
bcOmega70.append(DirichletBC(V,e255,dOmega70nOmega86))
bcOmega70.append(BC_Omega70_only)
bcOmega71=bcs.copy()
e256=uOmega71_old+uOmega55_old
bcOmega71.append(DirichletBC(V,e256,dOmega71nOmega55))
e257=uOmega71_old+uOmega70_old
bcOmega71.append(DirichletBC(V,e257,dOmega71nOmega70))
e258=uOmega71_old+uOmega72_old
bcOmega71.append(DirichletBC(V,e258,dOmega71nOmega72))
e259=uOmega71_old+uOmega87_old
bcOmega71.append(DirichletBC(V,e259,dOmega71nOmega87))
bcOmega71.append(BC_Omega71_only)
bcOmega72=bcs.copy()
e260=uOmega72_old+uOmega56_old
bcOmega72.append(DirichletBC(V,e260,dOmega72nOmega56))
e261=uOmega72_old+uOmega71_old
bcOmega72.append(DirichletBC(V,e261,dOmega72nOmega71))
e262=uOmega72_old+uOmega73_old
bcOmega72.append(DirichletBC(V,e262,dOmega72nOmega73))
e263=uOmega72_old+uOmega88_old
bcOmega72.append(DirichletBC(V,e263,dOmega72nOmega88))
bcOmega72.append(BC_Omega72_only)
bcOmega73=bcs.copy()
e264=uOmega73_old+uOmega57_old
bcOmega73.append(DirichletBC(V,e264,dOmega73nOmega57))
e265=uOmega73_old+uOmega72_old
bcOmega73.append(DirichletBC(V,e265,dOmega73nOmega72))
e266=uOmega73_old+uOmega74_old
bcOmega73.append(DirichletBC(V,e266,dOmega73nOmega74))
e267=uOmega73_old+uOmega89_old
bcOmega73.append(DirichletBC(V,e267,dOmega73nOmega89))
bcOmega73.append(BC_Omega73_only)
bcOmega74=bcs.copy()
e268=uOmega74_old+uOmega58_old
bcOmega74.append(DirichletBC(V,e268,dOmega74nOmega58))
e269=uOmega74_old+uOmega73_old
bcOmega74.append(DirichletBC(V,e269,dOmega74nOmega73))
e270=uOmega74_old+uOmega75_old
bcOmega74.append(DirichletBC(V,e270,dOmega74nOmega75))
e271=uOmega74_old+uOmega90_old
bcOmega74.append(DirichletBC(V,e271,dOmega74nOmega90))
bcOmega74.append(BC_Omega74_only)
bcOmega75=bcs.copy()
e272=uOmega75_old+uOmega59_old
bcOmega75.append(DirichletBC(V,e272,dOmega75nOmega59))
e273=uOmega75_old+uOmega74_old
bcOmega75.append(DirichletBC(V,e273,dOmega75nOmega74))
e274=uOmega75_old+uOmega76_old
bcOmega75.append(DirichletBC(V,e274,dOmega75nOmega76))
e275=uOmega75_old+uOmega91_old
bcOmega75.append(DirichletBC(V,e275,dOmega75nOmega91))
bcOmega75.append(BC_Omega75_only)
bcOmega76=bcs.copy()
e276=uOmega76_old+uOmega60_old
bcOmega76.append(DirichletBC(V,e276,dOmega76nOmega60))
e277=uOmega76_old+uOmega75_old
bcOmega76.append(DirichletBC(V,e277,dOmega76nOmega75))
e278=uOmega76_old+uOmega77_old
bcOmega76.append(DirichletBC(V,e278,dOmega76nOmega77))
e279=uOmega76_old+uOmega92_old
bcOmega76.append(DirichletBC(V,e279,dOmega76nOmega92))
bcOmega76.append(BC_Omega76_only)
bcOmega77=bcs.copy()
e280=uOmega77_old+uOmega61_old
bcOmega77.append(DirichletBC(V,e280,dOmega77nOmega61))
e281=uOmega77_old+uOmega76_old
bcOmega77.append(DirichletBC(V,e281,dOmega77nOmega76))
e282=uOmega77_old+uOmega78_old
bcOmega77.append(DirichletBC(V,e282,dOmega77nOmega78))
e283=uOmega77_old+uOmega93_old
bcOmega77.append(DirichletBC(V,e283,dOmega77nOmega93))
bcOmega77.append(BC_Omega77_only)
bcOmega78=bcs.copy()
e284=uOmega78_old+uOmega62_old
bcOmega78.append(DirichletBC(V,e284,dOmega78nOmega62))
e285=uOmega78_old+uOmega77_old
bcOmega78.append(DirichletBC(V,e285,dOmega78nOmega77))
e286=uOmega78_old+uOmega79_old
bcOmega78.append(DirichletBC(V,e286,dOmega78nOmega79))
e287=uOmega78_old+uOmega94_old
bcOmega78.append(DirichletBC(V,e287,dOmega78nOmega94))
bcOmega78.append(BC_Omega78_only)
bcOmega79=bcs.copy()
e288=uOmega79_old+uOmega63_old
bcOmega79.append(DirichletBC(V,e288,dOmega79nOmega63))
e289=uOmega79_old+uOmega78_old
bcOmega79.append(DirichletBC(V,e289,dOmega79nOmega78))
e290=uOmega79_old+uOmega80_old
bcOmega79.append(DirichletBC(V,e290,dOmega79nOmega80))
e291=uOmega79_old+uOmega95_old
bcOmega79.append(DirichletBC(V,e291,dOmega79nOmega95))
bcOmega79.append(BC_Omega79_only)
bcOmega80=bcs.copy()
e292=uOmega80_old+uOmega64_old
bcOmega80.append(DirichletBC(V,e292,dOmega80nOmega64))
e293=uOmega80_old+uOmega79_old
bcOmega80.append(DirichletBC(V,e293,dOmega80nOmega79))
e294=uOmega80_old+uOmega96_old
bcOmega80.append(DirichletBC(V,e294,dOmega80nOmega96))
bcOmega80.append(BC_Omega80_only)
bcOmega81=bcs.copy()
e295=uOmega81_old+uOmega65_old
bcOmega81.append(DirichletBC(V,e295,dOmega81nOmega65))
e296=uOmega81_old+uOmega82_old
bcOmega81.append(DirichletBC(V,e296,dOmega81nOmega82))
e297=uOmega81_old+uOmega97_old
bcOmega81.append(DirichletBC(V,e297,dOmega81nOmega97))
bcOmega81.append(BC_Omega81_only)
bcOmega82=bcs.copy()
e298=uOmega82_old+uOmega66_old
bcOmega82.append(DirichletBC(V,e298,dOmega82nOmega66))
e299=uOmega82_old+uOmega81_old
bcOmega82.append(DirichletBC(V,e299,dOmega82nOmega81))
e300=uOmega82_old+uOmega83_old
bcOmega82.append(DirichletBC(V,e300,dOmega82nOmega83))
e301=uOmega82_old+uOmega98_old
bcOmega82.append(DirichletBC(V,e301,dOmega82nOmega98))
bcOmega82.append(BC_Omega82_only)
bcOmega83=bcs.copy()
e302=uOmega83_old+uOmega67_old
bcOmega83.append(DirichletBC(V,e302,dOmega83nOmega67))
e303=uOmega83_old+uOmega82_old
bcOmega83.append(DirichletBC(V,e303,dOmega83nOmega82))
e304=uOmega83_old+uOmega84_old
bcOmega83.append(DirichletBC(V,e304,dOmega83nOmega84))
e305=uOmega83_old+uOmega99_old
bcOmega83.append(DirichletBC(V,e305,dOmega83nOmega99))
bcOmega83.append(BC_Omega83_only)
bcOmega84=bcs.copy()
e306=uOmega84_old+uOmega68_old
bcOmega84.append(DirichletBC(V,e306,dOmega84nOmega68))
e307=uOmega84_old+uOmega83_old
bcOmega84.append(DirichletBC(V,e307,dOmega84nOmega83))
e308=uOmega84_old+uOmega85_old
bcOmega84.append(DirichletBC(V,e308,dOmega84nOmega85))
e309=uOmega84_old+uOmega100_old
bcOmega84.append(DirichletBC(V,e309,dOmega84nOmega100))
bcOmega84.append(BC_Omega84_only)
bcOmega85=bcs.copy()
e310=uOmega85_old+uOmega69_old
bcOmega85.append(DirichletBC(V,e310,dOmega85nOmega69))
e311=uOmega85_old+uOmega84_old
bcOmega85.append(DirichletBC(V,e311,dOmega85nOmega84))
e312=uOmega85_old+uOmega86_old
bcOmega85.append(DirichletBC(V,e312,dOmega85nOmega86))
e313=uOmega85_old+uOmega101_old
bcOmega85.append(DirichletBC(V,e313,dOmega85nOmega101))
bcOmega85.append(BC_Omega85_only)
bcOmega86=bcs.copy()
e314=uOmega86_old+uOmega70_old
bcOmega86.append(DirichletBC(V,e314,dOmega86nOmega70))
e315=uOmega86_old+uOmega85_old
bcOmega86.append(DirichletBC(V,e315,dOmega86nOmega85))
e316=uOmega86_old+uOmega87_old
bcOmega86.append(DirichletBC(V,e316,dOmega86nOmega87))
e317=uOmega86_old+uOmega102_old
bcOmega86.append(DirichletBC(V,e317,dOmega86nOmega102))
bcOmega86.append(BC_Omega86_only)
bcOmega87=bcs.copy()
e318=uOmega87_old+uOmega71_old
bcOmega87.append(DirichletBC(V,e318,dOmega87nOmega71))
e319=uOmega87_old+uOmega86_old
bcOmega87.append(DirichletBC(V,e319,dOmega87nOmega86))
e320=uOmega87_old+uOmega88_old
bcOmega87.append(DirichletBC(V,e320,dOmega87nOmega88))
e321=uOmega87_old+uOmega103_old
bcOmega87.append(DirichletBC(V,e321,dOmega87nOmega103))
bcOmega87.append(BC_Omega87_only)
bcOmega88=bcs.copy()
e322=uOmega88_old+uOmega72_old
bcOmega88.append(DirichletBC(V,e322,dOmega88nOmega72))
e323=uOmega88_old+uOmega87_old
bcOmega88.append(DirichletBC(V,e323,dOmega88nOmega87))
e324=uOmega88_old+uOmega89_old
bcOmega88.append(DirichletBC(V,e324,dOmega88nOmega89))
e325=uOmega88_old+uOmega104_old
bcOmega88.append(DirichletBC(V,e325,dOmega88nOmega104))
bcOmega88.append(BC_Omega88_only)
bcOmega89=bcs.copy()
e326=uOmega89_old+uOmega73_old
bcOmega89.append(DirichletBC(V,e326,dOmega89nOmega73))
e327=uOmega89_old+uOmega88_old
bcOmega89.append(DirichletBC(V,e327,dOmega89nOmega88))
e328=uOmega89_old+uOmega90_old
bcOmega89.append(DirichletBC(V,e328,dOmega89nOmega90))
e329=uOmega89_old+uOmega105_old
bcOmega89.append(DirichletBC(V,e329,dOmega89nOmega105))
bcOmega89.append(BC_Omega89_only)
bcOmega90=bcs.copy()
e330=uOmega90_old+uOmega74_old
bcOmega90.append(DirichletBC(V,e330,dOmega90nOmega74))
e331=uOmega90_old+uOmega89_old
bcOmega90.append(DirichletBC(V,e331,dOmega90nOmega89))
e332=uOmega90_old+uOmega91_old
bcOmega90.append(DirichletBC(V,e332,dOmega90nOmega91))
e333=uOmega90_old+uOmega106_old
bcOmega90.append(DirichletBC(V,e333,dOmega90nOmega106))
bcOmega90.append(BC_Omega90_only)
bcOmega91=bcs.copy()
e334=uOmega91_old+uOmega75_old
bcOmega91.append(DirichletBC(V,e334,dOmega91nOmega75))
e335=uOmega91_old+uOmega90_old
bcOmega91.append(DirichletBC(V,e335,dOmega91nOmega90))
e336=uOmega91_old+uOmega92_old
bcOmega91.append(DirichletBC(V,e336,dOmega91nOmega92))
e337=uOmega91_old+uOmega107_old
bcOmega91.append(DirichletBC(V,e337,dOmega91nOmega107))
bcOmega91.append(BC_Omega91_only)
bcOmega92=bcs.copy()
e338=uOmega92_old+uOmega76_old
bcOmega92.append(DirichletBC(V,e338,dOmega92nOmega76))
e339=uOmega92_old+uOmega91_old
bcOmega92.append(DirichletBC(V,e339,dOmega92nOmega91))
e340=uOmega92_old+uOmega93_old
bcOmega92.append(DirichletBC(V,e340,dOmega92nOmega93))
e341=uOmega92_old+uOmega108_old
bcOmega92.append(DirichletBC(V,e341,dOmega92nOmega108))
bcOmega92.append(BC_Omega92_only)
bcOmega93=bcs.copy()
e342=uOmega93_old+uOmega77_old
bcOmega93.append(DirichletBC(V,e342,dOmega93nOmega77))
e343=uOmega93_old+uOmega92_old
bcOmega93.append(DirichletBC(V,e343,dOmega93nOmega92))
e344=uOmega93_old+uOmega94_old
bcOmega93.append(DirichletBC(V,e344,dOmega93nOmega94))
e345=uOmega93_old+uOmega109_old
bcOmega93.append(DirichletBC(V,e345,dOmega93nOmega109))
bcOmega93.append(BC_Omega93_only)
bcOmega94=bcs.copy()
e346=uOmega94_old+uOmega78_old
bcOmega94.append(DirichletBC(V,e346,dOmega94nOmega78))
e347=uOmega94_old+uOmega93_old
bcOmega94.append(DirichletBC(V,e347,dOmega94nOmega93))
e348=uOmega94_old+uOmega95_old
bcOmega94.append(DirichletBC(V,e348,dOmega94nOmega95))
e349=uOmega94_old+uOmega110_old
bcOmega94.append(DirichletBC(V,e349,dOmega94nOmega110))
bcOmega94.append(BC_Omega94_only)
bcOmega95=bcs.copy()
e350=uOmega95_old+uOmega79_old
bcOmega95.append(DirichletBC(V,e350,dOmega95nOmega79))
e351=uOmega95_old+uOmega94_old
bcOmega95.append(DirichletBC(V,e351,dOmega95nOmega94))
e352=uOmega95_old+uOmega96_old
bcOmega95.append(DirichletBC(V,e352,dOmega95nOmega96))
e353=uOmega95_old+uOmega111_old
bcOmega95.append(DirichletBC(V,e353,dOmega95nOmega111))
bcOmega95.append(BC_Omega95_only)
bcOmega96=bcs.copy()
e354=uOmega96_old+uOmega80_old
bcOmega96.append(DirichletBC(V,e354,dOmega96nOmega80))
e355=uOmega96_old+uOmega95_old
bcOmega96.append(DirichletBC(V,e355,dOmega96nOmega95))
e356=uOmega96_old+uOmega112_old
bcOmega96.append(DirichletBC(V,e356,dOmega96nOmega112))
bcOmega96.append(BC_Omega96_only)
bcOmega97=bcs.copy()
e357=uOmega97_old+uOmega81_old
bcOmega97.append(DirichletBC(V,e357,dOmega97nOmega81))
e358=uOmega97_old+uOmega98_old
bcOmega97.append(DirichletBC(V,e358,dOmega97nOmega98))
e359=uOmega97_old+uOmega113_old
bcOmega97.append(DirichletBC(V,e359,dOmega97nOmega113))
bcOmega97.append(BC_Omega97_only)
bcOmega98=bcs.copy()
e360=uOmega98_old+uOmega82_old
bcOmega98.append(DirichletBC(V,e360,dOmega98nOmega82))
e361=uOmega98_old+uOmega97_old
bcOmega98.append(DirichletBC(V,e361,dOmega98nOmega97))
e362=uOmega98_old+uOmega99_old
bcOmega98.append(DirichletBC(V,e362,dOmega98nOmega99))
e363=uOmega98_old+uOmega114_old
bcOmega98.append(DirichletBC(V,e363,dOmega98nOmega114))
bcOmega98.append(BC_Omega98_only)
bcOmega99=bcs.copy()
e364=uOmega99_old+uOmega83_old
bcOmega99.append(DirichletBC(V,e364,dOmega99nOmega83))
e365=uOmega99_old+uOmega98_old
bcOmega99.append(DirichletBC(V,e365,dOmega99nOmega98))
e366=uOmega99_old+uOmega100_old
bcOmega99.append(DirichletBC(V,e366,dOmega99nOmega100))
e367=uOmega99_old+uOmega115_old
bcOmega99.append(DirichletBC(V,e367,dOmega99nOmega115))
bcOmega99.append(BC_Omega99_only)
bcOmega100=bcs.copy()
e368=uOmega100_old+uOmega84_old
bcOmega100.append(DirichletBC(V,e368,dOmega100nOmega84))
e369=uOmega100_old+uOmega99_old
bcOmega100.append(DirichletBC(V,e369,dOmega100nOmega99))
e370=uOmega100_old+uOmega101_old
bcOmega100.append(DirichletBC(V,e370,dOmega100nOmega101))
e371=uOmega100_old+uOmega116_old
bcOmega100.append(DirichletBC(V,e371,dOmega100nOmega116))
bcOmega100.append(BC_Omega100_only)
bcOmega101=bcs.copy()
e372=uOmega101_old+uOmega85_old
bcOmega101.append(DirichletBC(V,e372,dOmega101nOmega85))
e373=uOmega101_old+uOmega100_old
bcOmega101.append(DirichletBC(V,e373,dOmega101nOmega100))
e374=uOmega101_old+uOmega102_old
bcOmega101.append(DirichletBC(V,e374,dOmega101nOmega102))
e375=uOmega101_old+uOmega117_old
bcOmega101.append(DirichletBC(V,e375,dOmega101nOmega117))
bcOmega101.append(BC_Omega101_only)
bcOmega102=bcs.copy()
e376=uOmega102_old+uOmega86_old
bcOmega102.append(DirichletBC(V,e376,dOmega102nOmega86))
e377=uOmega102_old+uOmega101_old
bcOmega102.append(DirichletBC(V,e377,dOmega102nOmega101))
e378=uOmega102_old+uOmega103_old
bcOmega102.append(DirichletBC(V,e378,dOmega102nOmega103))
e379=uOmega102_old+uOmega118_old
bcOmega102.append(DirichletBC(V,e379,dOmega102nOmega118))
bcOmega102.append(BC_Omega102_only)
bcOmega103=bcs.copy()
e380=uOmega103_old+uOmega87_old
bcOmega103.append(DirichletBC(V,e380,dOmega103nOmega87))
e381=uOmega103_old+uOmega102_old
bcOmega103.append(DirichletBC(V,e381,dOmega103nOmega102))
e382=uOmega103_old+uOmega104_old
bcOmega103.append(DirichletBC(V,e382,dOmega103nOmega104))
e383=uOmega103_old+uOmega119_old
bcOmega103.append(DirichletBC(V,e383,dOmega103nOmega119))
bcOmega103.append(BC_Omega103_only)
bcOmega104=bcs.copy()
e384=uOmega104_old+uOmega88_old
bcOmega104.append(DirichletBC(V,e384,dOmega104nOmega88))
e385=uOmega104_old+uOmega103_old
bcOmega104.append(DirichletBC(V,e385,dOmega104nOmega103))
e386=uOmega104_old+uOmega105_old
bcOmega104.append(DirichletBC(V,e386,dOmega104nOmega105))
e387=uOmega104_old+uOmega120_old
bcOmega104.append(DirichletBC(V,e387,dOmega104nOmega120))
bcOmega104.append(BC_Omega104_only)
bcOmega105=bcs.copy()
e388=uOmega105_old+uOmega89_old
bcOmega105.append(DirichletBC(V,e388,dOmega105nOmega89))
e389=uOmega105_old+uOmega104_old
bcOmega105.append(DirichletBC(V,e389,dOmega105nOmega104))
e390=uOmega105_old+uOmega106_old
bcOmega105.append(DirichletBC(V,e390,dOmega105nOmega106))
e391=uOmega105_old+uOmega121_old
bcOmega105.append(DirichletBC(V,e391,dOmega105nOmega121))
bcOmega105.append(BC_Omega105_only)
bcOmega106=bcs.copy()
e392=uOmega106_old+uOmega90_old
bcOmega106.append(DirichletBC(V,e392,dOmega106nOmega90))
e393=uOmega106_old+uOmega105_old
bcOmega106.append(DirichletBC(V,e393,dOmega106nOmega105))
e394=uOmega106_old+uOmega107_old
bcOmega106.append(DirichletBC(V,e394,dOmega106nOmega107))
e395=uOmega106_old+uOmega122_old
bcOmega106.append(DirichletBC(V,e395,dOmega106nOmega122))
bcOmega106.append(BC_Omega106_only)
bcOmega107=bcs.copy()
e396=uOmega107_old+uOmega91_old
bcOmega107.append(DirichletBC(V,e396,dOmega107nOmega91))
e397=uOmega107_old+uOmega106_old
bcOmega107.append(DirichletBC(V,e397,dOmega107nOmega106))
e398=uOmega107_old+uOmega108_old
bcOmega107.append(DirichletBC(V,e398,dOmega107nOmega108))
e399=uOmega107_old+uOmega123_old
bcOmega107.append(DirichletBC(V,e399,dOmega107nOmega123))
bcOmega107.append(BC_Omega107_only)
bcOmega108=bcs.copy()
e400=uOmega108_old+uOmega92_old
bcOmega108.append(DirichletBC(V,e400,dOmega108nOmega92))
e401=uOmega108_old+uOmega107_old
bcOmega108.append(DirichletBC(V,e401,dOmega108nOmega107))
e402=uOmega108_old+uOmega109_old
bcOmega108.append(DirichletBC(V,e402,dOmega108nOmega109))
e403=uOmega108_old+uOmega124_old
bcOmega108.append(DirichletBC(V,e403,dOmega108nOmega124))
bcOmega108.append(BC_Omega108_only)
bcOmega109=bcs.copy()
e404=uOmega109_old+uOmega93_old
bcOmega109.append(DirichletBC(V,e404,dOmega109nOmega93))
e405=uOmega109_old+uOmega108_old
bcOmega109.append(DirichletBC(V,e405,dOmega109nOmega108))
e406=uOmega109_old+uOmega110_old
bcOmega109.append(DirichletBC(V,e406,dOmega109nOmega110))
e407=uOmega109_old+uOmega125_old
bcOmega109.append(DirichletBC(V,e407,dOmega109nOmega125))
bcOmega109.append(BC_Omega109_only)
bcOmega110=bcs.copy()
e408=uOmega110_old+uOmega94_old
bcOmega110.append(DirichletBC(V,e408,dOmega110nOmega94))
e409=uOmega110_old+uOmega109_old
bcOmega110.append(DirichletBC(V,e409,dOmega110nOmega109))
e410=uOmega110_old+uOmega111_old
bcOmega110.append(DirichletBC(V,e410,dOmega110nOmega111))
e411=uOmega110_old+uOmega126_old
bcOmega110.append(DirichletBC(V,e411,dOmega110nOmega126))
bcOmega110.append(BC_Omega110_only)
bcOmega111=bcs.copy()
e412=uOmega111_old+uOmega95_old
bcOmega111.append(DirichletBC(V,e412,dOmega111nOmega95))
e413=uOmega111_old+uOmega110_old
bcOmega111.append(DirichletBC(V,e413,dOmega111nOmega110))
e414=uOmega111_old+uOmega112_old
bcOmega111.append(DirichletBC(V,e414,dOmega111nOmega112))
e415=uOmega111_old+uOmega127_old
bcOmega111.append(DirichletBC(V,e415,dOmega111nOmega127))
bcOmega111.append(BC_Omega111_only)
bcOmega112=bcs.copy()
e416=uOmega112_old+uOmega96_old
bcOmega112.append(DirichletBC(V,e416,dOmega112nOmega96))
e417=uOmega112_old+uOmega111_old
bcOmega112.append(DirichletBC(V,e417,dOmega112nOmega111))
e418=uOmega112_old+uOmega128_old
bcOmega112.append(DirichletBC(V,e418,dOmega112nOmega128))
bcOmega112.append(BC_Omega112_only)
bcOmega113=bcs.copy()
e419=uOmega113_old+uOmega97_old
bcOmega113.append(DirichletBC(V,e419,dOmega113nOmega97))
e420=uOmega113_old+uOmega114_old
bcOmega113.append(DirichletBC(V,e420,dOmega113nOmega114))
e421=uOmega113_old+uOmega129_old
bcOmega113.append(DirichletBC(V,e421,dOmega113nOmega129))
bcOmega113.append(BC_Omega113_only)
bcOmega114=bcs.copy()
e422=uOmega114_old+uOmega98_old
bcOmega114.append(DirichletBC(V,e422,dOmega114nOmega98))
e423=uOmega114_old+uOmega113_old
bcOmega114.append(DirichletBC(V,e423,dOmega114nOmega113))
e424=uOmega114_old+uOmega115_old
bcOmega114.append(DirichletBC(V,e424,dOmega114nOmega115))
e425=uOmega114_old+uOmega130_old
bcOmega114.append(DirichletBC(V,e425,dOmega114nOmega130))
bcOmega114.append(BC_Omega114_only)
bcOmega115=bcs.copy()
e426=uOmega115_old+uOmega99_old
bcOmega115.append(DirichletBC(V,e426,dOmega115nOmega99))
e427=uOmega115_old+uOmega114_old
bcOmega115.append(DirichletBC(V,e427,dOmega115nOmega114))
e428=uOmega115_old+uOmega116_old
bcOmega115.append(DirichletBC(V,e428,dOmega115nOmega116))
e429=uOmega115_old+uOmega131_old
bcOmega115.append(DirichletBC(V,e429,dOmega115nOmega131))
bcOmega115.append(BC_Omega115_only)
bcOmega116=bcs.copy()
e430=uOmega116_old+uOmega100_old
bcOmega116.append(DirichletBC(V,e430,dOmega116nOmega100))
e431=uOmega116_old+uOmega115_old
bcOmega116.append(DirichletBC(V,e431,dOmega116nOmega115))
e432=uOmega116_old+uOmega117_old
bcOmega116.append(DirichletBC(V,e432,dOmega116nOmega117))
e433=uOmega116_old+uOmega132_old
bcOmega116.append(DirichletBC(V,e433,dOmega116nOmega132))
bcOmega116.append(BC_Omega116_only)
bcOmega117=bcs.copy()
e434=uOmega117_old+uOmega101_old
bcOmega117.append(DirichletBC(V,e434,dOmega117nOmega101))
e435=uOmega117_old+uOmega116_old
bcOmega117.append(DirichletBC(V,e435,dOmega117nOmega116))
e436=uOmega117_old+uOmega118_old
bcOmega117.append(DirichletBC(V,e436,dOmega117nOmega118))
e437=uOmega117_old+uOmega133_old
bcOmega117.append(DirichletBC(V,e437,dOmega117nOmega133))
bcOmega117.append(BC_Omega117_only)
bcOmega118=bcs.copy()
e438=uOmega118_old+uOmega102_old
bcOmega118.append(DirichletBC(V,e438,dOmega118nOmega102))
e439=uOmega118_old+uOmega117_old
bcOmega118.append(DirichletBC(V,e439,dOmega118nOmega117))
e440=uOmega118_old+uOmega119_old
bcOmega118.append(DirichletBC(V,e440,dOmega118nOmega119))
e441=uOmega118_old+uOmega134_old
bcOmega118.append(DirichletBC(V,e441,dOmega118nOmega134))
bcOmega118.append(BC_Omega118_only)
bcOmega119=bcs.copy()
e442=uOmega119_old+uOmega103_old
bcOmega119.append(DirichletBC(V,e442,dOmega119nOmega103))
e443=uOmega119_old+uOmega118_old
bcOmega119.append(DirichletBC(V,e443,dOmega119nOmega118))
e444=uOmega119_old+uOmega120_old
bcOmega119.append(DirichletBC(V,e444,dOmega119nOmega120))
e445=uOmega119_old+uOmega135_old
bcOmega119.append(DirichletBC(V,e445,dOmega119nOmega135))
bcOmega119.append(BC_Omega119_only)
bcOmega120=bcs.copy()
e446=uOmega120_old+uOmega104_old
bcOmega120.append(DirichletBC(V,e446,dOmega120nOmega104))
e447=uOmega120_old+uOmega119_old
bcOmega120.append(DirichletBC(V,e447,dOmega120nOmega119))
e448=uOmega120_old+uOmega121_old
bcOmega120.append(DirichletBC(V,e448,dOmega120nOmega121))
e449=uOmega120_old+uOmega136_old
bcOmega120.append(DirichletBC(V,e449,dOmega120nOmega136))
bcOmega120.append(BC_Omega120_only)
bcOmega121=bcs.copy()
e450=uOmega121_old+uOmega105_old
bcOmega121.append(DirichletBC(V,e450,dOmega121nOmega105))
e451=uOmega121_old+uOmega120_old
bcOmega121.append(DirichletBC(V,e451,dOmega121nOmega120))
e452=uOmega121_old+uOmega122_old
bcOmega121.append(DirichletBC(V,e452,dOmega121nOmega122))
e453=uOmega121_old+uOmega137_old
bcOmega121.append(DirichletBC(V,e453,dOmega121nOmega137))
bcOmega121.append(BC_Omega121_only)
bcOmega122=bcs.copy()
e454=uOmega122_old+uOmega106_old
bcOmega122.append(DirichletBC(V,e454,dOmega122nOmega106))
e455=uOmega122_old+uOmega121_old
bcOmega122.append(DirichletBC(V,e455,dOmega122nOmega121))
e456=uOmega122_old+uOmega123_old
bcOmega122.append(DirichletBC(V,e456,dOmega122nOmega123))
e457=uOmega122_old+uOmega138_old
bcOmega122.append(DirichletBC(V,e457,dOmega122nOmega138))
bcOmega122.append(BC_Omega122_only)
bcOmega123=bcs.copy()
e458=uOmega123_old+uOmega107_old
bcOmega123.append(DirichletBC(V,e458,dOmega123nOmega107))
e459=uOmega123_old+uOmega122_old
bcOmega123.append(DirichletBC(V,e459,dOmega123nOmega122))
e460=uOmega123_old+uOmega124_old
bcOmega123.append(DirichletBC(V,e460,dOmega123nOmega124))
e461=uOmega123_old+uOmega139_old
bcOmega123.append(DirichletBC(V,e461,dOmega123nOmega139))
bcOmega123.append(BC_Omega123_only)
bcOmega124=bcs.copy()
e462=uOmega124_old+uOmega108_old
bcOmega124.append(DirichletBC(V,e462,dOmega124nOmega108))
e463=uOmega124_old+uOmega123_old
bcOmega124.append(DirichletBC(V,e463,dOmega124nOmega123))
e464=uOmega124_old+uOmega125_old
bcOmega124.append(DirichletBC(V,e464,dOmega124nOmega125))
e465=uOmega124_old+uOmega140_old
bcOmega124.append(DirichletBC(V,e465,dOmega124nOmega140))
bcOmega124.append(BC_Omega124_only)
bcOmega125=bcs.copy()
e466=uOmega125_old+uOmega109_old
bcOmega125.append(DirichletBC(V,e466,dOmega125nOmega109))
e467=uOmega125_old+uOmega124_old
bcOmega125.append(DirichletBC(V,e467,dOmega125nOmega124))
e468=uOmega125_old+uOmega126_old
bcOmega125.append(DirichletBC(V,e468,dOmega125nOmega126))
e469=uOmega125_old+uOmega141_old
bcOmega125.append(DirichletBC(V,e469,dOmega125nOmega141))
bcOmega125.append(BC_Omega125_only)
bcOmega126=bcs.copy()
e470=uOmega126_old+uOmega110_old
bcOmega126.append(DirichletBC(V,e470,dOmega126nOmega110))
e471=uOmega126_old+uOmega125_old
bcOmega126.append(DirichletBC(V,e471,dOmega126nOmega125))
e472=uOmega126_old+uOmega127_old
bcOmega126.append(DirichletBC(V,e472,dOmega126nOmega127))
e473=uOmega126_old+uOmega142_old
bcOmega126.append(DirichletBC(V,e473,dOmega126nOmega142))
bcOmega126.append(BC_Omega126_only)
bcOmega127=bcs.copy()
e474=uOmega127_old+uOmega111_old
bcOmega127.append(DirichletBC(V,e474,dOmega127nOmega111))
e475=uOmega127_old+uOmega126_old
bcOmega127.append(DirichletBC(V,e475,dOmega127nOmega126))
e476=uOmega127_old+uOmega128_old
bcOmega127.append(DirichletBC(V,e476,dOmega127nOmega128))
e477=uOmega127_old+uOmega143_old
bcOmega127.append(DirichletBC(V,e477,dOmega127nOmega143))
bcOmega127.append(BC_Omega127_only)
bcOmega128=bcs.copy()
e478=uOmega128_old+uOmega112_old
bcOmega128.append(DirichletBC(V,e478,dOmega128nOmega112))
e479=uOmega128_old+uOmega127_old
bcOmega128.append(DirichletBC(V,e479,dOmega128nOmega127))
e480=uOmega128_old+uOmega144_old
bcOmega128.append(DirichletBC(V,e480,dOmega128nOmega144))
bcOmega128.append(BC_Omega128_only)
bcOmega129=bcs.copy()
e481=uOmega129_old+uOmega113_old
bcOmega129.append(DirichletBC(V,e481,dOmega129nOmega113))
e482=uOmega129_old+uOmega130_old
bcOmega129.append(DirichletBC(V,e482,dOmega129nOmega130))
e483=uOmega129_old+uOmega145_old
bcOmega129.append(DirichletBC(V,e483,dOmega129nOmega145))
bcOmega129.append(BC_Omega129_only)
bcOmega130=bcs.copy()
e484=uOmega130_old+uOmega114_old
bcOmega130.append(DirichletBC(V,e484,dOmega130nOmega114))
e485=uOmega130_old+uOmega129_old
bcOmega130.append(DirichletBC(V,e485,dOmega130nOmega129))
e486=uOmega130_old+uOmega131_old
bcOmega130.append(DirichletBC(V,e486,dOmega130nOmega131))
e487=uOmega130_old+uOmega146_old
bcOmega130.append(DirichletBC(V,e487,dOmega130nOmega146))
bcOmega130.append(BC_Omega130_only)
bcOmega131=bcs.copy()
e488=uOmega131_old+uOmega115_old
bcOmega131.append(DirichletBC(V,e488,dOmega131nOmega115))
e489=uOmega131_old+uOmega130_old
bcOmega131.append(DirichletBC(V,e489,dOmega131nOmega130))
e490=uOmega131_old+uOmega132_old
bcOmega131.append(DirichletBC(V,e490,dOmega131nOmega132))
e491=uOmega131_old+uOmega147_old
bcOmega131.append(DirichletBC(V,e491,dOmega131nOmega147))
bcOmega131.append(BC_Omega131_only)
bcOmega132=bcs.copy()
e492=uOmega132_old+uOmega116_old
bcOmega132.append(DirichletBC(V,e492,dOmega132nOmega116))
e493=uOmega132_old+uOmega131_old
bcOmega132.append(DirichletBC(V,e493,dOmega132nOmega131))
e494=uOmega132_old+uOmega133_old
bcOmega132.append(DirichletBC(V,e494,dOmega132nOmega133))
e495=uOmega132_old+uOmega148_old
bcOmega132.append(DirichletBC(V,e495,dOmega132nOmega148))
bcOmega132.append(BC_Omega132_only)
bcOmega133=bcs.copy()
e496=uOmega133_old+uOmega117_old
bcOmega133.append(DirichletBC(V,e496,dOmega133nOmega117))
e497=uOmega133_old+uOmega132_old
bcOmega133.append(DirichletBC(V,e497,dOmega133nOmega132))
e498=uOmega133_old+uOmega134_old
bcOmega133.append(DirichletBC(V,e498,dOmega133nOmega134))
e499=uOmega133_old+uOmega149_old
bcOmega133.append(DirichletBC(V,e499,dOmega133nOmega149))
bcOmega133.append(BC_Omega133_only)
bcOmega134=bcs.copy()
e500=uOmega134_old+uOmega118_old
bcOmega134.append(DirichletBC(V,e500,dOmega134nOmega118))
e501=uOmega134_old+uOmega133_old
bcOmega134.append(DirichletBC(V,e501,dOmega134nOmega133))
e502=uOmega134_old+uOmega135_old
bcOmega134.append(DirichletBC(V,e502,dOmega134nOmega135))
e503=uOmega134_old+uOmega150_old
bcOmega134.append(DirichletBC(V,e503,dOmega134nOmega150))
bcOmega134.append(BC_Omega134_only)
bcOmega135=bcs.copy()
e504=uOmega135_old+uOmega119_old
bcOmega135.append(DirichletBC(V,e504,dOmega135nOmega119))
e505=uOmega135_old+uOmega134_old
bcOmega135.append(DirichletBC(V,e505,dOmega135nOmega134))
e506=uOmega135_old+uOmega136_old
bcOmega135.append(DirichletBC(V,e506,dOmega135nOmega136))
e507=uOmega135_old+uOmega151_old
bcOmega135.append(DirichletBC(V,e507,dOmega135nOmega151))
bcOmega135.append(BC_Omega135_only)
bcOmega136=bcs.copy()
e508=uOmega136_old+uOmega120_old
bcOmega136.append(DirichletBC(V,e508,dOmega136nOmega120))
e509=uOmega136_old+uOmega135_old
bcOmega136.append(DirichletBC(V,e509,dOmega136nOmega135))
e510=uOmega136_old+uOmega137_old
bcOmega136.append(DirichletBC(V,e510,dOmega136nOmega137))
e511=uOmega136_old+uOmega152_old
bcOmega136.append(DirichletBC(V,e511,dOmega136nOmega152))
bcOmega136.append(BC_Omega136_only)
bcOmega137=bcs.copy()
e512=uOmega137_old+uOmega121_old
bcOmega137.append(DirichletBC(V,e512,dOmega137nOmega121))
e513=uOmega137_old+uOmega136_old
bcOmega137.append(DirichletBC(V,e513,dOmega137nOmega136))
e514=uOmega137_old+uOmega138_old
bcOmega137.append(DirichletBC(V,e514,dOmega137nOmega138))
e515=uOmega137_old+uOmega153_old
bcOmega137.append(DirichletBC(V,e515,dOmega137nOmega153))
bcOmega137.append(BC_Omega137_only)
bcOmega138=bcs.copy()
e516=uOmega138_old+uOmega122_old
bcOmega138.append(DirichletBC(V,e516,dOmega138nOmega122))
e517=uOmega138_old+uOmega137_old
bcOmega138.append(DirichletBC(V,e517,dOmega138nOmega137))
e518=uOmega138_old+uOmega139_old
bcOmega138.append(DirichletBC(V,e518,dOmega138nOmega139))
e519=uOmega138_old+uOmega154_old
bcOmega138.append(DirichletBC(V,e519,dOmega138nOmega154))
bcOmega138.append(BC_Omega138_only)
bcOmega139=bcs.copy()
e520=uOmega139_old+uOmega123_old
bcOmega139.append(DirichletBC(V,e520,dOmega139nOmega123))
e521=uOmega139_old+uOmega138_old
bcOmega139.append(DirichletBC(V,e521,dOmega139nOmega138))
e522=uOmega139_old+uOmega140_old
bcOmega139.append(DirichletBC(V,e522,dOmega139nOmega140))
e523=uOmega139_old+uOmega155_old
bcOmega139.append(DirichletBC(V,e523,dOmega139nOmega155))
bcOmega139.append(BC_Omega139_only)
bcOmega140=bcs.copy()
e524=uOmega140_old+uOmega124_old
bcOmega140.append(DirichletBC(V,e524,dOmega140nOmega124))
e525=uOmega140_old+uOmega139_old
bcOmega140.append(DirichletBC(V,e525,dOmega140nOmega139))
e526=uOmega140_old+uOmega141_old
bcOmega140.append(DirichletBC(V,e526,dOmega140nOmega141))
e527=uOmega140_old+uOmega156_old
bcOmega140.append(DirichletBC(V,e527,dOmega140nOmega156))
bcOmega140.append(BC_Omega140_only)
bcOmega141=bcs.copy()
e528=uOmega141_old+uOmega125_old
bcOmega141.append(DirichletBC(V,e528,dOmega141nOmega125))
e529=uOmega141_old+uOmega140_old
bcOmega141.append(DirichletBC(V,e529,dOmega141nOmega140))
e530=uOmega141_old+uOmega142_old
bcOmega141.append(DirichletBC(V,e530,dOmega141nOmega142))
e531=uOmega141_old+uOmega157_old
bcOmega141.append(DirichletBC(V,e531,dOmega141nOmega157))
bcOmega141.append(BC_Omega141_only)
bcOmega142=bcs.copy()
e532=uOmega142_old+uOmega126_old
bcOmega142.append(DirichletBC(V,e532,dOmega142nOmega126))
e533=uOmega142_old+uOmega141_old
bcOmega142.append(DirichletBC(V,e533,dOmega142nOmega141))
e534=uOmega142_old+uOmega143_old
bcOmega142.append(DirichletBC(V,e534,dOmega142nOmega143))
e535=uOmega142_old+uOmega158_old
bcOmega142.append(DirichletBC(V,e535,dOmega142nOmega158))
bcOmega142.append(BC_Omega142_only)
bcOmega143=bcs.copy()
e536=uOmega143_old+uOmega127_old
bcOmega143.append(DirichletBC(V,e536,dOmega143nOmega127))
e537=uOmega143_old+uOmega142_old
bcOmega143.append(DirichletBC(V,e537,dOmega143nOmega142))
e538=uOmega143_old+uOmega144_old
bcOmega143.append(DirichletBC(V,e538,dOmega143nOmega144))
e539=uOmega143_old+uOmega159_old
bcOmega143.append(DirichletBC(V,e539,dOmega143nOmega159))
bcOmega143.append(BC_Omega143_only)
bcOmega144=bcs.copy()
e540=uOmega144_old+uOmega128_old
bcOmega144.append(DirichletBC(V,e540,dOmega144nOmega128))
e541=uOmega144_old+uOmega143_old
bcOmega144.append(DirichletBC(V,e541,dOmega144nOmega143))
e542=uOmega144_old+uOmega160_old
bcOmega144.append(DirichletBC(V,e542,dOmega144nOmega160))
bcOmega144.append(BC_Omega144_only)
bcOmega145=bcs.copy()
e543=uOmega145_old+uOmega129_old
bcOmega145.append(DirichletBC(V,e543,dOmega145nOmega129))
e544=uOmega145_old+uOmega146_old
bcOmega145.append(DirichletBC(V,e544,dOmega145nOmega146))
e545=uOmega145_old+uOmega161_old
bcOmega145.append(DirichletBC(V,e545,dOmega145nOmega161))
bcOmega145.append(BC_Omega145_only)
bcOmega146=bcs.copy()
e546=uOmega146_old+uOmega130_old
bcOmega146.append(DirichletBC(V,e546,dOmega146nOmega130))
e547=uOmega146_old+uOmega145_old
bcOmega146.append(DirichletBC(V,e547,dOmega146nOmega145))
e548=uOmega146_old+uOmega147_old
bcOmega146.append(DirichletBC(V,e548,dOmega146nOmega147))
e549=uOmega146_old+uOmega162_old
bcOmega146.append(DirichletBC(V,e549,dOmega146nOmega162))
bcOmega146.append(BC_Omega146_only)
bcOmega147=bcs.copy()
e550=uOmega147_old+uOmega131_old
bcOmega147.append(DirichletBC(V,e550,dOmega147nOmega131))
e551=uOmega147_old+uOmega146_old
bcOmega147.append(DirichletBC(V,e551,dOmega147nOmega146))
e552=uOmega147_old+uOmega148_old
bcOmega147.append(DirichletBC(V,e552,dOmega147nOmega148))
e553=uOmega147_old+uOmega163_old
bcOmega147.append(DirichletBC(V,e553,dOmega147nOmega163))
bcOmega147.append(BC_Omega147_only)
bcOmega148=bcs.copy()
e554=uOmega148_old+uOmega132_old
bcOmega148.append(DirichletBC(V,e554,dOmega148nOmega132))
e555=uOmega148_old+uOmega147_old
bcOmega148.append(DirichletBC(V,e555,dOmega148nOmega147))
e556=uOmega148_old+uOmega149_old
bcOmega148.append(DirichletBC(V,e556,dOmega148nOmega149))
e557=uOmega148_old+uOmega164_old
bcOmega148.append(DirichletBC(V,e557,dOmega148nOmega164))
bcOmega148.append(BC_Omega148_only)
bcOmega149=bcs.copy()
e558=uOmega149_old+uOmega133_old
bcOmega149.append(DirichletBC(V,e558,dOmega149nOmega133))
e559=uOmega149_old+uOmega148_old
bcOmega149.append(DirichletBC(V,e559,dOmega149nOmega148))
e560=uOmega149_old+uOmega150_old
bcOmega149.append(DirichletBC(V,e560,dOmega149nOmega150))
e561=uOmega149_old+uOmega165_old
bcOmega149.append(DirichletBC(V,e561,dOmega149nOmega165))
bcOmega149.append(BC_Omega149_only)
bcOmega150=bcs.copy()
e562=uOmega150_old+uOmega134_old
bcOmega150.append(DirichletBC(V,e562,dOmega150nOmega134))
e563=uOmega150_old+uOmega149_old
bcOmega150.append(DirichletBC(V,e563,dOmega150nOmega149))
e564=uOmega150_old+uOmega151_old
bcOmega150.append(DirichletBC(V,e564,dOmega150nOmega151))
e565=uOmega150_old+uOmega166_old
bcOmega150.append(DirichletBC(V,e565,dOmega150nOmega166))
bcOmega150.append(BC_Omega150_only)
bcOmega151=bcs.copy()
e566=uOmega151_old+uOmega135_old
bcOmega151.append(DirichletBC(V,e566,dOmega151nOmega135))
e567=uOmega151_old+uOmega150_old
bcOmega151.append(DirichletBC(V,e567,dOmega151nOmega150))
e568=uOmega151_old+uOmega152_old
bcOmega151.append(DirichletBC(V,e568,dOmega151nOmega152))
e569=uOmega151_old+uOmega167_old
bcOmega151.append(DirichletBC(V,e569,dOmega151nOmega167))
bcOmega151.append(BC_Omega151_only)
bcOmega152=bcs.copy()
e570=uOmega152_old+uOmega136_old
bcOmega152.append(DirichletBC(V,e570,dOmega152nOmega136))
e571=uOmega152_old+uOmega151_old
bcOmega152.append(DirichletBC(V,e571,dOmega152nOmega151))
e572=uOmega152_old+uOmega153_old
bcOmega152.append(DirichletBC(V,e572,dOmega152nOmega153))
e573=uOmega152_old+uOmega168_old
bcOmega152.append(DirichletBC(V,e573,dOmega152nOmega168))
bcOmega152.append(BC_Omega152_only)
bcOmega153=bcs.copy()
e574=uOmega153_old+uOmega137_old
bcOmega153.append(DirichletBC(V,e574,dOmega153nOmega137))
e575=uOmega153_old+uOmega152_old
bcOmega153.append(DirichletBC(V,e575,dOmega153nOmega152))
e576=uOmega153_old+uOmega154_old
bcOmega153.append(DirichletBC(V,e576,dOmega153nOmega154))
e577=uOmega153_old+uOmega169_old
bcOmega153.append(DirichletBC(V,e577,dOmega153nOmega169))
bcOmega153.append(BC_Omega153_only)
bcOmega154=bcs.copy()
e578=uOmega154_old+uOmega138_old
bcOmega154.append(DirichletBC(V,e578,dOmega154nOmega138))
e579=uOmega154_old+uOmega153_old
bcOmega154.append(DirichletBC(V,e579,dOmega154nOmega153))
e580=uOmega154_old+uOmega155_old
bcOmega154.append(DirichletBC(V,e580,dOmega154nOmega155))
e581=uOmega154_old+uOmega170_old
bcOmega154.append(DirichletBC(V,e581,dOmega154nOmega170))
bcOmega154.append(BC_Omega154_only)
bcOmega155=bcs.copy()
e582=uOmega155_old+uOmega139_old
bcOmega155.append(DirichletBC(V,e582,dOmega155nOmega139))
e583=uOmega155_old+uOmega154_old
bcOmega155.append(DirichletBC(V,e583,dOmega155nOmega154))
e584=uOmega155_old+uOmega156_old
bcOmega155.append(DirichletBC(V,e584,dOmega155nOmega156))
e585=uOmega155_old+uOmega171_old
bcOmega155.append(DirichletBC(V,e585,dOmega155nOmega171))
bcOmega155.append(BC_Omega155_only)
bcOmega156=bcs.copy()
e586=uOmega156_old+uOmega140_old
bcOmega156.append(DirichletBC(V,e586,dOmega156nOmega140))
e587=uOmega156_old+uOmega155_old
bcOmega156.append(DirichletBC(V,e587,dOmega156nOmega155))
e588=uOmega156_old+uOmega157_old
bcOmega156.append(DirichletBC(V,e588,dOmega156nOmega157))
e589=uOmega156_old+uOmega172_old
bcOmega156.append(DirichletBC(V,e589,dOmega156nOmega172))
bcOmega156.append(BC_Omega156_only)
bcOmega157=bcs.copy()
e590=uOmega157_old+uOmega141_old
bcOmega157.append(DirichletBC(V,e590,dOmega157nOmega141))
e591=uOmega157_old+uOmega156_old
bcOmega157.append(DirichletBC(V,e591,dOmega157nOmega156))
e592=uOmega157_old+uOmega158_old
bcOmega157.append(DirichletBC(V,e592,dOmega157nOmega158))
e593=uOmega157_old+uOmega173_old
bcOmega157.append(DirichletBC(V,e593,dOmega157nOmega173))
bcOmega157.append(BC_Omega157_only)
bcOmega158=bcs.copy()
e594=uOmega158_old+uOmega142_old
bcOmega158.append(DirichletBC(V,e594,dOmega158nOmega142))
e595=uOmega158_old+uOmega157_old
bcOmega158.append(DirichletBC(V,e595,dOmega158nOmega157))
e596=uOmega158_old+uOmega159_old
bcOmega158.append(DirichletBC(V,e596,dOmega158nOmega159))
e597=uOmega158_old+uOmega174_old
bcOmega158.append(DirichletBC(V,e597,dOmega158nOmega174))
bcOmega158.append(BC_Omega158_only)
bcOmega159=bcs.copy()
e598=uOmega159_old+uOmega143_old
bcOmega159.append(DirichletBC(V,e598,dOmega159nOmega143))
e599=uOmega159_old+uOmega158_old
bcOmega159.append(DirichletBC(V,e599,dOmega159nOmega158))
e600=uOmega159_old+uOmega160_old
bcOmega159.append(DirichletBC(V,e600,dOmega159nOmega160))
e601=uOmega159_old+uOmega175_old
bcOmega159.append(DirichletBC(V,e601,dOmega159nOmega175))
bcOmega159.append(BC_Omega159_only)
bcOmega160=bcs.copy()
e602=uOmega160_old+uOmega144_old
bcOmega160.append(DirichletBC(V,e602,dOmega160nOmega144))
e603=uOmega160_old+uOmega159_old
bcOmega160.append(DirichletBC(V,e603,dOmega160nOmega159))
e604=uOmega160_old+uOmega176_old
bcOmega160.append(DirichletBC(V,e604,dOmega160nOmega176))
bcOmega160.append(BC_Omega160_only)
bcOmega161=bcs.copy()
e605=uOmega161_old+uOmega145_old
bcOmega161.append(DirichletBC(V,e605,dOmega161nOmega145))
e606=uOmega161_old+uOmega162_old
bcOmega161.append(DirichletBC(V,e606,dOmega161nOmega162))
e607=uOmega161_old+uOmega177_old
bcOmega161.append(DirichletBC(V,e607,dOmega161nOmega177))
bcOmega161.append(BC_Omega161_only)
bcOmega162=bcs.copy()
e608=uOmega162_old+uOmega146_old
bcOmega162.append(DirichletBC(V,e608,dOmega162nOmega146))
e609=uOmega162_old+uOmega161_old
bcOmega162.append(DirichletBC(V,e609,dOmega162nOmega161))
e610=uOmega162_old+uOmega163_old
bcOmega162.append(DirichletBC(V,e610,dOmega162nOmega163))
e611=uOmega162_old+uOmega178_old
bcOmega162.append(DirichletBC(V,e611,dOmega162nOmega178))
bcOmega162.append(BC_Omega162_only)
bcOmega163=bcs.copy()
e612=uOmega163_old+uOmega147_old
bcOmega163.append(DirichletBC(V,e612,dOmega163nOmega147))
e613=uOmega163_old+uOmega162_old
bcOmega163.append(DirichletBC(V,e613,dOmega163nOmega162))
e614=uOmega163_old+uOmega164_old
bcOmega163.append(DirichletBC(V,e614,dOmega163nOmega164))
e615=uOmega163_old+uOmega179_old
bcOmega163.append(DirichletBC(V,e615,dOmega163nOmega179))
bcOmega163.append(BC_Omega163_only)
bcOmega164=bcs.copy()
e616=uOmega164_old+uOmega148_old
bcOmega164.append(DirichletBC(V,e616,dOmega164nOmega148))
e617=uOmega164_old+uOmega163_old
bcOmega164.append(DirichletBC(V,e617,dOmega164nOmega163))
e618=uOmega164_old+uOmega165_old
bcOmega164.append(DirichletBC(V,e618,dOmega164nOmega165))
e619=uOmega164_old+uOmega180_old
bcOmega164.append(DirichletBC(V,e619,dOmega164nOmega180))
bcOmega164.append(BC_Omega164_only)
bcOmega165=bcs.copy()
e620=uOmega165_old+uOmega149_old
bcOmega165.append(DirichletBC(V,e620,dOmega165nOmega149))
e621=uOmega165_old+uOmega164_old
bcOmega165.append(DirichletBC(V,e621,dOmega165nOmega164))
e622=uOmega165_old+uOmega166_old
bcOmega165.append(DirichletBC(V,e622,dOmega165nOmega166))
e623=uOmega165_old+uOmega181_old
bcOmega165.append(DirichletBC(V,e623,dOmega165nOmega181))
bcOmega165.append(BC_Omega165_only)
bcOmega166=bcs.copy()
e624=uOmega166_old+uOmega150_old
bcOmega166.append(DirichletBC(V,e624,dOmega166nOmega150))
e625=uOmega166_old+uOmega165_old
bcOmega166.append(DirichletBC(V,e625,dOmega166nOmega165))
e626=uOmega166_old+uOmega167_old
bcOmega166.append(DirichletBC(V,e626,dOmega166nOmega167))
e627=uOmega166_old+uOmega182_old
bcOmega166.append(DirichletBC(V,e627,dOmega166nOmega182))
bcOmega166.append(BC_Omega166_only)
bcOmega167=bcs.copy()
e628=uOmega167_old+uOmega151_old
bcOmega167.append(DirichletBC(V,e628,dOmega167nOmega151))
e629=uOmega167_old+uOmega166_old
bcOmega167.append(DirichletBC(V,e629,dOmega167nOmega166))
e630=uOmega167_old+uOmega168_old
bcOmega167.append(DirichletBC(V,e630,dOmega167nOmega168))
e631=uOmega167_old+uOmega183_old
bcOmega167.append(DirichletBC(V,e631,dOmega167nOmega183))
bcOmega167.append(BC_Omega167_only)
bcOmega168=bcs.copy()
e632=uOmega168_old+uOmega152_old
bcOmega168.append(DirichletBC(V,e632,dOmega168nOmega152))
e633=uOmega168_old+uOmega167_old
bcOmega168.append(DirichletBC(V,e633,dOmega168nOmega167))
e634=uOmega168_old+uOmega169_old
bcOmega168.append(DirichletBC(V,e634,dOmega168nOmega169))
e635=uOmega168_old+uOmega184_old
bcOmega168.append(DirichletBC(V,e635,dOmega168nOmega184))
bcOmega168.append(BC_Omega168_only)
bcOmega169=bcs.copy()
e636=uOmega169_old+uOmega153_old
bcOmega169.append(DirichletBC(V,e636,dOmega169nOmega153))
e637=uOmega169_old+uOmega168_old
bcOmega169.append(DirichletBC(V,e637,dOmega169nOmega168))
e638=uOmega169_old+uOmega170_old
bcOmega169.append(DirichletBC(V,e638,dOmega169nOmega170))
e639=uOmega169_old+uOmega185_old
bcOmega169.append(DirichletBC(V,e639,dOmega169nOmega185))
bcOmega169.append(BC_Omega169_only)
bcOmega170=bcs.copy()
e640=uOmega170_old+uOmega154_old
bcOmega170.append(DirichletBC(V,e640,dOmega170nOmega154))
e641=uOmega170_old+uOmega169_old
bcOmega170.append(DirichletBC(V,e641,dOmega170nOmega169))
e642=uOmega170_old+uOmega171_old
bcOmega170.append(DirichletBC(V,e642,dOmega170nOmega171))
e643=uOmega170_old+uOmega186_old
bcOmega170.append(DirichletBC(V,e643,dOmega170nOmega186))
bcOmega170.append(BC_Omega170_only)
bcOmega171=bcs.copy()
e644=uOmega171_old+uOmega155_old
bcOmega171.append(DirichletBC(V,e644,dOmega171nOmega155))
e645=uOmega171_old+uOmega170_old
bcOmega171.append(DirichletBC(V,e645,dOmega171nOmega170))
e646=uOmega171_old+uOmega172_old
bcOmega171.append(DirichletBC(V,e646,dOmega171nOmega172))
e647=uOmega171_old+uOmega187_old
bcOmega171.append(DirichletBC(V,e647,dOmega171nOmega187))
bcOmega171.append(BC_Omega171_only)
bcOmega172=bcs.copy()
e648=uOmega172_old+uOmega156_old
bcOmega172.append(DirichletBC(V,e648,dOmega172nOmega156))
e649=uOmega172_old+uOmega171_old
bcOmega172.append(DirichletBC(V,e649,dOmega172nOmega171))
e650=uOmega172_old+uOmega173_old
bcOmega172.append(DirichletBC(V,e650,dOmega172nOmega173))
e651=uOmega172_old+uOmega188_old
bcOmega172.append(DirichletBC(V,e651,dOmega172nOmega188))
bcOmega172.append(BC_Omega172_only)
bcOmega173=bcs.copy()
e652=uOmega173_old+uOmega157_old
bcOmega173.append(DirichletBC(V,e652,dOmega173nOmega157))
e653=uOmega173_old+uOmega172_old
bcOmega173.append(DirichletBC(V,e653,dOmega173nOmega172))
e654=uOmega173_old+uOmega174_old
bcOmega173.append(DirichletBC(V,e654,dOmega173nOmega174))
e655=uOmega173_old+uOmega189_old
bcOmega173.append(DirichletBC(V,e655,dOmega173nOmega189))
bcOmega173.append(BC_Omega173_only)
bcOmega174=bcs.copy()
e656=uOmega174_old+uOmega158_old
bcOmega174.append(DirichletBC(V,e656,dOmega174nOmega158))
e657=uOmega174_old+uOmega173_old
bcOmega174.append(DirichletBC(V,e657,dOmega174nOmega173))
e658=uOmega174_old+uOmega175_old
bcOmega174.append(DirichletBC(V,e658,dOmega174nOmega175))
e659=uOmega174_old+uOmega190_old
bcOmega174.append(DirichletBC(V,e659,dOmega174nOmega190))
bcOmega174.append(BC_Omega174_only)
bcOmega175=bcs.copy()
e660=uOmega175_old+uOmega159_old
bcOmega175.append(DirichletBC(V,e660,dOmega175nOmega159))
e661=uOmega175_old+uOmega174_old
bcOmega175.append(DirichletBC(V,e661,dOmega175nOmega174))
e662=uOmega175_old+uOmega176_old
bcOmega175.append(DirichletBC(V,e662,dOmega175nOmega176))
e663=uOmega175_old+uOmega191_old
bcOmega175.append(DirichletBC(V,e663,dOmega175nOmega191))
bcOmega175.append(BC_Omega175_only)
bcOmega176=bcs.copy()
e664=uOmega176_old+uOmega160_old
bcOmega176.append(DirichletBC(V,e664,dOmega176nOmega160))
e665=uOmega176_old+uOmega175_old
bcOmega176.append(DirichletBC(V,e665,dOmega176nOmega175))
e666=uOmega176_old+uOmega192_old
bcOmega176.append(DirichletBC(V,e666,dOmega176nOmega192))
bcOmega176.append(BC_Omega176_only)
bcOmega177=bcs.copy()
e667=uOmega177_old+uOmega161_old
bcOmega177.append(DirichletBC(V,e667,dOmega177nOmega161))
e668=uOmega177_old+uOmega178_old
bcOmega177.append(DirichletBC(V,e668,dOmega177nOmega178))
e669=uOmega177_old+uOmega193_old
bcOmega177.append(DirichletBC(V,e669,dOmega177nOmega193))
bcOmega177.append(BC_Omega177_only)
bcOmega178=bcs.copy()
e670=uOmega178_old+uOmega162_old
bcOmega178.append(DirichletBC(V,e670,dOmega178nOmega162))
e671=uOmega178_old+uOmega177_old
bcOmega178.append(DirichletBC(V,e671,dOmega178nOmega177))
e672=uOmega178_old+uOmega179_old
bcOmega178.append(DirichletBC(V,e672,dOmega178nOmega179))
e673=uOmega178_old+uOmega194_old
bcOmega178.append(DirichletBC(V,e673,dOmega178nOmega194))
bcOmega178.append(BC_Omega178_only)
bcOmega179=bcs.copy()
e674=uOmega179_old+uOmega163_old
bcOmega179.append(DirichletBC(V,e674,dOmega179nOmega163))
e675=uOmega179_old+uOmega178_old
bcOmega179.append(DirichletBC(V,e675,dOmega179nOmega178))
e676=uOmega179_old+uOmega180_old
bcOmega179.append(DirichletBC(V,e676,dOmega179nOmega180))
e677=uOmega179_old+uOmega195_old
bcOmega179.append(DirichletBC(V,e677,dOmega179nOmega195))
bcOmega179.append(BC_Omega179_only)
bcOmega180=bcs.copy()
e678=uOmega180_old+uOmega164_old
bcOmega180.append(DirichletBC(V,e678,dOmega180nOmega164))
e679=uOmega180_old+uOmega179_old
bcOmega180.append(DirichletBC(V,e679,dOmega180nOmega179))
e680=uOmega180_old+uOmega181_old
bcOmega180.append(DirichletBC(V,e680,dOmega180nOmega181))
e681=uOmega180_old+uOmega196_old
bcOmega180.append(DirichletBC(V,e681,dOmega180nOmega196))
bcOmega180.append(BC_Omega180_only)
bcOmega181=bcs.copy()
e682=uOmega181_old+uOmega165_old
bcOmega181.append(DirichletBC(V,e682,dOmega181nOmega165))
e683=uOmega181_old+uOmega180_old
bcOmega181.append(DirichletBC(V,e683,dOmega181nOmega180))
e684=uOmega181_old+uOmega182_old
bcOmega181.append(DirichletBC(V,e684,dOmega181nOmega182))
e685=uOmega181_old+uOmega197_old
bcOmega181.append(DirichletBC(V,e685,dOmega181nOmega197))
bcOmega181.append(BC_Omega181_only)
bcOmega182=bcs.copy()
e686=uOmega182_old+uOmega166_old
bcOmega182.append(DirichletBC(V,e686,dOmega182nOmega166))
e687=uOmega182_old+uOmega181_old
bcOmega182.append(DirichletBC(V,e687,dOmega182nOmega181))
e688=uOmega182_old+uOmega183_old
bcOmega182.append(DirichletBC(V,e688,dOmega182nOmega183))
e689=uOmega182_old+uOmega198_old
bcOmega182.append(DirichletBC(V,e689,dOmega182nOmega198))
bcOmega182.append(BC_Omega182_only)
bcOmega183=bcs.copy()
e690=uOmega183_old+uOmega167_old
bcOmega183.append(DirichletBC(V,e690,dOmega183nOmega167))
e691=uOmega183_old+uOmega182_old
bcOmega183.append(DirichletBC(V,e691,dOmega183nOmega182))
e692=uOmega183_old+uOmega184_old
bcOmega183.append(DirichletBC(V,e692,dOmega183nOmega184))
e693=uOmega183_old+uOmega199_old
bcOmega183.append(DirichletBC(V,e693,dOmega183nOmega199))
bcOmega183.append(BC_Omega183_only)
bcOmega184=bcs.copy()
e694=uOmega184_old+uOmega168_old
bcOmega184.append(DirichletBC(V,e694,dOmega184nOmega168))
e695=uOmega184_old+uOmega183_old
bcOmega184.append(DirichletBC(V,e695,dOmega184nOmega183))
e696=uOmega184_old+uOmega185_old
bcOmega184.append(DirichletBC(V,e696,dOmega184nOmega185))
e697=uOmega184_old+uOmega200_old
bcOmega184.append(DirichletBC(V,e697,dOmega184nOmega200))
bcOmega184.append(BC_Omega184_only)
bcOmega185=bcs.copy()
e698=uOmega185_old+uOmega169_old
bcOmega185.append(DirichletBC(V,e698,dOmega185nOmega169))
e699=uOmega185_old+uOmega184_old
bcOmega185.append(DirichletBC(V,e699,dOmega185nOmega184))
e700=uOmega185_old+uOmega186_old
bcOmega185.append(DirichletBC(V,e700,dOmega185nOmega186))
e701=uOmega185_old+uOmega201_old
bcOmega185.append(DirichletBC(V,e701,dOmega185nOmega201))
bcOmega185.append(BC_Omega185_only)
bcOmega186=bcs.copy()
e702=uOmega186_old+uOmega170_old
bcOmega186.append(DirichletBC(V,e702,dOmega186nOmega170))
e703=uOmega186_old+uOmega185_old
bcOmega186.append(DirichletBC(V,e703,dOmega186nOmega185))
e704=uOmega186_old+uOmega187_old
bcOmega186.append(DirichletBC(V,e704,dOmega186nOmega187))
e705=uOmega186_old+uOmega202_old
bcOmega186.append(DirichletBC(V,e705,dOmega186nOmega202))
bcOmega186.append(BC_Omega186_only)
bcOmega187=bcs.copy()
e706=uOmega187_old+uOmega171_old
bcOmega187.append(DirichletBC(V,e706,dOmega187nOmega171))
e707=uOmega187_old+uOmega186_old
bcOmega187.append(DirichletBC(V,e707,dOmega187nOmega186))
e708=uOmega187_old+uOmega188_old
bcOmega187.append(DirichletBC(V,e708,dOmega187nOmega188))
e709=uOmega187_old+uOmega203_old
bcOmega187.append(DirichletBC(V,e709,dOmega187nOmega203))
bcOmega187.append(BC_Omega187_only)
bcOmega188=bcs.copy()
e710=uOmega188_old+uOmega172_old
bcOmega188.append(DirichletBC(V,e710,dOmega188nOmega172))
e711=uOmega188_old+uOmega187_old
bcOmega188.append(DirichletBC(V,e711,dOmega188nOmega187))
e712=uOmega188_old+uOmega189_old
bcOmega188.append(DirichletBC(V,e712,dOmega188nOmega189))
e713=uOmega188_old+uOmega204_old
bcOmega188.append(DirichletBC(V,e713,dOmega188nOmega204))
bcOmega188.append(BC_Omega188_only)
bcOmega189=bcs.copy()
e714=uOmega189_old+uOmega173_old
bcOmega189.append(DirichletBC(V,e714,dOmega189nOmega173))
e715=uOmega189_old+uOmega188_old
bcOmega189.append(DirichletBC(V,e715,dOmega189nOmega188))
e716=uOmega189_old+uOmega190_old
bcOmega189.append(DirichletBC(V,e716,dOmega189nOmega190))
e717=uOmega189_old+uOmega205_old
bcOmega189.append(DirichletBC(V,e717,dOmega189nOmega205))
bcOmega189.append(BC_Omega189_only)
bcOmega190=bcs.copy()
e718=uOmega190_old+uOmega174_old
bcOmega190.append(DirichletBC(V,e718,dOmega190nOmega174))
e719=uOmega190_old+uOmega189_old
bcOmega190.append(DirichletBC(V,e719,dOmega190nOmega189))
e720=uOmega190_old+uOmega191_old
bcOmega190.append(DirichletBC(V,e720,dOmega190nOmega191))
e721=uOmega190_old+uOmega206_old
bcOmega190.append(DirichletBC(V,e721,dOmega190nOmega206))
bcOmega190.append(BC_Omega190_only)
bcOmega191=bcs.copy()
e722=uOmega191_old+uOmega175_old
bcOmega191.append(DirichletBC(V,e722,dOmega191nOmega175))
e723=uOmega191_old+uOmega190_old
bcOmega191.append(DirichletBC(V,e723,dOmega191nOmega190))
e724=uOmega191_old+uOmega192_old
bcOmega191.append(DirichletBC(V,e724,dOmega191nOmega192))
e725=uOmega191_old+uOmega207_old
bcOmega191.append(DirichletBC(V,e725,dOmega191nOmega207))
bcOmega191.append(BC_Omega191_only)
bcOmega192=bcs.copy()
e726=uOmega192_old+uOmega176_old
bcOmega192.append(DirichletBC(V,e726,dOmega192nOmega176))
e727=uOmega192_old+uOmega191_old
bcOmega192.append(DirichletBC(V,e727,dOmega192nOmega191))
e728=uOmega192_old+uOmega208_old
bcOmega192.append(DirichletBC(V,e728,dOmega192nOmega208))
bcOmega192.append(BC_Omega192_only)
bcOmega193=bcs.copy()
e729=uOmega193_old+uOmega177_old
bcOmega193.append(DirichletBC(V,e729,dOmega193nOmega177))
e730=uOmega193_old+uOmega194_old
bcOmega193.append(DirichletBC(V,e730,dOmega193nOmega194))
e731=uOmega193_old+uOmega209_old
bcOmega193.append(DirichletBC(V,e731,dOmega193nOmega209))
bcOmega193.append(BC_Omega193_only)
bcOmega194=bcs.copy()
e732=uOmega194_old+uOmega178_old
bcOmega194.append(DirichletBC(V,e732,dOmega194nOmega178))
e733=uOmega194_old+uOmega193_old
bcOmega194.append(DirichletBC(V,e733,dOmega194nOmega193))
e734=uOmega194_old+uOmega195_old
bcOmega194.append(DirichletBC(V,e734,dOmega194nOmega195))
e735=uOmega194_old+uOmega210_old
bcOmega194.append(DirichletBC(V,e735,dOmega194nOmega210))
bcOmega194.append(BC_Omega194_only)
bcOmega195=bcs.copy()
e736=uOmega195_old+uOmega179_old
bcOmega195.append(DirichletBC(V,e736,dOmega195nOmega179))
e737=uOmega195_old+uOmega194_old
bcOmega195.append(DirichletBC(V,e737,dOmega195nOmega194))
e738=uOmega195_old+uOmega196_old
bcOmega195.append(DirichletBC(V,e738,dOmega195nOmega196))
e739=uOmega195_old+uOmega211_old
bcOmega195.append(DirichletBC(V,e739,dOmega195nOmega211))
bcOmega195.append(BC_Omega195_only)
bcOmega196=bcs.copy()
e740=uOmega196_old+uOmega180_old
bcOmega196.append(DirichletBC(V,e740,dOmega196nOmega180))
e741=uOmega196_old+uOmega195_old
bcOmega196.append(DirichletBC(V,e741,dOmega196nOmega195))
e742=uOmega196_old+uOmega197_old
bcOmega196.append(DirichletBC(V,e742,dOmega196nOmega197))
e743=uOmega196_old+uOmega212_old
bcOmega196.append(DirichletBC(V,e743,dOmega196nOmega212))
bcOmega196.append(BC_Omega196_only)
bcOmega197=bcs.copy()
e744=uOmega197_old+uOmega181_old
bcOmega197.append(DirichletBC(V,e744,dOmega197nOmega181))
e745=uOmega197_old+uOmega196_old
bcOmega197.append(DirichletBC(V,e745,dOmega197nOmega196))
e746=uOmega197_old+uOmega198_old
bcOmega197.append(DirichletBC(V,e746,dOmega197nOmega198))
e747=uOmega197_old+uOmega213_old
bcOmega197.append(DirichletBC(V,e747,dOmega197nOmega213))
bcOmega197.append(BC_Omega197_only)
bcOmega198=bcs.copy()
e748=uOmega198_old+uOmega182_old
bcOmega198.append(DirichletBC(V,e748,dOmega198nOmega182))
e749=uOmega198_old+uOmega197_old
bcOmega198.append(DirichletBC(V,e749,dOmega198nOmega197))
e750=uOmega198_old+uOmega199_old
bcOmega198.append(DirichletBC(V,e750,dOmega198nOmega199))
e751=uOmega198_old+uOmega214_old
bcOmega198.append(DirichletBC(V,e751,dOmega198nOmega214))
bcOmega198.append(BC_Omega198_only)
bcOmega199=bcs.copy()
e752=uOmega199_old+uOmega183_old
bcOmega199.append(DirichletBC(V,e752,dOmega199nOmega183))
e753=uOmega199_old+uOmega198_old
bcOmega199.append(DirichletBC(V,e753,dOmega199nOmega198))
e754=uOmega199_old+uOmega200_old
bcOmega199.append(DirichletBC(V,e754,dOmega199nOmega200))
e755=uOmega199_old+uOmega215_old
bcOmega199.append(DirichletBC(V,e755,dOmega199nOmega215))
bcOmega199.append(BC_Omega199_only)
bcOmega200=bcs.copy()
e756=uOmega200_old+uOmega184_old
bcOmega200.append(DirichletBC(V,e756,dOmega200nOmega184))
e757=uOmega200_old+uOmega199_old
bcOmega200.append(DirichletBC(V,e757,dOmega200nOmega199))
e758=uOmega200_old+uOmega201_old
bcOmega200.append(DirichletBC(V,e758,dOmega200nOmega201))
e759=uOmega200_old+uOmega216_old
bcOmega200.append(DirichletBC(V,e759,dOmega200nOmega216))
bcOmega200.append(BC_Omega200_only)
bcOmega201=bcs.copy()
e760=uOmega201_old+uOmega185_old
bcOmega201.append(DirichletBC(V,e760,dOmega201nOmega185))
e761=uOmega201_old+uOmega200_old
bcOmega201.append(DirichletBC(V,e761,dOmega201nOmega200))
e762=uOmega201_old+uOmega202_old
bcOmega201.append(DirichletBC(V,e762,dOmega201nOmega202))
e763=uOmega201_old+uOmega217_old
bcOmega201.append(DirichletBC(V,e763,dOmega201nOmega217))
bcOmega201.append(BC_Omega201_only)
bcOmega202=bcs.copy()
e764=uOmega202_old+uOmega186_old
bcOmega202.append(DirichletBC(V,e764,dOmega202nOmega186))
e765=uOmega202_old+uOmega201_old
bcOmega202.append(DirichletBC(V,e765,dOmega202nOmega201))
e766=uOmega202_old+uOmega203_old
bcOmega202.append(DirichletBC(V,e766,dOmega202nOmega203))
e767=uOmega202_old+uOmega218_old
bcOmega202.append(DirichletBC(V,e767,dOmega202nOmega218))
bcOmega202.append(BC_Omega202_only)
bcOmega203=bcs.copy()
e768=uOmega203_old+uOmega187_old
bcOmega203.append(DirichletBC(V,e768,dOmega203nOmega187))
e769=uOmega203_old+uOmega202_old
bcOmega203.append(DirichletBC(V,e769,dOmega203nOmega202))
e770=uOmega203_old+uOmega204_old
bcOmega203.append(DirichletBC(V,e770,dOmega203nOmega204))
e771=uOmega203_old+uOmega219_old
bcOmega203.append(DirichletBC(V,e771,dOmega203nOmega219))
bcOmega203.append(BC_Omega203_only)
bcOmega204=bcs.copy()
e772=uOmega204_old+uOmega188_old
bcOmega204.append(DirichletBC(V,e772,dOmega204nOmega188))
e773=uOmega204_old+uOmega203_old
bcOmega204.append(DirichletBC(V,e773,dOmega204nOmega203))
e774=uOmega204_old+uOmega205_old
bcOmega204.append(DirichletBC(V,e774,dOmega204nOmega205))
e775=uOmega204_old+uOmega220_old
bcOmega204.append(DirichletBC(V,e775,dOmega204nOmega220))
bcOmega204.append(BC_Omega204_only)
bcOmega205=bcs.copy()
e776=uOmega205_old+uOmega189_old
bcOmega205.append(DirichletBC(V,e776,dOmega205nOmega189))
e777=uOmega205_old+uOmega204_old
bcOmega205.append(DirichletBC(V,e777,dOmega205nOmega204))
e778=uOmega205_old+uOmega206_old
bcOmega205.append(DirichletBC(V,e778,dOmega205nOmega206))
e779=uOmega205_old+uOmega221_old
bcOmega205.append(DirichletBC(V,e779,dOmega205nOmega221))
bcOmega205.append(BC_Omega205_only)
bcOmega206=bcs.copy()
e780=uOmega206_old+uOmega190_old
bcOmega206.append(DirichletBC(V,e780,dOmega206nOmega190))
e781=uOmega206_old+uOmega205_old
bcOmega206.append(DirichletBC(V,e781,dOmega206nOmega205))
e782=uOmega206_old+uOmega207_old
bcOmega206.append(DirichletBC(V,e782,dOmega206nOmega207))
e783=uOmega206_old+uOmega222_old
bcOmega206.append(DirichletBC(V,e783,dOmega206nOmega222))
bcOmega206.append(BC_Omega206_only)
bcOmega207=bcs.copy()
e784=uOmega207_old+uOmega191_old
bcOmega207.append(DirichletBC(V,e784,dOmega207nOmega191))
e785=uOmega207_old+uOmega206_old
bcOmega207.append(DirichletBC(V,e785,dOmega207nOmega206))
e786=uOmega207_old+uOmega208_old
bcOmega207.append(DirichletBC(V,e786,dOmega207nOmega208))
e787=uOmega207_old+uOmega223_old
bcOmega207.append(DirichletBC(V,e787,dOmega207nOmega223))
bcOmega207.append(BC_Omega207_only)
bcOmega208=bcs.copy()
e788=uOmega208_old+uOmega192_old
bcOmega208.append(DirichletBC(V,e788,dOmega208nOmega192))
e789=uOmega208_old+uOmega207_old
bcOmega208.append(DirichletBC(V,e789,dOmega208nOmega207))
e790=uOmega208_old+uOmega224_old
bcOmega208.append(DirichletBC(V,e790,dOmega208nOmega224))
bcOmega208.append(BC_Omega208_only)
bcOmega209=bcs.copy()
e791=uOmega209_old+uOmega193_old
bcOmega209.append(DirichletBC(V,e791,dOmega209nOmega193))
e792=uOmega209_old+uOmega210_old
bcOmega209.append(DirichletBC(V,e792,dOmega209nOmega210))
e793=uOmega209_old+uOmega225_old
bcOmega209.append(DirichletBC(V,e793,dOmega209nOmega225))
bcOmega209.append(BC_Omega209_only)
bcOmega210=bcs.copy()
e794=uOmega210_old+uOmega194_old
bcOmega210.append(DirichletBC(V,e794,dOmega210nOmega194))
e795=uOmega210_old+uOmega209_old
bcOmega210.append(DirichletBC(V,e795,dOmega210nOmega209))
e796=uOmega210_old+uOmega211_old
bcOmega210.append(DirichletBC(V,e796,dOmega210nOmega211))
e797=uOmega210_old+uOmega226_old
bcOmega210.append(DirichletBC(V,e797,dOmega210nOmega226))
bcOmega210.append(BC_Omega210_only)
bcOmega211=bcs.copy()
e798=uOmega211_old+uOmega195_old
bcOmega211.append(DirichletBC(V,e798,dOmega211nOmega195))
e799=uOmega211_old+uOmega210_old
bcOmega211.append(DirichletBC(V,e799,dOmega211nOmega210))
e800=uOmega211_old+uOmega212_old
bcOmega211.append(DirichletBC(V,e800,dOmega211nOmega212))
e801=uOmega211_old+uOmega227_old
bcOmega211.append(DirichletBC(V,e801,dOmega211nOmega227))
bcOmega211.append(BC_Omega211_only)
bcOmega212=bcs.copy()
e802=uOmega212_old+uOmega196_old
bcOmega212.append(DirichletBC(V,e802,dOmega212nOmega196))
e803=uOmega212_old+uOmega211_old
bcOmega212.append(DirichletBC(V,e803,dOmega212nOmega211))
e804=uOmega212_old+uOmega213_old
bcOmega212.append(DirichletBC(V,e804,dOmega212nOmega213))
e805=uOmega212_old+uOmega228_old
bcOmega212.append(DirichletBC(V,e805,dOmega212nOmega228))
bcOmega212.append(BC_Omega212_only)
bcOmega213=bcs.copy()
e806=uOmega213_old+uOmega197_old
bcOmega213.append(DirichletBC(V,e806,dOmega213nOmega197))
e807=uOmega213_old+uOmega212_old
bcOmega213.append(DirichletBC(V,e807,dOmega213nOmega212))
e808=uOmega213_old+uOmega214_old
bcOmega213.append(DirichletBC(V,e808,dOmega213nOmega214))
e809=uOmega213_old+uOmega229_old
bcOmega213.append(DirichletBC(V,e809,dOmega213nOmega229))
bcOmega213.append(BC_Omega213_only)
bcOmega214=bcs.copy()
e810=uOmega214_old+uOmega198_old
bcOmega214.append(DirichletBC(V,e810,dOmega214nOmega198))
e811=uOmega214_old+uOmega213_old
bcOmega214.append(DirichletBC(V,e811,dOmega214nOmega213))
e812=uOmega214_old+uOmega215_old
bcOmega214.append(DirichletBC(V,e812,dOmega214nOmega215))
e813=uOmega214_old+uOmega230_old
bcOmega214.append(DirichletBC(V,e813,dOmega214nOmega230))
bcOmega214.append(BC_Omega214_only)
bcOmega215=bcs.copy()
e814=uOmega215_old+uOmega199_old
bcOmega215.append(DirichletBC(V,e814,dOmega215nOmega199))
e815=uOmega215_old+uOmega214_old
bcOmega215.append(DirichletBC(V,e815,dOmega215nOmega214))
e816=uOmega215_old+uOmega216_old
bcOmega215.append(DirichletBC(V,e816,dOmega215nOmega216))
e817=uOmega215_old+uOmega231_old
bcOmega215.append(DirichletBC(V,e817,dOmega215nOmega231))
bcOmega215.append(BC_Omega215_only)
bcOmega216=bcs.copy()
e818=uOmega216_old+uOmega200_old
bcOmega216.append(DirichletBC(V,e818,dOmega216nOmega200))
e819=uOmega216_old+uOmega215_old
bcOmega216.append(DirichletBC(V,e819,dOmega216nOmega215))
e820=uOmega216_old+uOmega217_old
bcOmega216.append(DirichletBC(V,e820,dOmega216nOmega217))
e821=uOmega216_old+uOmega232_old
bcOmega216.append(DirichletBC(V,e821,dOmega216nOmega232))
bcOmega216.append(BC_Omega216_only)
bcOmega217=bcs.copy()
e822=uOmega217_old+uOmega201_old
bcOmega217.append(DirichletBC(V,e822,dOmega217nOmega201))
e823=uOmega217_old+uOmega216_old
bcOmega217.append(DirichletBC(V,e823,dOmega217nOmega216))
e824=uOmega217_old+uOmega218_old
bcOmega217.append(DirichletBC(V,e824,dOmega217nOmega218))
e825=uOmega217_old+uOmega233_old
bcOmega217.append(DirichletBC(V,e825,dOmega217nOmega233))
bcOmega217.append(BC_Omega217_only)
bcOmega218=bcs.copy()
e826=uOmega218_old+uOmega202_old
bcOmega218.append(DirichletBC(V,e826,dOmega218nOmega202))
e827=uOmega218_old+uOmega217_old
bcOmega218.append(DirichletBC(V,e827,dOmega218nOmega217))
e828=uOmega218_old+uOmega219_old
bcOmega218.append(DirichletBC(V,e828,dOmega218nOmega219))
e829=uOmega218_old+uOmega234_old
bcOmega218.append(DirichletBC(V,e829,dOmega218nOmega234))
bcOmega218.append(BC_Omega218_only)
bcOmega219=bcs.copy()
e830=uOmega219_old+uOmega203_old
bcOmega219.append(DirichletBC(V,e830,dOmega219nOmega203))
e831=uOmega219_old+uOmega218_old
bcOmega219.append(DirichletBC(V,e831,dOmega219nOmega218))
e832=uOmega219_old+uOmega220_old
bcOmega219.append(DirichletBC(V,e832,dOmega219nOmega220))
e833=uOmega219_old+uOmega235_old
bcOmega219.append(DirichletBC(V,e833,dOmega219nOmega235))
bcOmega219.append(BC_Omega219_only)
bcOmega220=bcs.copy()
e834=uOmega220_old+uOmega204_old
bcOmega220.append(DirichletBC(V,e834,dOmega220nOmega204))
e835=uOmega220_old+uOmega219_old
bcOmega220.append(DirichletBC(V,e835,dOmega220nOmega219))
e836=uOmega220_old+uOmega221_old
bcOmega220.append(DirichletBC(V,e836,dOmega220nOmega221))
e837=uOmega220_old+uOmega236_old
bcOmega220.append(DirichletBC(V,e837,dOmega220nOmega236))
bcOmega220.append(BC_Omega220_only)
bcOmega221=bcs.copy()
e838=uOmega221_old+uOmega205_old
bcOmega221.append(DirichletBC(V,e838,dOmega221nOmega205))
e839=uOmega221_old+uOmega220_old
bcOmega221.append(DirichletBC(V,e839,dOmega221nOmega220))
e840=uOmega221_old+uOmega222_old
bcOmega221.append(DirichletBC(V,e840,dOmega221nOmega222))
e841=uOmega221_old+uOmega237_old
bcOmega221.append(DirichletBC(V,e841,dOmega221nOmega237))
bcOmega221.append(BC_Omega221_only)
bcOmega222=bcs.copy()
e842=uOmega222_old+uOmega206_old
bcOmega222.append(DirichletBC(V,e842,dOmega222nOmega206))
e843=uOmega222_old+uOmega221_old
bcOmega222.append(DirichletBC(V,e843,dOmega222nOmega221))
e844=uOmega222_old+uOmega223_old
bcOmega222.append(DirichletBC(V,e844,dOmega222nOmega223))
e845=uOmega222_old+uOmega238_old
bcOmega222.append(DirichletBC(V,e845,dOmega222nOmega238))
bcOmega222.append(BC_Omega222_only)
bcOmega223=bcs.copy()
e846=uOmega223_old+uOmega207_old
bcOmega223.append(DirichletBC(V,e846,dOmega223nOmega207))
e847=uOmega223_old+uOmega222_old
bcOmega223.append(DirichletBC(V,e847,dOmega223nOmega222))
e848=uOmega223_old+uOmega224_old
bcOmega223.append(DirichletBC(V,e848,dOmega223nOmega224))
e849=uOmega223_old+uOmega239_old
bcOmega223.append(DirichletBC(V,e849,dOmega223nOmega239))
bcOmega223.append(BC_Omega223_only)
bcOmega224=bcs.copy()
e850=uOmega224_old+uOmega208_old
bcOmega224.append(DirichletBC(V,e850,dOmega224nOmega208))
e851=uOmega224_old+uOmega223_old
bcOmega224.append(DirichletBC(V,e851,dOmega224nOmega223))
e852=uOmega224_old+uOmega240_old
bcOmega224.append(DirichletBC(V,e852,dOmega224nOmega240))
bcOmega224.append(BC_Omega224_only)
bcOmega225=bcs.copy()
e853=uOmega225_old+uOmega209_old
bcOmega225.append(DirichletBC(V,e853,dOmega225nOmega209))
e854=uOmega225_old+uOmega226_old
bcOmega225.append(DirichletBC(V,e854,dOmega225nOmega226))
e855=uOmega225_old+uOmega241_old
bcOmega225.append(DirichletBC(V,e855,dOmega225nOmega241))
bcOmega225.append(BC_Omega225_only)
bcOmega226=bcs.copy()
e856=uOmega226_old+uOmega210_old
bcOmega226.append(DirichletBC(V,e856,dOmega226nOmega210))
e857=uOmega226_old+uOmega225_old
bcOmega226.append(DirichletBC(V,e857,dOmega226nOmega225))
e858=uOmega226_old+uOmega227_old
bcOmega226.append(DirichletBC(V,e858,dOmega226nOmega227))
e859=uOmega226_old+uOmega242_old
bcOmega226.append(DirichletBC(V,e859,dOmega226nOmega242))
bcOmega226.append(BC_Omega226_only)
bcOmega227=bcs.copy()
e860=uOmega227_old+uOmega211_old
bcOmega227.append(DirichletBC(V,e860,dOmega227nOmega211))
e861=uOmega227_old+uOmega226_old
bcOmega227.append(DirichletBC(V,e861,dOmega227nOmega226))
e862=uOmega227_old+uOmega228_old
bcOmega227.append(DirichletBC(V,e862,dOmega227nOmega228))
e863=uOmega227_old+uOmega243_old
bcOmega227.append(DirichletBC(V,e863,dOmega227nOmega243))
bcOmega227.append(BC_Omega227_only)
bcOmega228=bcs.copy()
e864=uOmega228_old+uOmega212_old
bcOmega228.append(DirichletBC(V,e864,dOmega228nOmega212))
e865=uOmega228_old+uOmega227_old
bcOmega228.append(DirichletBC(V,e865,dOmega228nOmega227))
e866=uOmega228_old+uOmega229_old
bcOmega228.append(DirichletBC(V,e866,dOmega228nOmega229))
e867=uOmega228_old+uOmega244_old
bcOmega228.append(DirichletBC(V,e867,dOmega228nOmega244))
bcOmega228.append(BC_Omega228_only)
bcOmega229=bcs.copy()
e868=uOmega229_old+uOmega213_old
bcOmega229.append(DirichletBC(V,e868,dOmega229nOmega213))
e869=uOmega229_old+uOmega228_old
bcOmega229.append(DirichletBC(V,e869,dOmega229nOmega228))
e870=uOmega229_old+uOmega230_old
bcOmega229.append(DirichletBC(V,e870,dOmega229nOmega230))
e871=uOmega229_old+uOmega245_old
bcOmega229.append(DirichletBC(V,e871,dOmega229nOmega245))
bcOmega229.append(BC_Omega229_only)
bcOmega230=bcs.copy()
e872=uOmega230_old+uOmega214_old
bcOmega230.append(DirichletBC(V,e872,dOmega230nOmega214))
e873=uOmega230_old+uOmega229_old
bcOmega230.append(DirichletBC(V,e873,dOmega230nOmega229))
e874=uOmega230_old+uOmega231_old
bcOmega230.append(DirichletBC(V,e874,dOmega230nOmega231))
e875=uOmega230_old+uOmega246_old
bcOmega230.append(DirichletBC(V,e875,dOmega230nOmega246))
bcOmega230.append(BC_Omega230_only)
bcOmega231=bcs.copy()
e876=uOmega231_old+uOmega215_old
bcOmega231.append(DirichletBC(V,e876,dOmega231nOmega215))
e877=uOmega231_old+uOmega230_old
bcOmega231.append(DirichletBC(V,e877,dOmega231nOmega230))
e878=uOmega231_old+uOmega232_old
bcOmega231.append(DirichletBC(V,e878,dOmega231nOmega232))
e879=uOmega231_old+uOmega247_old
bcOmega231.append(DirichletBC(V,e879,dOmega231nOmega247))
bcOmega231.append(BC_Omega231_only)
bcOmega232=bcs.copy()
e880=uOmega232_old+uOmega216_old
bcOmega232.append(DirichletBC(V,e880,dOmega232nOmega216))
e881=uOmega232_old+uOmega231_old
bcOmega232.append(DirichletBC(V,e881,dOmega232nOmega231))
e882=uOmega232_old+uOmega233_old
bcOmega232.append(DirichletBC(V,e882,dOmega232nOmega233))
e883=uOmega232_old+uOmega248_old
bcOmega232.append(DirichletBC(V,e883,dOmega232nOmega248))
bcOmega232.append(BC_Omega232_only)
bcOmega233=bcs.copy()
e884=uOmega233_old+uOmega217_old
bcOmega233.append(DirichletBC(V,e884,dOmega233nOmega217))
e885=uOmega233_old+uOmega232_old
bcOmega233.append(DirichletBC(V,e885,dOmega233nOmega232))
e886=uOmega233_old+uOmega234_old
bcOmega233.append(DirichletBC(V,e886,dOmega233nOmega234))
e887=uOmega233_old+uOmega249_old
bcOmega233.append(DirichletBC(V,e887,dOmega233nOmega249))
bcOmega233.append(BC_Omega233_only)
bcOmega234=bcs.copy()
e888=uOmega234_old+uOmega218_old
bcOmega234.append(DirichletBC(V,e888,dOmega234nOmega218))
e889=uOmega234_old+uOmega233_old
bcOmega234.append(DirichletBC(V,e889,dOmega234nOmega233))
e890=uOmega234_old+uOmega235_old
bcOmega234.append(DirichletBC(V,e890,dOmega234nOmega235))
e891=uOmega234_old+uOmega250_old
bcOmega234.append(DirichletBC(V,e891,dOmega234nOmega250))
bcOmega234.append(BC_Omega234_only)
bcOmega235=bcs.copy()
e892=uOmega235_old+uOmega219_old
bcOmega235.append(DirichletBC(V,e892,dOmega235nOmega219))
e893=uOmega235_old+uOmega234_old
bcOmega235.append(DirichletBC(V,e893,dOmega235nOmega234))
e894=uOmega235_old+uOmega236_old
bcOmega235.append(DirichletBC(V,e894,dOmega235nOmega236))
e895=uOmega235_old+uOmega251_old
bcOmega235.append(DirichletBC(V,e895,dOmega235nOmega251))
bcOmega235.append(BC_Omega235_only)
bcOmega236=bcs.copy()
e896=uOmega236_old+uOmega220_old
bcOmega236.append(DirichletBC(V,e896,dOmega236nOmega220))
e897=uOmega236_old+uOmega235_old
bcOmega236.append(DirichletBC(V,e897,dOmega236nOmega235))
e898=uOmega236_old+uOmega237_old
bcOmega236.append(DirichletBC(V,e898,dOmega236nOmega237))
e899=uOmega236_old+uOmega252_old
bcOmega236.append(DirichletBC(V,e899,dOmega236nOmega252))
bcOmega236.append(BC_Omega236_only)
bcOmega237=bcs.copy()
e900=uOmega237_old+uOmega221_old
bcOmega237.append(DirichletBC(V,e900,dOmega237nOmega221))
e901=uOmega237_old+uOmega236_old
bcOmega237.append(DirichletBC(V,e901,dOmega237nOmega236))
e902=uOmega237_old+uOmega238_old
bcOmega237.append(DirichletBC(V,e902,dOmega237nOmega238))
e903=uOmega237_old+uOmega253_old
bcOmega237.append(DirichletBC(V,e903,dOmega237nOmega253))
bcOmega237.append(BC_Omega237_only)
bcOmega238=bcs.copy()
e904=uOmega238_old+uOmega222_old
bcOmega238.append(DirichletBC(V,e904,dOmega238nOmega222))
e905=uOmega238_old+uOmega237_old
bcOmega238.append(DirichletBC(V,e905,dOmega238nOmega237))
e906=uOmega238_old+uOmega239_old
bcOmega238.append(DirichletBC(V,e906,dOmega238nOmega239))
e907=uOmega238_old+uOmega254_old
bcOmega238.append(DirichletBC(V,e907,dOmega238nOmega254))
bcOmega238.append(BC_Omega238_only)
bcOmega239=bcs.copy()
e908=uOmega239_old+uOmega223_old
bcOmega239.append(DirichletBC(V,e908,dOmega239nOmega223))
e909=uOmega239_old+uOmega238_old
bcOmega239.append(DirichletBC(V,e909,dOmega239nOmega238))
e910=uOmega239_old+uOmega240_old
bcOmega239.append(DirichletBC(V,e910,dOmega239nOmega240))
e911=uOmega239_old+uOmega255_old
bcOmega239.append(DirichletBC(V,e911,dOmega239nOmega255))
bcOmega239.append(BC_Omega239_only)
bcOmega240=bcs.copy()
e912=uOmega240_old+uOmega224_old
bcOmega240.append(DirichletBC(V,e912,dOmega240nOmega224))
e913=uOmega240_old+uOmega239_old
bcOmega240.append(DirichletBC(V,e913,dOmega240nOmega239))
e914=uOmega240_old+uOmega256_old
bcOmega240.append(DirichletBC(V,e914,dOmega240nOmega256))
bcOmega240.append(BC_Omega240_only)
bcOmega241=bcs.copy()
e915=uOmega241_old+uOmega225_old
bcOmega241.append(DirichletBC(V,e915,dOmega241nOmega225))
e916=uOmega241_old+uOmega242_old
bcOmega241.append(DirichletBC(V,e916,dOmega241nOmega242))
bcOmega241.append(BC_Omega241_only)
bcOmega242=bcs.copy()
e917=uOmega242_old+uOmega226_old
bcOmega242.append(DirichletBC(V,e917,dOmega242nOmega226))
e918=uOmega242_old+uOmega241_old
bcOmega242.append(DirichletBC(V,e918,dOmega242nOmega241))
e919=uOmega242_old+uOmega243_old
bcOmega242.append(DirichletBC(V,e919,dOmega242nOmega243))
bcOmega242.append(BC_Omega242_only)
bcOmega243=bcs.copy()
e920=uOmega243_old+uOmega227_old
bcOmega243.append(DirichletBC(V,e920,dOmega243nOmega227))
e921=uOmega243_old+uOmega242_old
bcOmega243.append(DirichletBC(V,e921,dOmega243nOmega242))
e922=uOmega243_old+uOmega244_old
bcOmega243.append(DirichletBC(V,e922,dOmega243nOmega244))
bcOmega243.append(BC_Omega243_only)
bcOmega244=bcs.copy()
e923=uOmega244_old+uOmega228_old
bcOmega244.append(DirichletBC(V,e923,dOmega244nOmega228))
e924=uOmega244_old+uOmega243_old
bcOmega244.append(DirichletBC(V,e924,dOmega244nOmega243))
e925=uOmega244_old+uOmega245_old
bcOmega244.append(DirichletBC(V,e925,dOmega244nOmega245))
bcOmega244.append(BC_Omega244_only)
bcOmega245=bcs.copy()
e926=uOmega245_old+uOmega229_old
bcOmega245.append(DirichletBC(V,e926,dOmega245nOmega229))
e927=uOmega245_old+uOmega244_old
bcOmega245.append(DirichletBC(V,e927,dOmega245nOmega244))
e928=uOmega245_old+uOmega246_old
bcOmega245.append(DirichletBC(V,e928,dOmega245nOmega246))
bcOmega245.append(BC_Omega245_only)
bcOmega246=bcs.copy()
e929=uOmega246_old+uOmega230_old
bcOmega246.append(DirichletBC(V,e929,dOmega246nOmega230))
e930=uOmega246_old+uOmega245_old
bcOmega246.append(DirichletBC(V,e930,dOmega246nOmega245))
e931=uOmega246_old+uOmega247_old
bcOmega246.append(DirichletBC(V,e931,dOmega246nOmega247))
bcOmega246.append(BC_Omega246_only)
bcOmega247=bcs.copy()
e932=uOmega247_old+uOmega231_old
bcOmega247.append(DirichletBC(V,e932,dOmega247nOmega231))
e933=uOmega247_old+uOmega246_old
bcOmega247.append(DirichletBC(V,e933,dOmega247nOmega246))
e934=uOmega247_old+uOmega248_old
bcOmega247.append(DirichletBC(V,e934,dOmega247nOmega248))
bcOmega247.append(BC_Omega247_only)
bcOmega248=bcs.copy()
e935=uOmega248_old+uOmega232_old
bcOmega248.append(DirichletBC(V,e935,dOmega248nOmega232))
e936=uOmega248_old+uOmega247_old
bcOmega248.append(DirichletBC(V,e936,dOmega248nOmega247))
e937=uOmega248_old+uOmega249_old
bcOmega248.append(DirichletBC(V,e937,dOmega248nOmega249))
bcOmega248.append(BC_Omega248_only)
bcOmega249=bcs.copy()
e938=uOmega249_old+uOmega233_old
bcOmega249.append(DirichletBC(V,e938,dOmega249nOmega233))
e939=uOmega249_old+uOmega248_old
bcOmega249.append(DirichletBC(V,e939,dOmega249nOmega248))
e940=uOmega249_old+uOmega250_old
bcOmega249.append(DirichletBC(V,e940,dOmega249nOmega250))
bcOmega249.append(BC_Omega249_only)
bcOmega250=bcs.copy()
e941=uOmega250_old+uOmega234_old
bcOmega250.append(DirichletBC(V,e941,dOmega250nOmega234))
e942=uOmega250_old+uOmega249_old
bcOmega250.append(DirichletBC(V,e942,dOmega250nOmega249))
e943=uOmega250_old+uOmega251_old
bcOmega250.append(DirichletBC(V,e943,dOmega250nOmega251))
bcOmega250.append(BC_Omega250_only)
bcOmega251=bcs.copy()
e944=uOmega251_old+uOmega235_old
bcOmega251.append(DirichletBC(V,e944,dOmega251nOmega235))
e945=uOmega251_old+uOmega250_old
bcOmega251.append(DirichletBC(V,e945,dOmega251nOmega250))
e946=uOmega251_old+uOmega252_old
bcOmega251.append(DirichletBC(V,e946,dOmega251nOmega252))
bcOmega251.append(BC_Omega251_only)
bcOmega252=bcs.copy()
e947=uOmega252_old+uOmega236_old
bcOmega252.append(DirichletBC(V,e947,dOmega252nOmega236))
e948=uOmega252_old+uOmega251_old
bcOmega252.append(DirichletBC(V,e948,dOmega252nOmega251))
e949=uOmega252_old+uOmega253_old
bcOmega252.append(DirichletBC(V,e949,dOmega252nOmega253))
bcOmega252.append(BC_Omega252_only)
bcOmega253=bcs.copy()
e950=uOmega253_old+uOmega237_old
bcOmega253.append(DirichletBC(V,e950,dOmega253nOmega237))
e951=uOmega253_old+uOmega252_old
bcOmega253.append(DirichletBC(V,e951,dOmega253nOmega252))
e952=uOmega253_old+uOmega254_old
bcOmega253.append(DirichletBC(V,e952,dOmega253nOmega254))
bcOmega253.append(BC_Omega253_only)
bcOmega254=bcs.copy()
e953=uOmega254_old+uOmega238_old
bcOmega254.append(DirichletBC(V,e953,dOmega254nOmega238))
e954=uOmega254_old+uOmega253_old
bcOmega254.append(DirichletBC(V,e954,dOmega254nOmega253))
e955=uOmega254_old+uOmega255_old
bcOmega254.append(DirichletBC(V,e955,dOmega254nOmega255))
bcOmega254.append(BC_Omega254_only)
bcOmega255=bcs.copy()
e956=uOmega255_old+uOmega239_old
bcOmega255.append(DirichletBC(V,e956,dOmega255nOmega239))
e957=uOmega255_old+uOmega254_old
bcOmega255.append(DirichletBC(V,e957,dOmega255nOmega254))
e958=uOmega255_old+uOmega256_old
bcOmega255.append(DirichletBC(V,e958,dOmega255nOmega256))
bcOmega255.append(BC_Omega255_only)
bcOmega256=bcs.copy()
e959=uOmega256_old+uOmega240_old
bcOmega256.append(DirichletBC(V,e959,dOmega256nOmega240))
e960=uOmega256_old+uOmega255_old
bcOmega256.append(DirichletBC(V,e960,dOmega256nOmega255))
bcOmega256.append(BC_Omega256_only)
nSchwarz=10
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
	uOmega65_old=uOmega65
	uOmega66_old=uOmega66
	uOmega67_old=uOmega67
	uOmega68_old=uOmega68
	uOmega69_old=uOmega69
	uOmega70_old=uOmega70
	uOmega71_old=uOmega71
	uOmega72_old=uOmega72
	uOmega73_old=uOmega73
	uOmega74_old=uOmega74
	uOmega75_old=uOmega75
	uOmega76_old=uOmega76
	uOmega77_old=uOmega77
	uOmega78_old=uOmega78
	uOmega79_old=uOmega79
	uOmega80_old=uOmega80
	uOmega81_old=uOmega81
	uOmega82_old=uOmega82
	uOmega83_old=uOmega83
	uOmega84_old=uOmega84
	uOmega85_old=uOmega85
	uOmega86_old=uOmega86
	uOmega87_old=uOmega87
	uOmega88_old=uOmega88
	uOmega89_old=uOmega89
	uOmega90_old=uOmega90
	uOmega91_old=uOmega91
	uOmega92_old=uOmega92
	uOmega93_old=uOmega93
	uOmega94_old=uOmega94
	uOmega95_old=uOmega95
	uOmega96_old=uOmega96
	uOmega97_old=uOmega97
	uOmega98_old=uOmega98
	uOmega99_old=uOmega99
	uOmega100_old=uOmega100
	uOmega101_old=uOmega101
	uOmega102_old=uOmega102
	uOmega103_old=uOmega103
	uOmega104_old=uOmega104
	uOmega105_old=uOmega105
	uOmega106_old=uOmega106
	uOmega107_old=uOmega107
	uOmega108_old=uOmega108
	uOmega109_old=uOmega109
	uOmega110_old=uOmega110
	uOmega111_old=uOmega111
	uOmega112_old=uOmega112
	uOmega113_old=uOmega113
	uOmega114_old=uOmega114
	uOmega115_old=uOmega115
	uOmega116_old=uOmega116
	uOmega117_old=uOmega117
	uOmega118_old=uOmega118
	uOmega119_old=uOmega119
	uOmega120_old=uOmega120
	uOmega121_old=uOmega121
	uOmega122_old=uOmega122
	uOmega123_old=uOmega123
	uOmega124_old=uOmega124
	uOmega125_old=uOmega125
	uOmega126_old=uOmega126
	uOmega127_old=uOmega127
	uOmega128_old=uOmega128
	uOmega129_old=uOmega129
	uOmega130_old=uOmega130
	uOmega131_old=uOmega131
	uOmega132_old=uOmega132
	uOmega133_old=uOmega133
	uOmega134_old=uOmega134
	uOmega135_old=uOmega135
	uOmega136_old=uOmega136
	uOmega137_old=uOmega137
	uOmega138_old=uOmega138
	uOmega139_old=uOmega139
	uOmega140_old=uOmega140
	uOmega141_old=uOmega141
	uOmega142_old=uOmega142
	uOmega143_old=uOmega143
	uOmega144_old=uOmega144
	uOmega145_old=uOmega145
	uOmega146_old=uOmega146
	uOmega147_old=uOmega147
	uOmega148_old=uOmega148
	uOmega149_old=uOmega149
	uOmega150_old=uOmega150
	uOmega151_old=uOmega151
	uOmega152_old=uOmega152
	uOmega153_old=uOmega153
	uOmega154_old=uOmega154
	uOmega155_old=uOmega155
	uOmega156_old=uOmega156
	uOmega157_old=uOmega157
	uOmega158_old=uOmega158
	uOmega159_old=uOmega159
	uOmega160_old=uOmega160
	uOmega161_old=uOmega161
	uOmega162_old=uOmega162
	uOmega163_old=uOmega163
	uOmega164_old=uOmega164
	uOmega165_old=uOmega165
	uOmega166_old=uOmega166
	uOmega167_old=uOmega167
	uOmega168_old=uOmega168
	uOmega169_old=uOmega169
	uOmega170_old=uOmega170
	uOmega171_old=uOmega171
	uOmega172_old=uOmega172
	uOmega173_old=uOmega173
	uOmega174_old=uOmega174
	uOmega175_old=uOmega175
	uOmega176_old=uOmega176
	uOmega177_old=uOmega177
	uOmega178_old=uOmega178
	uOmega179_old=uOmega179
	uOmega180_old=uOmega180
	uOmega181_old=uOmega181
	uOmega182_old=uOmega182
	uOmega183_old=uOmega183
	uOmega184_old=uOmega184
	uOmega185_old=uOmega185
	uOmega186_old=uOmega186
	uOmega187_old=uOmega187
	uOmega188_old=uOmega188
	uOmega189_old=uOmega189
	uOmega190_old=uOmega190
	uOmega191_old=uOmega191
	uOmega192_old=uOmega192
	uOmega193_old=uOmega193
	uOmega194_old=uOmega194
	uOmega195_old=uOmega195
	uOmega196_old=uOmega196
	uOmega197_old=uOmega197
	uOmega198_old=uOmega198
	uOmega199_old=uOmega199
	uOmega200_old=uOmega200
	uOmega201_old=uOmega201
	uOmega202_old=uOmega202
	uOmega203_old=uOmega203
	uOmega204_old=uOmega204
	uOmega205_old=uOmega205
	uOmega206_old=uOmega206
	uOmega207_old=uOmega207
	uOmega208_old=uOmega208
	uOmega209_old=uOmega209
	uOmega210_old=uOmega210
	uOmega211_old=uOmega211
	uOmega212_old=uOmega212
	uOmega213_old=uOmega213
	uOmega214_old=uOmega214
	uOmega215_old=uOmega215
	uOmega216_old=uOmega216
	uOmega217_old=uOmega217
	uOmega218_old=uOmega218
	uOmega219_old=uOmega219
	uOmega220_old=uOmega220
	uOmega221_old=uOmega221
	uOmega222_old=uOmega222
	uOmega223_old=uOmega223
	uOmega224_old=uOmega224
	uOmega225_old=uOmega225
	uOmega226_old=uOmega226
	uOmega227_old=uOmega227
	uOmega228_old=uOmega228
	uOmega229_old=uOmega229
	uOmega230_old=uOmega230
	uOmega231_old=uOmega231
	uOmega232_old=uOmega232
	uOmega233_old=uOmega233
	uOmega234_old=uOmega234
	uOmega235_old=uOmega235
	uOmega236_old=uOmega236
	uOmega237_old=uOmega237
	uOmega238_old=uOmega238
	uOmega239_old=uOmega239
	uOmega240_old=uOmega240
	uOmega241_old=uOmega241
	uOmega242_old=uOmega242
	uOmega243_old=uOmega243
	uOmega244_old=uOmega244
	uOmega245_old=uOmega245
	uOmega246_old=uOmega246
	uOmega247_old=uOmega247
	uOmega248_old=uOmega248
	uOmega249_old=uOmega249
	uOmega250_old=uOmega250
	uOmega251_old=uOmega251
	uOmega252_old=uOmega252
	uOmega253_old=uOmega253
	uOmega254_old=uOmega254
	uOmega255_old=uOmega255
	uOmega256_old=uOmega256
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
	solve(aOmega65==LOmega65,uOmega65,bcs=bcOmega65,solver_parameters=params)
	solve(aOmega66==LOmega66,uOmega66,bcs=bcOmega66,solver_parameters=params)
	solve(aOmega67==LOmega67,uOmega67,bcs=bcOmega67,solver_parameters=params)
	solve(aOmega68==LOmega68,uOmega68,bcs=bcOmega68,solver_parameters=params)
	solve(aOmega69==LOmega69,uOmega69,bcs=bcOmega69,solver_parameters=params)
	solve(aOmega70==LOmega70,uOmega70,bcs=bcOmega70,solver_parameters=params)
	solve(aOmega71==LOmega71,uOmega71,bcs=bcOmega71,solver_parameters=params)
	solve(aOmega72==LOmega72,uOmega72,bcs=bcOmega72,solver_parameters=params)
	solve(aOmega73==LOmega73,uOmega73,bcs=bcOmega73,solver_parameters=params)
	solve(aOmega74==LOmega74,uOmega74,bcs=bcOmega74,solver_parameters=params)
	solve(aOmega75==LOmega75,uOmega75,bcs=bcOmega75,solver_parameters=params)
	solve(aOmega76==LOmega76,uOmega76,bcs=bcOmega76,solver_parameters=params)
	solve(aOmega77==LOmega77,uOmega77,bcs=bcOmega77,solver_parameters=params)
	solve(aOmega78==LOmega78,uOmega78,bcs=bcOmega78,solver_parameters=params)
	solve(aOmega79==LOmega79,uOmega79,bcs=bcOmega79,solver_parameters=params)
	solve(aOmega80==LOmega80,uOmega80,bcs=bcOmega80,solver_parameters=params)
	solve(aOmega81==LOmega81,uOmega81,bcs=bcOmega81,solver_parameters=params)
	solve(aOmega82==LOmega82,uOmega82,bcs=bcOmega82,solver_parameters=params)
	solve(aOmega83==LOmega83,uOmega83,bcs=bcOmega83,solver_parameters=params)
	solve(aOmega84==LOmega84,uOmega84,bcs=bcOmega84,solver_parameters=params)
	solve(aOmega85==LOmega85,uOmega85,bcs=bcOmega85,solver_parameters=params)
	solve(aOmega86==LOmega86,uOmega86,bcs=bcOmega86,solver_parameters=params)
	solve(aOmega87==LOmega87,uOmega87,bcs=bcOmega87,solver_parameters=params)
	solve(aOmega88==LOmega88,uOmega88,bcs=bcOmega88,solver_parameters=params)
	solve(aOmega89==LOmega89,uOmega89,bcs=bcOmega89,solver_parameters=params)
	solve(aOmega90==LOmega90,uOmega90,bcs=bcOmega90,solver_parameters=params)
	solve(aOmega91==LOmega91,uOmega91,bcs=bcOmega91,solver_parameters=params)
	solve(aOmega92==LOmega92,uOmega92,bcs=bcOmega92,solver_parameters=params)
	solve(aOmega93==LOmega93,uOmega93,bcs=bcOmega93,solver_parameters=params)
	solve(aOmega94==LOmega94,uOmega94,bcs=bcOmega94,solver_parameters=params)
	solve(aOmega95==LOmega95,uOmega95,bcs=bcOmega95,solver_parameters=params)
	solve(aOmega96==LOmega96,uOmega96,bcs=bcOmega96,solver_parameters=params)
	solve(aOmega97==LOmega97,uOmega97,bcs=bcOmega97,solver_parameters=params)
	solve(aOmega98==LOmega98,uOmega98,bcs=bcOmega98,solver_parameters=params)
	solve(aOmega99==LOmega99,uOmega99,bcs=bcOmega99,solver_parameters=params)
	solve(aOmega100==LOmega100,uOmega100,bcs=bcOmega100,solver_parameters=params)
	solve(aOmega101==LOmega101,uOmega101,bcs=bcOmega101,solver_parameters=params)
	solve(aOmega102==LOmega102,uOmega102,bcs=bcOmega102,solver_parameters=params)
	solve(aOmega103==LOmega103,uOmega103,bcs=bcOmega103,solver_parameters=params)
	solve(aOmega104==LOmega104,uOmega104,bcs=bcOmega104,solver_parameters=params)
	solve(aOmega105==LOmega105,uOmega105,bcs=bcOmega105,solver_parameters=params)
	solve(aOmega106==LOmega106,uOmega106,bcs=bcOmega106,solver_parameters=params)
	solve(aOmega107==LOmega107,uOmega107,bcs=bcOmega107,solver_parameters=params)
	solve(aOmega108==LOmega108,uOmega108,bcs=bcOmega108,solver_parameters=params)
	solve(aOmega109==LOmega109,uOmega109,bcs=bcOmega109,solver_parameters=params)
	solve(aOmega110==LOmega110,uOmega110,bcs=bcOmega110,solver_parameters=params)
	solve(aOmega111==LOmega111,uOmega111,bcs=bcOmega111,solver_parameters=params)
	solve(aOmega112==LOmega112,uOmega112,bcs=bcOmega112,solver_parameters=params)
	solve(aOmega113==LOmega113,uOmega113,bcs=bcOmega113,solver_parameters=params)
	solve(aOmega114==LOmega114,uOmega114,bcs=bcOmega114,solver_parameters=params)
	solve(aOmega115==LOmega115,uOmega115,bcs=bcOmega115,solver_parameters=params)
	solve(aOmega116==LOmega116,uOmega116,bcs=bcOmega116,solver_parameters=params)
	solve(aOmega117==LOmega117,uOmega117,bcs=bcOmega117,solver_parameters=params)
	solve(aOmega118==LOmega118,uOmega118,bcs=bcOmega118,solver_parameters=params)
	solve(aOmega119==LOmega119,uOmega119,bcs=bcOmega119,solver_parameters=params)
	solve(aOmega120==LOmega120,uOmega120,bcs=bcOmega120,solver_parameters=params)
	solve(aOmega121==LOmega121,uOmega121,bcs=bcOmega121,solver_parameters=params)
	solve(aOmega122==LOmega122,uOmega122,bcs=bcOmega122,solver_parameters=params)
	solve(aOmega123==LOmega123,uOmega123,bcs=bcOmega123,solver_parameters=params)
	solve(aOmega124==LOmega124,uOmega124,bcs=bcOmega124,solver_parameters=params)
	solve(aOmega125==LOmega125,uOmega125,bcs=bcOmega125,solver_parameters=params)
	solve(aOmega126==LOmega126,uOmega126,bcs=bcOmega126,solver_parameters=params)
	solve(aOmega127==LOmega127,uOmega127,bcs=bcOmega127,solver_parameters=params)
	solve(aOmega128==LOmega128,uOmega128,bcs=bcOmega128,solver_parameters=params)
	solve(aOmega129==LOmega129,uOmega129,bcs=bcOmega129,solver_parameters=params)
	solve(aOmega130==LOmega130,uOmega130,bcs=bcOmega130,solver_parameters=params)
	solve(aOmega131==LOmega131,uOmega131,bcs=bcOmega131,solver_parameters=params)
	solve(aOmega132==LOmega132,uOmega132,bcs=bcOmega132,solver_parameters=params)
	solve(aOmega133==LOmega133,uOmega133,bcs=bcOmega133,solver_parameters=params)
	solve(aOmega134==LOmega134,uOmega134,bcs=bcOmega134,solver_parameters=params)
	solve(aOmega135==LOmega135,uOmega135,bcs=bcOmega135,solver_parameters=params)
	solve(aOmega136==LOmega136,uOmega136,bcs=bcOmega136,solver_parameters=params)
	solve(aOmega137==LOmega137,uOmega137,bcs=bcOmega137,solver_parameters=params)
	solve(aOmega138==LOmega138,uOmega138,bcs=bcOmega138,solver_parameters=params)
	solve(aOmega139==LOmega139,uOmega139,bcs=bcOmega139,solver_parameters=params)
	solve(aOmega140==LOmega140,uOmega140,bcs=bcOmega140,solver_parameters=params)
	solve(aOmega141==LOmega141,uOmega141,bcs=bcOmega141,solver_parameters=params)
	solve(aOmega142==LOmega142,uOmega142,bcs=bcOmega142,solver_parameters=params)
	solve(aOmega143==LOmega143,uOmega143,bcs=bcOmega143,solver_parameters=params)
	solve(aOmega144==LOmega144,uOmega144,bcs=bcOmega144,solver_parameters=params)
	solve(aOmega145==LOmega145,uOmega145,bcs=bcOmega145,solver_parameters=params)
	solve(aOmega146==LOmega146,uOmega146,bcs=bcOmega146,solver_parameters=params)
	solve(aOmega147==LOmega147,uOmega147,bcs=bcOmega147,solver_parameters=params)
	solve(aOmega148==LOmega148,uOmega148,bcs=bcOmega148,solver_parameters=params)
	solve(aOmega149==LOmega149,uOmega149,bcs=bcOmega149,solver_parameters=params)
	solve(aOmega150==LOmega150,uOmega150,bcs=bcOmega150,solver_parameters=params)
	solve(aOmega151==LOmega151,uOmega151,bcs=bcOmega151,solver_parameters=params)
	solve(aOmega152==LOmega152,uOmega152,bcs=bcOmega152,solver_parameters=params)
	solve(aOmega153==LOmega153,uOmega153,bcs=bcOmega153,solver_parameters=params)
	solve(aOmega154==LOmega154,uOmega154,bcs=bcOmega154,solver_parameters=params)
	solve(aOmega155==LOmega155,uOmega155,bcs=bcOmega155,solver_parameters=params)
	solve(aOmega156==LOmega156,uOmega156,bcs=bcOmega156,solver_parameters=params)
	solve(aOmega157==LOmega157,uOmega157,bcs=bcOmega157,solver_parameters=params)
	solve(aOmega158==LOmega158,uOmega158,bcs=bcOmega158,solver_parameters=params)
	solve(aOmega159==LOmega159,uOmega159,bcs=bcOmega159,solver_parameters=params)
	solve(aOmega160==LOmega160,uOmega160,bcs=bcOmega160,solver_parameters=params)
	solve(aOmega161==LOmega161,uOmega161,bcs=bcOmega161,solver_parameters=params)
	solve(aOmega162==LOmega162,uOmega162,bcs=bcOmega162,solver_parameters=params)
	solve(aOmega163==LOmega163,uOmega163,bcs=bcOmega163,solver_parameters=params)
	solve(aOmega164==LOmega164,uOmega164,bcs=bcOmega164,solver_parameters=params)
	solve(aOmega165==LOmega165,uOmega165,bcs=bcOmega165,solver_parameters=params)
	solve(aOmega166==LOmega166,uOmega166,bcs=bcOmega166,solver_parameters=params)
	solve(aOmega167==LOmega167,uOmega167,bcs=bcOmega167,solver_parameters=params)
	solve(aOmega168==LOmega168,uOmega168,bcs=bcOmega168,solver_parameters=params)
	solve(aOmega169==LOmega169,uOmega169,bcs=bcOmega169,solver_parameters=params)
	solve(aOmega170==LOmega170,uOmega170,bcs=bcOmega170,solver_parameters=params)
	solve(aOmega171==LOmega171,uOmega171,bcs=bcOmega171,solver_parameters=params)
	solve(aOmega172==LOmega172,uOmega172,bcs=bcOmega172,solver_parameters=params)
	solve(aOmega173==LOmega173,uOmega173,bcs=bcOmega173,solver_parameters=params)
	solve(aOmega174==LOmega174,uOmega174,bcs=bcOmega174,solver_parameters=params)
	solve(aOmega175==LOmega175,uOmega175,bcs=bcOmega175,solver_parameters=params)
	solve(aOmega176==LOmega176,uOmega176,bcs=bcOmega176,solver_parameters=params)
	solve(aOmega177==LOmega177,uOmega177,bcs=bcOmega177,solver_parameters=params)
	solve(aOmega178==LOmega178,uOmega178,bcs=bcOmega178,solver_parameters=params)
	solve(aOmega179==LOmega179,uOmega179,bcs=bcOmega179,solver_parameters=params)
	solve(aOmega180==LOmega180,uOmega180,bcs=bcOmega180,solver_parameters=params)
	solve(aOmega181==LOmega181,uOmega181,bcs=bcOmega181,solver_parameters=params)
	solve(aOmega182==LOmega182,uOmega182,bcs=bcOmega182,solver_parameters=params)
	solve(aOmega183==LOmega183,uOmega183,bcs=bcOmega183,solver_parameters=params)
	solve(aOmega184==LOmega184,uOmega184,bcs=bcOmega184,solver_parameters=params)
	solve(aOmega185==LOmega185,uOmega185,bcs=bcOmega185,solver_parameters=params)
	solve(aOmega186==LOmega186,uOmega186,bcs=bcOmega186,solver_parameters=params)
	solve(aOmega187==LOmega187,uOmega187,bcs=bcOmega187,solver_parameters=params)
	solve(aOmega188==LOmega188,uOmega188,bcs=bcOmega188,solver_parameters=params)
	solve(aOmega189==LOmega189,uOmega189,bcs=bcOmega189,solver_parameters=params)
	solve(aOmega190==LOmega190,uOmega190,bcs=bcOmega190,solver_parameters=params)
	solve(aOmega191==LOmega191,uOmega191,bcs=bcOmega191,solver_parameters=params)
	solve(aOmega192==LOmega192,uOmega192,bcs=bcOmega192,solver_parameters=params)
	solve(aOmega193==LOmega193,uOmega193,bcs=bcOmega193,solver_parameters=params)
	solve(aOmega194==LOmega194,uOmega194,bcs=bcOmega194,solver_parameters=params)
	solve(aOmega195==LOmega195,uOmega195,bcs=bcOmega195,solver_parameters=params)
	solve(aOmega196==LOmega196,uOmega196,bcs=bcOmega196,solver_parameters=params)
	solve(aOmega197==LOmega197,uOmega197,bcs=bcOmega197,solver_parameters=params)
	solve(aOmega198==LOmega198,uOmega198,bcs=bcOmega198,solver_parameters=params)
	solve(aOmega199==LOmega199,uOmega199,bcs=bcOmega199,solver_parameters=params)
	solve(aOmega200==LOmega200,uOmega200,bcs=bcOmega200,solver_parameters=params)
	solve(aOmega201==LOmega201,uOmega201,bcs=bcOmega201,solver_parameters=params)
	solve(aOmega202==LOmega202,uOmega202,bcs=bcOmega202,solver_parameters=params)
	solve(aOmega203==LOmega203,uOmega203,bcs=bcOmega203,solver_parameters=params)
	solve(aOmega204==LOmega204,uOmega204,bcs=bcOmega204,solver_parameters=params)
	solve(aOmega205==LOmega205,uOmega205,bcs=bcOmega205,solver_parameters=params)
	solve(aOmega206==LOmega206,uOmega206,bcs=bcOmega206,solver_parameters=params)
	solve(aOmega207==LOmega207,uOmega207,bcs=bcOmega207,solver_parameters=params)
	solve(aOmega208==LOmega208,uOmega208,bcs=bcOmega208,solver_parameters=params)
	solve(aOmega209==LOmega209,uOmega209,bcs=bcOmega209,solver_parameters=params)
	solve(aOmega210==LOmega210,uOmega210,bcs=bcOmega210,solver_parameters=params)
	solve(aOmega211==LOmega211,uOmega211,bcs=bcOmega211,solver_parameters=params)
	solve(aOmega212==LOmega212,uOmega212,bcs=bcOmega212,solver_parameters=params)
	solve(aOmega213==LOmega213,uOmega213,bcs=bcOmega213,solver_parameters=params)
	solve(aOmega214==LOmega214,uOmega214,bcs=bcOmega214,solver_parameters=params)
	solve(aOmega215==LOmega215,uOmega215,bcs=bcOmega215,solver_parameters=params)
	solve(aOmega216==LOmega216,uOmega216,bcs=bcOmega216,solver_parameters=params)
	solve(aOmega217==LOmega217,uOmega217,bcs=bcOmega217,solver_parameters=params)
	solve(aOmega218==LOmega218,uOmega218,bcs=bcOmega218,solver_parameters=params)
	solve(aOmega219==LOmega219,uOmega219,bcs=bcOmega219,solver_parameters=params)
	solve(aOmega220==LOmega220,uOmega220,bcs=bcOmega220,solver_parameters=params)
	solve(aOmega221==LOmega221,uOmega221,bcs=bcOmega221,solver_parameters=params)
	solve(aOmega222==LOmega222,uOmega222,bcs=bcOmega222,solver_parameters=params)
	solve(aOmega223==LOmega223,uOmega223,bcs=bcOmega223,solver_parameters=params)
	solve(aOmega224==LOmega224,uOmega224,bcs=bcOmega224,solver_parameters=params)
	solve(aOmega225==LOmega225,uOmega225,bcs=bcOmega225,solver_parameters=params)
	solve(aOmega226==LOmega226,uOmega226,bcs=bcOmega226,solver_parameters=params)
	solve(aOmega227==LOmega227,uOmega227,bcs=bcOmega227,solver_parameters=params)
	solve(aOmega228==LOmega228,uOmega228,bcs=bcOmega228,solver_parameters=params)
	solve(aOmega229==LOmega229,uOmega229,bcs=bcOmega229,solver_parameters=params)
	solve(aOmega230==LOmega230,uOmega230,bcs=bcOmega230,solver_parameters=params)
	solve(aOmega231==LOmega231,uOmega231,bcs=bcOmega231,solver_parameters=params)
	solve(aOmega232==LOmega232,uOmega232,bcs=bcOmega232,solver_parameters=params)
	solve(aOmega233==LOmega233,uOmega233,bcs=bcOmega233,solver_parameters=params)
	solve(aOmega234==LOmega234,uOmega234,bcs=bcOmega234,solver_parameters=params)
	solve(aOmega235==LOmega235,uOmega235,bcs=bcOmega235,solver_parameters=params)
	solve(aOmega236==LOmega236,uOmega236,bcs=bcOmega236,solver_parameters=params)
	solve(aOmega237==LOmega237,uOmega237,bcs=bcOmega237,solver_parameters=params)
	solve(aOmega238==LOmega238,uOmega238,bcs=bcOmega238,solver_parameters=params)
	solve(aOmega239==LOmega239,uOmega239,bcs=bcOmega239,solver_parameters=params)
	solve(aOmega240==LOmega240,uOmega240,bcs=bcOmega240,solver_parameters=params)
	solve(aOmega241==LOmega241,uOmega241,bcs=bcOmega241,solver_parameters=params)
	solve(aOmega242==LOmega242,uOmega242,bcs=bcOmega242,solver_parameters=params)
	solve(aOmega243==LOmega243,uOmega243,bcs=bcOmega243,solver_parameters=params)
	solve(aOmega244==LOmega244,uOmega244,bcs=bcOmega244,solver_parameters=params)
	solve(aOmega245==LOmega245,uOmega245,bcs=bcOmega245,solver_parameters=params)
	solve(aOmega246==LOmega246,uOmega246,bcs=bcOmega246,solver_parameters=params)
	solve(aOmega247==LOmega247,uOmega247,bcs=bcOmega247,solver_parameters=params)
	solve(aOmega248==LOmega248,uOmega248,bcs=bcOmega248,solver_parameters=params)
	solve(aOmega249==LOmega249,uOmega249,bcs=bcOmega249,solver_parameters=params)
	solve(aOmega250==LOmega250,uOmega250,bcs=bcOmega250,solver_parameters=params)
	solve(aOmega251==LOmega251,uOmega251,bcs=bcOmega251,solver_parameters=params)
	solve(aOmega252==LOmega252,uOmega252,bcs=bcOmega252,solver_parameters=params)
	solve(aOmega253==LOmega253,uOmega253,bcs=bcOmega253,solver_parameters=params)
	solve(aOmega254==LOmega254,uOmega254,bcs=bcOmega254,solver_parameters=params)
	solve(aOmega255==LOmega255,uOmega255,bcs=bcOmega255,solver_parameters=params)
	solve(aOmega256==LOmega256,uOmega256,bcs=bcOmega256,solver_parameters=params)
end_time2=time.time()
print("{},{},{},{}".format(N,N_subdom,end_time1-start_time1,end_time2-start_time2))
# does this line exist?
# perhaps some other lines after the fact?
