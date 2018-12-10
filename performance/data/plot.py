import matplotlib.pyplot as plt
import numpy as np
n_subdom=np.array([2,4,8,16,32,64])
n_subdom=n_subdom**2.
times_direct=np.array([0.2395,0.29485,0.25468])
times_iter=np.array([0.12351298,0.6867,3.8147])
ns=np.array([4,16,64])
lines=np.array([173,509,1901,7309,29389,118093])
lines_abst=np.array([76,76,76,76,76,76])
# Uncomment for time to solution plot
#plt.loglog(ns,times_direct, label="Direct solution time",basex=2,linewidth=3)
#plt.loglog(ns,times_iter, label="Domain decomposition solution time",basex=2,linewidth=3)
#plt.legend()
#plt.title("Execution time for direct method versus Alternating Schwarz")
#plt.xlabel("Number of subdomains")
#plt.ylabel("Execution time, s")
#plt.grid()
#plt.show()
# Uncomment for complexity plot
#plt.loglog(n_subdom,lines, label="Lines output",basex=2)
#plt.loglog(n_subdom,lines_abst, label="Lines input",basex=2)
#plt.legend()
#plt.title("Complexity of generated output code as a function of subdomains")
#plt.xlabel("Number of subdomains")
#plt.ylabel("Number of lines")
#plt.grid()
#plt.show()

# Uncomment for parallel execution plot
nelems=np.array([128,384,1280,4608,17408,67584,266240])
#nelems=np.array([17408,67584,266240])
data=np.array([
        0.2861051559448242,0.35235261,
        0.26036620140075684,0.35889911651611,
        0.2926771640777588,0.470905065536499,
        0.32793378829956055,0.52205467224121,
        0.2260420322418213,0.290619611740112,
        0.2357163429260254,0.370650291442871,
        0.30976366996765137,0.40284156799316,
        0.9006919860839844,0.582341909408569,
        0.25136780738830566,0.31157279014587,
        0.2580869197845459,0.428766489028930,
        0.5075592994689941,0.514225721359252,
        0.6142938137054443,0.587920665740966,
        0.31852197647094727,0.53030896186828,
        0.306685209274292,0.5662755966186523,
        0.3787832260131836,0.661444664001464,
        0.4463632106781006,0.738361835479736,
        0.7018179893493652,1.635366678237915,
        0.6834075450897217,1.503889322280883,
        0.8878288269042969,1.857781410217285,
        0.9698021411895752,1.820297241210937,
        3.5174331665039062,8.341603755950928,
        3.8931384086608887,8.545975685119629,
        4.3235907554626465,9.615485429763794,
        5.351849317550659,12.53217887878418,
        19.31284785270691,42.58852005004883,
        22.09125328063965,47.3184654712677,
        26.699352264404297,54.45480394363403,
        33.17833089828491,70.13560962677002
        ])

# 8 data points for each element type

direct_sol=np.zeros(7)
iterative_sol=np.zeros(7)
for i in range(nelems.shape[0]):
    num_elems=str(nelems[i])
    #direct_sol=np.zeros(4)
    direct_sol[i]=data[8*i]
    iterative_sol[i]=data[8*i+1]
    #direct_sol[0]=data[8*i+0]
    #direct_sol[1]=data[8*i+2]
    #direct_sol[2]=data[8*i+4]
    #direct_sol[3]=data[8*i+6]

    #iterative_sol=np.zeros(4)
    #iterative_sol[0]=data[8*i+1]
    #iterative_sol[1]=data[8*i+1]
    #iterative_sol[2]=data[8*i+1]
    #iterative_sol[3]=data[8*i+1]
    #plt.loglog([1,2,3,4],direct_sol,label="Direct solution time, nElems={}".format(num_elems), basex=2)
    #plt.loglog([1,2,3,4],iterative_sol,label="Alternating Schwarz solution time, nElems={}".format(num_elems), basex=2)

plt.loglog(nelems,direct_sol,label="Direct solution time",linewidth=3)
plt.loglog(nelems,iterative_sol,label="Alternating Schwarz solution time",linewidth=3)
plt.title("Scaling performance plot, function of number of mesh elements")
plt.xlabel("Number of elements")
plt.ylabel("Execution time, s")
plt.legend()
plt.grid()
plt.show()
