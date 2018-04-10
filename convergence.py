import numpy as np
import matplotlib.pyplot as plt
import simple_lax as sl

m=8 #2^m, gives number of points in space

def spatial_convergence(m,solver):
    convergence_list=np.zeros(m)
    rho_exact,v_exact=solver
    for i in range(m):

        convergence_list[i]=calculate_spatial_error(i,)
    return 0

def calculate_spatial_error(iteration):
    return 0
"""
#print(U_exact, U)
    len1=len(U)
    X=np.linspace(0, 1, len1)
    #plt.figure()
    #plt.plot(X, U_exact, label="Exact")
    #plt.plot(X, U, label="approx")
    #plt.legend()
    #plt.show()
    len2=len(U_exact)
    number=(len2-1)/(len1-1)
    newlist=np.zeros(len1)
    for i in range(len1):
        newlist[i]=U_exact[int(i*number)]
"""
















def time_convergence():
    return 0


