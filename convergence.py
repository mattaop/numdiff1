import numpy as np
import matplotlib.pyplot as plt
import simple_lax as sl

m=8 #2^m, gives number of points in space
max_time=5*60 #seconds
T=1000
X=2**(m)
V0=120
rho_max=140
E=100
rho0=40
L=5000 #meter
delta_t=0.001
delta_x=L/(X-1)  #meter

sigma=56.7
my=600
tau=0.5
c=54
grid_rho=np.zeros((T,X))
grid_v=np.zeros((T,X))
grid_rho[0]=np.ones(X)*rho0
grid_v[0]=sl.safe_v(grid_rho[0], V0, rho_max, E)



def spatial_convergence(m,solver,grid_rho, grid_v, T, X, rho0,delta_t,delta_x,L,sigma,V0,rho_max,E,tau,c,my):
    convergence_list=np.zeros((2,m+1))
    print(convergence_list)
    rho_exact,v_exact=solver(grid_rho, grid_v, T, X, rho0,delta_t,delta_x,L,sigma,V0,rho_max,E,tau,c,my)    # T,delta_t
    x_list = np.linspace(-L, L, len(rho_exact[0]))
    #plt.plot(x_list,rho_exact[-1])
    #plt.plot(x_list, v_exact[-1])
    #plt.show()
    exact_list=np.array([rho_exact[-1],v_exact[-1]])
    #print(exact_list)
    step_length_list=np.zeros(m+1)
    for i in range(1,m+1):
        len_points = 2 ** i
        new_exact_list=np.zeros((2,len_points))
        ratio=(len(exact_list[0])-1)/(len_points-1)

        for j in range(len_points):
            new_exact_list[:,j]=exact_list[:,int(j*ratio)]

        #print(new_exact_list)
        delta_x=L/(len_points-1)    #with endpoints
        print(delta_x)
        step_length_list[i-1]=delta_x
        grid_rho = np.zeros((T, len_points))
        grid_v = np.zeros((T, len_points))
        grid_rho[0] = np.ones(len_points) * rho0
        grid_v[0] = sl.safe_v(grid_rho[0], V0, rho_max, E)

        rho_i,v_i=solver(grid_rho, grid_v, T, len_points, rho0,delta_t,delta_x,L,sigma,V0,rho_max,E,tau,c,my)
        i_list = np.array([rho_i[-1], v_i[-1]])

        convergence_list[0][i-1]=np.linalg.norm(new_exact_list[0]-i_list[0],np.inf)
        convergence_list[1][i-1] = np.linalg.norm(new_exact_list[1] - i_list[1],np.inf)
        print(len(convergence_list[0]), len(step_length_list))
        x_list2=np.linspace(-L,L,len(new_exact_list[0]))
        plt.plot(x_list,exact_list[0])
        plt.plot(x_list2,i_list[0])
        plt.show()
        #print(convergence_list)

    return convergence_list,step_length_list




def plot_convergence():
    conv_list,step_length_list=spatial_convergence(m,sl.solve_simple_lax,grid_rho, grid_v, T, X, rho0,delta_t,delta_x,L,sigma,V0,rho_max,E,tau,c,my)
    print(len(conv_list),len(step_length_list))
    plt.loglog(step_length_list,conv_list[0])
    plt.loglog(step_length_list,conv_list[1])
    plt.show()

plot_convergence()



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


