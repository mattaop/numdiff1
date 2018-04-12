import numpy as np
import matplotlib.pyplot as plt
import simple_lax as sl
import simple_lax_vectorized as sl_v

import upwind_vectorized as up_v

import upwind as up
import constants as c


#solve_simple_lax(time_points, space_points, rho0, delta_t,delta_x)
#Laget kun for å funke for Simple-Lax foreløpig:
def time_error(solver, space_points, delta_x,T_max,T_ex,u_ex):
    n = 12 #
    error_list_rho = np.zeros(n)
    error_list_v = np.zeros(n)
    delta_t_list = np.zeros(n)
    for i in range(n):
        time_points = 2**(i+1) #Number of time points in each iteration
        delta_t = T_max/(time_points-1) #delta t in each iteration
        print(delta_t)
        u = solver(time_points, space_points, delta_t,delta_x)
        error_rho = u_ex[:,:,0][T_ex-1]-u[:,:,0][time_points-1]
        error_v = u_ex[:,:,1][T_ex-1]-u[:,:,1][time_points-1]
        error_list_rho[i] = np.sqrt(delta_x*delta_t)*np.linalg.norm(error_rho,2)
        error_list_v[i] = np.sqrt(delta_x*delta_t)*np.linalg.norm(error_v,2)
        delta_t_list[i] = delta_t
    return delta_t_list,error_list_rho,error_list_v

def time_convergence():

    T_max = 5 #Time (minutes?) until we stop the simulation
    T_ex = 10000 #Number of time steps in the reference (exact) solution 
    
    delta_t_min = T_max/(T_ex-1) #The delta T-value used in the exact solution
    
    u_ex = sl_v.solve_simple_lax(T_ex, c.SPACE_POINTS,delta_t_min,c.delta_x)
    
    delta_t_list,error_rho,error_v = time_error(sl_v.solve_simple_lax,c.SPACE_POINTS,c.delta_x,T_max, T_ex,u_ex)
    print(error_rho)
    print(error_v)
    
    print(delta_t_list)
    plt.figure()
    plt.plot(delta_t_list,error_rho,label=r"$\rho$")
    plt.plot(delta_t_list,error_v,label= "v")
    plt.title("Convergence plot in time")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("Error")
    plt.semilogx()
    plt.semilogy()
    plt.legend()
    plt.show()  
    
def plot_time_convergence(solver)
time_convergence()