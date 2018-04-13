import numpy as np
import matplotlib.pyplot as plt
import constants as c


def f(u_last):
    f_step = np.zeros(2)
    f_step[:] = u_last[0]*u_last[1], (u_last[1]**2)/2 + c.C**2*np.log(u_last[0])
    return f_step

def s(time, position, u_last, delta_t, delta_x, j, tau, V0, my, rho_max, E):
    s_step = np.zeros(2)
    s_step[:] = c.q_in(time)*c.phi(position), (1/tau)*((V0*(1-u_last[j,0]/rho_max))/(1+E*(u_last[j,0]/rho_max)**4)
                                                         - u_last[j,1])
    return s_step

def u_approx_mac_cormack(u_last, delta_t, delta_x, j, time, position, tau, V0, my, rho_max, E):
    return u_last[j] - delta_t/delta_x*(f(u_last[j])-f(u_last[j-1]))\
               +delta_t*s(time, position, u_last, delta_t, delta_x, j, tau, V0, my, rho_max, E)

def u_next_mac_cormack(u_last, u_approx, delta_t, delta_x, j, time, position, tau, V0, my, rho_max, E):
    u_approx[j+1] = u_approx_mac_cormack(u_last, delta_t, delta_x, j+1, time, position, tau, V0, my, rho_max, E)
    return (u_approx[j]+u_last[j]- delta_t/delta_x*(f(u_approx[j+1])-f(u_approx[j]))
            +delta_t*s(time, position, u_approx, delta_t, delta_x, j, tau, V0, my, rho_max, E))/2

def one_step_mac_cormack(u_last, X, delta_t, delta_x ,time, rho0, L, tau, V0, my, rho_max, E):
    u_next = np.zeros((X,2))
    u_next[0,:] = rho0, c.safe_v(rho0)
    u_approx = np.zeros((X, 2))
    u_approx[0, :] = u_approx_mac_cormack(u_last, delta_t, delta_x, 0, time, -L, tau, V0, my, rho_max, E)
    for j in range(1,X-1):
        position=j*delta_x-L
        u_next[j] = u_next_mac_cormack(u_last, u_approx, delta_t, delta_x, j, time, position, tau, V0, my, rho_max, E)
    u_next[X-1]=2*u_next[X-2]-u_next[X-3]
    return u_next

def solve_mac_cormack(T, X, delta_t, delta_x):
    rho0, L, tau , V0, my, rho_max, E = c.RHO_0, c.L, c.TAU, c.V0, c.MY, c.RHO_MAX, c.E
    grid_u = c.initialize_grid(T, X, rho0)
    for i in range(1, T):
        #print(i)
        time=i*delta_t
        grid_u[i]=one_step_mac_cormack(grid_u[i-1], X, delta_t, delta_x, time, rho0, L, tau, V0, my, rho_max, E)
    return grid_u

def plot_mac_cormack(T, X, delta_x, grid_u):
    x=np.linspace(-X*delta_x,X*delta_x,X)
    plt.figure()
    plt.plot(x,grid_u[T-1])
    plt.show()

def main():
    grid_u = solve_mac_cormack(c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)
    plot_mac_cormack(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:,:,0])
    plot_mac_cormack(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:,:,1])
main()
