import numpy as np
import matplotlib.pyplot as plt
import constants as c


def f2(u_last, u_m):
    f_step = np.zeros(2)
    f_step[:] = 0, c.C **2*u_last[0]/u_m[0]
    return f_step

def f(u_last):
    f_step = np.zeros(2)
    f_step[:] = u_last[0]*u_last[1], (u_last[1]**2)/2
    return f_step

def s(time, position, u_last, delta_t, delta_x, j, tau, V0, my, rho_max, E):
    s_step = np.zeros(2)
    s_step[:] = c.q_in(time)*c.phi(position), (1/tau)*((V0*(1-u_last[j,0]/rho_max))/(1+E*(u_last[j,0]/rho_max)**4)
                                                         - u_last[j,1])
    return s_step

def u_next_half_step(u_last, delta_t, delta_x, j, time, position, tau, V0, my, rho_max, E):
    return (u_last[j+1] + u_last[j] - delta_t /delta_x*(f(u_last[j+1]) - f(u_last[j])) \
           - delta_t/delta_x*(f2(u_last[j+1], u_last[j])- f2(u_last[j],u_last[j]))
           + delta_t/2*(s(time, position, u_last, delta_t, delta_x, j+1, tau,  V0, my,rho_max, E)\
           + s(time, position, u_last, delta_t, delta_x, j, tau, V0, my,rho_max, E)))/2


def u_next_lax_wendroff(u_last, u_halfstep, delta_t, delta_x, j, time, position, tau, V0, my, rho_max, E):
    u_halfstep[j] = u_next_half_step(u_last, delta_t, delta_x, j, time, position, tau, V0, my, rho_max, E)
    return u_last[j] - delta_t/delta_x*(f(u_halfstep[j])-f(u_halfstep[j-1])) \
           - delta_t/delta_x*(f2(u_halfstep[j+1],u_halfstep[j])- f2(u_halfstep[j], u_halfstep[j]))\
           + delta_t/2*(s(time, position, u_halfstep, delta_t, delta_x, j, tau, V0, my, rho_max, E)\
           + s(time, position, u_halfstep, delta_t, delta_x, j-1, tau, V0, my, rho_max, E))

def one_step_lax_wendroff(u_last, X, delta_t, delta_x ,time, rho0, L, tau, V0, my, rho_max, E):
    u_next = np.zeros((X,2))
    u_next[0,:] = rho0, c.safe_v(rho0)
    u_halfstep = np.zeros((X,2))
    u_halfstep[0,:] = u_next_half_step(u_last, delta_t, delta_x, 0, time, -L/2, tau, V0, my, rho_max, E)
    for j in range(1,X-1):
        position=j*delta_x-L
        u_next[j] = u_next_lax_wendroff(u_last, u_halfstep, delta_t, delta_x, j, time, position, tau, V0, my, rho_max, E)
    u_next[X-1]=2*u_next[X-2]-u_next[X-3]
    return u_next

def solve_lax_wendroff(T, X, delta_t, delta_x):
    rho0, L, tau , V0, my, rho_max, E = c.RHO_0, c.L, c.TAU, c.V0, c.MY, c.RHO_MAX, c.E
    grid_u = c.initialize_grid(T, X, rho0)
    for i in range(1, T):
        #print(i)
        time=i*delta_t
        grid_u[i]=one_step_lax_wendroff(grid_u[i-1], X, delta_t, delta_x, time, rho0, L, tau, V0, my, rho_max, E)
    return grid_u

def plot_lax_wendroff(T, X, delta_x, grid_u):
    x=np.linspace(-X*delta_x,X*delta_x,X)
    plt.figure()
    plt.plot(x,grid_u[T-1])
    plt.show()

def main():
    grid_u = solve_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)
    plot_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:,:,0])
    plot_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:,:,1])
main()
