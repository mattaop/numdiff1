import numpy as np
import matplotlib.pyplot as plt
import constants as c

def initialize_grid(T, X, rho0):
    grid_u = np.zeros((T, X, 2))
    grid_u[0, :, 0] = np.ones(X) * rho0
    grid_u[0, :, 1] = safe_v(grid_u[0, :, 0])
    return grid_u

def safe_v(rho):
    return c.V0 * (1 - rho / c.RHO_MAX) / (1 + c.E * (rho / c.RHO_MAX) ** 4)

def q_in(time):
    return 1000

def phi(x):
    return (2*np.pi*c.SIGMA**2)**(-0.5)*np.exp(-x**2/(2*c.SIGMA**2))

def f(u_last):
    f_step = np.zeros(2)
    f_step[:] = u_last[0]*u_last[1], (u_last[1]**2)/2 + c.C**2*np.log(u_last[0])
    return f_step

def s(time, position, u_last, delta_t, delta_x, j, tau, V0, my, rho_max, E):
    s_step = np.zeros(2)
    s_step[:] = q_in(time)*phi(position), (1/tau)*((V0*(1-u_last[j,0]/rho_max))/(1+E*(u_last[j,0]/rho_max)**4)
                                                         - u_last[j,1])+my*delta_t*(u_last[j+1,1]-2*u_last[j,1]
                                                         + u_last[j-1,1])/(u_last[j,0]*delta_x**2)
    return s_step

def u_next_simple_lax(u_last, delta_t, delta_x, j, time, position, tau, V0, my, rho_max, E):
    return (u_last[j+1]+u_last[j-1])/2 - delta_t/(2*delta_x)*(f(u_last[j+1])-f(u_last[j-1])) + delta_t*s(time, position, u_last, delta_t, delta_x, j, tau, V0, my, rho_max, E)

def one_step_simple_lax(u_last, X, delta_t, delta_x ,time, rho0, L, tau, V0, my, rho_max, E):
    u_next = np.zeros((X,2))
    u_next[0,:] = rho0, safe_v(rho0)
    for j in range(1,X-1):
        position=j*delta_x-L/2
        u_next[j] = u_next_simple_lax(u_last, delta_t, delta_x, j, time, position, tau, V0, my, rho_max, E)
    u_next[X-1]=2*u_next[X-2]-u_next[X-3]
    return u_next

def solve_simple_lax(T, X, delta_t, delta_x):
    rho0, L, tau , V0, my, rho_max, E = c.RHO_0, c.L, c.TAU, c.V0, c.MY, c.RHO_MAX, c.E
    grid_u = initialize_grid(T, X, rho0)
    for i in range(1, T):
        time=i*delta_t
        grid_u[i]=one_step_simple_lax(grid_u[i-1], X, delta_t, delta_x, time, rho0, L, tau, V0, my, rho_max, E)
    return grid_u

def plot_simple_lax(T, X, delta_x, grid_u):
    x=np.linspace(-X*delta_x,X*delta_x,X)
    plt.figure()
    plt.plot(x,grid_u[T-1])
    plt.show()

def main():
    grid_u = solve_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)
    plot_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:,:,0])
    plot_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:,:,1])
main()
