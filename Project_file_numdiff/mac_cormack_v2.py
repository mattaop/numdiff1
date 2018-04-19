import numpy as np
import matplotlib.pyplot as plt
import constants as c


def g(u_last, u_m, delta_x):
    g_step = np.zeros(2)
    g_step[:] = 0, - c.C **2*u_last[0]/u_m[0]/delta_x
    # +c.MY*(u_last[j+1,1]-2*u_last[j,1]+ u_last[j-1,1])/(u_last[j,0]*delta_x**2)
    return g_step

def f(u_last):
    f_step = np.zeros(2)
    f_step[:] = u_last[0]*u_last[1], (u_last[1]**2)/2
    return f_step

def s(time, position, u_last, delta_t, delta_x, j):
    s_step = np.zeros(2)
    s_step[:] = c.q_in(time)*c.phi(position), \
                (1/c.TAU)*((c.V0*(1-u_last[j,0]/c.RHO_MAX))/(1+c.E*(u_last[j,0]/c.RHO_MAX)**4)- u_last[j,1])
    return s_step

def u_approx_mac_cormack(u_last, delta_t, delta_x, j, time, position):
    return u_last[j] - delta_t / delta_x * (f(u_last[j+1]) - f(u_last[j])) \
               + delta_t*(g(u_last[j+1], u_last[j], delta_x)-g(u_last[j], u_last[j], delta_x))\
               + delta_t * s(time, position+delta_x, u_last, delta_t, delta_x, j)

def u_next_mac_cormack(u_last, u_approx, delta_t, delta_x, j, time, position):
    return (u_approx[j]+u_last[j]- (delta_t)/(2*delta_x)*(f(u_approx[j])- f(u_approx[j-1])) \
            + delta_t/(2)*(g(u_approx[j+1], u_approx[j], delta_x)-g(u_approx[j], u_approx[j], delta_x))\
            + delta_t*s(time, position, u_approx, delta_t, delta_x, j))/2

def one_step_mac_cormack(u_last, X, delta_t, delta_x ,time, rho0, L):
    u_next = np.zeros((X,2))
    u_next[0,:] = rho0, c.safe_v(rho0)
    u_approx = np.zeros((X, 2))
    u_approx[1] = u_approx_mac_cormack(u_last, delta_t, delta_x, 1, time, delta_x-L / 2)
    for j in range(1,X-2):
        position=j*delta_x-L/2
        u_approx[j+1] = u_approx_mac_cormack(u_last, delta_t, delta_x, j+1, time, position + delta_x)
        u_next[j] = u_next_mac_cormack(u_last, u_approx, delta_t, delta_x, j, time, position)
        u_next[j][0] = min(c.RHO_MAX, u_next[j][0])
        u_next[j][1] = max(0, u_next[j][1])
    u_next[X-2]=2*u_next[X-3]-u_next[X-4]
    u_next[X-1]=2*u_next[X-2]-u_next[X-3]
    return u_next

def solve_mac_cormack(T, X, delta_t, delta_x):
    rho0, L = c.RHO_0, c.L
    grid_u = c.initialize_grid(T, X, rho0)
    for i in range(1, T):
        #print(i)
        time=i*delta_t
        grid_u[i]=one_step_mac_cormack(grid_u[i-1], X, delta_t, delta_x, time, rho0, L)
    return grid_u

def plot_mac_cormack(T, X, delta_x, grid_u):
    x=np.linspace(-X*delta_x,X*delta_x,X)
    plt.figure()
    plt.plot(x,grid_u[T-1])
    plt.show()

def main():
    grid_u = solve_mac_cormack(c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)
    plot_mac_cormack(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 0])
    plot_mac_cormack(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 1])

main()
