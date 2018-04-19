import numpy as np
import matplotlib.pyplot as plt
import constants as c
import functions as func

def u_approx_mac_cormack(u_last, delta_t, delta_x, j, time, position):
    return u_last[j] - delta_t/delta_x*(func.f(u_last[j]) - func.f(u_last[j-1])) \
               + delta_t*func.g(u_last, delta_x, j)\
               + delta_t*func.s(time, position+delta_x, u_last, j)

def u_next_mac_cormack(u_last, u_approx, delta_t, delta_x, j, time, position):
    return (u_approx[j]+u_last[j]- (delta_t)/(2*delta_x)*(func.f(u_approx[j+1])- func.f(u_approx[j]))\
            + delta_t*func.g(u_approx, delta_x, j)\
            + delta_t*func.s(time, position, u_approx, j))/2

def one_step_mac_cormack(u_last, X, delta_t, delta_x ,time, rho0, L):
    u_next = np.zeros((X,2))
    u_next[0,:] = u_last[1,0], func.safe_v(u_last[1,0])
    u_approx = np.zeros((X, 2))
    u_approx[0] = u_approx_mac_cormack(u_last, delta_t, delta_x, 0, time, delta_x)
    u_approx[1] = u_approx_mac_cormack(u_last, delta_t, delta_x, 1, time, delta_x-L/2)
    for j in range(1,X-2):
        position=j*delta_x-L/2
        u_approx[j+1] = u_approx_mac_cormack(u_last, delta_t, delta_x, j+1, time, position + delta_x)
        u_next[j] = u_next_mac_cormack(u_last, u_approx, delta_t, delta_x, j, time, position)
        u_next[j][0] = min(c.RHO_MAX, u_next[j][0])
        u_next[j][1] = max(0, u_next[j][1])
    u_next[X-2]=2*u_next[X-3]-u_next[X-4]
    u_next[X-1]=2*u_next[X-2]-u_next[X-3]
    return u_next

def solve_mac_cormack(T, X, MAX_TIME):
    rho0, L = c.RHO_0, c.L
    grid_u = func.initialize_grid(T, X, rho0)
    delta_x = L/(X-1)
    delta_t = MAX_TIME/(T-1)
    for i in range(1, T):
        time=i*delta_t
        grid_u[i]=one_step_mac_cormack(grid_u[i-1], X, delta_t, delta_x, time, rho0, L)
    return grid_u

def plot_mac_cormack(T, X, delta_x, grid_u):
    x=np.linspace(-X*delta_x,X*delta_x,X)
    plt.figure()
    plt.plot(x,grid_u[T-1])
    plt.show()

def main():
    grid_u = solve_mac_cormack(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME)
    plot_mac_cormack(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 0])
    plot_mac_cormack(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 1])

#main()
