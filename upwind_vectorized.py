import numpy as np
import matplotlib.pyplot as plt

def safe_v(rho, V0, rho_max, E):
    return V0*(1-rho/rho_max)/(1+E*(rho/rho_max)**4)

def q_in(time):
    return 40

def phi(x, sigma):
    return (2*np.pi*sigma**2)**(-0.5)*np.exp(-x**2/(2*sigma**2))

def f(u_last, c, j):
    f_step = np.zeros(2)
    f_step[:] = u_last[j,0]*u_last[j,1], (u_last[j,1]**2)/2 + c**2*np.log(u_last[j,0])
    return f_step

def s(time, position, sigma, tau, u_last, rho_max, delta_t, V0, j ,E):
    s_step = np.zeros(2)
    s_step[:] = q_in(time)*phi(position,sigma), (1/tau)*((V0*(1-u_last[j,0]/rho_max))/(1+E*(u_last[j,0]/rho_max)**4)-u_last[j,1])
    return s_step

def u_next_upwind(u_last, delta_t, delta_x, j, time, position, sigma, tau, V0, rho_max, E, c, my):
    return u_last[j] - delta_t/delta_x*(f(u_last,c,j)-f(u_last,c,j-1)) + delta_t*s(time, position, sigma, tau, u_last, rho_max, delta_t, V0, j, E)


def one_step_upwind(u_last, X, delta_t, delta_x ,time, L, sigma, rho0, V0, rho_max, E, tau, c, my):
    u_next = np.zeros((X,2))
    u_next[0,:] = rho0, safe_v(rho0, V0, rho_max, E)
    for j in range(1,X-1):
        position=j*delta_x-L/2
        u_next[j] = u_next_upwind(u_last, delta_t, delta_x, j, time, position, sigma, tau, V0, rho_max, E, c, my)
    u_next[X-1]=2*u_next[X-2]-u_next[X-3]

    return u_next

def solve_upwind(grid_u, T, X, rho0,delta_t,delta_x,L,sigma,V0,rho_max,E,tau,c,my):
    for i in range(1,T):
        time=i*delta_t
        grid_u[i]=one_step_upwind(grid_u[i-1], X, delta_t,delta_x ,time,L,sigma, rho0,V0,rho_max,E,tau,c,my)
    return grid_u

def plot_upwind(T,X,delta_x,grid_rho):
    x=np.linspace(-X*delta_x,X*delta_x,X)
    plt.plot(x,grid_rho[T-1])
    plt.show()

def main():
    T = 100
    X = 100
    V0 = 80
    rho_max = 140
    E = 100
    rho0 = 10
    delta_t = 0.01
    delta_x = 20  # meter
    L = delta_x * X  # meter
    sigma = 56.7
    my = 600
    tau = 0.5
    c = 54

    grid_u = np.zeros((T,X,2))
    grid_u[0,:,0] = np.ones(X)*rho0
    grid_u[0,:,1] = safe_v(grid_u[0,:,0], V0, rho_max, E)
    grid_u = solve_upwind(grid_u, T, X, rho0, delta_t, delta_x, L, sigma, V0, rho_max, E, tau, c, my)

    plot_upwind(T, X, delta_x, grid_u[:,:,0])
main()
