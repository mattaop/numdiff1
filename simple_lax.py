import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import constants as c
from matplotlib import cm

def safe_v(rho):
    return c.V0*(1-rho/c.RHO_MAX)/(1+c.E*(rho/c.RHO_MAX)**4)

def q_in(time):
    if time>100:
        return 2
    else:
        return 0
def phi(x):
	return (2*np.pi*c.SIGMA**2)**(-0.5)*np.exp(-x**2/(2*c.SIGMA**2))


def rho_next_simple_lax(rho_last, v_last, delta_t, delta_x, j, time, position):
    return  min(c.RHO_MAX,1/2*(rho_last[j+1]+rho_last[j-1]) - v_last[j]*delta_t/(2*delta_x)*(rho_last[j+1]-rho_last[j-1])-\
            rho_last[j]*delta_t/(2*delta_x)*(v_last[j+1]- v_last[j-1]) + delta_t*q_in(time)*phi(position))

def v_next_simple_lax(rho_last, v_last, delta_t, delta_x,j):
    return max(0,((v_last[j+1]+v_last[j-1])/2-v_last[j]*delta_t*(v_last[j+1]-v_last[j-1])/(2*delta_x))\
           +(delta_t/c.TAU)*((c.V0*(1-rho_last[j]/c.RHO_MAX))/(1+c.E*(rho_last[j]/c.RHO_MAX)**4)-v_last[j])-\
           (rho_last[j+1]-rho_last[j-1])*delta_t/(rho_last[j]*2*delta_x)*c.C**2+c.MY*delta_t*(v_last[j+1]-2*v_last[j]+v_last[j-1])/(rho_last[j]*delta_x**2))
"""
def f2_simple_lax(rho_last, v_last, rho_temp, v_temp, delta_t, delta_x,j, tau, V0, rho_max, E, c, my):
    return rho_last[j]*(v_temp/delta_t - (v_last[j+1]+v_last[j-1])/(2*delta_t)+v_last[j]*(v_last[j+1]-v_last[j-1])/(2*delta_x))\
           -(rho_last[j]/tau)*((V0*(1-rho_last[j]/rho_max))/(1+E*(rho_last[j]/rho_max)**4)-v_last[j])+\
           (rho_last[j+1]-rho_last[j-1])/(2*delta_x)*c**2-my*(v_last[j+1]-2*v_last[j]+v_last[j+1])/delta_x**2

def f1_simple_lax(rho_last, v_last, rho_temp, v_temp, delta_t, delta_x, j, time, position, sigma):
    return rho_temp / delta_t - 1 / (2 * delta_t) * (rho_last[j + 1] + rho_last[j - 1])\
           + v_last[j] / (2 * delta_x) * (rho_last[j + 1] - rho_last[j - 1]) + rho_last[j] / (2 * delta_x) * (
                       v_last[j + 1] - v_last[j - 1])\
           - q_in(time) * phi(position, sigma)
"""
#no need
'''def jacobi_matrix(rho_last, v_last, rho_temp, v_temp, delta_t, delta_x, j, V0, tau, rho_max, E):
    j1=1/delta_t+(v_last[j+1]-v_last[j-1])/(2*delta_x)
    j2=(rho_last[j+1]-rho_last[j-1])/(2*delta_x)
    j3=(v_temp/delta_t - (v_last[j+1]+v_last[j-1])/(2*delta_t)+v_temp*(v_last[j+1]-v_last[j-1])/(2*delta_x))-V0/tau*(rho_max**3*(rho_max**5-2*rho_max**4*rho_temp-3*rho_max*E*rho_temp**4+2*E*rho_temp**5)/(rho_max**4+E*rho_temp**4)**2)+v_temp/tau
    j4=rho_temp/delta_t+rho_temp*(v_last[j+1]-v_last[j-1])/(2*delta_x)+rho_temp/tau
    jacobi=np.matrix([[j1, j2],[j3,j4]])
    return jacobi'''

'''def newtons(rho_last, v_last, delta_t, delta_x, j, V0, tau, rho_max, E):
    rho_temp, v_temp=rho_last, v_last
    f_vector=np.ndarray([f1_simple_lax, f2_simple_lax])
    while max(f_vector[0], f_vector[1])>1e-4:
        jacobi=jacobi_matrix(rho_last, v_last, rho_temp, v_temp, delta_t, delta_x, j, V0, tau, rho_max, E)
        delta_X=np.'''


def one_step_simple_lax(rho_last, v_last, X, delta_t,delta_x ,time):
    rho_next=np.zeros(X)
    v_next=np.zeros(X)
    rho_next[0]=rho_last[1]
    v_next[0]=safe_v(rho_last[1])
    for j in range(1,X-1):
        position=j*delta_x-c.L/2
        rho_next[j], v_next[j]=rho_next_simple_lax(rho_last,v_last,delta_t,delta_x, j ,time,position),\
                                 v_next_simple_lax(rho_last,v_last,delta_t,delta_x, j)
    rho_next[X-1]=2*rho_next[X-2]-rho_next[X-3]
    v_next[X-1]=2*v_next[X-2]-v_next[X-3]

    return rho_next, v_next

def solve_simple_lax(grid_rho, grid_v, T, X,delta_t,delta_x):

    for i in range(1,T):
        time=i*delta_t
        grid_rho[i], grid_v[i]=one_step_simple_lax(grid_rho[i-1],grid_v[i-1],X,
                                                   delta_t,delta_x ,time)
    return grid_rho,grid_v

def plot_simple_lax(T,delta_t,X,delta_x,grid_rho,grid_v):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x=np.arange(-X*delta_x/2,X*delta_x/2,delta_x)
    y=np.arange(0,T*delta_t,delta_t)
    x,y=np.meshgrid(x,y)
    ax.plot_surface(x, y, grid_rho,cmap=cm.coolwarm)

    plt.show()

def plot_simple_lax_3d(T,delta_t,X,delta_x,grid_rho,grid_v):

    x=np.linspace(-X*delta_x,X*delta_x,X)
    plt.plot(x,grid_rho[T-1])
    plt.show()
    plt.figure()
    plt.plot(x,grid_v[T-1])
    plt.show()

def main():
    grid_rho=np.zeros((c.TIME_POINTS,c.SPACE_POINTS))
    grid_v=np.zeros((c.TIME_POINTS,c.SPACE_POINTS))
    grid_rho[0]=np.ones(c.SPACE_POINTS)*c.RHO_0
    grid_v[0]=safe_v(grid_rho[0])
    solve_simple_lax(grid_rho, grid_v, c.TIME_POINTS, c.SPACE_POINTS,c.delta_t,c.delta_x)
    #print(grid_v)
    #print(grid_rho)

    #plot_simple_lax(c.TIME_POINTS,c.delta_t,c.SPACE_POINTS,c.delta_x,grid_rho,grid_v)
    plot_simple_lax_3d(c.TIME_POINTS, c.delta_t, c.SPACE_POINTS, c.delta_x, grid_rho, grid_v)
main()


