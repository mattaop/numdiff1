import numpy as np

def safe_v(rho, V0, rho_max, E):
    return V0*(1-rho/rho_max)/(1+E*(rho/rho_max)**4)

def q_in(time):
    return 0

def phi(x, sigma):

	return (2*np.pi*sigma**2)**(-0.5)*np.exp(-x**2/(2*sigma**2))


def f1_simple_lax(rho_last, v_last, init_rho, init_v, delta_t, delta_x,j, time, position):
    return init_rho/delta_t-(rho_last[j+1]+rho_last[j-1])/(2*delta_t)+init_v*(rho_last[j+1]-rho_last[j-1])/(2*delta_x)+\
           init_rho*(v_last[j+1]-v_last[j-1])/(2*delta_x)-q_in(time)*phi(position)

def f2_simple_lax(rho_last, v_last, init_rho, init_v, delta_t, delta_x,j, tau, V0, rho_max, E, c, my):
    return (init_rho*(init_v/delta_t - (v_last[j+1]+v_last[j-1])/(2*delta_t)+init_v*(v_last[j+1]-v_last[j-1])/(2*delta_x))-\
           (init_rho/tau)*((V0*(1-init_rho/rho_max))/(1+E*(init_rho/rho_max)**4)-init_v)+\
           (rho_last[j+1]-rho_last[j-1])/(2*delta_x)*c**2-my*(v_last[j+1]-2*v_last[j]+v_last[j+1])/delta_x**2)


def jacobi_matrix(rho_last, v_last, init_rho, init_v, delta_t, delta_x, j, V0, tau, rho_max, E):
    j1=1/delta_t+(v_last[j+1]-v_last[j-1])/(2*delta_x)
    j2=(rho_last[j+1]-rho_last[j-1])/(2*delta_x)
    j3=(init_v/delta_t - (v_last[j+1]+v_last[j-1])/(2*delta_t)+init_v*(v_last[j+1]-v_last[j-1])/(2*delta_x))-V0/tau*(rho_max**3*(rho_max**5-2*rho_max**4*init_rho-3*rho_max*E*init_rho**4+2*E*init_rho**5)/(rho_max**4+E*init_rho**4)**2)+init_v/tau
    j4=init_rho/delta_t+init_rho*(v_last[j+1]-v_last[j-1])/(2*delta_x)+init_rho/tau
    jacobi=np.matrix([[j1, j2],[j3,j4]])
    return jacobi

def newtons(rho_last, v_last, delta_t, delta_x, j, V0, tau, rho_max, E):
    init_rho, init_v=rho_last, v_last
    f_vector=np.ndarrau([f1_simple_lax, f2_simple_lax])
    while max(f_vector[0], f_vector[1])>1e-4:
        jacobi=jacobi_matrix(rho_last, v_last, init_rho, init_v, delta_t, delta_x, j, V0, tau, rho_max, E)
        delta_X=np.


def one_step_simple_lax(rho_last, v_last, X, rho0):
    rho_next=np.zeros(X)
    v_next=np.zeros(X)
    rho_next[0]=rho0
    v_next[0]=safe_v(rho0)
    for i in range(1,X-1):
        rho_next[i], v_next[i]=newtons(rho_last, v_last,i)
    return rho_next, v_next

def solve_simple_lax(grid_rho, grid_v, T, X, rho0):
    for i in range(1,T):
        time=i*delta_t
        grid_rho[i], grid_v[i]=one_step_simple_lax(grid_rho[i-1], grid_v[i-1], X, rho0)



def main():
    T=10
    X=5
    V0=120
    rho_max=140
    E=100
    rho0=0
    grid_rho=np.zeros((T,X))
    grid_v=np.zeros((T,X))
    grid_rho[0]=np.ones(X)*rho0
    grid_v[0]=safe_v(grid_rho[0], V0, rho_max, E)
    solve_simple_lax(grid_rho, grid_v, T, X, rho0)
    print(grid_v)
    print(grid_rho)


main()
