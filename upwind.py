import numpy as np

def safe_v(rho, V0, rho_max, E):
    return V0*(1-rho/rho_max)/(1+E*(rho/rho_max)**4)

def q_in(time):
    return 0

def phi(x, sigma):
	return (2*np.pi*sigma**2)**(-0.5)*np.exp(-x**2/(2*sigma**2))


def rho_next_upwind(rho_last, v_last, delta_t, delta_x, j, time, position,sigma):
    return  rho_last[j] - delta_t/(delta_x)*(rho_last[j]*v_last[j]-rho_last[j-1]*v_last[j-1])+ q_in(time)*phi(position,sigma)

def v_next_upwind(rho_last, v_last, delta_t, delta_x,j, tau, V0, rho_max, E, c, my):
    return v_last[j] - delta_t/(2*delta_x)*(v_last[j]**2+c**2*np.log(rho_last[j])) +(delta_t/tau)*((V0*(1-rho_last[j]/rho_max))/(1+E*(rho_last[j]/rho_max)**4)-v_last[j])-\
           (rho_last[j]-rho_last[j-1])*delta_t/(rho_last[j]*delta_x)*c**2+my*delta_t*(v_last[j]-2*v_last[j]+v_last[j+1])/(rho_last[j]*delta_x**2)

def f2_simple_lax(rho_last, v_last, rho_temp, v_temp, delta_t, delta_x,j, tau, V0, rho_max, E, c, my):
    return rho_last[j]*(v_temp/delta_t - (v_last[j+1]+v_last[j-1])/(2*delta_t)+v_last[j]*(v_last[j+1]-v_last[j-1])/(2*delta_x))\
           -(rho_last[j]/tau)*((V0*(1-rho_last[j]/rho_max))/(1+E*(rho_last[j]/rho_max)**4)-v_last[j])+\
           (rho_last[j+1]-rho_last[j-1])/(2*delta_x)*c**2-my*(v_last[j+1]-2*v_last[j]+v_last[j+1])/delta_x**2

def f1_simple_lax(rho_last, v_last, rho_temp, v_temp, delta_t, delta_x,j, time, position,sigma):
    return rho_temp / delta_t - 1 / (2 * delta_t) * (rho_last[j + 1] + rho_last[j - 1])\
           + v_last[j] / (2 * delta_x) * (rho_last[j + 1] - rho_last[j - 1]) + rho_last[j] / (2 * delta_x) * (v_last[j + 1] - v_last[j - 1])\
           - q_in(time) * phi(position, sigma)

def one_step_upwind(rho_last, v_last, X, delta_t,delta_x ,time,L,sigma, rho0,V0,rho_max,E,tau,c,my):
    rho_next=np.zeros(X)
    v_next=np.zeros(X)
    rho_next[0]=rho0
    v_next[0]=safe_v(rho0,V0,rho_max,E)
    for j in range(1,X-1):

        position=j*delta_x-L/2
        rho_next[j], v_next[j]= rho_next_upwind(rho_last,v_last,delta_t,delta_x, j ,time,position,sigma),\
                                v_next_upwind(rho_last,v_last,delta_t,delta_x, j, tau, V0, rho_max, E, c, my)
    return rho_next, v_next

def solve_upwind(grid_rho, grid_v, T, X, rho0,delta_t,delta_x,L,sigma,V0,rho_max,E,tau,c,my):
    for i in range(1,T):
        time=i*delta_t
        grid_rho[i], grid_v[i]=one_step_upwind(grid_rho[i-1],grid_v[i-1],X,delta_t,delta_x ,time,L,sigma, rho0,V0,rho_max,E,tau,c,my)


def main():
    T=10
    X=5
    V0=120
    rho_max=140
    E=100
    rho0=0
    delta_t=0.01
    delta_x=37
    L=100
    sigma=56.7
    my=600
    tau=0.5
    c=54

    grid_rho=np.zeros((T,X))
    grid_v=np.zeros((T,X))
    grid_rho[0]=np.ones(X)*rho0
    grid_v[0]=safe_v(grid_rho[0], V0, rho_max, E)
    solve_upwind(grid_rho, grid_v, T, X, rho0,delta_t,delta_x,L,sigma,V0,rho_max,E,tau,c,my)
    print(grid_v)
    print(grid_rho)


main()
