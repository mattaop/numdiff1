import numpy as np



M=6
MAX_TIME=5*60 #seconds

TIME_POINTS= 1000
SPACE_POINTS=2**(M)

L=5000 #meter

delta_t=0.001
delta_x=L/(SPACE_POINTS-1)  #meter

V0=33.33
RHO_MAX=140
E=100
RHO_0=40
Q_IN=2000

# hei
SIGMA=56.7
MY=0
TAU=0.5
C=54

def initialize_grid(T, X, rho0):
    grid_u = np.zeros((T, X, 2))
    grid_u[0, :, 0] = np.ones(X) * rho0
    grid_u[0, :, 1] = safe_v(grid_u[0, :, 0])
    return grid_u

def safe_v(rho):
    return V0 * (1 - rho / RHO_MAX) / (1 + E * (rho / RHO_MAX) ** 4)

def q_in(time):
    return 1000

def phi(x):
    return (2*np.pi*SIGMA**2)**(-0.5)*np.exp(-x**2/(2*SIGMA**2))