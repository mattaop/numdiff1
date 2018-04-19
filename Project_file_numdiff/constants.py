import numpy as np



M = 12
MAX_TIME = 500*60 #seconds




TIME_POINTS = 1000


#SPACE_POINTS = 2**(M)

#SPACE_POINTS=2**(M)
SPACE_POINTS=1000

L=5000 #meter

delta_t=0.25


delta_x=L/(SPACE_POINTS-1)  #meter

V0=33.33
RHO_MAX=0.140
E=100
RHO_0=0.05

SIGMA=300
MY=2
TAU=30
C=15

def initialize_grid(T, X, rho0):
    grid_u = np.zeros((T, X, 2))
    grid_u[0, :, 0] = np.ones(X) * rho0
    grid_u[0, :, 1] = safe_v(grid_u[0, :, 0])
    return grid_u

def safe_v(rho):
    return V0 * (1 - rho / RHO_MAX) / (1 + E * (rho / RHO_MAX) ** 4)
print(safe_v(RHO_0))
def q_in(time):
    if time<30:
        return 0.01
    else:
        return 0


def phi(x):
    return (2*np.pi*SIGMA**2)**(-0.5)*np.exp(-x**2/(2*SIGMA**2))