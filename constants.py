import numpy as np

M=8 #2^m, gives number of points in space
MAX_TIME=5*60 #seconds
TIME_POINTS=1000
SPACE_POINTS=2**(M)
V0=120
RHO_MAX=140
E=100
RHO_0=40
L=5000 #meter
delta_t=0.001
delta_x=L/(SPACE_POINTS-1)  #meter

SIGMA=56.7
MY=0
TAU=0.5
C=54