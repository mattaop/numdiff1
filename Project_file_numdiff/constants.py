import numpy as np


#TIMEVALUES
MAX_TIME = 60 #seconds
TIME_POINTS = 1000
delta_t=MAX_TIME/(TIME_POINTS-1) #seconds

#SPACTIAL VALUES
L=5000 #meter
M = 8
SPACE_POINTS = 2**(M)
delta_x=L/(SPACE_POINTS-1)  #meter

#CONSTANTS
V0=33.33 #m/s
RHO_MAX=0.140 #vehicles/m
E=100 #
RHO_0=0.05 #vehicles/m
SIGMA=300 #
MY=0 #
TAU=30 #sec
C=15 #m/s
