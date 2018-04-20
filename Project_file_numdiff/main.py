import numpy as np
import matplotlib.pyplot as plt

import constants as c
import lax_wendroff as lw
import lax_friedrichs as lf
import spatial_convergence as sc
import time_convergence as tc
import upwind as up
import mac_cormack as mc

if __name__ == "__main__":

    Master_Flag = {
                    0: 'Lax-Friedrichs',
                    1: 'Upwind',
                    2: 'Lax-Wendroff',
                    3: 'MacCormack',
                    4: 'Time Convergence',
                    5: 'Spatial Convergence',


            }[3]        #<-------Write number of the function you want to test.


    if Master_Flag =='Lax-Friedrichs':
        grid_u = lf.solve_lax_friedrichs(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME)
        lf.plot_lax_friedrichs(c.TIME_POINTS, c.SPACE_POINTS, grid_u)
        lf.plot_lax_friedrichs_3d_rho(c.TIME_POINTS,c.SPACE_POINTS, c.MAX_TIME, grid_u[:, :, 0])
        lf.plot_lax_friedrichs_3d_v(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME, grid_u[:, :, 1])


    elif Master_Flag=='Upwind':
        grid_u = up.solve_upwind(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME)
        up.plot_upwind(c.TIME_POINTS, c.SPACE_POINTS, grid_u[:, :, 0])
        up.plot_upwind(c.TIME_POINTS, c.SPACE_POINTS, grid_u[:, :, 1])
        up.plot_upwind_3d_rho(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME, grid_u[:, :, 0])
        up.plot_upwind_3d_v(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME, grid_u[:, :, 1])

    elif Master_Flag=='Lax-Wendroff':
        grid_u = lw.solve_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME)
        lw.plot_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, grid_u[:, :, 0])
        lw.plot_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, grid_u[:, :, 1])
        lw.plot_lax_wendroff_3d_rho(c.TIME_POINTS,c.SPACE_POINTS,c.MAX_TIME,grid_u[:,:,0])
        lw.plot_lax_wendroff_3d_v(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME,grid_u[:, :, 1])

    elif Master_Flag == 'MacCormack':
        grid_u = mc.solve_mac_cormack(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME)
        mc.plot_mac_cormack(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x,grid_u[:, :, 0])
        mc.plot_mac_cormack(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x,grid_u[:, :, 1])
        mc.plot_mac_cormack_3d_rho(c.TIME_POINTS,c.SPACE_POINTS,c.MAX_TIME,grid_u[:,:,0])
        mc.plot_mac_cormack_3d_v(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME, grid_u[:, :, 0])

    elif Master_Flag=='Time Convergence':
        tc.plot_time_convergence(lf.solve_lax_friedrichs, up.solve_upwind, lw.solve_lax_wendroff, mc.solve_mac_cormack)


    elif Master_Flag=='Spatial Convergence':
        sc.plot_spatial_convergence(lf.solve_lax_friedrichs, up.solve_upwind, lw.solve_lax_wendroff, mc.solve_mac_cormack)

