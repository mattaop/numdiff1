import numpy as np
import matplotlib.pyplot as plt

import constants as c

import lax_wendroff as lw
import lax_friedrichs as lf
import upwind as up
import mac_cormack as mc

import time_convergence as tc
import spatial_convergence as sc
import general_convergence as gc

if __name__ == "__main__":

    Master_Flag = {
                    0: 'Lax-Friedrichs',
                    1: 'Upwind',
                    2: 'Lax Wendroff',
                    3: 'Time Convergence',
                    4: 'Spatial Convergence',
                    5: 'General Convergence',
                    6: ' 3d plot'



            }[0]        #<-------Write number of the function you want to test.

    if Master_Flag =='Lax-Friedrichs':
        grid_u = lf.solve_lax_friedrichs(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME)
        lf.plot_lax_friedrichs_3d_rho(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 0])
        lf.plot_lax_friedrichs_3d_v(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 0])
        #plot_lax_friedrichs(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 0])


    elif Master_Flag=='Upwind':
        for i in range(10,15):
            space_points=2**i
            print(space_points)
            d_x = c.L / (space_points - 1)
            grid_u = up.solve_upwind(c.TIME_POINTS, space_points, c.MAX_TIME)
            up.plot_upwind(c.TIME_POINTS,space_points, grid_u[:, :, 0])
            #up_v2.plot_upwind(c.TIME_POINTS, space_points, grid_u[:, :, 1])
            plt.show()

    elif Master_Flag=='Lax Wendroff':
        grid_u = lw.solve_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME)
        lw.plot_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, grid_u[:, :, 0])
        lw.plot_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, grid_u[:, :, 1])

    elif Master_Flag=='Time Convergence':
        tc.plot_time_convergence_2(lf.solve_lax_friedrichs, up.solve_upwind,
                                lw.solve_lax_wendroff, mc.solve_mac_cormack)


    elif Master_Flag=='Spatial Convergence':
        sc.plot_spatial_convergence(lf.solve_lax_friedrichs, mc.solve_mac_cormack)

    elif Master_Flag=='General Convergence':
        gc.plot_general_convergence(up.solve_upwind)

    elif Master_Flag==' 3d plot':
        grid_u = lf.solve_lax_friedrichs(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME)
        lf.plot_lax_friedrichs2_3d_rho(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME, grid_u[:, :, 0])
        lf.plot_lax_friedrichs2_3d_v(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME, grid_u[:, :, 1])
        #grid_u = sl_v.solve_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)
        #sl_v.plot_simple_lax_3d_rho(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME, grid_u[:,:,0])
        #sl_v.plot_simple_lax_3d_v(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME, grid_u[:,:,1])