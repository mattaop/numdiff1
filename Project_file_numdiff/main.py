import numpy as np
import matplotlib.pyplot as plt
import constants as c
import convergence as conv
import lax_wendroff as lw
import simple_lax as sl
import simple_lax_vectorized as sl_v
import simple_lax_vectorized_v2 as sl_v2
import spatial_convergence as sc
import time_convergence as tc

import upwind as up
import upwind_vectorized as up_v
import upwind_vectorized_v2 as up_v2
import mac_cormack_v2 as mc_v2

if __name__ == "__main__":

    Master_Flag = {
                    0: 'Lax-Friedrich',
                    1: 'Upwind',
                    2: 'Lax Wendroff',
                    3: 'Time Convergence',
                    4: 'Spatial Convergence',
                    5: ' 3d plot'





            }[4]        #<-------Write number of the function you want to test. For example, for finding the best sensor location, write 8 in the [ ].
    if Master_Flag =='Lax-Friedrich':
        grid_u = sl_v.solve_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)
        sl_v.plot_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 0])
        #sl_v.plot_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 1])


    elif Master_Flag=='Upwind':
        for i in range(10,15):
            space_points=2**i
            print(space_points)
            d_x = c.L / (space_points - 1)
            grid_u = up_v2.solve_upwind(c.TIME_POINTS,space_points, c.delta_t, d_x)
            up_v2.plot_upwind(c.TIME_POINTS,space_points, d_x, grid_u[:, :, 0])
            #up_v2.plot_upwind(c.TIME_POINTS,space_points, d_x, grid_u[:, :, 1]# )
            plt.show()

    elif Master_Flag=='Lax Wendroff':
        grid_u = lw.solve_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)
        lw.plot_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 0])
        lw.plot_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 1])

    elif Master_Flag=='Time Convergence':
        tc.plot_time_convergence_2(sl_v.solve_simple_lax, up_v2.solve_upwind,
                                lw.solve_lax_wendroff, mc_v2.solve_mac_cormack)


    elif Master_Flag=='Spatial Convergence':
        sc.plot_spatial_convergence(sl_v.solve_simple_lax, mc_v2.solve_mac_cormack)


    elif Master_Flag==' 3d plot':
        grid_u = sl_v.solve_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)
        sl_v.plot_simple_lax_3d_rho(c.TIME_POINTS,c.delta_t,c.SPACE_POINTS,c.delta_x,grid_u[:,:,0])
        sl_v.plot_simple_lax_3d_v(c.TIME_POINTS,c.delta_t,c.SPACE_POINTS,c.delta_x,grid_u[:,:,1])