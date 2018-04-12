import numpy as np
import matplotlib.pyplot as plt
import constants as c
import convergence as conv
import lax_wendroff as lw
import simple_lax as sl
import simple_lax_vectorized as sl_v
import upwind as up
import upwind_vectorized as up_v


if __name__ == "__main__":

    Master_Flag = {
                    0: 'Simple Lax',
                    1: 'Upwind',
                    2: 'Lax Wendroff',
                    3: 'Time Convergence'





            }[1]        #<-------Write number of the function you want to test. For example, for finding the best sensor location, write 8 in the [ ].
    if Master_Flag =='Simple Lax':
        grid_u = sl_v.solve_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)
        sl_v.plot_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 0])
        sl_v.plot_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 1])


    elif Master_Flag=='Upwind':
        grid_u = up_v.solve_upwind(c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)
        up_v.plot_upwind(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 0])
        up_v.plot_upwind(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 1])

    elif Master_Flag=='Lax Wendroff':
        grid_u = lw.solve_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)
        lw.plot_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 0])
        lw.plot_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:, :, 1])

    elif Master_Flag=='Time Convergence':
