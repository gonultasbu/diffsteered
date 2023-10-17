"""
Author: Burak M Gonultas

MIT License

Copyright (c) 2023 Burak Mert Gonultas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import sys
import os 
from datetime import datetime
start_time = datetime.now()
script_name = os.path.split(sys.argv[0])[-1].split(".")[0]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from scipy.signal import place_poles
# matplotlib.use('Agg')

fig = plt.figure()
plt.axis('equal')
plt.grid()
# plt.tight_layout()
ax = fig.add_subplot(111)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
plt.ylabel("z-axis")
plt.xlabel("x-axis")
traj_arr = np.loadtxt("xztheta_traj.csv", delimiter=",")
state_x_list = traj_arr[::10,0]
state_z_list = traj_arr[::10,1]
state_theta_list = traj_arr[::10,2]
state_e_1_list = traj_arr[::10,3]
state_e_1_dot_list = traj_arr[::10,4]
state_e_2_list = traj_arr[::10,5]
state_e_2_dot_list = traj_arr[::10,6]
center_x = traj_arr[0,7]
center_y = traj_arr[0,8]
circle_radius = traj_arr[0,9]
lateral_axis_xvals = np.arange(-30,30,0.01)

patch = patches.Rectangle((0, 0), 0, 0, angle=0.0, fc='k')
m_lateral = -1.0/np.tan(state_theta_list[0])
b_lateral_translated = (state_z_list[0]) - (m_lateral*(state_x_list[0]))
line, = ax.plot(lateral_axis_xvals, (m_lateral*lateral_axis_xvals)+b_lateral_translated)
label = ax.text(-8, -5, "", ha='center', va='center', fontsize=10, color="Red")
label_2 = ax.text(-8, -6, "", ha='center', va='center', fontsize=10, color="Blue")
label_3 = ax.text(-8, -7, "", ha='center', va='center', fontsize=10, color="Green")
label_4 = ax.text(-8, -8, "", ha='center', va='center', fontsize=10, color="Purple")
label_5 = ax.text(-8, -9, "", ha='center', va='center', fontsize=10, color="Black")

def init():
    ax.add_patch(patches.Circle((center_x, center_y),radius=circle_radius, fill=False, color="navy"))
    ax.add_patch(patch)
    return patch,

def animate(i):
    patch.set_width(1.0)
    patch.set_height(0.2)
    patch.set_xy([state_x_list[i], state_z_list[i]])
    patch.angle = np.rad2deg(state_theta_list[i])
    m_lateral = -1.0/np.tan(state_theta_list[i])
    b_lateral_translated = (state_z_list[i]) - (m_lateral*(state_x_list[i]))
    line.set_ydata((m_lateral*lateral_axis_xvals)+b_lateral_translated)
    label.set_text("$\\theta$: "+str(state_theta_list[i]))
    label_2.set_text("$e_1$: "+str(state_e_1_list[i]))
    label_3.set_text("$e_2$: "+str(state_e_2_list[i]))
    label_4.set_text("$e_{1dot}$: "+str(state_e_1_dot_list[i]))
    label_5.set_text("$e_{2dot}$: "+str(state_e_2_dot_list[i]))
    # print("Euler heading is:"+str(state_theta_list[i]))
    return patch, line, label, label_2, label_3, label_4, label_5,

anim = animation.FuncAnimation(fig, 
                            animate,
                            init_func=init,
                            frames=len(state_x_list),
                            interval=0.001,
                            blit=True)
plt.show()

# writervideo = animation.FFMpegWriter(fps=60)
# anim.save('animation.mp4', writer=writervideo)
# plt.close()