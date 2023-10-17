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
import rosbag
import numpy as np
import sys
import os
from datetime import datetime
start_time = datetime.now()
script_name = os.path.split(sys.argv[0])[-1].split(".")[0]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from os import path as op
from tqdm import tqdm
import pandas as pd


def plot_f1tenth_bagdata(bagdir,save_folder):
	bag = rosbag.Bag(bagdir)
	x_coords_list = []
	z_coords_list = []
	phasespace_t_list = []
	speed_cmd_list = []
	steering_cmd_list = []
	cmds_t_list = []
	headings_list = []
	phasespace_topic = "/phasespace/rigids_throttled"
	teleop_topic = "/vesc/low_level/ackermann_cmd_mux/input/navigation"
	bagname = bagdir.split(".")[-2].split("/")[-1]
	for topic, msg, t in tqdm(bag.read_messages(topics=[phasespace_topic, teleop_topic]), ascii=True):
		if(topic==phasespace_topic):
			x_coords_list.append(msg.rigids[1].x)
			z_coords_list.append(msg.rigids[1].z)
			phasespace_t_list.append(msg.rigids[1].time*1e-9)
			r = R.from_quat([msg.rigids[1].qx, msg.rigids[1].qy, msg.rigids[1].qz, msg.rigids[1].qw])
			headings_list.append(-r.as_euler('YZX', degrees=False)[0])
		elif(topic==teleop_topic):
			speed_cmd_list.append(msg.drive.speed)
			steering_cmd_list.append(msg.drive.steering_angle)
			cmds_t_list.append(t.secs+(t.nsecs*1e-9))

	bag.close()
	plt.scatter(x=np.array(x_coords_list)/1000.0,y=np.array(z_coords_list)/1000.0,
			c=np.tan(np.linspace(-0.99, 0.99, len(x_coords_list))),
            s=0.5,
            linewidths=0.5)
	
	plt.axis('equal')
	plt.grid()
	plt.ylabel("z-axis")
	plt.xlabel("x-axis")
	plt.savefig(save_folder+bagname+"trajectory.pdf")
	plt.clf()

	plt.plot(np.array(cmds_t_list)-np.array(cmds_t_list)[0], 
		np.array(speed_cmd_list),
		".",markersize=2)
	plt.grid()
	plt.ylabel("input speed command(m/s)")
	plt.xlabel("time(s)")
	plt.savefig(save_folder+bagname+"speed_cmds.pdf")
	plt.clf()
	
	plt.plot(np.array(cmds_t_list)-np.array(cmds_t_list)[0], 
		np.array(steering_cmd_list),
		".",markersize=2)
	plt.grid()
	plt.ylabel("steering command in rads")
	plt.xlabel("time(s)")
	plt.savefig(save_folder+bagname+"steer_cmds.pdf")
	plt.clf()

	kernel_size = 20
	kernel = np.ones(kernel_size) / kernel_size
	
	plt.plot((np.array(phasespace_t_list)-np.array(phasespace_t_list)[0])[1:], 
		np.convolve(np.divide((np.linalg.norm(np.vstack((np.ediff1d(x_coords_list)/1000.0, np.ediff1d(z_coords_list)/1000.0)),axis=0)), np.ediff1d(phasespace_t_list)), kernel, mode='same'),
		".",markersize=2)
	plt.grid()
	plt.ylabel("velocity measured(m/s)")
	plt.xlabel("time(s)")
	# plt.ylim([0,3.0])
	plt.savefig(save_folder+bagname+"vels.pdf")
	plt.clf()
	
	traj_arr = np.vstack([x_coords_list, z_coords_list, phasespace_t_list, headings_list]).T
	np.savetxt(save_folder+bagname+"_traj.csv",traj_arr, delimiter=',', fmt='%.4f')
	cmds_arr = np.vstack([speed_cmd_list, steering_cmd_list, cmds_t_list]).T
	np.savetxt(save_folder+bagname+"_cmds.csv",cmds_arr, delimiter=',', fmt='%.4f')
	return
if __name__ == "__main__":
	folder = "bag/bags_2023_02_28/wiggle/"
	for fname in os.listdir(os.path.join(os.getcwd(),folder)):
		plot_f1tenth_bagdata(folder+fname,folder)
		
		