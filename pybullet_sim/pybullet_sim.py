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
import numpy as np
import time
import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt
# import scipy.linalg as sp
import sys, os, fire, math
import pybullet
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from scipy.signal import place_poles
start_time = datetime.now().strftime("%Y-%m-%d%H-%M-%S")
script_name = os.path.split(sys.argv[0])[-1].split(".")[0]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
eps = 1e-12


class BulletSim():
	def __init__(self):
		self.i = 0
		self.data_timestamp_prev = 0
		self.e_1_dot = 0.0
		self.e_2_dot = 0.0
		self.angle_error_prev = 0.0
		self.angle_error_prev_2 = 0.0 
		self.vel_error_prev = 0.0
		self.e_1_prev = 10*eps
		self.e_2_prev = 10*eps
		self.target_angles_list = []
		self.target_vels_list = []
		self.e_list = []
		""" 
		Refer to https://github.com/jainachin/bayesrace/blob/master/bayes_race/params/f110.py 
		for these params. Official repo has the same params as well apparently.
		"""
		self.state_x_list, self.state_z_list, self.state_theta_list = [],[],[]
		self.e_1_list,self.e_2_list,self.e_1_dot_list,self.e_2_dot_list = [],[],[],[]
		self.topic_throttle_counter = 0
		self.center_x_list, self.center_y_list, self.r_list = [],[],[]
		# self.m = 3.74
		# self.Iz = 0.04712
		# self.Caf = 4.718 * 0.2652515
		# self.Car = 5.4562 * 0.2652515
		# self.lf = 0.15875
		# self.lr = 0.17145
		self.m = 3.1
		self.Iz = 0.04712
		self.g = 9.81
		self.lf = 0.15875  #0.1621 #0.15875         #0.1653
		self.lr =  0.17145 #0.1425 #0.17145 	      #0.1494
		self.multiplier_f = ((self.m * self.g) / (1 + (self.lf/self.lr)))#  * (np.pi/180) 
		self.multiplier_r = ((self.m*self.g) - ((self.m*self.g) / (1 + (self.lf/self.lr))))#  * (np.pi/180)
		self.Caf =  4.718  * self.multiplier_f  #4.718 * 0.275  #10.310 * 0.2513  #5.1134
		self.Car =  5.4562 * self.multiplier_r #5.4562 * 0.254 #8.522 * 0.2779 #4.9883
		self.Vx = 1.0
		self.create_K_matrix()
		self.sim_ts = 0.002
		self.controller_freq = 25
		self.sim_num_ts = 30000
		self.text_id = None
		self.text_id_2 = None
		return 
		"""'C_Sf': 11.07013852453441, 'C_Sr': 9.654900033605438, 'lf': 0.1568723900063424, 'lr': 0.14049784589354194"""
	def create_K_matrix(self):
		a_1 = np.array([0,1,0,0],dtype=np.float64)
		a_2 = np.hstack((0,-((2*self.Caf)+(2*self.Car))/(self.m*self.Vx),((2*self.Caf)+(2*self.Car))/(self.m),((-2*self.Caf*self.lf)+(2*self.Car*self.lr))/(self.m*self.Vx)))
		a_3 = np.array([0,0,0,1],dtype=np.float64)
		a_4 = np.hstack((0, -((2*self.Caf*self.lf)-(2*self.Car*self.lr))/(self.Iz*self.Vx), ((2*self.Caf*self.lf)-(2*self.Car*self.lr))/(self.Iz), -((2*self.Caf*(self.lf**2))+(2*self.Car*(self.lr**2)))/(self.Iz*self.Vx)))
		A = np.vstack((a_1,a_2,a_3,a_4))
		B1 = np.vstack((
				np.array(0,dtype=np.float64),
				(2*self.Caf)/(self.m),
				np.array(0,dtype=np.float64),
				(2*self.Caf*self.lf)/(self.Iz)))
		# Pol = np.array([-5,-4,-7,-10], dtype=np.float64)
		Pol = np.array([-5-3j,-5+3j,-7,-10])
		self.K = place_poles(A, B1, Pol).gain_matrix
		self.A = A
		self.B1 = B1
		return 


	def run_sim(self):
		pybullet.connect(pybullet.GUI)
		pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_WIREFRAME,0)
		pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)

		pybullet.resetSimulation()
		ground = pybullet.loadURDF(os.path.join(
					os.path.dirname(__file__), "./urdf/terrain/big_ground.urdf"),useFixedBase=1)

		init_heading_euler = R.from_euler("YZX",[0.0,0.0,-90.0] , degrees=True)
		init_heading_quat  = init_heading_euler.as_quat()

		agent = pybullet.loadURDF(os.path.join(
					os.path.dirname(__file__), "./urdf/uva-f1tenth/uva-f1tenth-model.urdf"),[0.0, 0.204, 0.0],init_heading_euler.as_quat())

		# base_p_orient = pybullet.getBasePositionAndOrientation(agent) 

		pybullet.setGravity(0, -9.81, 0)
		pybullet.setTimeStep(self.sim_ts)
		pybullet.setRealTimeSimulation(0)
		long_velocity = self.Vx/0.05
					
		pybullet.changeDynamics(ground,0,lateralFriction=1.0,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		pybullet.changeDynamics(agent,2,lateralFriction=1.0,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		pybullet.changeDynamics(agent,3,lateralFriction=1.0,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		pybullet.changeDynamics(agent,5,lateralFriction=1.0,spinningFriction=0.0,rollingFriction=0.0,restitution=0)
		pybullet.changeDynamics(agent,7,lateralFriction=1.0,spinningFriction=0.0,rollingFriction=0.0,restitution=0)

		if self.center_y>0:
			for line_idx in np.arange(0,int(self.ref_traj.shape[0]-1)):
				pybullet.addUserDebugLine(lineFromXYZ=self.ref_traj[int(line_idx),:],lineToXYZ=self.ref_traj[int(line_idx+1),:],lineColorRGB=[0, 0, 1.0],lineWidth=7.5)
		else:
			for line_idx in np.arange(0,int(self.ref_traj.shape[0]-1)):
				pybullet.addUserDebugLine(lineFromXYZ=self.ref_traj[int(line_idx),:],lineToXYZ=self.ref_traj[int(line_idx+1),:],lineColorRGB=[0, 0, 1.0],lineWidth=7.5)	

		for i in np.arange(self.sim_num_ts):
			
			pybullet.stepSimulation()
			# time.sleep(0.000001)
			
			if not (i%int((1/self.controller_freq) * (1/self.sim_ts))): 
				pybullet.setJointMotorControl2(bodyUniqueId=agent,
					jointIndex=2,
					controlMode=pybullet.VELOCITY_CONTROL,
					targetVelocity = long_velocity)
				pybullet.setJointMotorControl2(bodyUniqueId=agent,
					jointIndex=3,
					controlMode=pybullet.VELOCITY_CONTROL,
					targetVelocity = long_velocity)
				
				pybullet.setJointMotorControl2(bodyUniqueId=agent,
					jointIndex=5,
					controlMode=pybullet.VELOCITY_CONTROL,
					targetVelocity = long_velocity)
				pybullet.setJointMotorControl2(bodyUniqueId=agent,
					jointIndex=7,
					controlMode=pybullet.VELOCITY_CONTROL,
					targetVelocity = long_velocity)	

			# cg_pos_3d_xyz = np.array(pybullet.getLinkState(agent,8)[0])
			cg_pos_3d_xyz = np.array(pybullet.getBasePositionAndOrientation(agent)[0])
			self.veh_2d_world_pos = cg_pos_3d_xyz[[0,2]]
			# cg_heading_3d_quat = R.from_quat(np.array(pybullet.getLinkState(agent,8)[1]))
			cg_heading_3d_quat = R.from_quat(np.array(pybullet.getBasePositionAndOrientation(agent)[1]))
			self.veh_2d_world_heading  = -cg_heading_3d_quat.as_euler('YZX', degrees=False)[0]

			if ((i>0) and (not (i%int((1/self.controller_freq) * (1/self.sim_ts))))):
				e_state = self.compute_e_state_new(i=i, 
					data_timestamp=(i)*self.sim_ts)	
				self.target_steering_angle = (self.linear_Cntrl(self.K,e_state))
				# print(f"steering angle is: {target_steering_angle}")
				# print(f"errors are as follows: {e_state}")
				self.e_list.append(e_state)
				pybullet.setJointMotorControl2(bodyUniqueId=agent,
					jointIndex=4,
					controlMode=pybullet.POSITION_CONTROL,
					targetPosition = self.target_steering_angle)
				pybullet.setJointMotorControl2(bodyUniqueId=agent,
					jointIndex=6,
					controlMode=pybullet.POSITION_CONTROL,
					targetPosition = self.target_steering_angle)

				if self.text_id_2 is not None:
					pybullet.removeUserDebugItem(self.text_id_2)
				# self.text_id_2 = pybullet.addUserDebugText(str(np.round(e_state[2],3)),[self.center_x,1.5,self.center_y])

		
		return 

	def compute_e_state_new(self,i, data_timestamp):
		"""
		Within the error computation function global nimblephysics 
		coordinates are transformed in order to comply with the Rajamani's 
		definitions and the output is in Rajamani's coordinates.
		collision algo source:
		https://cp-algorithms.com/geometry/circle-line-intersection.html
		"""
		local_veh_2d_pos_state = self.veh_2d_world_pos[[-1,0]] + eps  # Keep in mind that we are reordering indices. Final variable is in zx format.
		locally_computed_veh_heading_2d = self.veh_2d_world_heading + eps
		# Lateral axis in line equation in mx+b=y form
		m_lateral = -1 / np.tan(locally_computed_veh_heading_2d)
		# b = y - mx, translate it because the ref circle is translated as well
		b_lateral_translated = (local_veh_2d_pos_state[0]-self.center_y) - (m_lateral*(local_veh_2d_pos_state[1]-self.center_x))
		# Transform line equation to Ax+By+C = 0 form
		A_lateral = m_lateral
		B_lateral = -1.0
		C_lateral = b_lateral_translated
		# Solve line-circle intersection problem for centered radius (we have translated the line equation)
		x_0 = - (A_lateral*C_lateral)/(A_lateral**2 + B_lateral**2)
		y_0 = - (B_lateral*C_lateral)/(A_lateral**2 + B_lateral**2)
		intersection_d = np.sqrt(self.r**2-((C_lateral**2)/(A_lateral**2 + B_lateral**2)))
		intersection_m = np.sqrt((intersection_d**2)/(A_lateral**2 + B_lateral**2))
		ax = x_0 + B_lateral*intersection_m
		ay = y_0 - A_lateral*intersection_m
		bx = x_0 - B_lateral*intersection_m
		by = y_0 + A_lateral*intersection_m
		# Translate back the intersection points
		ax = ax + self.center_x
		ay = ay + self.center_y 
		bx = bx + self.center_x 
		by = by + self.center_y 
		pt_a_dist = np.linalg.norm(np.hstack((ay, ax)) - local_veh_2d_pos_state)
		pt_b_dist = np.linalg.norm(np.hstack((by, bx)) - local_veh_2d_pos_state)
		if pt_a_dist > pt_b_dist: # Investigate this if condition whether it breaks differentiability.
			intersection_pt = np.hstack((by, bx))
		else:
			intersection_pt = np.hstack((ay, ax))
		center_pt = np.hstack((self.center_y, self.center_x))
		"""
		Heading is important for e_1, not the sign of center y. 
		Get the vehicle frame of reference coordinates of the intersection point to for e_1.
		"""
		T_matrix, inv_T_matrix = self.global_to_local_rf_2d_T_matrix(local_veh_2d_pos_state[[1,0]], locally_computed_veh_heading_2d)
		self.e_1 = np.squeeze(np.matmul(inv_T_matrix,np.expand_dims(np.hstack((intersection_pt[1],intersection_pt[0],[1])),axis=1))[1])
		
		# gt_trajectory_intersection_m = -1.0/((intersection_pt - center_pt)[0] / (intersection_pt - center_pt)[1])
		gt_trajectory_heading = -np.arctan2((intersection_pt - center_pt)[1],(intersection_pt - center_pt)[0])
		self.e_2 = (locally_computed_veh_heading_2d - gt_trajectory_heading)
		self.e_2 = ((self.e_2 + np.pi) % (2*np.pi)) - np.pi
		self.e_1_dot = (self.e_1 - self.e_1_prev) / (data_timestamp-self.data_timestamp_prev)
		self.e_2_dot = -(self.e_2 - self.e_2_prev) / (data_timestamp-self.data_timestamp_prev)
		self.local_veh_2d_pos_state_prev = local_veh_2d_pos_state
		self.e_1_prev = self.e_1
		self.e_2_prev = self.e_2
		if (np.isnan(self.e_2) or np.isnan(self.e_1) or np.isnan(self.e_2_dot) or np.isnan(self.e_1_dot)):
			print("NaN detected") 
		stacked_e = np.stack((self.e_1, self.e_1_dot, self.e_2, self.e_2_dot))
		self.data_timestamp_prev = data_timestamp

		if self.text_id is not None:
			pybullet.removeUserDebugItem(self.text_id)
		# self.text_id = pybullet.addUserDebugText(f"gt heading {np.round(gt_trajectory_heading,3)} veh heading: {np.round(locally_computed_veh_heading_2d,3)}",
		# 	[self.center_x,0.5,self.center_y])
		return stacked_e

	def linear_Cntrl(self, K_gains, e_state):
		# \sigma = -Kx 
		cntrl = -np.clip(np.squeeze(np.matmul(K_gains, e_state)),a_min=np.deg2rad(-19.48057),a_max=np.deg2rad(19.48057))
		return cntrl

	def global_to_local_rf_2d_T_matrix(self, local_pos, local_heading):
		o_0 = np.hstack((np.cos(local_heading), -np.sin(local_heading), local_pos[0]))
		o_1 = np.hstack((np.sin(local_heading),np.cos(local_heading), local_pos[1]))
		o_2 = np.array([0.0,0.0,1.0], dtype=np.float64)
		output = np.vstack((o_0, o_1, o_2))
		inverse_output = np.linalg.pinv(output)
		return output, inverse_output

	def points_on_circumference(self,center=(0, 0), r=50, n=100):
		x_pts = center[0] + (np.cos(2 * np.pi / n * np.arange(0, n)) * r)
		y_pts = center[1] + (np.sin(2 * np.pi / n * np.arange(0, n)) * r)  
		return np.hstack((np.expand_dims(x_pts,1), np.expand_dims(y_pts,1)))

	def three_pts_define_circle(self, p1, p2, p3):
		"""
		Returns the center and radius of the circle passing the given 3 points.
		In case the 3 points form a line, returns (None, infinity).
		"""
		temp = p2[0] * p2[0] + p2[1] * p2[1]
		bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
		cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
		det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
		
		if abs(det) < 1.0e-6:
			return (None, np.inf)
		
		# Center of circle
		cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
		cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
		
		radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
		return ((cx, cy), radius)

	def pick_circle_drone_room(self, fixed_r=None, fixed_center=None):
		# Define circle radius
		if fixed_r is None:
			self.r = np.random.randint(low=30, high=35, size=1)
			# self.r = torch.tensor(25, dtype=torch.float64)
			if np.random.randint(2,size=1):
				self.center_y = -self.r
			else:
				self.center_y = -self.r
			self.center_x = 0.48
			print(f"randomly picked radius is: {self.r}")
			# print(f"Randomly picked circle center is: {self.center_x.detach().numpy()}")
			# with open(f"experiment_data_saves/{start_time}_{script_name}_data.txt", "a") as d:
			# 	d.write(f"Randomly picked circle center is: {self.center_x.detach().numpy()}\n")
		else:
			self.r = fixed_r
			self.center_x = fixed_center[0]
			# print(f"Selected circle center is: {self.center_x.detach().numpy()}")
			# with open(f"experiment_data_saves/{start_time}_{script_name}_data.txt", "a") as d:
			# 	d.write(f"Selected circle center is: {self.center_x.detach().numpy()}\n")
			self.center_y = fixed_center[1]
		
 
		circle_sample_n = 50
		circle_pts = self.points_on_circumference(center=(self.center_x, self.center_y),
								r=self.r,
								n=circle_sample_n)
		height = 0.121
		self.ref_traj = np.zeros((circle_sample_n,3))
		
		self.ref_traj[:,1] = np.repeat(height, circle_sample_n)
		self.ref_traj[:,0] = circle_pts[:,0]
		self.ref_traj[:,2] = circle_pts[:,1]			
		return 


	def plot_error_states(self):
		np_arr_e_list = np.array(self.e_list)
		sns.set_style('darkgrid')
		sns.lineplot(x=int((1/self.controller_freq) * (1/self.sim_ts))*np.arange(np_arr_e_list[:,0].shape[0]),y=np_arr_e_list[:,0],label="$e_{1}$")
		sns.lineplot(x=int((1/self.controller_freq) * (1/self.sim_ts))*np.arange(np_arr_e_list[:,2].shape[0]),y=np_arr_e_list[:,2],label="$e_{2}$")
		# sns.lineplot(x=int((1/self.controller_freq) * (1/self.sim_ts))*np.arange(np_arr_e_list[:,1].shape[0]),y=np_arr_e_list[:,1],label="$\dot{e_{1}}$")
		# sns.lineplot(x=int((1/self.controller_freq) * (1/self.sim_ts))*np.arange(np_arr_e_list[:,3].shape[0]),y=np_arr_e_list[:,3],label="$\dot{e_{2}}$")
		# plt.plot(np_arr_e_list)
		# plt.grid(True)
		# plt.ylim(bottom=-2.0, top=2.0)
		# e_profile.set(xlim=(0, 7000))
		# plt.legend(["e1", "e1_dot", "e2", "e2_dot"])
		plt.legend()
		plt.xlabel("Simulation timesteps")
		plt.ylabel("Error")
		plt.savefig(f"bullet_{script_name}_{start_time}_errors.pdf")
		plt.clf()
		return


if __name__ == "__main__":
	bs = BulletSim()
	center, radius = bs.three_pts_define_circle(p1=[0.11, -0.018], 
					p2=[3.43, -2.206],
					p3=[3.48, -2.4])
	bs.pick_circle_drone_room(fixed_r=radius, fixed_center=center)
	bs.run_sim()
	bs.plot_error_states()
	quit()