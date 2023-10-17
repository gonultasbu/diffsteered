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
import rospy
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import std_msgs
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from phasespace_msgs.msg import Rigids
from visualization_msgs.msg import Marker
from scipy.signal import place_poles
import fire

eps = 1e-12

class SimpleFeedbackController():
	def __init__(self, save_to_csv=False, publisher_active=False):
		self.i = 0
		self.data_timestamp_prev = 0
		self.e_1_dot = 0.0
		self.e_2_dot = 0.0
		self.angle_error_prev = 0.0
		self.angle_error_prev_2 = 0.0 
		self.vel_error_prev = 0.0
		self.xz_prev = np.array([0.0,0.0])
		self.e_1_prev = 0.0
		self.e_2_prev = 0.0
		self.target_angles_list = []
		self.target_vels_list = []
		self.e_list = []
		""" 
		You can refer to https://github.com/jainachin/bayesrace/blob/master/bayes_race/params/f110.py 
		for these params.
		"""
		self.m = 3.1
		self.Iz = 0.04712
		self.g = 9.81
		self.lf = 0.15875  #0.1621 #0.15875         #0.1653
		self.lr =  0.17145 #0.1425 #0.17145 	      #0.1494

		self.multiplier_f = ((self.m * self.g) / (1 + (self.lf/self.lr)))# *(3.14/180)

		self.multiplier_r = ((self.m*self.g) - ((self.m*self.g) / (1 + (self.lf/self.lr))))# *(3.14/180)

		self.Caf =  4.718 * self.multiplier_f  #4.718 * 0.275  #10.310 * 0.2513  #5.1134
		self.Car =  5.4562  * self.multiplier_r  #5.4562 * 0.254 #8.522 * 0.2779 #4.9883
		self.Vx = 1.0
		self.create_K_matrix()
		self.state_x_list, self.state_z_list, self.state_theta_list = [],[],[]
		self.save_to_csv = save_to_csv
		self.publisher_active = publisher_active
		self.e_1_list,self.e_2_list,self.e_1_dot_list,self.e_2_dot_list = [],[],[],[]
		self.topic_throttle_counter = 0
		self.center_x_list, self.center_y_list, self.r_list = [],[],[]
		self.steering_angle_list = []
		return 

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
		# A = np.array([[0, 1, 0, 0],
		# 	[-7438, -1128, -543.6, 503.3],
		# 	[0, 0, 0, 1],
		# 	[-16400, -2472, -1199, 1110]])
		# B1 = np.array([[-125.9, -2103, 344.5, -2529]]).T
		self.K = place_poles(A, B1, Pol).gain_matrix
		self.A = A
		self.B1 = B1
		return 

	def csv_saver_hook(self):
		csv_array = np.vstack([self.state_x_list,
					self.state_z_list,
					self.state_theta_list,
					self.e_1_list,
					self.e_1_dot_list,
					self.e_2_list,
					self.e_2_dot_list,
					self.center_x_list,
					self.center_y_list,
					self.r_list,
					self.steering_angle_list]).T
		np.savetxt("xztheta_traj.csv",csv_array, delimiter=',', fmt='%.4f')
		return

	def run_controller(self):
		rospy.init_node('controller_node', anonymous=True)
		self.nav_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1) 
		self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
		# self.rate = rospy.Rate(100)
		if (self.save_to_csv):
			rospy.on_shutdown(self.csv_saver_hook)
		self.ps_sub = rospy.Subscriber('/phasespace/rigids_throttled', Rigids, self.feedback_callback)
		rospy.spin()
		return 

	def feedback_callback(self, data):
		if (np.isclose(data.rigids[1].x,0) and np.isclose(data.rigids[1].z,0)):
			"""
			If the phasespace data is incorrect, don't apply the feedback.
			"""
			return 
		self.veh_2d_world_pos = np.array([data.rigids[1].x, data.rigids[1].z])/1000.0 # convert to meters
		r = R.from_quat([data.rigids[1].qx, data.rigids[1].qy, data.rigids[1].qz, data.rigids[1].qw])
		data_timestamp = 1e-9*(data.rigids[1].time)
		# rot_mat = r.as_dcm()
		# self.veh_2d_world_heading = np.arctan2(-(rot_mat[2,0]), np.sqrt((rot_mat[2,1])**2 + (rot_mat[2,2])**2))
		self.veh_2d_world_heading = -r.as_euler('YZX', degrees=False)[0]
		self.i+=1
		e_state = self.compute_e_state_new(i=0, data_timestamp=data_timestamp)
		
		self.topic_throttle_counter += 1
		if ((self.save_to_csv) and (not(self.i%200))):
		 	# print("2D world position is: " + str(self.veh_2d_world_pos))
			print("Euler Heading is: " + str(self.veh_2d_world_heading))
			# print("As rotation vector:" + str(r.apply([1.0,0.0,0.0])))
			# print("Rotation matrix is: \n" + str(r.as_dcm()))
			# print("Euler angle around y-axis is: " + str(np.arctan2(-(rot_mat[2,0]), np.sqrt((rot_mat[2,1])**2 + (rot_mat[2,2])**2))))
			# print(e_state)

		# self.vis_state()
		steering_angle = (self.linear_Cntrl(K_gains=self.K, e_state=e_state))
		if (self.save_to_csv):
			self.steering_angle_list.append(steering_angle)
		# Throttle the steering command to prevent backups.
		if (not np.isnan(steering_angle)):
			drive = AckermannDrive(steering_angle=steering_angle, speed=self.Vx)
			h = std_msgs.msg.Header()
			h.stamp = rospy.Time.now() 
			drive_cmd = AckermannDriveStamped(header=h, drive=drive)
			# if not(self.i%200):
			
			# while ((self.publisher_active) and (not rospy.is_shutdown())):
			if (self.publisher_active):
				rospy.loginfo(drive_cmd)
				rospy.loginfo(e_state)
				self.nav_pub.publish(drive_cmd)
			self.topic_throttle_counter = 0
		elif ((np.isnan(steering_angle)) and (not(self.i%200))):
			print("Steering angle is NaN.")

		marker = Marker()

		marker.header.frame_id = "/map"
		marker.header.stamp = rospy.Time.now()

		# set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
		marker.type = 0
		marker.id = 0

		# Set the scale of the marker
		marker.scale.x = .1
		marker.scale.y = .1
		marker.scale.z = .1

		# Set the color
		marker.color.r = 0.0
		marker.color.g = 1.0
		marker.color.b = 0.0
		marker.color.a = 1.0

		# Set the pose of the marker
		marker.pose.position.x = self.veh_2d_world_pos[0]
		marker.pose.position.y = self.veh_2d_world_pos[1]
		marker.pose.position.z = 0.1
		marker.pose.orientation.x = ((R.from_euler('YZX',[0.0,self.veh_2d_world_heading,0.0], degrees=False)).as_quat())[0]
		marker.pose.orientation.y = ((R.from_euler('YZX',[0.0,self.veh_2d_world_heading,0.0], degrees=False)).as_quat())[1]
		marker.pose.orientation.z = ((R.from_euler('YZX',[0.0,self.veh_2d_world_heading,0.0], degrees=False)).as_quat())[2]
		marker.pose.orientation.w = ((R.from_euler('YZX',[0.0,self.veh_2d_world_heading,0.0], degrees=False)).as_quat())[3]		
		self.marker_pub.publish(marker)
		return 

	def linear_Cntrl(self, K_gains, e_state):
		# \sigma = -Kx 
		cntrl = -np.clip(np.squeeze(np.matmul(K_gains, e_state)),a_min=np.deg2rad(-19.48057),a_max=np.deg2rad(19.48057))
		return cntrl

	def pick_circle_drone_room(self, fixed_r=None, fixed_center=None):
		# Define circle radius
		if fixed_r is None:
			self.r = np.random.randint(low=25, high=35, size=1)
			# self.r = torch.tensor(25, dtype=torch.float64)
			if np.random.randint(2,size=1):
				self.center_x = +self.r
			else:
				self.center_x = -self.r
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
 
		circle_sample_n = 2000
		circle_pts = self.points_on_circumference(center=(self.center_x, self.center_y),
								r=self.r,
								n=circle_sample_n)
		height = 0.121
		self.ref_traj = np.zeros((circle_sample_n,3))
		
		self.ref_traj[:,1] = np.repeat(height, circle_sample_n)
		self.ref_traj[:,0] = circle_pts[:,0]
		self.ref_traj[:,2] = circle_pts[:,1]			
		return 

	def compute_e_state_new(self,i,data_timestamp):
		"""
		Within the error computation function global simulator 
		coordinates are transformed in order to comply with the Rajamani's 
		definitions and the output is in Rajamani's coordinates.
		collision algo source:
		https://cp-algorithms.com/geometry/circle-line-intersection.html
		"""
		local_veh_2d_pos_state = self.veh_2d_world_pos[[-1,0]] # Keep in mind that we are reordering indices. Final variable is in zx format.
		locally_computed_veh_heading_2d = self.veh_2d_world_heading
		if (self.save_to_csv): 
			self.state_x_list.append(local_veh_2d_pos_state[1])
			self.state_z_list.append(local_veh_2d_pos_state[0])
			self.state_theta_list.append(self.veh_2d_world_heading)
			self.center_x_list.append(self.center_x)
			self.center_y_list.append(self.center_y)
			self.r_list.append(self.r)

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
		if pt_a_dist > pt_b_dist:
			intersection_pt = np.hstack((by, bx))
		else:
			intersection_pt = np.hstack((ay, ax))

		center_pt = np.hstack((self.center_y, self.center_x))
		# Heading is important for e_1, not the sign of center y. 
		# Get the vehicle frame of reference coordinates of the intersection point to for e_1.

		T_matrix, inv_T_matrix = self.global_to_local_rf_2d_T_matrix(local_veh_2d_pos_state[[1,0]], locally_computed_veh_heading_2d)

		self.e_1 = np.squeeze(np.matmul(inv_T_matrix,np.expand_dims(np.hstack((intersection_pt[1],intersection_pt[0],[1])),axis=1))[1])
		
		# gt_trajectory_intersection_m = -1.0/((intersection_pt - center_pt)[0] / (intersection_pt - center_pt)[1])
		gt_trajectory_heading = -np.arctan2((intersection_pt - center_pt)[1],(intersection_pt - center_pt)[0])
		self.e_2 = (locally_computed_veh_heading_2d - gt_trajectory_heading)
		# Angular difference formula.
		self.e_2 = ((self.e_2 + np.pi) % (2*np.pi)) - np.pi

		self.e_1_dot = (self.e_1 - self.e_1_prev) / (data_timestamp-self.data_timestamp_prev)
		self.e_2_dot = (self.e_2 - self.e_2_prev) / (data_timestamp-self.data_timestamp_prev)
		self.local_veh_2d_pos_state_prev = local_veh_2d_pos_state

		self.e_1_prev = self.e_1
		self.e_2_prev = self.e_2
		# if (np.isnan(self.e_2) or np.isnan(self.e_1) or np.isnan(self.e_2_dot) or np.isnan(self.e_1_dot)):
			# print("NaN detected") 
		stacked_e = np.stack((self.e_1, self.e_1_dot, self.e_2, self.e_2_dot))
		if (self.save_to_csv): 
			self.e_1_list.append(self.e_1)
			self.e_1_dot_list.append(self.e_1_dot)
			self.e_2_list.append(self.e_2)
			self.e_2_dot_list.append(self.e_2_dot)
			self.e_list.append(stacked_e)

		self.data_timestamp_prev = data_timestamp
		return stacked_e


	def points_on_circumference(self, center=(0, 0), r=50, n=100):
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

	# def vis_state(self):
	# 	ref_traj_circle = patches.Circle((self.center_x, self.center_y), radius=self.r, color='navy', fill=False)
	# 	self.ax.add_patch(ref_traj_circle)
	# 	plt.grid(True)
	# 	# self.ax.autoscale(True)
	# 	self.ax.axis('equal')
	# 	self.fig.tight_layout()
	#	plt.show()
	#	plt.clf()

	def global_to_local_rf_2d_T_matrix(self, local_pos, local_heading):
		o_0 = np.hstack((np.cos(local_heading), -np.sin(local_heading), local_pos[0]))
		o_1 = np.hstack((np.sin(local_heading),np.cos(local_heading), local_pos[1]))
		o_2 = np.array([0.0,0.0,1.0], dtype=np.float64)
		output = np.vstack((o_0, o_1, o_2))
		inverse_output = np.linalg.pinv(output)
		return output, inverse_output


def call_all_funcs(save_to_csv=True, publisher_active=False):
	sfc = SimpleFeedbackController(save_to_csv=save_to_csv, publisher_active=publisher_active)
	center, radius = sfc.three_pts_define_circle(p1=[3.370, 0.739], 
					p2=[5.005, -0.896],
					p3=[3.370, -2.571])
	sfc.pick_circle_drone_room(fixed_r=radius, fixed_center=center)
	sfc.run_controller()

if __name__ == "__main__":
	fire.Fire(call_all_funcs)
