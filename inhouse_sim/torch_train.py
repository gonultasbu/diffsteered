# Copyright 2020 Technical University of Munich, Professorship of Cyber-Physical Systems, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Prototype of vehicle dynamics functions and classes for simulating 2D Single
Track dynamic model
Following the implementation of commanroad's Single Track Dynamics model
Original implementation: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/
Author: Hongrui Zheng
"""
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
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from scipy.signal import place_poles
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from pytorch3d.loss import chamfer_distance
start_time = datetime.now().strftime("%Y-%m-%d%H-%M-%S")
script_name = os.path.split(sys.argv[0])[-1].split(".")[0]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
eps = 1e-12
torch.pi = torch.acos(torch.zeros(1)).item() * 2
# from discrete import FastDiscreteFrechetMatrix, earth_haversine, DiscreteFrechet
from soft_dtw_cuda import SoftDTW
agent = 0
def find_nearest_idx(array, value):
	idx = (np.abs(array - value)).argmin()
	return idx

class F1TenthModel():
	def __init__(self, data_dir, validation_dir):
		# gravity constant m/s^2
		self.g = torch.tensor(9.81, dtype=torch.float32, requires_grad=False, device=dev)
		self.bagnames = [os.path.join(data_dir,f.split('.')[0]) for f in os.listdir(data_dir) if f.endswith('.bag')]
		self.v_bagnames = [os.path.join(validation_dir,f.split('.')[0]) for f in os.listdir(validation_dir) if f.endswith('.bag')]
		self.official_params = {'mu': 1.0489, 
			'C_Sf': 4.718, 
			'C_Sr': 5.4562, 
			'lf': 0.15875*28.57, 
			'lr': 0.17145*28.57, 
			'h': 0.074, 
			'm': 3.1, 
			'I': 0.04712, 
			's_min': -0.4189, 
			's_max': 0.4189, 
			'sv_min': -3.2, 
			'sv_max': 3.2, 
			'v_switch': 7.319, 
			'a_max': 9.51, 
			'v_min':-5.0, 
			'v_max': 20.0, 
			'width': 0.31, 
			'length': 0.58}
		c_random_init = np.random.uniform(low=2.0, high=10.0, size=1)[0]
		l_random_init = np.random.uniform(low=2.857, high=10.0, size=1)[0]
		self.params = {'mu': torch.tensor(1.0489, dtype=torch.float32, requires_grad=False, device=dev),  
			'C_Sf': torch.as_tensor(c_random_init,dtype=torch.float32).requires_grad_(True),
			'C_Sr': torch.as_tensor(c_random_init,dtype=torch.float32).requires_grad_(True), 
			'lf': torch.as_tensor(l_random_init).requires_grad_(True), 
			'lr': torch.as_tensor(l_random_init).requires_grad_(True),
			'm': torch.tensor(3.1, dtype=torch.float32, requires_grad=False, device=dev), 
			'I': torch.tensor(0.04712, dtype=torch.float32, requires_grad=False, device=dev),
			'h': torch.tensor(0.074, dtype=torch.float32, requires_grad=False, device=dev), 
			's_min': torch.tensor(-0.4189, dtype=torch.float32, requires_grad=False, device=dev), 
			's_max': torch.tensor(0.4189, dtype=torch.float32, requires_grad=False, device=dev), 
			'sv_min': torch.tensor(-3.2, dtype=torch.float32, requires_grad=False, device=dev), 
			'sv_max': torch.tensor(3.2, dtype=torch.float32, requires_grad=False, device=dev), 
			'v_switch': torch.tensor(7.319, dtype=torch.float32, requires_grad=False, device=dev), 
			'a_max': torch.tensor(9.51, dtype=torch.float32, requires_grad=False, device=dev), 
			'v_min': torch.tensor(-5.0, dtype=torch.float32, requires_grad=False, device=dev), 
			'v_max': torch.tensor(20.0, dtype=torch.float32, requires_grad=False, device=dev), 
			'width': torch.tensor(0.31, dtype=torch.float32, requires_grad=False, device=dev), 
			'length': torch.tensor(0.58, dtype=torch.float32, requires_grad=False, device=dev)}
		self.optimization_dict = {'C_Sf':[],'C_Sr':[],'lf':[],'lr':[],'m':[],'I':[]}
		self.time_step = 0.002
		self.optimized_params_list = []
		for key,value in self.params.items():
			if value.requires_grad:
				if ((key=="C_Sf")or(key=="C_Sr")):
					self.optimized_params_list.append({'params': value, 'lr':0.2})
				elif (key=="I"):
					self.optimized_params_list.append({'params': value, 'lr':0.01})
				elif ((key=="lf")or(key=="lr")or(key=="m")):
					self.optimized_params_list.append({'params': value, 'lr':1.5})
			else: 
				self.params[key] = torch.tensor(self.official_params[key], dtype=torch.float32, requires_grad=False, device=dev)
		self.optimizer = torch.optim.Adam(self.optimized_params_list)	
		self.losses_list = []
		self.v_losses_list = []
		return 

	# @profile
	def vehicle_dynamics_st(self, x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I):
		"""
		Single Track Dynamic Vehicle Dynamics.

			Args:
				x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
					x1: x position in global coordinates
					x2: y position in global coordinates
					x3: steering angle of front wheels
					x4: velocity in x direction
					x5: yaw angle
					x6: yaw rate
					x7: slip angle at vehicle center
				u (numpy.ndarray (2, )): control input vector (u1, u2)
					u1: steering angle velocity of front wheels
					u2: longitudinal acceleration

			Returns:
				f (numpy.ndarray): right hand side of differential equations
		"""

		# gravity constant m/s^2
		g = torch.tensor(9.81, dtype=torch.float32, requires_grad=False, device=dev)

		# constraints
		u = u_init
		relu_lf = relu(lf)
		relu_lr =  relu(lr)
		relu_I = relu(I)
		relu_C_Sr = relu(C_Sr)
		relu_C_Sf = relu(C_Sf)

		# system dynamics
		f = torch.hstack([x[3]*torch.cos(x[6] + x[4]),
			x[3]*torch.sin(x[6] + x[4]),
			u[0],
			u[1],
			x[5],
			-mu*m/(x[3]*relu_I*(relu_lr+relu_lf))*(relu_lf**2*relu_C_Sf*(self.g*relu_lr-u[1]*h) + relu_lr**2*relu_C_Sr*(self.g*relu_lf + u[1]*h))*x[5] \
				+mu*m/(relu_I*(relu_lr+relu_lf))*(relu_lr*relu_C_Sr*(self.g*relu_lf + u[1]*h) - relu_lf*relu_C_Sf*(self.g*relu_lr - u[1]*h))*x[6] \
				+mu*m/(relu_I*(relu_lr+relu_lf))*relu_lf*relu_C_Sf*(self.g*relu_lr - u[1]*h)*x[2],
			(mu/(x[3]**2*(relu_lr+relu_lf))*(relu_C_Sr*(self.g*relu_lf + u[1]*h)*relu_lr - relu_C_Sf*(self.g*relu_lr - u[1]*h)*relu_lf)-1)*x[5] \
				-mu/(x[3]*(relu_lr+relu_lf))*(relu_C_Sr*(self.g*relu_lf + u[1]*h) + relu_C_Sf*(self.g*relu_lr-u[1]*h))*x[6] \
				+mu/(x[3]*(relu_lr+relu_lf))*(relu_C_Sf*(self.g*relu_lr-u[1]*h))*x[2]])
		return f

	def func_ST(self, x, t, u, mu, C_Sf, C_Sr, lf, lr, h, m, I):
		f = self.vehicle_dynamics_st(x, u, mu, C_Sf, C_Sr, lf, lr, h, m, I)
		return f

	def euler_integrator(self, f):
		self.state = self.state + self.time_step * f
		# Bound the yaw angle between [-pi,pi]
		self.state[4] = self.state[4] - (torch.ceil((self.state[4] + torch.pi)/(2*torch.pi))-1)*2*torch.pi
		return 

	def plot_agent_states(self, traj_arr=None, traj_interpolation=None, t_start_traj=None, t_end_traj=None, epoch_num=None):
		np_states = np.array(torch.vstack(self.states).detach().cpu().numpy())
		plt.scatter(x=np_states[:,0], y=np_states[:,1],
			c="red",
			s=0.5,
			linewidths=0.5)
		legend_list = ["Simulated trajectory"]
		if (traj_arr is not None):
			plt.scatter(x=traj_arr[:,0]/1000.0, y=traj_arr[:,1]/1000.0,
				c="navy",
				s=0.5,
				linewidths=0.5)		
			legend_list.append("Real robot datapoints")
		if (traj_interpolation is not None):
			plt.scatter(
				x=traj_interpolation(np.arange(0,t_end_traj-t_start_traj,agent.time_step))[:,0],
				y=traj_interpolation(np.arange(0,t_end_traj-t_start_traj,agent.time_step))[:,1],
				c="green",
				s=0.5,
				linewidths=0.5,
				alpha=0.1)
			legend_list.append("Interpolated robot datapoints")
		plt.legend(legend_list)
		plt.axis('equal')
		# plt.grid()
		plt.ylabel("y-axis")
		plt.xlabel("x-axis")
		plt.tight_layout()
		# plt.show()
		y_coord=-1.25
		# plt.xlim(-5, 5)
		# plt.ylim(-5, 5)
		added_text = f""
		if epoch_num is not None:
			for key,value in self.params.items():
				if value.requires_grad:
					if key in ["lr","lf"]:
						added_text += f"{key}:{np.round(value.detach().cpu().numpy()/28.57,3)} "
					else:
						added_text += f"{key}:{np.round(value.detach().cpu().numpy(),3)} "
			plt.text(3,y_coord,s=added_text)
			plt.savefig(f"pdf/e_{epoch_num}_{str(start_time).split('.')[0]}.pdf")
		plt.clf()
		return 
	
	def plot_loss(self):
		plt.plot(self.losses_list)
		# plt.grid()
		# plt.ylim(bottom=0)
		# plt.xlim(left=0)
		plt.ylabel(f"Loss")
		plt.xlabel(f"Epochs, batch size: {self.batch_size}")
		plt.tight_layout()
		plt.savefig(f"pdf/loss_{str(start_time).split('.')[0]}.pdf")
		plt.clf()
		return

	def sns_params(self):
		for key,value in self.params.items():
			if value.requires_grad:
				np_vals = np.array(self.optimization_dict[key])
				sns.set_style('darkgrid')
				sns.lineplot(x=np.arange(len(np_vals)),y=np_vals, label=f"${key}$")
				plt.xlabel(f"Epochs")
				plt.ylabel(f"{key}")
				plt.legend()
				plt.hlines(self.official_params[key],xmin=0,xmax=len(np_vals)-1,color="k")
				plt.tight_layout()
				plt.savefig(f"pdf/{key}_{str(start_time).split('.')[0]}.pdf")
				plt.clf()
		# plt.plot(np_arr_e_list)
		# plt.grid(True)
		# plt.ylim(bottom=-2.0, top=2.0)
		# e_profile.set(xlim=(0, 7000))
		# plt.legend(["e1", "e1_dot", "e2", "e2_dot"])
		# plt.legend()
		return 
	
	# @profile
	def train(self, epochs_num=1000, batch_size=1, plot_pdf=True):
		self.batch_size = batch_size
		# Poor Man's EarlyStopping Implementation
		best_loss = np.Inf
		self.best_v_loss = np.Inf
		es_patience = 50
		es_patience_ctr = 0
		# Poor Man's EarlyStopping Implementation
		# mse_loss_func = nn.MSELoss(reduction="sum")
		self.dtw_loss_func = SoftDTW(use_cuda=False, gamma=0.1)

		for epoch in tqdm(np.arange(epochs_num)):
			loss = torch.tensor(0.0,dtype=torch.float32)

			for key,value in self.params.items(): 
				if value.requires_grad:
					if (key in ["lr","lf"]):
						try: print(f"{key} value:{value.detach().cpu().numpy()/28.57:.4f}")
						except: pass
						(self.optimization_dict[key]).append(value.detach().cpu().clone().numpy()/28.57)
					else:
						try: print(f"{key} value:{value.detach().cpu().numpy():.4f}")
						except: pass
						(self.optimization_dict[key]).append(value.detach().cpu().clone().numpy())
					

			for i_batch in np.arange(batch_size):
				print(f"Epoch:{epoch}, batch:{i_batch}")
				print(f"Using {self.bagnames[((epoch+1)+i_batch)%len(self.bagnames)]}")
				cmds_arr = np.loadtxt(f"{self.bagnames[((epoch+1)+i_batch)%len(self.bagnames)]}_cmds.csv", delimiter=',')
				traj_arr = np.loadtxt(f"{self.bagnames[((epoch+1)+i_batch)%len(self.bagnames)]}_traj.csv", delimiter=',')
				# traj_arr= traj_arr[:int(len(traj_arr)/2),:] # Shorten the trajectory.
				traj_arr = traj_arr[traj_arr[:,0].nonzero()[0],:]
				t_start = cmds_arr[np.abs(cmds_arr[:,1]) > 0.02,2][0]+2.0
				t_end = cmds_arr[np.abs(cmds_arr[:,1]) > 0.02,2][-1]
				traj_arr = traj_arr[traj_arr[:,2]>t_start,:]
				traj_arr = traj_arr[traj_arr[:,2]<t_end,:]
				t_start_traj = traj_arr[0,2]
				t_end_traj = traj_arr[-1,2]
				# filter out the beginning and end without commands
				ref_traj_interped = interp1d(x=(traj_arr[:,[2]]-traj_arr[0,[2]]).squeeze(),y=traj_arr[:,[0,1]]/1000.0,axis=0, kind='cubic')
				self.state_2_table = torch.zeros_like(torch.from_numpy(np.arange(0,t_end_traj-t_start_traj,self.time_step)))
				for idx,ts in enumerate(np.arange(0,t_end_traj-t_start_traj,self.time_step)):
					self.state_2_table[idx] = -cmds_arr[find_nearest_idx(cmds_arr[:,2],ts + t_start_traj),1]
				delta0 = torch.tensor(-cmds_arr[np.abs(cmds_arr[:,1]) > 0.02,1][0], dtype=torch.float32, requires_grad=False, device=dev) # Keep in mind steering is inverted between real data and xy simulated data.
				vel0 = torch.tensor(1.0, dtype=torch.float32, requires_grad=False, device=dev)
				Psi0 = torch.tensor(traj_arr[0,3], dtype=torch.float32, requires_grad=False, device=dev) # initial cond 
				dotPsi0 = torch.tensor(0.0, dtype=torch.float32, requires_grad=False, device=dev)
				beta0 = torch.tensor(0.0, dtype=torch.float32, requires_grad=False, device=dev)
				sx0 = torch.tensor(traj_arr[0,0]/1000.0, dtype=torch.float32, requires_grad=False, device=dev) # initial cond 
				sy0 = torch.tensor(traj_arr[0,1]/1000.0, dtype=torch.float32, requires_grad=False, device=dev) # initial cond 
				initial_state = torch.hstack([sx0,sy0,delta0,vel0,Psi0,dotPsi0,beta0])
				u = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=False, device=dev) # Do not give any steering velocity or longitidunal acceleration.
				self.state = initial_state
				self.states = []

				for idx,ts in enumerate(np.arange(0,t_end_traj-t_start_traj,self.time_step)):
					self.state[2] = self.state_2_table[idx]
					self.states.append(self.state)
					f = self.func_ST(x=self.state, 
						t=ts, 
						u=u, 
						mu=self.params["mu"],
						C_Sf=self.params["C_Sf"],
						C_Sr=self.params["C_Sr"],
						lf=self.params["lf"]/28.57,
						lr=self.params["lr"]/28.57,
						h=self.params["h"],
						m=self.params["m"],
						I=self.params["I"])
					self.euler_integrator(f)
					
				loss += (self.dtw_loss_func(torch.unsqueeze(torch.vstack(self.states)[::100,[0,1]],dim=0).float(),
					torch.unsqueeze(torch.tensor(ref_traj_interped(np.arange(0,t_end_traj-t_start_traj,100*self.time_step)),
					requires_grad=False, dtype=torch.float32, device=dev),dim=0).float()).mean())/batch_size
				
				loss += 100*chamfer_distance(torch.unsqueeze(torch.vstack(self.states)[:,[0,1]],dim=0).float(),
				   	torch.unsqueeze(torch.tensor(ref_traj_interped(np.arange(0,t_end_traj-t_start_traj,self.time_step)),requires_grad=False, dtype=torch.float32, device=dev),dim=0).float())[0]/self.batch_size

			if (best_loss > loss.detach().cpu().numpy()):
				best_loss = loss.detach().cpu().numpy()
				es_patience_ctr = 0
				print(f"Loss improved!")
			else:
				es_patience_ctr += 1
				# if es_patience <= es_patience_ctr: 
					# print(f"EarlyStopping triggered, terminating training")
					# return 
			if ((plot_pdf)and(not epoch%10)):
				# with torch.no_grad():
					# self.validate(epoch)
				if (batch_size==1):
					self.plot_agent_states(traj_arr, traj_interpolation=ref_traj_interped, t_start_traj=t_start_traj, t_end_traj=t_end_traj, epoch_num=epoch)

			if ((epoch>0)and(plot_pdf)and(not(epoch%10))):
				self.plot_loss()
				self.sns_params()
				
			print(f"Loss is {loss.detach().cpu().numpy():.4f}, Best loss is {best_loss:.4f}")
			for key,value in self.params.items():
				if value.requires_grad:
					try: print(f"{key}.grad:{value.grad.detach().cpu().numpy():.4f}")
					except: pass
					
			self.optimizer.zero_grad()
			loss.backward()
			self.losses_list.append(loss.detach().cpu().numpy())
			self.optimizer.step()

		self.save_csv_data()
		return
	
	def validate(self, epoch):
		print(f"Starting validation round after epoch: {epoch}")
		for sample_idx in np.arange(len(self.v_bagnames)):
			v_loss = torch.tensor(0.0,dtype=torch.float32)
			print(f"Epoch:{epoch}")
			print(f"Using {self.bagnames[(epoch+1)%len(self.bagnames)]}")
			cmds_arr = np.loadtxt(f"{self.v_bagnames[sample_idx]}_cmds.csv", delimiter=',')
			traj_arr = np.loadtxt(f"{self.v_bagnames[sample_idx]}_traj.csv", delimiter=',')
			# traj_arr= traj_arr[:int(len(traj_arr)/2),:] # Shorten the trajectory.
			traj_arr = traj_arr[traj_arr[:,0].nonzero()[0],:]
			t_start = cmds_arr[np.abs(cmds_arr[:,1]) > 0.02,2][0]+2.0
			t_end = cmds_arr[np.abs(cmds_arr[:,1]) > 0.02,2][-1]
			traj_arr = traj_arr[traj_arr[:,2]>t_start,:]
			traj_arr = traj_arr[traj_arr[:,2]<t_end,:]
			t_start_traj = traj_arr[0,2]
			t_end_traj = traj_arr[-1,2]
			# filter out the beginning and end without commands
			ref_traj_interped = interp1d(x=(traj_arr[:,[2]]-traj_arr[0,[2]]).squeeze(),y=traj_arr[:,[0,1]]/1000.0,axis=0, kind='cubic')
			self.state_2_table = torch.zeros_like(torch.from_numpy(np.arange(0,t_end_traj-t_start_traj,self.time_step)))
			for idx,ts in enumerate(np.arange(0,t_end_traj-t_start_traj,self.time_step)):
				self.state_2_table[idx] = -cmds_arr[find_nearest_idx(cmds_arr[:,2],ts + t_start_traj),1]
			delta0 = torch.tensor(-cmds_arr[np.abs(cmds_arr[:,1]) > 0.02,1][0], dtype=torch.float32, requires_grad=False, device=dev) # Keep in mind steering is inverted between real data and xy simulated data.
			vel0 = torch.tensor(1.0, dtype=torch.float32, requires_grad=False, device=dev)
			Psi0 = torch.tensor(traj_arr[0,3], dtype=torch.float32, requires_grad=False, device=dev) # initial cond 
			dotPsi0 = torch.tensor(0.0, dtype=torch.float32, requires_grad=False, device=dev)
			beta0 = torch.tensor(0.0, dtype=torch.float32, requires_grad=False, device=dev)
			sx0 = torch.tensor(traj_arr[0,0]/1000.0, dtype=torch.float32, requires_grad=False, device=dev) # initial cond 
			sy0 = torch.tensor(traj_arr[0,1]/1000.0, dtype=torch.float32, requires_grad=False, device=dev) # initial cond 
			initial_state = torch.hstack([sx0,sy0,delta0,vel0,Psi0,dotPsi0,beta0])
			u = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=False, device=dev) # Do not give any steering velocity or longitidunal acceleration.
			self.state = initial_state
			self.states = []
			for idx,ts in enumerate(np.arange(0,t_end_traj-t_start_traj,self.time_step)):
				self.state[2] = self.state_2_table[idx]
				self.states.append(self.state)
				f = self.func_ST(x=self.state, 
					t=ts, 
					u=u, 
					mu=self.params["mu"],
					C_Sf=self.params["C_Sf"],
					C_Sr=self.params["C_Sr"],
					lf=self.params["lf"]/28.57,
					lr=self.params["lr"]/28.57,
					h=self.params["h"],
					m=self.params["m"],
					I=self.params["I"])
				self.euler_integrator(f)
			v_loss += (self.dtw_loss_func(torch.unsqueeze(torch.vstack(self.states)[::100,[0,1]],dim=0).float(),
				torch.unsqueeze(torch.tensor(ref_traj_interped(np.arange(0,t_end_traj-t_start_traj,100*self.time_step)),
				requires_grad=False, dtype=torch.float32, device=dev),dim=0).float()).mean()) / len(self.v_bagnames)
			v_loss += 100*chamfer_distance(torch.unsqueeze(torch.vstack(self.states)[:,[0,1]],dim=0).float(),
				torch.unsqueeze(torch.tensor(ref_traj_interped(np.arange(0,t_end_traj-t_start_traj,self.time_step)),requires_grad=False, dtype=torch.float32, device=dev),dim=0).float())[0] / len(self.v_bagnames)
		
		if (self.best_v_loss > v_loss.detach().cpu().numpy()):
			self.best_v_loss = v_loss.detach().cpu().numpy()
			print(f"Validation Loss improved!")
		print(f"=== Validation Loss is {v_loss.detach().cpu().numpy():.4f}, Best validation loss is {self.best_v_loss:.4f} ===")
		self.v_losses_list.append(v_loss.detach().cpu().numpy())
		return 


	def save_csv_data(self):
		df = pd.DataFrame()
		df['loss'] = np.array(self.losses_list)
		df['bs'] = np.repeat(self.batch_size, len(np.array(self.losses_list)))
		# df['v_loss'] = np.array(self.v_losses_list)
		for key,value in self.params.items():
			if value.requires_grad:
				np_vals = np.array(self.optimization_dict[key])
				df[key] = np_vals
		df.to_csv(f"csv/gradbased/{str(start_time).split('.')[0]}.csv",index_label="epoch")
		return 


def execute_training():
	global start_time, agent
	print(f"{torch.cuda.is_available()}")
	global agent
	agent = F1TenthModel(data_dir="bag/bags_2023_02_28/leftright/", 
		validation_dir="bag/bags_2023_02_27/wiggle2/validation/")
	train_start_time = time.time()
	start_time = datetime.now().strftime("%Y-%m-%d%H-%M-%S")
	agent.train(100,plot_pdf=True,batch_size=4)
	print(f"Training took {time.time()-train_start_time} seconds")

if __name__ == '__main__':
	# dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	dev = torch.device("cpu")
	for _ in np.arange(5):
		execute_training()
