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
import optuna
from multiprocessing import Pool
import pandas as pd 
from pytorch3d.loss import chamfer_distance
start_time = datetime.now().strftime("%Y-%m-%d%H-%M-%S")
script_name = os.path.split(sys.argv[0])[-1].split(".")[0]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
eps = 1e-12
# from discrete import FastDiscreteFrechetMatrix, earth_haversine, DiscreteFrechet
from soft_dtw_cuda import SoftDTW
from optuna_train import no_grad_F1TenthModel
agent = 0
def relu(x):
	return np.maximum(0,x)

def find_nearest_idx(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

class validation_F1TenthModel(no_grad_F1TenthModel):
	def gradfree_validate(self, param_list=['C_Sf','C_Sr'], plot_pdf=True, params_csvs_folder=None):
		self.plot_pdf = plot_pdf
		self.param_list = param_list		
		self.mse_loss_func = nn.MSELoss(reduction="mean")
		self.dtw_loss_func = SoftDTW(use_cuda=False, gamma=0.1)
		read_params_dict = {p:[] for p in param_list}
		if params_csvs_folder is not None:
			for f in [os.path.join(params_csvs_folder,i) for i in os.listdir(params_csvs_folder) if i.endswith(".csv")]:
				df = pd.read_csv(f,index_col=0)
				for p in param_list:
					read_params_dict[p].append(np.mean(df[p].iloc[-10:-1]))
			for k,v in read_params_dict.items():
				self.params[k] = np.mean(v)
		else:
			self.params = self.official_params
		print(self.params)
		for epoch in np.arange(len(self.bagnames)):
			loss = 0.0
			print(f"Epoch:{epoch}")
			print(f"Using {self.bagnames[epoch]}")
			cmds_arr = np.loadtxt(f"{self.bagnames[epoch]}_cmds.csv", delimiter=',')
			traj_arr = np.loadtxt(f"{self.bagnames[epoch]}_traj.csv", delimiter=',')
			# traj_arr= traj_arr[:int(len(traj_arr)/2),:] # Shorten the trajectory.
			traj_arr = traj_arr[traj_arr[:,0].nonzero()[0],:]
			t_start = cmds_arr[np.abs(cmds_arr[:,1]) > 0.02,2][0]+2.0
			t_end = cmds_arr[np.abs(cmds_arr[:,1]) > 0.02,2][-1]
			traj_arr = traj_arr[traj_arr[:,2]>t_start,:]
			traj_arr = traj_arr[traj_arr[:,2]<t_end,:]
			t_start_traj = traj_arr[0,2]
			t_end_traj = traj_arr[-1,2]
			ref_traj_interped = interp1d(x=(traj_arr[:,[2]]-traj_arr[0,[2]]).squeeze(),y=traj_arr[:,[0,1]]/1000.0,axis=0, kind='cubic')
			self.state_2_table = torch.zeros_like(torch.from_numpy(np.arange(0,t_end_traj-t_start_traj,self.time_step)))
			for idx,ts in enumerate(np.arange(0,t_end_traj-t_start_traj,self.time_step)):
				self.state_2_table[idx] = -cmds_arr[find_nearest_idx(cmds_arr[:,2],ts + t_start_traj),1]
			delta0 = -cmds_arr[np.abs(cmds_arr[:,1]) > 0.02,1][0] # Keep in mind steering is inverted between real data and xy simulated data.
			vel0 = 1.0
			Psi0 = traj_arr[0,3] # initial cond 
			dotPsi0 = 0.0
			beta0 = 0.0
			sx0 = traj_arr[0,0]/1000.0 # initial cond 
			sy0 = traj_arr[0,1]/1000.0 # initial cond 
			initial_state = np.hstack([sx0,sy0,delta0,vel0,Psi0,dotPsi0,beta0])
			u = [0.0, 0.0] # Do not give any steering velocity or longitidunal acceleration.
			self.state = initial_state
			self.states = []		
			for idx,ts in enumerate(np.arange(0,t_end_traj-t_start_traj,self.time_step)):
				self.state[2] = self.state_2_table[idx]
				self.states.append(self.state)
				f = self.func_ST(
					x=self.state, 
					t=ts, 
					u=u, 
					mu=self.params["mu"],
					C_Sf=self.params["C_Sf"],
					C_Sr=self.params["C_Sr"],
					lf=self.params["lf"],
					lr=self.params["lr"],
					h=self.params["h"],
					m=self.params["m"],
					I=self.params["I"],
					s_min=self.params["s_min"],
					s_max=self.params["s_max"],
					sv_min=self.params["sv_min"],
					sv_max=self.params["sv_max"],
					v_switch=self.params["v_switch"],
					a_max=self.params["a_max"],
					v_min=self.params["v_min"],
					v_max=self.params["v_max"])
				self.euler_integrator(f)
			# print(f"\nC_Sf: {self.params['C_Sf']}, C_Sr: {self.params['C_Sr']}, I: {self.params['I']}")
			# loss = self.mse_loss_func(torch.as_tensor(np.vstack(self.states)[:,[0,1]]), 
			# 			torch.tensor(self.ref_traj_interped(np.arange(0,self.t_end_traj-self.t_start_traj,self.time_step)),requires_grad=False, dtype=torch.float32)).detach().cpu().numpy()
			loss += (self.dtw_loss_func(torch.as_tensor(np.expand_dims(np.vstack(self.states)[::100,[0,1]],axis=0),dtype=torch.float32),
			    torch.as_tensor(np.expand_dims(ref_traj_interped(np.arange(0,t_end_traj-t_start_traj,100*self.time_step)),
				axis=0),dtype=torch.float32)).mean().detach().cpu().numpy())
			loss += 100*chamfer_distance(torch.unsqueeze(torch.from_numpy(np.vstack(self.states)[:,[0,1]]),dim=0).float(),
			   	torch.unsqueeze(torch.tensor(ref_traj_interped(np.arange(0,t_end_traj-t_start_traj,self.time_step)),requires_grad=False, dtype=torch.float32),dim=0).float())[0].detach().cpu().numpy()

			for key in ['C_Sf','C_Sr','lf','lr']:
				print(f"{key} value:{self.params[key]:.4f}")
				(self.optimization_dict[key]).append(self.params[key])

			if ((self.plot_pdf)):
				self.plot_agent_states(traj_arr, traj_interpolation=ref_traj_interped, t_start_traj=t_start_traj, t_end_traj=t_end_traj, epoch_num=epoch)

			if ((epoch>0)and(self.plot_pdf)and(not(epoch%10))):
				self.plot_loss()
				self.sns_params()

			self.losses_list.append(loss)
			print(f"Loss: {loss:.4f}")
		self.save_csv_data()
		print(f"== Mean validation loss is: {np.mean(self.losses_list):.4f} ==")
		return 
	
	def plot_agent_states(self, traj_arr=None, traj_interpolation=None, t_start_traj=None, t_end_traj=None, epoch_num=None):
		np_states = np.array(np.vstack(self.states))
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
			for key in self.param_list:
				added_text += f"{key}:{np.round(self.params[key],3)} "
			plt.text(3,y_coord,s=added_text)
			plt.savefig(f"pdf/valid_e_{epoch_num}_{str(start_time).split('.')[0]}.pdf")
		plt.clf()
		return 
	
	def sns_params(self):
		for key in self.param_list:
			np_vals = np.array(self.optimization_dict[key])
			sns.set_style('darkgrid')
			sns.lineplot(x=np.arange(len(np_vals)),y=np_vals, label=f"{key}")
			plt.xlabel(f"Epochs")
			plt.ylabel(f"{key}")
			plt.legend()
			plt.hlines(self.official_params[key],xmin=0,xmax=len(np_vals)-1,color="k")
			plt.tight_layout()
			plt.savefig(f"pdf/valid_{key}_{str(start_time).split('.')[0]}.pdf")
			plt.clf()
		return 
	
	def save_csv_data(self):
		df = pd.DataFrame()
		df['loss'] = np.array(self.losses_list)
		for key in self.param_list:
			np_vals = np.array(self.optimization_dict[key])
			df[key] = np_vals
		df.to_csv(f"csv/gradfree/valid_{str(start_time).split('.')[0]}.csv",index_label="epoch")
		return 

	def plot_loss(self):
		plt.plot(self.losses_list)

		plt.ylabel(f"Validation Loss")
		plt.xlabel(f"Validation Epochs")
		plt.tight_layout()
		plt.savefig(f"pdf/valid_loss_{str(start_time).split('.')[0]}.pdf")
		plt.clf()
		return
def execute_validation():
	global start_time, agent
	with torch.no_grad():
		agent = validation_F1TenthModel(data_dir="bag/bags_2023_02_28/wiggle/")
		train_start_time = time.time()
		start_time = datetime.now().strftime("%Y-%m-%d%H-%M-%S")
		# agent.gradfree_validate(param_list=['C_Sf','C_Sr','lr','lf'],plot_pdf=True,params_csvs_folder=None)
		# agent.gradfree_validate(param_list=['C_Sf','C_Sr'],plot_pdf=True,params_csvs_folder="csv/gradfree/exp_9/")
		agent.gradfree_validate(param_list=['C_Sf','C_Sr','lr','lf'],plot_pdf=True,params_csvs_folder="csv/gradbased/exp_11-untaped/")
		print(f"Training took {time.time()-train_start_time} seconds")
		
if __name__ == '__main__':
	execute_validation()
