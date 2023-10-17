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
import pandas as pd 
from pytorch3d.loss import chamfer_distance
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
start_time = datetime.now().strftime("%Y-%m-%d%H-%M-%S")
script_name = os.path.split(sys.argv[0])[-1].split(".")[0]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')

def errs_csv(fnames):
    labels = [ "Ours","Ground truth parameters"]
    for idx,fname in enumerate(fnames):
        df = pd.read_csv(fname, header=None)
        e1 = df[3].to_numpy()[:1000]
        t = df[11].to_numpy()[:1000]
        t=t-t[0]
        # interped = interp1d(kind='linear', x=np.arange(len(e1)),y=e1)(x=np.arange(len(e1)))
        sns.set_style('darkgrid')
        sns.lineplot(x=t,y=e1,label=labels[idx])
        # sns.lineplot(x=np.arange(len(e1)),y=interped)
    plt.xlabel(f"Time(s)")
    plt.ylabel(f"$e_1$(m)")
    # plt.xlim()
    plt.ylim([-0.1,0.0])
    # plt.legend()
    # plt.hlines(self.official_params[key],xmin=0,xmax=len(np_vals)-1,color="k")
    plt.tight_layout()
    plt.savefig(f"C:\\Users\\burak\\Documents\\GitHub\\rsn-f1tenth\\experiments_results\\untaped\\Burak_A_B\\five\\untaped_ep.pdf")
    plt.clf()
    return 

def ref_traj_csv(fnames):

    # labels = [ "Ours","Ground truth parameters"]
    for idx,fname in enumerate(fnames):
        df = pd.read_csv(fname, header=None)
        traj_x = df[0].to_numpy()[::3]
        traj_y = df[1].to_numpy()[::3]
        radius = df[9].to_numpy()[0]
        center_x = df[7].to_numpy()[0]
        center_y = df[8].to_numpy()[0]
        # sns.set_style('darkgrid')
        # ax=sns.scatterplot(x=traj_x,y=traj_y,s=7,c='red')
        fig, ax = plt.subplots() 
        robot_t=ax.scatter(x=np.array(traj_x),y=np.array(traj_y),
			c="red",
            s=0.5,
            linewidths=0.5)
        # sns.lineplot(x=np.arange(len(e1)),y=interped)
    circle1 = plt.Circle(xy=(center_x, center_y), radius=radius, color='green', fill=False)
    ax.add_patch(circle1)
    plt.xlabel(f"x(m)")
    plt.ylabel(f"y(m)")
    plt.axis('equal')

    custom_lines = [Line2D([0], [0], color='red', lw=2),
                Line2D([0], [0], color='green', lw=2)]
    plt.legend(custom_lines,["Robot trajectory","Reference"])
    # plt.xlim()
    # plt.ylim([-0.1,0.0])
    # plt.legend()
    # plt.hlines(self.official_params[key],xmin=0,xmax=len(np_vals)-1,color="k")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"C:\\Users\\burak\\Documents\\GitHub\\rsn-f1tenth\\experiments_results\\untaped\\Burak_A_B\\five\\untaped_five_burak_traj.pdf")
    plt.clf()
    return 


if __name__ == '__main__':
    errs_csv([f"C:\\Users\\burak\\Documents\\GitHub\\rsn-f1tenth\\experiments_results\\untaped\\Burak_A_B\\five\\xztheta_traj.csv",
    f"C:\\Users\\burak\\Documents\\GitHub\\rsn-f1tenth\\experiments_results\\untaped\\initial_A_B\\five\\xztheta_traj.csv"])
    # process_csv(f"csv\\gradbased\\")
    # ref_traj_csv([f"C:\\Users\\burak\\Documents\\GitHub\\rsn-f1tenth\\experiments_results\\untaped\\Burak_A_B\\five\\xztheta_traj.csv"])
    quit()
