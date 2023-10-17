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
import csv
import numpy as np  
import matplotlib
import matplotlib.pyplot as plt


def points_on_circumference(center=(0, 0), r=50, n=100):
		x_pts = center[0] + (np.cos(2 * np.pi / n * np.arange(0, n)) * r)
		y_pts = center[1] + (np.sin(2 * np.pi / n * np.arange(0, n)) * r)  
		return np.hstack((np.expand_dims(x_pts,1), np.expand_dims(y_pts,1)))




# Open the CSV file

def plotting_errs(csv_filename):

    circle_pts_x = []
    circle_pts_y = []


    with open(csv_filename, 'r') as csvfile:
        
        csvreader =  csv.reader(csvfile)

        data = np.array([row for row in csvreader], dtype=float)
        '''
        self.state_x_B_UT,  0
        self.state_z_B_UT,
        self.state_x_B_T
        self.state_z_B_T
        self.state_x_GT_UT  
        self.state_z_GT_UT 
        self.state_x_GT_T,  6
        self.state_z_GT_T,  7
        
        e1_vals_B_UT        8
        e1_vals_B_T       
        e1_vals_GT_UT
        e1_vals_GT_T        11

        e2_vals_B_UT        12
        e2_vals_B_T
        e2_vals_GT_UT
        e2_vals_GT_T        15
 
        center_x            16
        center_y            17
        r                   18
        '''
        # last 3 columns of CSV file
        center_x = data[0,16]
        center_y = data[0,17]
        r = data[0,18]
        
        circle_sample_n = 2000 
        circle_pts = points_on_circumference(center=(center_x, center_y),r=r,
                                    n=circle_sample_n)

        for point in circle_pts:
            circle_pts_x.append(point[0])
            circle_pts_y.append(point[1])

        #Setting states values
        unique_color_list = ["'-b'", "'-r'", "'-g'", "'-m'"]
        for i in range(0,4,2):
            state_x = data[:,i]
            state_z = data[:,i+1]
            plt.plot(state_z,state_x,unique_color_list[i/2])
            #plt.scatter(x=state_x,y=state_z,c=np.tan(np.linspace(-0.99, 0.99, len(state_x))),s=0.5, linewidths=0.5)
        
        plt.scatter(x=np.array(circle_pts_x),y=np.array(circle_pts_y),s=0.5,linewidths=0.5)
        plt.axis('equal')
        plt.grid()
        plt.ylabel("x-axis (m)")
        plt.xlabel("z-axis (m)")
        plt.savefig("trajectory_path_Burak.pdf")
        plt.clf()
        
        for i in range(4,7,2):
            state_x = data[:,i]
            state_z = data[:,i+1]
            plt.plot(state_z,state_x,unique_color_list[i/2])
            #plt.scatter(x=state_x,y=state_z,c=np.tan(np.linspace(-0.99, 0.99, len(state_x))),s=0.5, linewidths=0.5)
        
        plt.scatter(x=np.array(circle_pts_x),y=np.array(circle_pts_y),s=0.5,linewidths=0.5)
        plt.axis('equal')
        plt.grid()
        plt.ylabel("x-axis (m)")
        plt.xlabel("z-axis (m)")
        plt.savefig("trajectory_path_Repo.pdf")
        plt.clf()


        #e1 plotting 

        e1_vals = data[:,3]
        len_e1 = len(e1_vals)

        plt.plot(np.arange(len_e1), np.array(e1_vals),".",markersize=2)
        plt.grid()
        plt.ylabel("e1 Error")
        plt.xlabel("Number of Samples")
        plt.savefig("e1_plot.pdf")
        plt.clf()



        #e2 plotting 

        e2_vals = data[:,5]
        len_e2 = len(e2_vals)
        
        plt.plot(np.arange(len_e2), np.array(e2_vals),".",markersize=2)
        plt.grid()
        plt.ylabel("e2 Error")
        plt.xlabel("Number of Samples")
        plt.savefig("e2_plot.pdf")
        plt.clf()


if __name__ == "__main__":
	plotting_errs("xztheta_traj.csv")