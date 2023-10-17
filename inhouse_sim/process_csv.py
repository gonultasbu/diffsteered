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
start_time = datetime.now().strftime("%Y-%m-%d%H-%M-%S")
script_name = os.path.split(sys.argv[0])[-1].split(".")[0]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')

def process_csv(path):
    for fname in tqdm(os.listdir(os.path.join(os.getcwd(),path))):
        df = pd.read_csv(os.path.join(path,fname),index_col=0)
        for c in df.columns:
            sns.set_style('darkgrid')
            sns.lineplot(x=np.arange(len(df[c].to_numpy())),y=df[c].to_numpy(), label=f"${c}$")
            plt.xlabel(f"Epochs")
            plt.ylabel(f"{c}")
            plt.legend()
            # plt.hlines(self.official_params[key],xmin=0,xmax=len(np_vals)-1,color="k")
            plt.tight_layout()
            plt.savefig(f"csv_pdfs/{c}_{str(start_time).split('.')[0]}_{fname.split('.')[0]}.pdf")
            plt.clf()
    return 

if __name__ == '__main__':
    process_csv(f"csv\\gradfree\\")
    process_csv(f"csv\\gradbased\\")
    quit()
