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
import os 
import sys
from datetime import datetime
import re
import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["figure.autolayout"]
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import pandas as pd
start_time = datetime.now()
script_name = os.path.split(sys.argv[0])[-1].split(".")[0]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
palettes = sns.color_palette()
def plot_aggregated_loss(input_dirs, save_dir):
    labels_dict = {"gradfree":"CMA-ES", "gradbased":"Ours"}
    labels = [labels_dict[input_dirs[0].split("/")[1]], labels_dict[input_dirs[1].split("/")[1]]]
    savename = input_dirs[0].split("/")[-2]
    # labels = [x.split("/")[1] for x in input_dirs]
    for idx, input_folder in enumerate(input_dirs):
        input_files_list = [i for i in os.listdir(input_folder) if i.endswith(".csv")]
        batch_average_losses_list = []
        bal_xvals_list = []
        for i_file in input_files_list:
            bal_nan = np.full(1024, np.nan)
            with open(f"{input_folder}/{i_file}") as f:
                df = pd.read_csv(os.path.join(input_folder,i_file),index_col=0)
                batch_average_losses = df['loss'].to_numpy()[:100]
                bs = df['bs'][0]
                bal_nan[:batch_average_losses.shape[0]] = batch_average_losses
            bal_xvals = np.arange(len(bal_nan))
            batch_average_losses_list.append(bal_nan)
            bal_xvals_list.append(bal_xvals)
            print(f"{i_file} completed!")

        bal_stacked = np.hstack(batch_average_losses_list)
        bal_xvals_stacked = np.hstack(bal_xvals_list)

        sns.set_style('darkgrid')
        loss_plot = sns.lineplot(x=bal_xvals_stacked, y=bal_stacked, color=palettes[idx], label = f"{labels[idx]}", estimator=lambda x: np.nanmean(x))

    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel(f"Epochs")
    plt.tight_layout()
    plt.savefig(f"{save_dir}_{savename}_loss.pdf")
    plt.clf()

def plot_aggregated_CSf(input_dirs, save_dir):
    # labels = ["Ours", "CMA-ES"]
    savename = input_dirs[0].split("/")[-2]
    labels_dict = {"gradfree":"CMA-ES", "gradbased":"Ours"}
    labels = [labels_dict[input_dirs[0].split("/")[1]], labels_dict[input_dirs[1].split("/")[1]]]
    for idx, input_folder in enumerate(input_dirs):
        input_files_list = [i for i in os.listdir(input_folder) if i.endswith(".csv")]
        batch_average_losses_list = []
        bal_xvals_list = []
        for i_file in input_files_list:
            bal_nan = np.full(1024, np.nan)
            with open(f"{input_folder}/{i_file}") as f:
                df = pd.read_csv(os.path.join(input_folder,i_file),index_col=0)
                batch_average_losses = df['C_Sf'].to_numpy()
                bs = df['bs'][0]
                bal_nan[:batch_average_losses.shape[0]] = batch_average_losses
            bal_xvals = np.arange(len(bal_nan))
            batch_average_losses_list.append(bal_nan)
            bal_xvals_list.append(bal_xvals)
            print(f"{i_file} completed!")

        bal_stacked = np.hstack(batch_average_losses_list)
        bal_xvals_stacked = np.hstack(bal_xvals_list)

        sns.set_style('darkgrid')
        CSf_plot = sns.lineplot(x=bal_xvals_stacked, y=bal_stacked, color = palettes[idx], label = f"{labels[idx]} Training Loss", estimator=lambda x: np.nanmean(x))

    plt.legend()
    plt.ylabel("$C_{Sf}$")
    plt.xlabel(f"Epochs, batch size:{bs}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}_{savename}_C_Sf.pdf")
    plt.clf()

def plot_aggregated_CSr(input_dirs, save_dir):
    # labels = ["Ours", "CMA-ES"]
    savename = input_dirs[0].split("/")[-2]
    labels_dict = {"gradfree":"CMA-ES", "gradbased":"Ours"}
    labels = [labels_dict[input_dirs[0].split("/")[1]], labels_dict[input_dirs[1].split("/")[1]]]
    for idx, input_folder in enumerate(input_dirs):
        input_files_list = [i for i in os.listdir(input_folder) if i.endswith(".csv")]
        batch_average_losses_list = []
        bal_xvals_list = []
        for i_file in input_files_list:
            bal_nan = np.full(1024, np.nan)
            with open(f"{input_folder}/{i_file}") as f:
                df = pd.read_csv(os.path.join(input_folder,i_file),index_col=0)
                batch_average_losses = df['C_Sr'].to_numpy()
                bs = df['bs'][0]
                bal_nan[:batch_average_losses.shape[0]] = batch_average_losses
            bal_xvals = np.arange(len(bal_nan))
            batch_average_losses_list.append(bal_nan)
            bal_xvals_list.append(bal_xvals)
            print(f"{i_file} completed!")

        bal_stacked = np.hstack(batch_average_losses_list)
        bal_xvals_stacked = np.hstack(bal_xvals_list)

        sns.set_style('darkgrid')
        CSr_plot = sns.lineplot(x=bal_xvals_stacked, y=bal_stacked, color = palettes[idx], label = f"{labels[idx]} Training Loss", estimator=lambda x: np.nanmean(x))

    plt.legend()
    plt.ylabel("$C_{Sr}$")
    plt.xlabel(f"Epochs, batch size:{bs}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}_{savename}_C_Sr.pdf")
    plt.clf()


def plot_aggregated_lr(input_dirs, save_dir):
    # labels = ["Ours", "CMA-ES"]
    savename = input_dirs[0].split("/")[-2]
    labels_dict = {"gradfree":"CMA-ES", "gradbased":"Ours"}
    labels = [labels_dict[input_dirs[0].split("/")[1]], labels_dict[input_dirs[1].split("/")[1]]]
    for idx, input_folder in enumerate(input_dirs):
        input_files_list = [i for i in os.listdir(input_folder) if i.endswith(".csv")]
        batch_average_losses_list = []
        bal_xvals_list = []
        for i_file in input_files_list:
            bal_nan = np.full(1024, np.nan)
            with open(f"{input_folder}/{i_file}") as f:
                df = pd.read_csv(os.path.join(input_folder,i_file),index_col=0)
                batch_average_losses = df['lr'].to_numpy()
                bs = df['bs'][0]
                bal_nan[:batch_average_losses.shape[0]] = batch_average_losses
            bal_xvals = np.arange(len(bal_nan))
            batch_average_losses_list.append(bal_nan)
            bal_xvals_list.append(bal_xvals)
            print(f"{i_file} completed!")

        bal_stacked = np.hstack(batch_average_losses_list)
        bal_xvals_stacked = np.hstack(bal_xvals_list)

        sns.set_style('darkgrid')
        lr_plot = sns.lineplot(x=bal_xvals_stacked, y=bal_stacked, color = palettes[idx], label = f"{labels[idx]} Training Loss", estimator=lambda x: np.nanmean(x))

    plt.legend()
    plt.ylabel("$l_{r}$")
    plt.xlabel(f"Epochs, batch size:{bs}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}_{savename}_l_r.pdf")
    plt.clf()

def plot_aggregated_lf(input_dirs, save_dir):
    # labels = ["Ours", "CMA-ES"]
    savename = input_dirs[0].split("/")[-2]
    labels_dict = {"gradfree":"CMA-ES", "gradbased":"Ours"}
    labels = [labels_dict[input_dirs[0].split("/")[1]], labels_dict[input_dirs[1].split("/")[1]]]
    for idx, input_folder in enumerate(input_dirs):
        input_files_list = [i for i in os.listdir(input_folder) if i.endswith(".csv")]
        batch_average_losses_list = []
        bal_xvals_list = []
        for i_file in input_files_list:
            bal_nan = np.full(1024, np.nan)
            with open(f"{input_folder}/{i_file}") as f:
                df = pd.read_csv(os.path.join(input_folder,i_file),index_col=0)
                batch_average_losses = df['lf'].to_numpy()
                bs = df['bs'][0]
                bal_nan[:batch_average_losses.shape[0]] = batch_average_losses
            bal_xvals = np.arange(len(bal_nan))
            batch_average_losses_list.append(bal_nan)
            bal_xvals_list.append(bal_xvals)
            print(f"{i_file} completed!")

        bal_stacked = np.hstack(batch_average_losses_list)
        bal_xvals_stacked = np.hstack(bal_xvals_list)

        sns.set_style('darkgrid')
        lf_plot = sns.lineplot(x=bal_xvals_stacked, y=bal_stacked, color = palettes[idx], label = f"{labels[idx]} Training Loss", estimator=lambda x: np.nanmean(x))

    plt.legend()
    plt.ylabel("$l_{f}$")
    plt.xlabel(f"Epochs, batch size:{bs}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}_{savename}_l_f.pdf")
    plt.clf()

if __name__ == "__main__":
    i_1 = "csv/gradbased/exp_11-untaped/"
    i_2 = "csv/gradfree/exp_11-untaped/"
    plot_aggregated_loss([i_1,i_2],"csv/")
    plot_aggregated_CSr([i_1,i_2],"csv/")
    plot_aggregated_CSf([i_1,i_2],"csv/")
    plot_aggregated_lr([i_1,i_2],"csv/")
    plot_aggregated_lf([i_1,i_2],"csv/")

    quit()
