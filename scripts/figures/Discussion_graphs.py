#generates figures 5, 6, and 7
import os
import sys
DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(os.path.dirname(DIR))  # Go up two directories
os.chdir(DIR)
sys.path.append(DIR)

"""The sweep files from the Graph_data folder are loaded and used to generate the figures.
For auxinash, two files are loaded: one for the preference and one for the lambda.
This is necessary because fine-grained wandb sweep data for both parameters at once was impossible to obtain."""

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import pandas as pd
import math
import bisect
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20  # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 20  # Y-axis tick labels
plt.rcParams.update({'font.size': 20})



dataset = "M4"
variable = "auxinash"
dataset_list = ["M3","M4"]
variable_list = ["hyperstep_p","lambda","cosine"]


def get_actual_step(step,skip=93):
        #Every 93 steps, one is skipped: 0,1,..., 92, 94, 95, ..., 186, 188, 189, ...
        #We want the inverse of this function
        return math.ceil(step*(skip)/(skip+1))

def get_graph(variable, dataset):

    plt.style.use('science')
    if variable != "auxinash":
        plt.figure(figsize=(8, 6))

    
    
    #loed in csv
    data_variable = variable + " " + dataset
    print("-----------------------")
    print(data_variable)
    print("-----------------------")
    df = pd.read_csv(r"Graph_data\{}.csv".format(data_variable))
    #df = pd.read_csv(r"C:\Users\u0165132\OneDrive - KU Leuven\1-PhD\Thesis 2023\Adaptive-N-BEATS-S-main\Dynamic Weighting N-BEATS-S\scripts\lambda M4.csv")

    y = df.iloc[:,1].values
    if dataset == "M4":
        x= df.iloc[:,0].values
        x = [get_actual_step(step,93) for step in x]
    else:
        x = df.iloc[:,0].values
        x= [get_actual_step(step,2) for step in x]

    #print last x
    print(x[-1])
    # print(sd)
    if variable == "auxinash_lambda":
        #get rid of the last (max(x) - 10000) for M3 and ma
        if dataset == "M3":
            number = (max(x) - 10000)/10
        elif dataset == "M4":
            number = (max(x) - 18600)/10
        x_subsample = x[:-int(number)]
        y_subsample = y[:-int(number)]
    if variable == "auxinash":
        #get rid of the last (max(x) - 10000) for M3 and ma
        if dataset == "M3":
            number = (max(x) - 10000)/10
        if dataset == "M4":
            number = (max(x) - 18600)/10
        x_subsample = x[:-int(number)]
        y_subsample = y[:-int(number)]
        #divide y by 2
        y_subsample = y_subsample/2
        stab = 1 - y_subsample
    else:

        x_subsample = x[::8]
        y_subsample = y[::8]

    # plt.plot( x_subsample,y_subsample,alpha=0.8, label="Accuracy")
    if variable == "auxinash":
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.stackplot(x_subsample, stab, y_subsample, labels=['Stability', 'Accuracy'], alpha=0.8)

        ax.set_ylim(0, 1)
        ax.legend()

        # plt.plot(x_subsample,stab,alpha=0.8,label="Stability")#,color='red')
    if variable =="lambda":
        #make the plot a little wider
        plt.plot( x_subsample,y_subsample,alpha=0.8, label=r'$\lambda$',linewidth = 2)
    elif variable == "cosine":
        plt.plot( x_subsample,y_subsample,alpha=0.8, label="Cosine Similarity",linewidth = 2)
    plt.xlabel('Iterations')
    if variable == "lambda":
        plt.ylabel(r'$\lambda$')
    elif variable == "auxinash":
        plt.ylabel("Preference")
        plt.ylim(0, None)
        # plt.legend()
    elif variable == "auxinash_lambda":
        plt.ylabel(r'$\lambda$')
        plt.ylim(0, 1)
        
        
    else: 
        plt.ylabel("CosSim")
        plt.ylim(-1, 1)
    if not variable == "auxinash" and not variable == "auxinash_lambda" and not variable == "lambda":
        plt.axhline(0, color='black', linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1)
    # plt.axhline(0, color='black', linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1)
    folder = "figures"
    if not os.path.exists(folder):
        os.makedirs(folder)
    name = variable + "_" + dataset + ".pdf"
    name = os.path.join(folder, name)
    plt.savefig(name, format='pdf', dpi=300,bbox_inches='tight')
    plt.show()

def auxinash_lambda_graph(variable,dataset):
    plt.style.use('science')
    # plt.figure(figsize=(8, 6))
    data_variable = variable + " " + dataset
    df = pd.read_csv(r"Graph_data\{}.csv".format(data_variable))
    y = df.iloc[:,1].values
    x = df.iloc[:,0].values
    if dataset == "M4":
        x = [get_actual_step(step,93) for step in x]
    else:
        x = [get_actual_step(step,2) for step in x]

    #get lambda df
    data_variable = "auxinash_lambda " + dataset
    #get lambda samples
    df = pd.read_csv(r"Graph_data\{}.csv".format(data_variable))
    y_lambda = df.iloc[:,1].values
    x_lambda = df.iloc[:,0].values
    x_lambda = [get_actual_step(step,93) for step in x_lambda]
    #get rid of the last (max(x) - 10000) for M3 and ma
    if dataset == "M3":
        number = (max(x) - 10000)/10
    if dataset == "M4":
        number = (max(x) - 18600)/10
    x_target = x[-int(number)]    
    number_lambda = bisect.bisect_right(x_lambda, x_target) - 1



    x_subsample = x[:-int(number)]
    y_subsample = y[:-int(number)]
    y_subsample = y_subsample/2
    x_lambda_subsample = x_lambda[:int(number_lambda)]
    y_lambda_subsample = y_lambda[:int(number_lambda)]
    print(max(x_lambda_subsample))
    #subsample lambda
    x_lambda_subsample = x_lambda_subsample[::8]
    y_lambda_subsample = y_lambda_subsample[::8]

    #now do the stackplot + plot lambda on top
    #set color palette to viridis

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.stackplot(x_subsample, 1-y_subsample, y_subsample, labels=['Stability', 'Accuracy'], alpha=0.7, colors=["#ea801c","#b8b8b8"])
    ax2 = ax.twinx()
    line_lambda= ax2.plot(x_lambda_subsample,y_lambda_subsample,alpha=0.5, label=r"$\lambda$",color="#1a80bb")#b8b8b8
    ax.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax.set_xlim(0, max(x_subsample))
    
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    print(labels)
    # Add a single merged legend
    legend = ax2.legend(handles, labels, loc='upper right', frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('grey')
    legend.get_frame().set_alpha(0.9)
    plt.xlabel('Iterations')
    ax.set_ylabel(r'$\lambda$')
    ax2.set_ylabel("Preference")
    folder = "figures"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # plt.axhline(0, color='black', linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1)
    name = variable + "_lambda_" + dataset + ".pdf"
    name = os.path.join(folder, name)
    plt.savefig(name, format='pdf', dpi=300,bbox_inches='tight')
    plt.show()


    # plt.plot( x_subsample,y_subsample,alpha=0.8, label="Accuracy")
    # plt.xlabel('Iterations')
    # plt.ylabel(r'$\lambda$')
    # plt.axhline(0, color='black', linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1)
    # plt.savefig("auxinash_lambda_M4.pdf", format='pdf', dpi=300,bbox_inches='tight')
    # plt.show()


# auxinash_lambda_graph(variable,dataset)
# get_graph(variable, dataset)
for dataset in dataset_list:
    for variable in variable_list:
        if dataset == "M3":
            if variable == "hyperstep_p":
                auxinash_lambda_graph(variable,dataset)
            else:
                get_graph(variable, dataset)
        elif dataset == "M4":
            if variable == "hyperstep_p":
                auxinash_lambda_graph(variable,dataset)
            else:
                get_graph(variable, dataset)

