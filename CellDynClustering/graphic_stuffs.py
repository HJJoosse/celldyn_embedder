import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import time

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}
__author__ = "Chontira Chumsaeng and Huibert-Jan Joose"
"""
Utility functions graphing stuffs
"""
  

def multiple_line_plots(data, x):
    
    """
    Plot multiple line plots for a particular X variable.
    
    Parameters
    ---------
    data: dataframe
            data (including x variable) for plotting.
    
    x: String
            name of the X variable to be plotted.
                
    """
    
    df_shape = data.shape
    y_vars=data.loc[:, data.columns != x].columns
    for y in y_vars:
        plot = sns.lineplot(x=data[x], y = data[y])
        plt.show()
        

def plot_multi_scatter(data, row, col, subsample = False):

    plot_df = data
    if(type(subsample)==int):
        plot_df = plot_df.sample(subsample)

    fig, ax = plt.subplots(row,col,figsize = (30,50))
    i,j = 0,0

    for colx in plot_df:
        for coly in plot_df:
            ax[i,j].scatter(plot_df[colx],plot_df[coly], s = 3, alpha = 0.2)
            i = i+1 if j == col-1 else i
            j = j+1 if j < col-1 else 0

def plot_multi_scatter_with_labels(data,labels, row, col,subsample= False, remove_noise=False):

    plot_df = data

    if(type(subsample)==int):
        plot_df = plot_df.sample(subsample)
        labels = labels[plot_df.index]

    if(remove_noise):
        plot_df = plot_df[labels != -1]
        labels = labels[labels != -1]
        
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]


    fig, ax = plt.subplots(row,col,figsize = (30,50))
    i,j = 0,0

    for colx in plot_df:
        for coly in plot_df:
            ax[i,j].scatter(plot_df[colx],plot_df[coly],c=colors, s = 3, alpha = 0.2)
            i = i+1 if j == col-1 else i
            j = j+1 if j < col-1 else 0


def plot_predicted_multi_scatter(data, row, col, algorithm, kwds,subsample = False , remove_noise=False):

    plot_df = data
    if(type(subsample)==int):
        plot_df = plot_df.sample(subsample)

    labels = None
    if('metric' in kwds.keys()):
        if(kwds['metric'] == 'mahalanobis'):
            V=np.cov(plot_df, rowvar=False)
            labels = algorithm(**kwds, V = V).fit_predict(plot_df)
        else:
            labels = algorithm( **kwds).fit_predict(plot_df)
    else:
        labels = algorithm( **kwds).fit_predict(plot_df)


    if(remove_noise):
        plot_df = plot_df[labels != -1]
        labels = labels[labels != -1]
        

    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    fig, ax = plt.subplots(row,col,figsize = (30,50))
    i,j = 0,0

    for colx in plot_df:
        for coly in plot_df:
            ax[i,j].scatter(plot_df[colx],plot_df[coly], c=colors, s = 3, alpha = 0.2)
            i = i+1 if j == col-1 else i
            j = j+1 if j < col-1 else 0


def plot_predicted_clusters(data, algorithm, kwds,plotted_columns, subsample = False,remove_noise = False):

    plot_df = data

    start_time = time.time()
    if(type(subsample)==int):
        plot_df = plot_df.sample(subsample)
    
    labels = None
    if('metric' in kwds.keys()):
        if(kwds['metric'] == 'mahalanobis'):
            V=np.cov(plot_df, rowvar=False)
            labels = algorithm(**kwds, V = V).fit_predict(plot_df)
        else:
            labels = algorithm( **kwds).fit_predict(plot_df)
    else:
        labels = algorithm( **kwds).fit_predict(plot_df)    
        
        
    
    end_time = time.time()
    
    if(remove_noise):
        plot_df = plot_df[labels != -1]
        labels = labels[labels != -1]
    
    
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

    b = sns.scatterplot(plot_df.iloc[:,plotted_columns[0]], plot_df.iloc[:,plotted_columns[1]], c=colors,s = 3, alpha = 0.2)
    b.axes.set_title('Component '+ str(plotted_columns[0]+1) + ' vs component ' + str(str(plotted_columns[1]+1) ),fontsize=20)
    b.set_xlabel('Component '+ str(plotted_columns[0]+1),fontsize=20)
    b.set_ylabel('Component '+ str(plotted_columns[1]+1),fontsize=20)
    plt.show()

    print('Clusters found by {}'.format(str(algorithm.__name__)))
    print(-0.5, 0.7, 'Clustering took {:.2f} minutes'.format((end_time - start_time)/60))



def plot_scatter_preset(data,plotted_columns, subsample = False):

    plot_df = data

    if(type(subsample)==int):
        plot_df = plot_df.sample(subsample)


    b = sns.scatterplot(plot_df.iloc[:,plotted_columns[0]], plot_df.iloc[:,plotted_columns[1]],s = 3, alpha = 0.2)
    b.axes.set_title('Component '+ str(plotted_columns[0]+1) + ' vs component ' + str(str(plotted_columns[1]+1) ),fontsize=20)
    b.set_xlabel('Component '+ str(plotted_columns[0]+1),fontsize=20)
    b.set_ylabel('Component '+ str(plotted_columns[1]+1),fontsize=20)
    plt.show()



def plot_scatter_preset_with_labels(data,labels,plotted_columns, subsample = False,remove_noise = False):

    plot_df = data

    if(type(subsample)==int):
        plot_df = plot_df.sample(subsample)
        labels = labels[plot_df.index]


    if(remove_noise):
        plot_df = plot_df[labels != -1]
        labels = labels[labels != -1]

    colors = 'blue'
    if(any(labels != None)):
        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

    b = sns.scatterplot(plot_df.iloc[:,plotted_columns[0]], plot_df.iloc[:,plotted_columns[1]],c=colors,s = 3, alpha = 0.2)
    b.axes.set_title('Component '+ str(plotted_columns[0]+1) + ' vs component ' + str(str(plotted_columns[1]+1) ),fontsize=20)
    b.set_xlabel('Component '+ str(plotted_columns[0]+1),fontsize=20)
    b.set_ylabel('Component '+ str(plotted_columns[1]+1),fontsize=20)


    