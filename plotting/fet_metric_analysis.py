#!/usr/bin/python3
"""
Author: Leo Browning

Script contains funcitons to plot metrics from collections of FET data
"""

import os,glob,re,sys,time argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PAplot


def timeStampYMDH():
    # -> 'YYYY_MM_DD_HHMM' as a time stamp
    return time.strftime('%Y_%m_%d_%H%M')


def get_metrics(folder,search):
    """
    helper function to get all the metrics listed by file in a data folder

    uses only the first 16 characters of filename to identify the data entry
    At the moment it groups the files by deposition time in hours as specified in the filename. however will trip up on minutes for dep time.
    """

    # determines if there is a data folder in the given folder. if so use that
    if os.path.isdir(os.path.join(folder, "data")):
        dfolder=os.path.join(folder, "data")
    else:
        dfolder=folder
    # sets up filepaths
    filepaths =  glob.glob(os.path.join(folder,"*"+search+"*"))
    # sets up DataFrame
    df=pd.DataFrame(columns=['IDSmax','IDSmin','IGmax','hrs'],index=[os.path.basename(x)[:16] for x in filepaths])

    for i in range(len(filepaths)):
        data = np.genfromtxt(fname=filepaths[i], dtype=float, delimiter=',', skip_header=1)
        fname=os.path.basename(filepaths[i])
        try:
            # checks for a deposition time in hours
            hrs=int(re.search('(\d)hrdep', fname, flags=re.IGNORECASE).group(1))
            # exception to catch filenames that don't have 'Xhrdep' in name
        except Exception as e:
            print(e)
            print ("{}\n shows some issue with finding the dep time in hrs\n check the filename")
        # sets entry in the dataframe
        df.loc[fname[:16]] = [max(data[:,2]),min(data[:,2]),max(data[:,3]),hrs]
    # calculates on off ratio very simpyl. not robust with negatives etc.
    df['ONOFFratio']=df['IDSmax']/df['IDSmin']
    df=df.sort_index()
    return df

def plot_metrics(folder,save,show,search=''):
    """
    plots the key FET metrics between two folders
    metrics are On current, Off current, and on/off ratio

    can take a search term to select only data that has a given
    string in the filename
    """
    df=get_metrics(folder,search)
    fig = plt.figure(figsize=(10, 16), facecolor="white")
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)
    hrlim = (min(df['hrs'])-abs(min(df['hrs'])*0.1) , max(df['hrs'])+abs(max(df['hrs'])*0.1))
    ax1.set_xlim( hrlim )
    ax2.set_xlim( hrlim )
    ax3.set_xlim( hrlim )
    PAplot.scatter_plot(ax1,df['hrs'],df['IDSmax'],"deposition time (hrs)","max $ I_{DS}$",True)
    PAplot.scatter_plot(ax2,df['hrs'],df['IDSmin'],"deposition time (hrs)","min $ I_{DS}$",True)
    PAplot.scatter_plot(ax3,df['hrs'],df['ONOFFratio'],"deposition time (hrs)","ON/OFF ratio",True)
    ax1.set_title(folder+search+"\nMeasurement Metrics", fontsize=20,y=1.05)
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.85)

    if show:
        plt.show()
    if save:
        fig.savefig(os.path.join(folder,"{1}metrics_{0}.png".format(timeStampYMDH(),search)), bbox_inches='tight')
    plt.close(fig)
def plot_metric_difference(folder1,folder2,save,show,search=''):
    """
    plots the difference in the key FET metrics between two folders

    can take a search term to select only data that has a given
    string in the filename
    """
    df1=get_metrics(folder1, search)
    df2=get_metrics(folder2, search)
    if len(df2)<len(df1):
        df1=df1.loc[df1.index.isin(df2.index)]
    fig = plt.figure(figsize=(10, 16), facecolor="white")
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)
    hrlim = (min(df1['hrs'])-abs(min(df1['hrs'])*0.1) , max(df1['hrs'])+abs(max(df1['hrs'])*0.1))
    ax1.set_xlim( hrlim )
    ax2.set_xlim( hrlim )
    ax3.set_xlim( hrlim )
    PAplot.scatter_plot(ax1,df1['hrs'],df2['IDSmax']-df1['IDSmax'],"deposition time (hrs)","max $ I_{DS}$",False)
    PAplot.scatter_plot(ax2,df1['hrs'],df2['IDSmin']-df1['IDSmin'],"deposition time (hrs)","min $ I_{DS}$",False)
    PAplot.scatter_plot(ax3,df1['hrs'],df2['ONOFFratio']-df1['ONOFFratio'],"deposition time (hrs)","ON/OFF ratio",False)
    ax1.set_title("Delta {} to {}\nMeasurement Metrics".format(folder1,folder2), fontsize=20,y=1.05)
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.85)

    if show:
        plt.show()
    if save:
        fig.savefig("{1}delta_metrics_{0}.png".format(timeStampYMDH(),search), bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    #two cases
    # for plotting the metrics for a single data folder
    if len(sys.argv) == 3:
        plot_metrics(sys.argv[1],save=False,show=True,search=sys.argv[2])
    # for plotting the metrics for a single data folder
    elif len(sys.argv)== 4:
        plot_metrics(sys.argv[1],save=True,show=True,search=sys.argv[3])
        plot_metrics(sys.argv[2],save=True,show=True,search=sys.argv[3])
        plot_metric_difference(sys.argv[1], sys.argv[2], True, True ,search = sys.argv[3])
    else:
        print("this scritpt takes arguments\n    folder search(opt)\n    fodler1 folder2 search(opt)\n folders are high level data folders which have the structure folder/data/(datafiles)\n {} were given".format(len(sys.argv)-1))
