"""
Author: Leo Browning

Script contains functions for the plotting of data as collected
by my automated Parameter analyser scripts
"""

from plotting_fns import format_primary_axis,checkdir
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob, re, os, sys, time

def timeStampYMDH():
    # -> 'YYYY_MM_DD_HHMM' as a time stamp
    return time.strftime('%Y_%m_%d_%H%M')

def plotchip(directory,number):
    """
    This function simply plots all data from a chip, and whacks it in seperate
    plots inside a figure. The figures are saved individually in the
    /plots subdir of the parent directory.
    """
    fnames=glob.glob("{}data/*COL{}*".format(directory,number))
    if fnames==[]:
        return False
    fnames.sort()
    fig = plt.figure(figsize=(30, 8), facecolor="white")
    axes=[]
    for i in range(len(fnames)):
        axes.append(plt.subplot(len(fnames)//6,len(fnames)//2,i+1))

    for i in range(len(fnames)):
        if "output" in fnames[i]:
            data=pd.read_csv(fnames[i])
            axes[i].plot(data["VDS"],data["ID"], label="", linewidth=2)
            format_primary_axis(axes[i],"","","k",True,10)
        else:
            data=pd.read_csv(fnames[i])
            axes[i].semilogy(data["VG"],data["ID"], label="", linewidth=2)
            format_primary_axis(axes[i],"","","k",False,10)
    checkdir(os.path.join(directory,"plots"))
    plt.savefig("{}plots/COL{}_plot.png".format(directory,number))

def plot_all(directory, start, stop, search,show=True,save=False):
    """
    plots all of the data off a certain type across chips in a folder to
    compare the data across chips
    specifically designed to work for batches of 5 chips with
    2 devices per chip
    """
    filepaths =  glob.glob(os.path.join(directory+"/data", "*{}*".format( search)))
    newpaths=[]
    for index in range(int(start), int(stop)+1):
        for path in filepaths:
            if str(index) in path:
                newpaths.append(path)
    newpaths.sort()
    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax=plt.subplot(111)
    ax.set_color_cycle([plt.cm.Paired(i) for i in np.linspace(0, 0.8, len(newpaths))])
    for fname in newpaths:
        data=pd.read_csv(fname)
        ax.semilogy(data["VG"],data["ID"],label=os.path.basename(fname)[:9],linewidth=3)

    format_primary_axis(ax,xlabel="VG (V)", ylabel="ID (A)",title="COL{} -{} {}".format(start,stop,search), sci=False)
    if show:
        plt.show()
    if save:
        fig.savefig(os.path.join(directory,"{}_{}-{}_plots.png".format(search,start,stop)), bbox_inches='tight')
    plt.close(fig)

def plot_all_prepost(predirectory, postdirectory, search, show= True, save=True):
    """
    plots all of the data post encapsulation, for a given sweep type defined by search, against the equivalent data pre SU8 if that data exists
    """
    prepaths =  glob.glob(os.path.join(predirectory+"data", "*{}*".format( search)))
    prepaths.sort()
    postpaths =  glob.glob(os.path.join(postdirectory+"data", "*{}*".format( search)))
    postpaths.sort()

    fig = plt.figure(figsize=(8, 20), facecolor="white")
    axes=[plt.subplot(5,2,i+1) for i in range(len(postpaths))]


    for i  in range(len(postpaths)):
        postdata=pd.read_csv(postpaths[i])
        axes[i].semilogy(postdata["VG"],postdata["ID"],label=os.path.basename(postpaths[i])[:9]+"post",linewidth=3)
        for prepath in prepaths:
            if os.path.basename(postpaths[i])[:17] == os.path.basename(prepath)[:17]:
                predata=pd.read_csv(prepath)
                axes[i].semilogy(predata["VG"],predata["ID"],label=os.path.basename(postpaths[i])[:9]+"pre",linewidth=3)
        format_primary_axis(axes[i],xlabel="VG (V)", ylabel="ID (A)",title="{} \n {}".format(os.path.basename(postpaths[i])[:9], search), sci=False,fontsize=10)
    if show:
        plt.show()
    if save:
        fig.savefig(os.path.join("prepost_SU8-{}_plots.png".format(search)), bbox_inches='tight')
    plt.close(fig)





if __name__ == "__main__":
    assert len(sys.argv) in [4,5], "this script takes 3 arguments of a folder name and a start and stop chip index {} were given".format(len(sys.argv)-1)

    if sys.argv[1]=="prepost":
        plot_all_prepost(sys.argv[2], sys.argv[3],sys.argv[4])
    elif len(sys.argv)==4:
        # plots all the data for each chip in the given range
        for i in range(int(sys.argv[2]),int(sys.argv[3])+1):
            plotchip(sys.argv[1],i)
    elif len(sys.argv)==5:
        plot_all(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4],show=True,save=True)
