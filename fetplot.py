#!/usr/bin/python3
"""
Author: Leo Browning

Script contains functions for the plotting of data as collected
by my automated Parameter analyser scripts
"""

import glob, re, os, sys, time, argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from plotting_fns import format_primary_axis,checkdir


def timeStampYMDH():
    # -> 'YYYY_MM_DD_HHMM' as a time stamp
    return time.strftime('%Y_%m_%d_%H%M')

def plotchip(directory,number,v=True,display=True,save=True):
    """
    This function simply plots all data from a chip, and whacks it in seperate
    plots inside a figure. The figures are saved individually in the
    /plots subdir of the parent directory.
    """
    fnames=glob.glob("{}data/*COL{}*".format(directory,number))
    if fnames==[]:
        if v:
            print(" no data for:COL{}".format(number))
        return False
    fnames.sort()
    if v:
        print("plotting {} datasets for chip {}".format(len(fnames),number))
    fig = plt.figure(figsize=(30, 4), facecolor="white")
    axes=[]
    for i in range(len(fnames)):
        axes.append(plt.subplot(len(fnames)//6,len(fnames)//1,i+1))

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
    if save:
        plt.savefig("{}plots/COL{}_plot.png".format(directory,number))
    if display:
        plt.show()

def plot_all(directory, start, stop, search,display=True,save=False):
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
    if display:
        plt.show()
    if save:
        fig.savefig(os.path.join(directory,"{}_{}-{}_plots.png".format(search,start,stop)), bbox_inches='tight')
    plt.close(fig)

def plot_all_prepost(predirectory, postdirectory, search, display= True, save=True):
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
    if display:
        plt.show()
    if save:
        fig.savefig(os.path.join("prepost_SU8-{}_plots.png".format(search)), bbox_inches='tight')
    plt.close(fig)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""Description of the various functions available:\n    plotall - plots all data for each chip on a seperate plot and store in subdir plots/ \n    collect - plots all data of a type specified by search on a single plot""")

    parser.add_argument("function", type=str, choices=["plotall","collect"],
        help="what plotting function to use: %(choices)s")
    parser.add_argument("directory",type=str, help="The directory containing the data folder to analyse")

    parser.add_argument("start", type=int, help="Start of chip index")
    parser.add_argument("stop", type=int, help="end (inclusive) of chip index")

    #Flags
    parser.add_argument("--show", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    #optional Arguments
    parser.add_argument("--dir2", help="A second directory to compare against the first",action="store")
    parser.add_argument("--search", help="Search string to filter the filenames by.",action="store")




    args = parser.parse_args()
    if args.function=="plotall":
        for i in range(args.start,args.stop+1):
            plotchip(args.directory, i,save=args.s,show=args.d,v=args.v)
    if args.function=="collect":
        plot_all(args.directory,args.start,args.stop,args.search, save=args.s,show=args.d)
