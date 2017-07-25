#!/usr/bin/python3
"""
Author: Leo Browning

Script contains functions for the plotting of data as collected
by my automated Parameter analyser scripts
"""

import glob, re, os, sys, time, argparse, hashlib
import matplotlib.pyplot as plt
from IPython import embed
import pandas as pd
import numpy as np
import seaborn as sns
from plotting_fns import format_primary_axis,checkdir
############## COLORS ###################################

divergent=sns.color_palette(["#3778bf", "#feb308", "#441830", "#c13838", "#2f6830"])
hls=sns.color_palette("hls", 8)
##############   Helper Functions   #####################
# Functions for pulling important information out of filename strings
def find_deptime(fname):
    """returns the deptime in minutes from the filename"""
    if 'hrdep' in fname:
        return int(re.search('(\d{1,3})hrdep', fname, flags=re.IGNORECASE).group(1))*60
    elif 'mindep' in fname:
        return int(re.search('(\d{1,3})mindep', fname, flags=re.IGNORECASE).group(1))
    else: return "deptimeERROR"

def find_parameters(fname):
    try:
        if 'transfer' in fname:
            return re.search('(VG[0-9]\d*(:?\.\d+)?VDS[0-9]\d*(:?\.\d+)?)',fname).group(1)
        if 'output' in fname:
            return re.search('(VDS[0-9]\d*(:?\.\d+)?VG[0-9]\d*(:?\.\d+)?)',fname).group(1)
    except: return "parameterERROR"

def find_fabstep(fname):
    try:return re.search('dep(.[^_]*)',fname).group(1)
    except:return "fabstepERROR"

def find_numsweeps(fname):
    try:return re.search('x(\d*)',fname).group(1)
    except: return 1

def get_info(df):
    print("dataframe contains {} datapoints across {} runs".format(df.shape[0],len(set(df['uuid']))))
    print("the columns are: \n{}".format(list(df)))
    print(df[:10])
def get_runID(fname):
    return hashlib.md5(fname.encode()).hexdigest()[:8]
def get_timestamp(fname):
    try:return re.search('(\d{4}_\d{2}_\d{2}_\d{4})', fname).group(1)
    except: return "timestampERROR"
def find_gate(fname):
    try:return re.search('_([^_]*gate[^_]*)_', fname).group(1)
    except: return "backgate"


def plotchip(directory,number,v=True,show=True,save=True):
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
    fig = plt.figure(figsize=(16, 4), facecolor="white")
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
    if save:
        plt.savefig("{}plots/COL{}_plot.png".format(directory,number))
    if show:
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

####Tiling code
def load_to_dataframe(directory,v=True):
    fnames = glob.glob(os.path.join(directory,"data/*"))
    if len(directory)>1:
        for i in range(1,len(directory)):
            fnames=fnames+glob.glob(os.path.join(directory[i],"data/*"))
    fnames.sort()

    # process fnames
    frames= [ pd.read_csv(fname) for fname in fnames ]
    # sets index to fname and concatenates dataset

    df=pd.concat(frames,keys=[os.path.basename(fname) for fname in fnames])
    # moves the fname to a column
    df['temp']=df.index


    #makes columns for the various parameters held in the filename
    df['temp']=df['temp'].apply(lambda x:x[0])
    df['chip']= df['temp'].apply(lambda x:int(x[3:6]) )
    df['device']= df['temp'].apply(lambda x:x[3:9] )
    df['deptime']=df['temp'].apply(find_deptime)
    df['fabstep'] = df['temp'].apply(find_fabstep)
    df['parameters'] = df['temp'].apply(find_parameters)
    df['multi'] = df['temp'].apply(find_numsweeps)
    df['gate'] = df['temp'].apply(find_gate)
    df['timestamp']=df['temp'].apply(get_timestamp)
    df['uuid']=df['temp'].apply(get_runID)
    df['fname']=df['temp']
    df.drop(['temp'],axis=1,inplace=True)
    # resets the index to a unique identifier for a datarun
    df.reset_index(drop=True,inplace=True)
    #prints information about the dataframe
    if v:
        print("dataframe after load_to_dataframe()")
        get_info(df)
    return df

def mask_data(df,column,tag):
    data=df
    mask=data[column].isin([tag])
    return mask
def filter_data(df,column=["parameters",'multi'],tags=['VG10VDS0.01',1]):
    """filters the data in the parameter column by the filter value"""
    assert len(column)==len(tags), "number of columns and search parameters must match"
    assert type(column)==list
    assert type(tags)==list
    data=df
    total_mask=mask_data(data, column[0], tags[0])
    if len(column)!=1:
        for i in range(1,len(column)):
            total_mask=total_mask & mask_data(df, column[i], tags[i])
    data=data[total_mask]
    return data
def filter_fnames(df,search,remove=True):
    """removes data where the filenames include the search string"""
    data=df
    if remove:
        data=data[~data['fname'].str.contains("{}".format(search))]
    else:
        data=data[data['fname'].str.contains("{}".format(search))]
    return data
def match_data(df,column='fabstep',match='postSU8'):
    """filters the data by all runs on devices that have the match characteristic
    in their parameter column"""
    data=df
    ls=data[data[column].isin([match])]['device']
    data=data[data['device'].isin(ls)]
    return data

def tile_data(df, column="device",row=None, colwrap=2, color="fabstep", show=True, save=False, v=False,  xlim="", ylim="", sharey=True):
    #Seaborn plotting example
    if v:
        print("dataframe pre tile_data()")
        get_info(df)
    sns.set(style="ticks",font_scale=1,palette=divergent)
    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(df, col=column, hue=color,row=row, col_wrap=colwrap, size=4,sharey=sharey)
    # Draw a line plot to show the trajectory of each random walk
    grid.map(plt.semilogy, "VG", "ID").add_legend()
    # Adjust the tick positions and labels
    if xlim=='':
        pass
    else:
        grid.set(xlim=xlim)
    if ylim=='':
        pass
    else:
        grid.set(ylim=ylim)
    # Adjust the arrangement of the plots
    grid.fig.tight_layout(w_pad=1)

    if show:
        plt.show()
    if save:
        grid.fig.suptitle(save,y=1.05)
        grid.savefig("{}.png".format(save))
    plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""Description of the various functions available:\n    plotall - plots all data for each chip on a seperate plot and store in subdir plots/ \n    collect - plots all data of a type specified by search on a single plot\n    tile - used with the -i flag to open an interactive session which allows exploration of the data""")

    parser.add_argument("function", type=str, choices=["plotall", "collect", "tile"],
        help="what plotting function to use: %(choices)s")
    parser.add_argument("directory",type=str, help="The directory containing the data folder to analyse",nargs='+')

    parser.add_argument("--start", type=int, help="Start of chip index")
    parser.add_argument("--stop", type=int, help="end (inclusive) of chip index")

    #Flags
    parser.add_argument("--show", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true",default=False)
    parser.add_argument("-i","--interactive",action="store_true", help="If applicable opens up an ipython terminal")

    parser.add_argument("--search", help="Search string to filter the filenames by.",action="store")
    args = parser.parse_args()
    if args.function=="tile":
        df=pd.concat([load_to_dataframe(directory,v=args.verbose) for directory in args.directory])

        if args.interactive:
            embed()
        else:
            df=filter_data(df)
            df=match_data(df)
            tile_data(df,v=args.verbose,show=args.show,save=args.save)

    if args.function=="plotall":
        assert len(args.directory)==1, "this function takes only one directory"
        for i in range(args.start,args.stop+1):
            plotchip(args.directory[0], i,save=args.save,show=args.show,v=args.verbose)
    if args.function=="collect":
        assert len(args.directory)==1, "this function takes only one directory"
        plot_all(args.directory[0],args.start,args.stop,args.search, save=args.s,show=args.d)
