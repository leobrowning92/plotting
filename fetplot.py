#!/usr/bin/env python3
"""
Author: Leo Browning

Script contains functions for the plotting of data as collected
by my automated Parameter analyser scripts
"""

import glob, re, os, sys, time, argparse, hashlib, textwrap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from IPython import embed
import pandas as pd
import numpy as np
import seaborn as sns
from plotting_fns import format_primary_axis,checkdir
import matplotlib
matplotlib.use("TkAgg")

############## COLORS ###################################

divergent=sns.color_palette(["#3778bf", "#feb308", "#7F2C5A", "#c13838", "#2f6830"])
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
            return re.search('(VG[0-9]\d*(:?\.\d+)?VDS[0-9]\d*(:?\.\d+)?)', fname).group(1)
        if 'output' in fname:
            return re.search('(VDS[0-9]\d*(:?\.\d+)?VG[0-9]\d*(:?\.\d+)?)', fname).group(1)
    except: return "parameterERROR"
def find_fabstep(fname):
    try:return re.search('dep(.[^_]*)',fname).group(1)
    except:return "fabstepERROR"
def find_sweep(fname):
    if "transfer" in fname:
        return 'transfer'
    elif 'output' in fname:
        return 'output'
    else: return 'sweepERROR'
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
def load_AFMdensities(directory='AFM_densities'):
    fnames = glob.glob(os.path.join(directory,"*"))
    densities={os.path.basename(x)[7:13]:float(pd.read_csv(x)['1uMean'][0]) for x in fnames}
    deviations ={os.path.basename(x)[7:13]:float(pd.read_csv(x)['1uStd'][0]) for x in fnames}
    return densities,deviations
def dictsub(x,dictionary):
    return dictionary[x]

############### Plotting Functions ######################

def plotchips(directory,v=True,show=True,save=True,force=False):
    """
    This function simply plots all data from a chip, and whacks it in seperate
    plots inside a figure. The figures are saved individually in the
    /plots subdir of the parent directory.
    """
    checkdir(os.path.join(directory,"plots"))
    df=load_to_dataframe(directory,v=v,force=force)
    for n in set(df.chip):
        try:
            if v:
                print("plotting data for COL{}".format(n))
            tile_data(df[(df.chip == n)&(df.sweep == 'transfer')], column="parameters", row='device',color='gate',  save="{}plots/COL{}_transferplot".format(directory,n), show=show,sharey=False)
            tile_data(df[(df.chip == n)&(df.sweep == 'output')],column=None,row='device',color=None, save="{}plots/COL{}_outputplot".format(directory,n),show=show, x="VDS",log=False)
        except Exception as e:
            print(e)
            print("failed to plot data from these files:\n{}".format(set(df[df.chip == n].fname)))





##################### Tiling code ###########################
def load_to_dataframe(directory,v=True,force=True):
    if force==False and os.path.isfile(os.path.join(directory,"dataframe.h5")):
        try:
            print("loading dataframe from database in {}".format(directory))
            store =pd.HDFStore(os.path.join(directory,"dataframe.h5"))
            df=store['df']
        except Exception as e:
            print(e)
            print("there was an error loading the database")
    else:
        print("loading data from source files in {}".format(directory))
        fnames = glob.glob(os.path.join(directory,"data/COL*"))
        if len(directory)>1:
            for i in range(1,len(directory)):
                fnames=fnames+glob.glob(os.path.join(directory[i],"data/COL*"))
        fnames.sort()
        mean,std=load_AFMdensities()
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
        df['sweep'] = df['temp'].apply(find_sweep)
        df['timestamp']=df['temp'].apply(get_timestamp)
        df['uuid']=df['temp'].apply(get_runID)
        df['junctionMean']=df['device'].apply(lambda x: mean[x] if x in mean.keys() else np.nan)
        df['junctionStd']=df['device'].apply(lambda x: std[x] if x in std.keys() else np.nan)
        df['fname']=df['temp']
        df.drop(['temp'],axis=1,inplace=True)
        # resets the index to a unique identifier for a datarun
        df.reset_index(drop=True,inplace=True)
        #prints information about the dataframe
        store = pd.HDFStore(os.path.join(directory,"dataframe.h5"))
        store['df'] = df

    if v:
        print("dataframe from {} after load_to_dataframe()".format(directory))
        get_info(df)
    return df

################# Data Formatting functions

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

def updownplot(x,y,**kwargs):
    plt.semilogy(x[:len(x)//2],y[:len(y)//2],**kwargs)
    plt.semilogy(x[len(x)//2:],y[len(y)//2:],'--',**kwargs,)


############### Core Tiled data function ###########
def tile_data(df, column="parameters",row='device', colwrap=None, color="fabstep", show=True, save=False, v=False,  xlim="", ylim="", sharey=True, x="VG", y="ID", log=True, updown=False, palette=divergent, col_order=None, hue_order=None):
    #Seaborn plotting example
    if v:
        print("dataframe pre tile_data()")
        get_info(df)
    sns.set(style="ticks",font_scale=1,palette=palette)
    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(df, col=column, hue=color,row=row, col_wrap=colwrap, size=4,sharey=sharey, col_order=col_order, hue_order=hue_order)
    # Draw a line plot to show the trajectory of each random walk
    if updown:
        grid.map(updownplot, x, y).add_legend()
    elif log and not(updown):
        grid.map(plt.semilogy, x, y).add_legend()
    else:
        grid.map(plt.plot, x, y).add_legend( )
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
    # grid.fig.tight_layout(w_pad=1)
    plt.legend(["          "], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if show:
        plt.show()
    if save:
        grid.fig.suptitle(save,y=1.05)
        grid.savefig("{}.png".format(save))
    plt.close()



############## Tiling and data filtering presets ##############

def select_20Vbackgates_oftopgated(df, plot=False, save="backgate_acrossFabsteps_updown"):
    data = df[df.device.isin(set(df[df.fabstep=='postSU8'].device)) & (df.gate=='backgate') & (df.parameters=='VG20VDS0.1')]
    if plot:
        tile_data(data,  column='junctionMean', row=None, colwrap=2, sharey=True, updown=False, save=save,col_order= ['481_01','481_02','484_01','484_02','487_01','487_02','490_01','490_02','493_01','493_02',], hue_order=["postSD",'postSU8','postTop'])
    else:
        return data
def topgate_spread(df,plot=False):
    data = df[(df.fabstep=='postTop') & (df.parameters=='VG20VDS0.1')]
    if plot:
        tile_data(data, column='gate', row='device', color='gate', save='VG20VDS0.1_topgate_spread', sharey=False)
    else:
        return data
def single_topgate(df,plot=False):
    data = df[(df.fabstep=='postTop') & (df.parameters=='VG20VDS0.1') & (df.device=='490_02')]
    if plot:
        tile_data(data, column='device',row=None, color='gate', save='VG20VDS0.1_490_2_topgate', sharey=False)
    else:
        return data




if __name__ == "__main__":
    parser = argparse.ArgumentParser( formatter_class=argparse.RawDescriptionHelpFormatter, description=textwrap.dedent("""\
            Description of the various functions available:
                plotall   -  plots all data for each chip on a seperate plot and store in subdir plots/
                collect   -  plots all data of a type specified by search on a single plot
                tile      -  used with the -i flag to open an interactive session which allows exploration of the data"""))

    parser.add_argument("function", type=str, choices=["plotall", "collect", "tile","test"],
        help="what plotting function to use: %(choices)s")
    parser.add_argument("directory",type=str, help="The directory containing the data folder to analyse",nargs='+')

    parser.add_argument("--start", type=int, help="Start of chip index")
    parser.add_argument("--stop", type=int, help="end (inclusive) of chip index")

    #Flags
    parser.add_argument("--show", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true",default=False)
    parser.add_argument("-i","--interactive",action="store_true", help="If applicable opens up an ipython terminal")
    parser.add_argument("-f",'--force',action="store_true",help="Forces reloading the data from the source data")

    parser.add_argument("--search", help="Search string to filter the filenames by.",action="store")
    args = parser.parse_args()
    if args.verbose:
        print("directories loaded : {}".format(args.directory))
    if args.function=="tile":
        df=pd.concat([load_to_dataframe(directory, v=args.verbose, force=args.force) for directory in args.directory])

        if args.interactive:
            embed()
        else:
            # df=filter_data(df)
            # df=match_data(df)
            tile_data(df,v=args.verbose,show=args.show,save=args.save)

    if args.function=="plotall":
        assert len(args.directory)==1, "this function takes only one directory"
        plotchips(args.directory[0], save=args.save, show=args.show, v=args.verbose,force=args.force)
    if args.function=="collect":
        assert len(args.directory)==1, "this function takes only one directory"
        plot_all(args.directory[0],args.start,args.stop,args.search, save=args.s,show=args.d)
    if args.function=="test":
        df=pd.concat([load_to_dataframe(directory, v=args.verbose, force=args.force) for directory in args.directory])
        if args.interactive:
            embed()
