#!/usr/bin/env python3
"""
Author: Leo Browning

Script contains functions for the plotting of data as collected
by my automated Parameter analyser scripts
"""

import glob, re, os, sys, time, argparse, hashlib, textwrap
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from IPython import embed
import pandas as pd
import numpy as np
import seaborn as sns
from plotting_fns import format_primary_axis,checkdir


############## COLORS ###################################


divergent=sns.color_palette(["#3778bf", "#feb308", "#7F2C5A", "#c13838", "#2f6830"])
hls=sns.color_palette("hls", 8)
fab_order=['postSD','postSU8','postTop']
son_order=['preson','postson5min','postson10min','postson20min','postson25min','postson30min','postson40min','postson50min']
six=sns.cubehelix_palette(6, start=1.5, rot=-1,light=0.5,dark=0.2)


##############   Helper Functions   #####################
# Functions for pulling important information out of filename strings


def find_deptime(fname):
    """returns the deptime in minutes from the filename"""
    if 'hrdep' in fname:
        return int(re.search('(\d{1,3})hrdep', fname, flags=re.IGNORECASE).group(1))*60
    elif 'mindep' in fname:
        return int(re.search('(\d{1,3})mindep', fname,  flags=re.IGNORECASE).group(1))
    else: return "deptimeERROR"
def find_sontime(fname):
    """returns the sontime in minutes from the filename"""
    try: return int(re.search('postson(\d{1,2})min', fname,flags=re.IGNORECASE).group(1))
    except: return "deptimeERROR"
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
def find_run(fname):
    try:return int(re.search('run(\d{1,3})',fname).group(1))
    except:
        print("run error for"+fname)
        return 1
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


def get_metrics(df,fname,directory):
    df['IGmax']=max(df["IG"])
    df['IDmax']=max(df["ID"])
    df['IDmin']=abs(min(df["ID"]))
    if "transfer" in fname:
        #need to consider some error catching for -ve ids
        df['onoff']=df['IDmax']/abs(df['IDmin'])
        Ithresh=0.5*max(df.ID)
        try:
            Vthresh=df[(df.ID<Ithresh)].iloc[0]
        except:
            print("thresherror for {}".format(fname))
            Vthresh=np.nan
        df["Ithresh"]=Ithresh
        df["Vthresh"]=Vthresh
    else:
        df['onoff']=np.nan
        df["Ithresh"]=np.nan
        df["Vthresh"]=np.nan


    tubedata=pd.read_csv(os.path.join(directory, "all_density_data.csv"))
    for key in list(tubedata):
        if key!="device":
            df[key]=float( tubedata[key][ os.path.basename(fname)[0:9] ==tubedata.device ].values )
    return df
def gate_area(gate):
    areas={"backgate":3600,"topgate1":600,"topgate2":200,"topgate3":200,"topgate4":600}
    return areas[gate]

def normalize_onoff(df):
    backgates={device : df[(df["device"]==device) & (df["gate"]=="backgate") & (df.parameters=="VG20VDS0.1")].onoff.values[0] for device in df[df.gate=="backgate"].drop_duplicates(subset="device").device.values}
    print(backgates)
    df["backoff"]=df["device"].apply(lambda x: backgates[x])
    # df.loc[df[gate]=="top",'normalonoff']=df[df.chip=="back"]
    df["normalonoff"]=df.onoff/df.backoff
    return df




##################### Tiling code ###########################


def load_to_dataframe(directory,v=True,force=True):
    if force==False and os.path.isfile(os.path.join(directory,"dataframe.h5")):
        try:
            print("loading dataframe from HDF5 file in {}".format(directory))
            store =pd.HDFStore(os.path.join(directory,"dataframe.h5"))
            df=store['df']
        except Exception as e:
            print(e)
            print("there was an error loading the database from h5 file for {}".format(directory))
    else:
        print("loading data from source files in {}".format(directory))
        fnames = glob.glob(os.path.join(directory,"data/COL*"))
        if len(directory)>1:
            for i in range(1,len(directory)):
                fnames=fnames+glob.glob(os.path.join(directory[i],"data/COL*"))
        fnames.sort()
        # process fnames
        frames= [ get_metrics(pd.read_csv(fname),fname,directory) for fname in fnames ]
        # sets index to fname and concatenates dataset
        df=pd.concat(frames,keys=[os.path.basename(fname) for fname in fnames])
        # moves the fname to a column
        df['temp']=df.index
        #makes columns for the various parameters held in the filename
        df['temp']=df['temp'].apply(lambda x:x[0])
        df['chip']= df['temp'].apply(lambda x:int(x[3:6]) )
        df['run']=df['temp'].apply(find_run)
        df['device']= df['temp'].apply(lambda x:x[3:9] )
        df['deptime']=df['temp'].apply(find_deptime)
        df['sontime']=df['temp'].apply(find_sontime)
        df['fabstep'] = df['temp'].apply(find_fabstep)
        df['parameters'] = df['temp'].apply(find_parameters)
        df['gate'] = df['temp'].apply(find_gate)
        df['multi'] = df['temp'].apply(find_numsweeps)
        df['gateArea']=df['gate'].apply(gate_area)
        df['sweep'] = df['temp'].apply(find_sweep)
        df['timestamp']=df['temp'].apply(get_timestamp)
        df['poissonratio']=df.tubeStd**2/df.tubeMean
        df['uuid']=df['temp'].apply(get_runID)
        df['fname']=df['temp']
        df.drop(['temp'],axis=1,inplace=True)
        # resets the index to a unique identifier for a datarun
        df.reset_index(drop=True,inplace=True)
        #prints information about the dataframe
        df=normalize_onoff(df)
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


def tile_data(df, column=None,row='device', color="fabstep",
            colwrap=None,  show=True, save=False, v=False,  xlim="", ylim="", sharey=True, x="VG", y="ID", log=True, updown=False, palette=hls,
            col_order=None, hue_order=None, ls='-', marker='', xlabel='', ylabel='', fontscale=1, close=False,hue_kws=None,legend=True):
    if v:
        print("dataframe pre tile_data()")
        get_info(df)
    sns.set(style="ticks",font_scale=fontscale,palette=palette)
    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(df, col=column, hue=color,row=row, col_wrap=colwrap, size=4,sharey=sharey, col_order=col_order, hue_order=hue_order,legend_out=True,hue_kws=hue_kws)
    # Draw a line plot to show the trajectory of each random walk
    if updown:
        grid.map(updownplot, x, y,ls=ls,marker=marker)
    elif log and not(updown):
        grid.map(plt.semilogy, x, y,ls=ls,marker=marker)
    else:
        grid.map(plt.plot, x, y,ls=ls,marker=marker)
    # Adjust the tick positions and labels
    if xlim=='':
        pass
    else:
        grid.set(xlim=xlim)
    if ylim=='':
        pass
    else:
        grid.set(ylim=ylim)
    if xlabel and ylabel:
        grid.set_axis_labels(xlabel, ylabel)
    if legend:
        plt.legend(loc='best', fancybox=True, framealpha=0.5)
    # Adjust the arrangement of the plots
    # grid.fig.tight_layout(w_pad=1)
    # sns.plt.legend(loc='upper left',bbox_to_anchor=(1,0.5))
    # for ax in grid.axes.flat:
    #     box = ax.get_position()
    #     ax.set_position([box.x0,box.y0,box.width*0.85,box.height])
    # plt.legend(["                  "], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


    if show:
        plt.show()
    if save:
        grid.fig.suptitle(save,y=1.05)
        grid.savefig("{}.png".format(save))
        # grid.savefig("{}.pdf".format(save))
    if close:
        plt.close()
    return grid

############### Plotting Functions ######################


def plotchips(directory,v=True,show=True,save=True,force=False):
    """
    This function simply plots all data from a device, and whacks it in seperate
    plots inside a figure. The figures are saved individually in the
    /plots subdir of the parent directory.
    """
    checkdir(os.path.join(directory,"plots"))
    df=load_to_dataframe(directory,v=v,force=force)
    for n in set(df.device):
        try:
            if v:
                print("plotting data for COL{}".format(n))
            tile_data(df[(df.device == n)&(df.sweep == 'transfer')], column="parameters", row='fabstep',color='gate',  save="{}plots/COL{}_transferplot".format(directory,n), show=show,sharey=False)
            tile_data(df[(df.device == n)&(df.sweep == 'output')],column='gate',row='fabstep',color='VG', save="{}plots/COL{}_outputplot".format(directory,n),show=show, x="VDS",log=False)
        except Exception as e:
            print(e)
            print("failed to plot all data from this device:\n{}".format(set(df[df.device == n].fname)))



############## Tiling and data filtering presets ##############


def select_20Vbackgates_oftopgated(df, plot=False, save="backgate_acrossFabsteps_updown"):
    data = df[df.device.isin(set(df[df.fabstep=='postSU8'].device)) & (df.gate=='backgate') & (df.parameters=='VG20VDS0.1')]
    if plot:
        tile_data(data,  column='junctionMean', row=None, colwrap=2, sharey=True, updown=False, save=save,col_order= ['481_01','481_02','484_01','484_02','487_01','487_02','490_01','490_02','493_01','493_02',], hue_order=fab_order)
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
def metric_plot(df, p1, p2, show=True, save=False, hue_order=fab_order,gate=None,palette=hls,xlim='',column=None,row=None,colwrap=None,sharey=True,color="fabstep",log=True):
    if save:
        save="metric_{}vs{}_{}".format(p1,p2,save)
    tile_data(df.drop_duplicates(subset=[p1,p2]),column=column,row=row,colwrap=colwrap,x=p1,y=p2,ls='',marker='o',show=show,save=save,hue_order=hue_order,palette=palette,xlim=xlim,sharey=sharey,color=color,log=log)
def standard_metrics(df,show=True, save='backgate', palette=hls,xlim=''):
    data = df[(df.parameters=='VG20VDS0.1') & (df.gate=='backgate') ]
    for x in ["junctionMean","tubeMean","chip"]:
        for y in ["onoff","IDmax","IDmin"]:
            metric_plot(data, x, y, save=save, show=show, palette=palette,hue_order=None, color="chip")
def density_metrics(df,show=True,save="backgate", palette=hls,xlim='',log=True):
    if not log:
        save=save+"lineary"
    data = df[(df.parameters=='VG20VDS0.1') & (df.gate=='backgate') ]
    for y in ["junctionMean","tubeMean","lengthMean","percolation"]:
        metric_plot(data, "chip", y, save=save, show=show, palette=palette,hue_order=None, color="chip",log=log)
    metric_plot(data, "tubeMean", "junctionMean", save=save, show=show, palette=palette,hue_order=None, color="chip",log=log)

def topgate_metrics(df,show=True, save='topgate', palette=hls,xlim=''):
    data = df[(df.fabstep=='postTop') & (df.parameters=='VG20VDS0.1')]
    metric_plot(data,'gateArea', 'normalonoff',save=save,show=show, palette=palette,xlim=xlim,sharey=False,color="device",hue_order=None)
    metric_plot(data,'gateArea', 'onoff',save=save,show=show, palette=palette,xlim=xlim)
    metric_plot(data,'gateArea', 'IDmax',save=save,show=show, palette=palette,xlim=xlim)
    metric_plot(data,'gateArea', 'IDmin',save=save,show=show, palette=palette,xlim=xlim)
def sonication_metrics(df,show=True, save='sonication', palette=hls,xlim=''):
    data=df
    data.chip=data.device.apply(lambda x:float(x[0:3]) if x[-1]=='1' else float(x[0:3])+0.5)
    metric_plot(df, 'chip', 'IDmax', hue_order=son_order, xlim=(480,496), save=save, show=show)
    metric_plot(df, 'chip', 'IDmin', hue_order=son_order, xlim=(480,496), save=save, show=show)
    metric_plot(df, 'chip', 'onoff', hue_order=son_order, xlim=(480,496), save=save, show=show)
def sonication_sweeps(df,directory):
    for n in set(df.device):
        tile_data(df[(df.device == n)&(df.sweep == 'transfer')], column="parameters", row=None,color='fabstep',  save="{}plots/COL{}_transferplot".format(directory,n), show=False,sharey=False,hue_order=son_order)


if __name__ == "__main__":
    parser = argparse.ArgumentParser( formatter_class=argparse.RawDescriptionHelpFormatter, description=textwrap.dedent("""\
            Description of the various functions available:
                plotall   -  plots all data for each chip on a seperate plot and store in subdir plots/
                collect   -  plots all data of a type specified by search on a single plot
                tile      -  used with the -i flag to open an interactive session which allows exploration of the data
                test      -  simply loads datasets use -i for interactive and -f to force reload"""))

    # required arguments
    parser.add_argument("function", type=str, choices=["plotall", "collect", "tile","test"],
        help="what plotting function to use: %(choices)s")
    parser.add_argument("directory",type=str, help="The directory containing the data folder to analyse",nargs='+')

    # optional flags
    parser.add_argument("--start", type=int, help="Start of chip index")
    parser.add_argument("--stop", type=int, help="end (inclusive) of chip index")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument( "--clear", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true",default=False)
    parser.add_argument("-i","--interactive",action="store_true", help="If applicable opens up an ipython terminal")
    parser.add_argument("-f",'--force',action="store_true",help="Forces reloading the data from the source data")
    parser.add_argument("--search", help="Search string to filter the filenames by.",action="store")
    args = parser.parse_args()
    assert all([os.path.isdir(x) for x in args.directory]), "invalid directory path"
    if args.verbose:
        print("directories loaded : {}".format(args.directory))
    pd.options.display.max_rows=6

    #run funcitons
    if args.function=="tile":
        df=pd.concat([load_to_dataframe(directory, v=args.verbose, force=args.force) for directory in args.directory])

        if args.interactive:
            embed()
        else:
            df=df[df.parameters=='VG20VSD0.1']
            tile_data(df,v=args.verbose,show=args.show,save=args.save)
    if args.function=="plotall":
        assert len(args.directory)==1, "this function takes only one directory"
        if args.clear:
            for f in glob.glob(os.path.join(args.directory[0], "plots", "*")):
                os.remove(f)
        plotchips(args.directory[0], save=args.save, show=args.show, v=args.verbose,force=args.force)
    if args.function=="collect":
        assert len(args.directory)==1, "this function takes only one directory"
        plot_all(args.directory[0],args.start,args.stop,args.search, save=args.s,show=args.d)
    if args.function=="test":
        df=pd.concat([load_to_dataframe(directory, v=args.verbose, force=args.force) for directory in args.directory])
        if args.interactive:
            embed()
