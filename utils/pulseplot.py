#!/usr/bin/env python3
import argparse, glob, os, textwrap, hashlib,re,sys
import pandas as pd
from plotting_fns import format_primary_axis,checkdir
from IPython import embed
import matplotlib.pyplot as plt
import scipy.constants as const
divergent=["#3778bf", "#feb308", "#7F2C5A", "#c13838", "#2f6830"]
def get_info(df):
    print("dataframe contains {} datapoints across {} runs".format(df.shape[0],len(set(df['uuid']))))
    print("the columns are: \n{}".format(list(df)))
    print(df[:10])
def get_runID(fname):
    return hashlib.md5(fname.encode()).hexdigest()[:8]
def get_run(fname):
    try: return int(re.search('run(\d{1,2})', fname).group(1))
    except: return "runERROR"

def load_to_dataframe(directory,v=True,force=True):
    if force==False and os.path.isfile(os.path.join(directory,"dataframe.h5")):
        try:
            print("loading dataframe from HDF5 file in {}".format(directory))
            store =pd.HDFStore(os.path.join(directory,"dataframe.h5"))
            df=store['df']
        except Exception as e:
            print(e)
            print("there was an error loading the database")
    else:
        print("loading data from source files in {}".format(directory))
        fnames = glob.glob(os.path.join(directory,"data/*"))
        if len(directory)>1:
            for i in range(1,len(directory)):
                fnames=fnames+glob.glob(os.path.join(directory[i],"data/*"))
        fnames.sort()
        # load names into dataframes
        frames= [ pd.read_csv(fname) for fname in fnames ]
        # sets index to fname and concatenates dataset
        df=pd.concat(frames,keys=[os.path.basename(fname) for fname in fnames])
        # moves the fname to a column
        df['temp']=df.index
        #makes columns for the various parameters held in the filename
        df['temp']=df['temp'].apply(lambda x:x[0])
        df['fname']=df['temp']
        df["R"]=df['AV']/df['AI']
        df["G"]=df['AI']/df['AV']/const.physical_constants["conductance quantum"][0]
        df['run']=df['temp'].apply(get_run)
        df['uuid']=df['temp'].apply(get_runID)
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

def timetrace(ax1,data, x="Time", y1="AV", y2="IV", y2log=False, c1='r', c2='b', y1label="AV", y2label="IV"):

    ax1.plot(data.Time,data.AV,'.-',color=c1, markersize=15, linewidth=3)
    format_primary_axis(ax1, xlabel="Time (s)", ylabel=y1label, color=c1, sci=True, fontsize=20, title="")
    ax2=ax1.twinx()
    if y2log:
        if all(data[y2]>0):
            ax2.semilogy(data.Time,data[y2],'.-',color=c2, markersize=15, linewidth=3)
            format_primary_axis(ax2, xlabel="Time (s)", ylabel=y2label, color=c2, sci=not(y2log), fontsize=20, title="")
        else:
            ax2.plot(data.Time,data[y2],'.-',color=c2, markersize=15, linewidth=3)
            format_primary_axis(ax2, xlabel="Time (s)", ylabel=y2label, color=c2, sci=True, fontsize=20, title="")
    else:
        ax2.plot(data.Time,data[y2],'.-', color=c2, markersize=15, linewidth=3)
        format_primary_axis(ax2, xlabel="Time (s)", ylabel=y2label, color=c2, sci=True, fontsize=20, title="")
def sweep(ax1,data,x="AV",y="IV",ylog=False,c1='r',ylabel="IV"):
    ax1.plot(data.AV,data[y],'.-',color=c1, markersize=15, linewidth=3)
    format_primary_axis(ax1, xlabel="AV", ylabel=ylabel, color=c1, sci=ylog, fontsize=20, title="")

def plot_sequence(data):
    fig=plt.figure(facecolor='white',figsize=(10,20))
    ax1=plt.subplot(311)
    ax2=plt.subplot(312)
    ax3=plt.subplot(313)
    timetrace(ax1, data, y2="AI", y2log=False, c1="#3778bf", c2="#7F2C5A",y2label="I / A")
    timetrace(ax2, data, y2="R", y2log=True, c1="#3778bf", c2="#c13838", y2label="R / $\Omega$")
    timetrace(ax3, data, y2="G", y2log=True, c1="#3778bf", c2="#feb308", y2label="G / $G_0$")
def plot_sweep(data):
    fig=plt.figure(facecolor='white',figsize=(20,20))
    ax1=plt.subplot(321)
    ax2=plt.subplot(323)
    ax3=plt.subplot(325)
    timetrace(ax1, data, y2="AI", y2log=False, c1="#3778bf", c2="#7F2C5A",y2label="I / A")
    timetrace(ax2, data, y2="R", y2log=True, c1="#3778bf", c2="#c13838", y2label="R / $\Omega$")
    timetrace(ax3, data, y2="G", y2log=True, c1="#3778bf", c2="#feb308", y2label="G / $G_0$")
    ax4=plt.subplot(322)
    ax5=plt.subplot(324)
    ax6=plt.subplot(326)
    sweep(ax4, data, x="AV", y="AI", ylog=False, c1="#7F2C5A",ylabel="I / A")
    sweep(ax5, data, x="AV", y="R", ylog=True, c1="#c13838", ylabel= "R / $\Omega$")
    sweep(ax6, data, x="AV", y="G", ylog=True, c1="#feb308", ylabel="G / $G_0$")

def plot_all(directory,v=True,show=True,save=True,force=False):
    checkdir(os.path.join(directory,"plots"))
    df=load_to_dataframe(directory,v=v,force=force)
    for n in sorted(set(df.run)):
        data=df[df.run==n]
        fname=data.fname.iloc[0]
        try:
            if "sweep" in fname:
                plot_sweep(data)
            else:
                plot_sequence(data)


            if v:
                print("utils data for: {}".format(fname))
            if show:
                plt.show()
            if save:
                plt.savefig("{}.png".format(os.path.join(directory,'plots', fname[:-4])))
            plt.close()
        except Exception as e:
            print(e)
            print("failed to plot data from these files:\n{}".format(fname))

if __name__ == "__main__":
    parser = argparse.ArgumentParser( formatter_class=argparse.RawDescriptionHelpFormatter, description=textwrap.dedent("""\
            Description of the various functions available:
                plotall   -  plots all data for each chip on a seperate plot and store in subdir plots/
                test      -  simply loads datasets use -i for interactive and -f to force reload"""))
    parser.add_argument("function", type=str, choices=["test","plotall"],
        help="what utils function to use: %(choices)s")
    parser.add_argument("directory",type=str, help="The directory containing the data folder to analyse",nargs='+')
    parser.add_argument("--show", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true",default=False)
    parser.add_argument("-i","--interactive",action="store_true", help="If applicable opens up an ipython terminal")
    parser.add_argument("-f",'--force',action="store_true",help="Forces reloading the data from the source data")
    args = parser.parse_args()

    assert all([os.path.isdir(x) for x in args.directory]), "invalid directory path"
    if args.function=="test":
        print(args.directory)
        if len(args.directory)>1:
            df=pd.concat([load_to_dataframe(directory, v=args.verbose, force=args.force) for directory in args.directory])
        else:
            df=load_to_dataframe(args.directory[0], v=args.verbose, force=args.force)
        if args.interactive:
            embed()
    if args.function=="plotall":
        assert len(args.directory)==1, "Only one directory accepted"
        plot_all(args.directory[0],v=args.verbose,show=args.show,save=args.save,force=args.force)
