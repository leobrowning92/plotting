import os,glob,re,sys,time
import pandas as pd
import numpy as np
import PAplot
import matplotlib.pyplot as plt

def timeStampYMDH():
    # -> 'YYYY_MM_DD_HHMM' as a time stamp
    return time.strftime('%Y_%m_%d_%H%M')


def get_metrics(folder,search):
    filepaths =  glob.glob(os.path.join(folder,"*"+search+"*"))
    df=pd.DataFrame(columns=['IDSmax','IDSmin','IGmax','hrs'],index=[os.path.basename(x)[:16] for x in filepaths])

    for i in range(len(filepaths)):
        data = np.genfromtxt(fname=filepaths[i],dtype=float,delimiter=',',skip_header=1)
        fname=os.path.basename(filepaths[i])
        try:
            hrs=int(re.search('(\d)hrdep', fname,flags=re.IGNORECASE).group(1))
            df.loc[fname[:16]] = [max(data[:,2]),min(data[:,2]),max(data[:,3]),hrs]
        except Exception as e:
            print(e)
            print (fname)
    return df
def data_to_frame(folder,search):
    if os.path.isdir(os.path.join(folder, "data")):
        dfolder=os.path.join(folder, "data")
    else:
        dfolder=folder
    df=get_metrics(dfolder,search)
    df=df.sort_index()
    df['ratio']=df['IDSmax']/df['IDSmin']
    return df
def plot_metrics(folder,save,show,search=''):
    df=data_to_frame(folder,search)
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
    PAplot.scatter_plot(ax3,df['hrs'],df['ratio'],"deposition time (hrs)","ON/OFF ratio",True)
    ax1.set_title(folder+search+"\nMeasurement Metrics", fontsize=20,y=1.05)
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.85)

    if show:
        plt.show()
    if save:
        fig.savefig(os.path.join(folder,"{1}metrics_{0}.png".format(timeStampYMDH(),search)), bbox_inches='tight')
    plt.close(fig)
def plot_metric_difference(folder1,folder2,save,show,search=''):
    df1=data_to_frame(folder1, search)
    df2=data_to_frame(folder2, search)
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
    PAplot.scatter_plot(ax3,df1['hrs'],df2['ratio']-df1['ratio'],"deposition time (hrs)","ON/OFF ratio",False)
    ax1.set_title("Delta {} to {}\nMeasurement Metrics".format(folder1,folder2), fontsize=20,y=1.05)
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.85)

    if show:
        plt.show()
    if save:
        fig.savefig("{1}delta_metrics_{0}.png".format(timeStampYMDH(),search), bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        plot_metrics(sys.argv[1],save=False,show=True,search=sys.argv[2])
    elif len(sys.argv)== 4:
        plot_metrics(sys.argv[1],save=True,show=True,search=sys.argv[3])
        plot_metrics(sys.argv[2],save=True,show=True,search=sys.argv[3])
        plot_metric_difference(sys.argv[1], sys.argv[2], True, True ,search = sys.argv[3])
    else:
        print("this scritpt takes 1 argument of a folder name. and an optional argument of a filter {} were given".format(len(sys.argv)-1))
