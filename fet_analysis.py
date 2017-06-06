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
    df=pd.DataFrame(columns=['IDSmax','IDSmin','IGmax','hrs'],index=[os.path.basename(x) for x in filepaths])

    for i in range(len(filepaths)):
        data = np.genfromtxt(fname=filepaths[i],dtype=float,delimiter=',',skip_header=1)
        fname=os.path.basename(filepaths[i])
        try:
            hrs=int(re.search('(\d)hrdep', fname,flags=re.IGNORECASE).group(1))
            df.loc[fname] = [max(data[:,2]),min(data[:,2]),max(data[:,3]),hrs]
        except Exception as e:
            print(e)
            print (fname)
    return df

def plot_metrics(folder,save,show,search=''):
    if os.path.isdir(os.path.join(folder, "data")):
        dfolder=os.path.join(folder, "data")
    else:
        dfolder=folder
    df=get_metrics(dfolder,search)
    df=df.sort_index()
    df['ratio']=df['IDSmax']/df['IDSmin']
    fig = plt.figure(figsize=(10, 16), facecolor="white")
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)
    ax1.set_xlim( (min(df['hrs'])-abs(min(df['hrs'])*0.1) , max(df['hrs'])+abs(max(df['hrs'])*0.1)) )
    ax2.set_xlim( (min(df['hrs'])-abs(min(df['hrs'])*0.1) , max(df['hrs'])+abs(max(df['hrs'])*0.1)) )
    ax3.set_xlim( (min(df['hrs'])-abs(min(df['hrs'])*0.1) , max(df['hrs'])+abs(max(df['hrs'])*0.1)) )
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
if __name__ == "__main__":
    assert len(sys.argv) == 3 , "this scritpt takes 1 argument of a folder name. and an optional argument of a filter {} were given".format(len(sys.argv)-1)
    plot_metrics(sys.argv[1],save=True,show=True,search=sys.argv[2])
