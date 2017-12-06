#!/usr/bin/env python3
import argparse, textwrap
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.collections import LineCollection

#######NEW CODE
def plot_segments(ax,  segs=np.array([[[1,2],[2,2]]])  ):
    ax.set_xlim((0,1024))
    ax.set_ylim((1024,0))
    line_segments = LineCollection(segs,linewidth=0.8,colors='b')
    ax.add_collection(line_segments)
def plot_tubes(ax,segs,junctions):
    plot_segments(ax,segs)
    ax.scatter(junctions[:,0],junctions[:,1],c="c", s=30, linewidth=2, marker="x")
def check_intersect(s1,s2):
    #assert that x intervals overlap
    if max(s1[:,0])<min(s2[:,0]):
        return False # intervals do not overlap
    #gradients
    m1=(s1[0,1]-s1[1,1])/(s1[0,0]-s1[1,0])
    m2=(s2[0,1]-s2[1,1])/(s2[0,0]-s2[1,0])
    #intercepts
    b1=s1[0,1]-m1*s1[0,0]
    b2=s2[0,1]-m2*s2[0,0]
    if m1==m2:
        return False #lines are parallel
    #xi,yi on both lines
    xi=(b2-b1)/(m1-m2)
    yi=(b2*m1-b1*m2)/(m1-m2)
    if min(s1[:,0])<xi<max(s1[:,0]) and min(s2[:,0])<xi<max(s2[:,0]):
        return [xi,yi]
    else:
        return False
def get_junctions(segs):
    junctions=[]
    for i in range(len(segs)):
        for j in range(i,len(segs)):
            intersect=check_intersect(segs[i],segs[j])
            if intersect:
                junctions.append(intersect)
    return np.array(junctions)
def get_ends(row):
    xc,yc,angle,length = row[0],row[1],row[2],row[3]
    angle=angle/180*np.pi
    x1=xc-length/2*np.cos(angle)
    x2=xc+length/2*np.cos(angle)
    y1=yc+length/2*np.sin(angle)
    y2=yc-length/2*np.sin(angle)
    return np.array( [ [x1,y1],[x2,y2] ] )
def get_density_distribution(df,n):
    data=np.empty((n,n))
    for i in range(n):
        for j in range(n):
            data[-j,-i]=len(df[(df.x<(i+1)*1024//n) & (df.x>=i*1024//n)&(df.y<(j+1)*1024//n)&(df.y>=j*1024//n)])*(n/5.)**2
    return data
def save_density_metrics(tubedata,juncdata,fname):
    pd.DataFrame([(np.mean(juncdata),np.mean(juncdata),np.std(juncdata))], columns=["totalMean","1uMean","1uStd"]).to_csv( fname.replace("_tubes.txt", "_Totalmean_junctions.csv"), delimiter=',')
    pd.DataFrame([(np.mean(tubedata),np.mean(tubedata),np.std(tubedata))], columns=["totalMean","1uMean","1uStd"]).to_csv( fname.replace("_tubes.txt", "_Totalmean_tubes.csv"), delimiter=',')

def plot_image(fname,save=False, show=True):
    dftubes=pd.read_csv(fname, delimiter='\t',usecols=[2,3,4,5])
    dftubes.columns=['x','y','angle','length']
    segs=np.array([get_ends(row) for row in dftubes.values])
    junctions=get_junctions(segs)
    dfjunctions=pd.DataFrame(junctions,columns=["x","y"])


    tubedist=get_density_distribution(dftubes, 5)
    juncdist=get_density_distribution(dfjunctions, 5)
    if save:
        save_density_metrics(tubedist, juncdist, fname)


    fig=plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,2)
    ax2 = fig.add_subplot(1,2,1)


    img=mpimg.imread(fname.replace("_tubes.txt",".png"))
    ax1.imshow(img)
    plot_tubes(ax1,segs,junctions)
    major_ticks = np.arange(0, 1024, 204.7)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.set_xticklabels([0,1,2,3,4,5])
    ax1.set_yticklabels([0,1,2,3,4,5])

    n=5
    g=sns.distplot(np.reshape(tubedist,n**2),bins=10,ax=ax2,label="Tubes")
    g=sns.distplot(np.reshape(juncdist,n**2),bins=10,ax=ax2,label="Junctions")
    g.set_xlabel("Density $/\mu m^2$")
    g.set_ylabel("Count")
    ax2.legend()
    if save:
        plt.savefig(fname.replace("_tubes.txt", "_density_plots.png"))
    if show:
        plt.show()



#######NEW CODE

def process_points(fname,save=False, show=True):
    df=pd.read_csv(fname, delimiter='\t', usecols=[2,3])
    df.columns=['x','y']

    fig=plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,3,3)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,1)

    ax1.scatter(df.x,df.y, c="c", s=30, linewidth=0.8, marker="x")
    img=mpimg.imread(fname.replace("_junctions.txt",".png"))
    ax1.imshow(img)
    major_ticks = np.arange(0, 1024, 204.7)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.set_xticklabels([0,1,2,3,4,5])
    ax1.set_yticklabels([0,1,2,3,4,5])

    n=5
    data=np.empty((n,n))
    for i in range(n):
        for j in range(n):
            data[-j,-i]=len(df[(df.x<(i+1)*1024//n) & (df.x>=i*1024//n)&(df.y<(j+1)*1024//n)&(df.y>=j*1024//n)])*(n/5.)**2
    sns.heatmap(data,ax=ax2,cmap="PuBu")

    mean = np.mean(data)
    std = np.std(data)
    average=len(df)/25.

    g=sns.distplot(np.reshape(data,n**2),bins=10,ax=ax3)
    g.set_xlabel("Junction Density $/\mu m^2$")
    g.set_ylabel("Count")
    if save:
        plt.savefig(fname.replace("_junctions.txt", "_plots_junctions.png"))
    if show:
        plt.show()




    ################# Save data ###############
    if save:
        header = "junction density from 1um^2 sections:".format(average,mean,std)
        pd.DataFrame([(average,mean,std)],columns=["totalMean","1uMean","1uStd"]).to_csv(fname.replace("_junctions.txt", "_Totalmean_junctions.csv"),delimiter=',')
        np.savetxt(fname.replace("_junctions.txt", "_1umData_junctions.csv"),data, delimiter=',',header=header)

if __name__ == "__main__":
    parser = argparse.ArgumentParser( formatter_class=argparse.RawDescriptionHelpFormatter, description=textwrap.dedent("""\
            For plotting AFM junction densities"""))

    parser.add_argument("files", type=str, nargs='*', help="names of text files with point information from afm images")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true",default=False)
    args = parser.parse_args()
    for fname in args.files:
        assert fname[-4:]==".txt", "incorrect filetype: {}".format(fname)
        plot_image(fname,save=args.save,show=args.show)
