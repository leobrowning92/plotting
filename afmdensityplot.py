#!/usr/bin/env python3
import argparse, textwrap
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def process_points(fname,save=False, show=True):
    df=pd.read_csv(fname, delimiter='\t', usecols=[2,3])
    df.columns=['x','y']

    fig=plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,3,3)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,1)

    ax1.scatter(df.x,df.y, c="c", s=30, linewidth=0.8, marker="x")
    img=mpimg.imread(fname[:-4]+".png")
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
        plat.savefig(fname[:-4]+"_plots.png")
    plt.show()




    ################# Save data ###############
    if save:
        header = "junction density from 1um^2 sections:".format(average,mean,std)
        pd.DataFrame([(average,mean,std)],columns=["totalMean","1uMean","1uStd"]).to_csv(fname[:-4]+"_Totalmean.csv",delimiter=',')
        np.savetxt(fname[:-4]+"_1umData.csv",data, delimiter=',',header=header)

if __name__ == "__main__":
    parser = argparse.ArgumentParser( formatter_class=argparse.RawDescriptionHelpFormatter, description=textwrap.dedent("""\
            For plotting AFM junction densities"""))

    parser.add_argument("files", type=str, nargs='*', help="names of text files with point information from afm images")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true",default=False)
    args = parser.parse_args()
    for fname in args.files:
        assert fname[-4:]==".txt", "incorrect filetype"
        process_points(fname)
