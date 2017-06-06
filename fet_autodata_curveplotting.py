from plotting_fns import format_primary_axis,checkdir
import matplotlib.pyplot as plt
import pandas as pd
import glob, re, os, sys

def plotchip(directory,number):
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
            axes[i].plot(data["VDS"],data["ID"], label="postSU8", linewidth=2)
            format_primary_axis(axes[i],"","","k",True,10)
        else:
            data=pd.read_csv(fnames[i])
            axes[i].semilogy(data["VG"],data["ID"], label="postSU8", linewidth=2)
            format_primary_axis(axes[i],"","","k",False,10)
    checkdir(os.path.join(directory,"plots"))
    plt.savefig("{}plots/COL{}_plot.png".format(directory,number))



if __name__ == "__main__":
    assert len(sys.argv) == 4 , "this scritpt takes 3 arguments of a folder name and a start and stop chip index {} were given".format(len(sys.argv)-1)
    for i in range(int(sys.argv[2]),int(sys.argv[3])+1):
        plotchip(sys.argv[1],i)
# for i in range(len(fnames)):
#     for j in range (2):
#         if "transfer" in fnames[i] and :
#             data=pd.read_csv(fnames[i])
#             axes[j].plot(data["VDS"],data["ID"], label="postSU8", linewidth=2)
#             data=pd.read_csv(fnames[i])
#             axes[j].plot(data["VDS"],data["ID"], label="preSU8", linewidth=2)
#             format_primary_axis(axes[i],"","","k",True,10)
#     else:
#         data=pd.read_csv(fnames[i*2])
#         axes[i].semilogy(data["VG"],data["ID"], label="postSU8", linewidth=2)
#         data=pd.read_csv(fnames[i*2+1])
#         axes[i].semilogy(data["VG"],data["ID"], label="preSU8", linewidth=2)
#         format_primary_axis(axes[i],"","","k",False,10)

#
# plt.savefig("COL473_SU8encap_comparison.png")
# plt.show()
