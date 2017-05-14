import numpy as np

import matplotlib.pyplot as plt

import os, glob, sys


# Helper functions
def checkdir(directoryname):
    if os.path.isdir(directoryname) == False:
        os.system("mkdir " + directoryname)
    pass

# plotting functions
def plot_IV(path, skip=1, delim=",", show=False,
            title='IV', xlabel='V', ylabel='I', data1="y1", log=False):
    """
    for two terminal IV measurments
    specifically from the usual FET sript that has
    VG,VDS,ID,IG collumns in that order
    """

    title = os.path.basename(path).replace(".csv", "")
    # print(title)
    data = np.loadtxt(path, skiprows=skip, delimiter=delim, dtype=float)
    # print(data)

    x = np.array([row[0]for row in data])
    y1 = np.array([row[1]for row in data])

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    sub = plt.subplot(1, 1, 1)

    if log:
        sub.semilogy(x, y1, "r-", linewidth=2.0)
    else:
        sub.plot(x, y1, "r-", linewidth=2.0)
        sub.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    sub.legend((data1), loc=2, fontsize=30)
    sub.axis([min(x), max(x), min(y1), max(y1)], fontsize=20)
    sub.tick_params(axis='both', which='major', labelsize=20)

    sub.set_title(title, fontsize=20, y=1.08)
    sub.tick_params(axis='both', which='major', labelsize=20)
    sub.set_xlabel(xlabel, fontsize=20)
    sub.set_ylabel(ylabel, fontsize=20)

    if show:
        plt.show(block=True)
    checkdir("plots")

    name = os.path.basename(path).replace(".csv", "_plt.jpg")
    fig.savefig("plots/" + name, format="jpg")
    # print(name + ' plotted')
    plt.close(fig)
    pass


def overlap_axis_plot(ax1, x, y1, y2,  xlabel, y1label, y2label, log):
    """possible feature duplicate of plot_two_yscales()
    more suitable for scripting"""
    if log:
        ax1.semilogy(x, y1, "r-", linewidth=2.0)
    else:
        ax1.plot(x, y1, "r-", linewidth=2.0)

    ax1.tick_params(axis='both', which='major', labelsize=20)



    ax1.set_xlabel(xlabel, fontsize=20)
    ax1.set_ylabel(y1label, fontsize=20, color='r')
    # ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    offset_text = ax1.yaxis.get_offset_text()

    offset_text.set_size(20)
    offset_text.set_color('red')
    for tl in ax1.get_yticklabels():
        tl.set_color('r')

    ax2 = ax1.twinx()
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.axis([min(x), max(x), min(y2), max(y2)], fontsize=20)
    ax2.plot(x, y2, "b-", linewidth=2.0)

    ax2.set_ylabel(y2label, fontsize=20, color='b')
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    offset_text = ax2.yaxis.get_offset_text()

    offset_text.set_size(20)
    offset_text.set_color('blue')
    for tl in ax2.get_yticklabels():
        tl.set_color('b')


def plot_two_yscales(path, skip=1, delim=",", show=False, save=True, log=False,
                     title='x vs y', xlabel='x', y1label='y1', y2label='y2'):
    """plots x , y1 , y2 data from a 3 collumn csv file
    specific to the ones outputted from the parameter analyzer
    in the clean room by the download_data_as_matrix.py script
    which has collumns VG, VDS, ID, IG in that order.
    saves each plot to a directory called plots at the location of this script.
    """
    name=os.path.basename(path).replace(".csv", "")
    title = "FET characteristics for : \n" + name

    data = np.loadtxt(path, skiprows=skip, delimiter=delim, dtype=float)
    # print(data)

    x = np.array([row[0]for row in data])
    y1 = np.array([row[2]for row in data])
    y2 = np.array([row[3]for row in data])

    fig = plt.figure(figsize=(10, 16), facecolor="white")
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    overlap_axis_plot(ax1, x, y1, y2,  xlabel, y1label, y2label, True)
    overlap_axis_plot(ax2, x, y1, y2,  xlabel, y1label, y2label, False)

    fig.suptitle(title, fontsize=20)
    # plt.tight_layout()
    if show:
        plt.show(block=True)

    if save:
        checkdir("plots")

        name = os.path.basename(path).replace(".csv", "_plt.jpg")
        fig.savefig("plots/" + name, format="jpg")

    plt.close(fig)


def plot_folder(folder, xlabel="$V_G$", y1label="$I_{DS}$", y2label="$I_{G}$"):
    assert folder[-1]=="/", "Error: you must pass a folder to this script"
    filenames=glob.glob(folder+"*")
    for i in range(len(filenames)):
        data=np.genfromtxt(fname=filenames[i],dtype=float,delimiter=',',skip_header=1)
        title=filenames[i]

        fig = plt.figure(figsize=(10, 16), facecolor="white")
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)

        overlap_axis_plot(ax1, data[:,0], data[:,2], data[:,3],  xlabel, y1label, y2label, True)
        overlap_axis_plot(ax2, data[:,0], data[:,2], data[:,3],  xlabel, y1label, y2label, False)

        fig.tight_layout()
        fig.subplots_adjust(top=1.5)
        fig.suptitle(title, fontsize=20,y=1.05)
        print(i,filenames[i])


        checkdir("plots")
        path=filenames[i]
        name = os.path.basename(path).replace(".csv", "_plt.jpg")
        fig.savefig("plots/" + name,bbox_inches='tight', format="jpg")
        plt.close(fig)

if __name__ == "__main__":
    assert len(sys.argv) == 2 , "this scritpt takes 1 argument of a folder name. {} were given".format(len(sys.argv)-1)
    plot_folder(sys.argv[1])
