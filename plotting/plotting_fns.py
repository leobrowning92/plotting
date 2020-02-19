import matplotlib.pyplot as plt
import pandas as pd
import os



divergent=["#3778bf", "#feb308", "#7F2C5A", "#c13838", "#2f6830"]




def checkdir(directoryname):
    if os.path.isdir(directoryname) == False:
        os.system("mkdir " + directoryname)
    pass
def plot_output(ax,data,linewidth=2):
    for vg in data.drop_duplicates(subset="VG")["VG"]:
        ax.plot(data[data.VG==vg].VDS, data[data.VG==vg].ID , label="V_G={}".format(vg), linewidth=linewidth)
def format_primary_axis(ax, xlabel="", ylabel="", color="k", sci=True, fontsize=20, title="",axlabel='',colorboth=False):
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    if colorboth:
        for tl in ax.get_xticklabels():
            tl.set_color(color)
        ax.set_xlabel(xlabel, fontsize=fontsize,color=color)
    else:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize, color=color)
    ax.set_title(title,fontsize=fontsize*1.5)
    if sci:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0),fontsize=fontsize)
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_size(fontsize)
    offset_text.set_color(color)
    for tl in ax.get_yticklabels():
        tl.set_color(color)

    ax.legend(loc='best', fancybox=True, framealpha=0.5,fontsize=fontsize*0.6,ncol=len(ax.get_legend_handles_labels()[1])//10+1)
    ax.text(-0.4,1,axlabel, fontsize=fontsize*1.5,transform=ax.transAxes)
    ax.grid(True,which="Major")
    # ax.locator_params(axis='x', nticks=3)
    if len(ax.get_xticks())>5:
        ax.set_xticks(ax.get_xticks()[::2])
def format_whole_axis(ax, xlabel="", ylabel="", color="k", sci=True, fontsize=20, title="",axlabel=''):
    format_primary_axis(ax, xlabel="", ylabel="", color="k", sci=True, fontsize=20, title="",axlabel='')





def scatter_plot(ax, x, y,  xlabel, ylabel, color, log,fontsize=20):
    if log:
        ax.semilogy(x, y, "o",color=color, linewidth=2.0)
    else:
        ax.plot(x, y, "o",color=color, linewidth=2.0)
    format_primary_axis(ax, xlabel, ylabel, color)
