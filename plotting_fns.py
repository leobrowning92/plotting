import matplotlib.pyplot as plt
import os

def checkdir(directoryname):
    if os.path.isdir(directoryname) == False:
        os.system("mkdir " + directoryname)
    pass

def format_primary_axis(ax,xlabel="",ylabel="",color="k",sci=True,fontsize=20,title=""):
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize, color=color)
    ax.set_title(title,fontsize=fontsize*1.5)
    if sci:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_size(fontsize)
    offset_text.set_color(color)
    for tl in ax.get_yticklabels():
        tl.set_color(color)

    ax.legend(loc='best', fancybox=True, framealpha=0.5,fontsize=fontsize*0.6,ncol=len(ax.get_legend_handles_labels()[1])//10+1)
    ax.grid(True,which="Major")
    # ax.locator_params(axis='x', nticks=3)
    if len(ax.get_xticks())>5:
        ax.set_xticks(ax.get_xticks()[::2])


def scatter_plot(ax, x, y,  xlabel, ylabel, color, log,fontsize=20):
    if log:
        ax.semilogy(x, y, "o",color=color, linewidth=2.0)
    else:
        ax.plot(x, y, "o",color=color, linewidth=2.0)
    format_primary_axis(ax, xlabel, ylabel, color)
