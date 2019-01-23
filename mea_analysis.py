#!/usr/bin/env python3
"""
    File name: mea_analysis.py
    Author: Leo Browning
    email: leobrowning92@gmail.com
    Description:
    Bunch of functions for the analysis and plotting of multichanel voltage data collected from a PXI4303.
"""
import os, re, glob
import pandas as pd
import numpy as np
from nptdms import TdmsFile
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from matplotlib.font_manager import FontProperties
import scipy.optimize as opt

sublabels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
substyle = FontProperties()
substyle.set_size = 'large'
substyle.set_weight = 'bold'
def plot_all(f = ''):
    listall = glob.glob('**', recursive = True)
    for fname in listall:
        if "SMU" in fname and fname.endswith('.tdms') and f in fname:
            try:
                smudf, vdf = open_data(fname)
                plot_signal(vdf, smudf, tr = (0, 1e10), save = fname[:-5])
                print("DONE", fname)
            except Exception as e:
                print("ERROR:", fname)
                print(e)


def format_axis(ax):
    ax.tick_params(direction = 'in', which = 'both', right = True, top = True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)


def open_tdms_asdf(path, v = False):
    tfile = TdmsFile(path)
    rootobject = tfile.object()
    df = tfile.as_dataframe()
    cols = []
    for header in list(df):
        header = re.sub("['/]", '', header)[8:]
        header = header.replace('Voltage_', 'v')
        header = header.replace("Untitled", "current")
        header = header.replace("Time", "time")
        cols.append(header)
    df.columns = cols
    df['trel'] = df.time - df.time.iloc[0]
    df.trel = pd.to_numeric(df.trel)/1000000000

    if v:
        df.info()
    return df

def open_data(basepath):
    """
    Opens the data saved by the PXI system measuring current through a voltage
    sourcing SMU, and voltage across multiple channels of the multichanel
    voltage measurement.

    Args:
        basepath (str): the filename of the SMU data in tdms format. ie
            MEA015_postS_100mV_SMU.tdms

    Returns:
        smudf (pandas dataframe): The current, voltage and conductance over
            the whole device. timestamped
        vdf (pandas dataframe): The voltage across each channel of the
            multichannel PXI4303. timestamped

    IMPORTANT: this is based on each measurement producing two files with the
    naming convention:
        MEA015_postS_100mV_V.tdms
        MEA015_postS_100mV_SMU.tdms
    The code will not function without the driving voltage with units of mV or V and underscores in it ie "_10V_". Additionally there MUST be two files which are identical except for the SMU or V in the filename. This is something that could be changed, but for the moment is how I have written it.
    """
    smudf = open_tdms_asdf(basepath)
    vdf = open_tdms_asdf(basepath.replace("SMU", "V"))
    V = False
    if not(re.search("(\d+)V", basepath)):
        V = int(re.search("(\d+)mV", basepath).group(1))
    elif not(re.search("(\d+)mV", basepath)):
        V = int(re.search("(\d+)V", basepath).group(1))
    else:
        print("ERROR:couldnt detect conduction")
        print(basepath)
    if V:
        smudf['voltage'] = V
        g0 = 7.75e-5 #quantum conductance
        smudf['conductance'] = smudf.current/smudf.voltage/g0
    return smudf, vdf

def remap_pins(df):
    """This ONLY needs to be used for any data taken before 2018-06-19"""
    mapping = {"v0" :"v0", "v1" :"v1", "v2" :"v6", "v3" :"v2",
         "v4" :"v4", "v5" :"v9", "v6" :"v5", "v7" :"v3",
         "v8" :"v7", "v9" :"v10", "v10":"v14", "v11":"v12",
         "v12":"v8", "v13":"v13", "v14":"v15", "v15":"v11"}
    cols = [mapping[i] for i in list(df)[1:-1]]
    df.columns = ['time'] + cols + ['trel']
    df = df[['time', "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", 'trel']]
    return df

def vdf_to_tensor(vdf):
    size = len(vdf)
    data = vdf.iloc[:, 3:-1].values
    a = np.empty((size, 1))
    a[:] = np.nan
    tensor = np.concatenate((a, data, a), axis = 1).reshape((size, 4, 4))
    return tensor

def analyze_events(smudf, gthresh):
    """
    determines the conductance switching events in a time series that exceed a given conductance threshold

    Args:
        smudf (pandas dataframe): Conductance time series data. output of
            open_data()
        gthresh (float): The conductance threshold above which a change over
            one timestep is considered to be a switching event.
    Returns:
        dgthresh (numpy array): returns an array of switching events. their size and time of occurence
        iei (numpy array): array of the time between switching events
    """

    g = smudf.conductance.values # vector of conductance values
    t = smudf.trel.values # vector timestamps corresponding of equal length to g

    # array of the difference between successive g values and the time at which they occur
    dg = np.array([[g[i] - g[i-1], t[i]] for i in range(len(g))])
    dg[0] = [0,0] # corrects for g[0] - g[-1] in preveous line

    # all dg that are above gthresh, including their time data
    dgthresh = np.array([point for point in dg if abs(point[0])>= gthresh])

    #array of the time between switching events
    iei = np.array([dgthresh[i+1, 1] - dgthresh[i, 1] for i in range(len(dgthresh) - 1)])

    return dgthresh, iei

c3 = ['r', 'r', 'r', 'b',
    'r', 'r', 'b', 'k',
    'r', 'b', 'k', 'k',
    'b', 'k', 'k', 'k',
    ]
def plot_cmap(c, ax = False):
    """test docstring"""
    if not(ax):
        fig, ax = plt.subplots(1, figsize = (1, 1))
    # Create a Rectangle patch
    for i in range(4):
        for j in range(4):
            rect = patches.Rectangle((j*0.25, 0.75-i*0.25), 0.25, 0.25, linewidth = 2, edgecolor = 'grey', facecolor = c[i*4+j])

    # Add the patch to the Axes
            ax.add_patch(rect)
    ax.axis('off')

def plot_conductance(sdf, ax, tr = (0, 10000)):
    sdf = sdf[(sdf.trel>tr[0])&(sdf.trel<tr[1])]
    ax.plot(sdf.trel, sdf.conductance, 'g-')
    ax.set_xlabel("Time (s)", fontsize = 12)
    ax.set_ylabel("Conductance ($G_0$)", fontsize = 12)
    ax.ticklabel_format(style = 'sci', scilimits = (0, 0), axis = 'y')
    ax.tick_params(direction = 'in', which = 'both', top = True)
    ax.spines['top'].set_visible(True)
    ax.ticklabel_format(style = 'plain', axis = 'both', useOffset = False)

def plot_voltages(vdf, ax, tr = (0, 10000), vr = (3, 16), colors = c3, remove_channels = []):
    vdf = vdf[(vdf.trel>tr[0])&(vdf.trel<tr[1])]
    for i in range(*vr):
        if i in remove_channels:
            pass
        else:
            ax.plot(vdf.trel, vdf[list(vdf)[i+1]],  color = colors[(i-1)%len(colors)], alpha = 1, linewidth = 0.5)

    ax.set_xlabel("Time (s)", fontsize = 12)
    ax.set_ylabel("Voltage (V)", fontsize = 12)
    ax.tick_params(direction = 'in', which = 'both')

def plot_signal(vdf, sdfo, tr = (0, 10000), vr = (3, 16), colors = c3, save = '', show = False, remove_channels = [], time_correction = 0, title = True):
    sdf = sdfo.copy()
    sdf.trel = sdf.trel + time_correction
    fig, axes = plt.subplots(nrows = 2, figsize = (6.3, 6.3))
    plot_conductance(sdf, axes.flat[0], tr)
    plot_voltages(vdf, axes.flat[1], tr, vr, colors, remove_channels)
    for label in (axes.flat[1].get_xticklabels() + axes.flat[1].get_yticklabels() + axes.flat[0].get_xticklabels() + axes.flat[0].get_yticklabels()):
        label.set_fontsize(12)
    # axes.flat[2].legend(ncol = 2, fontsize = 12)
    if title:
        axes.flat[0].set_title(save)
    for i in range(len(axes)):
        axes[i].text(0.03, 0.91, sublabels[i], horizontalalignment = 'left',  verticalalignment = 'center', transform = axes[i].transAxes, color = 'k', size = 'large',
        weight = 'bold')

    plt.tight_layout()
    if save:
        plt.savefig(save+'.png')
        axes.flat[0].set_title('')
        plt.savefig(save+'.pdf')
    if show:
        pass
    else:
        plt.close()



def plot_principal_components(resd, displaynan = [], show = False, save = False, alpha = 0.5, ls = '', ends = True):
    """
        calculates the principal components of the data resd, and then plots their relative variance, the first to components and the projection of the data on to those first two components. asumes len(16) data.

        Args:
            resd (np.array): numpy array with columns = dimensions and rows = measurements
            displaynan (list): Columns to omit from display due to dead channels
            show (bool): show a figure when finished. Defaults to False.
            save (str or bool): filename to save, or False. Defaults to False.
            alpha (type): Alpha value of points in scatter plot of raw data projected on to (p0, p1). Defaults to 0.5.
            ls (type): scatter plot linestyle. Defaults to ''.
            ends (type): whether or not to plot the start and end points of the time series in projection. Defaults to True.

        Returns:
            fig (mpl figure object): figure object
            axes (mpl axes list): the axes used for plotting
    """

    # This step is VITAL for reasonable PCA. without some form of normalization
    # PCA will fail utterly
    resd=np.array([(point-point.mean())/point.mean() for point in resd])

    # from sklearn.decomposition import PCA
    pca = PCA()
    # resd is an np array where each column is a dimension (voltage channel)
    # and each row is a measurement of all dimensions (voltage at a time)
    pca.fit(resd)

    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (6.3, 6.3))

    # these are the principal component vectors
    pc = pca.components_
    # adds in empty values to allow 4x4 display of vectors
    display_pc = np.insert(pc, displaynan, np.nan, axis = 1)

    # Plots each of the first 3 principal components
    for i in range(1, 3):
        im = axes.flat[i].imshow(np.insert(display_pc[i-1], [0, 14], np.nan).reshape((4, 4)), cmap = 'viridis')
        axes.flat[i].axis('off')
        axes.flat[i].set_title("$s$ = {:0.2f}".format(pca.explained_variance_ratio_ [i-1]))
        divider = make_axes_locatable(axes.flat[i])
        cax = divider.append_axes('bottom', size = '10%', pad = 0.1)
        cb = plt.colorbar(im, cax = cax, orientation = 'horizontal', label = "V", format = '%.2f')
        cmin = np.nanmin(display_pc[i-1])           # colorbar min value
        cmax = np.nanmax(display_pc[i-1])
        span = cmax - cmin

        cb.set_ticks([cmin + (0.1*span), cmax - (0.1*span)])
        cb.set_ticklabels(["{:.2f}".format(cmin + (0.1*span)), "{:.2f}".format(cmax - (0.1*span))])
        cb.set_label("relative correlation")

    # plots the relative variance that is due to each of the pc vectors
    axes.flat[0].plot(pca.explained_variance_ratio_ , 'ro')
    axes.flat[0].set_ylabel('relative variation ratio')
    axes.flat[0].set_xlabel('principal component')

    # projects the observed measurements when projected
    # on to the first two principal components by default
    plot_pca_projection(resd, pca.explained_variance_ratio_ , pca.components_, axes.flat[3], alpha = alpha, ls = ls, ends = ends)
    axes.flat[3].set_ylabel('$p_1$')
    axes.flat[3].set_xlabel('$p_0$')

    #formatting
    for i in range(len(axes.flat)):
        axes.flat[i].text(-0.15, 1.05, sublabels[i], horizontalalignment = 'left',  verticalalignment = 'center', transform = axes.flat[i].transAxes, color = 'k', size = 'large',
        weight = 'bold')
    plt.tight_layout()
    if save:
        plt.savefig(save+'.pdf')
        plt.savefig(save+'.png')

    return fig, axes


def plot_pca_projection(resd, vari_ratio, pc, ax = None, components = (0, 1), ls = '', alpha = 0.5, show = False, save = False, ends = True):
    """
        Projects a dataset on to 2 of its principal components.
        and produces a 2D scatter plot of that data.

        Args:
            resd (np.array): numpy array with columns = dimensions and
                rows = measurements
            vari_ratio (np.array): the variance ratio of the principal components
            pc (np.array): principal component vectors as from
                sklearn.decomposition.PCA
            ax (mpl axis): optional axis. Defaults to None.
            components (2d tupple): which two principal components to
                project on to. Defaults to (0, 1).
            ls (string): mpl linestyle. Defaults to ''.
            alpha (float [0,1]): alpha of points. Defaults to 0.5.
            show (bool): show a figure when finished. Defaults to False.
            save (str or bool): filename to save, or False. Defaults to False.
            ends (bool): whether to plot start and end of projection.
                Defaults to True.

        Returns:
            Nothing, but optionally displays the plot or saves it to disk.
    """

    if not(ax):
        fig, ax = plt.subplots(figsize = (6.3, 6.3))
    # projection of the measurements in resd on to the principal components
    ax.plot(resd.dot(pc[components[0]]), resd.dot(pc[components[1]]), ls = ls, marker = '.', alpha = alpha, label = "data projection")
    if ends:
        ax.plot(resd.dot(pc[components[0]])[0], resd.dot(pc[components[1]])[0], 'ro', label = "start")
        ax.plot(resd.dot(pc[components[0]])[-1], resd.dot(pc[components[1]])[-1], 'kx', label = "end")
    ax.set_xlabel('s[{}] = {:0.2f}'.format(components[0], vari_ratio[components[0]]))
    ax.set_ylabel('s[{}] = {:0.2f}'.format(components[1], vari_ratio[components[1]]))
    # ax.legend()
    if save:
        plt.tight_layout()
        plt.savefig(save+'.png')
    if show:
        pass

def plot_timeslice(tensor, index, ax):
    im = ax.imshow(tensor[index], cmap = 'viridis', interpolation = 'nearest')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size = '10%', pad = 0.1)
    cb = plt.colorbar(im, cax = cax, orientation = 'horizontal', label = "V", format = '%.2f')
    cmin = np.nanmin(tensor[index])           # colorbar min value
    cmax = np.nanmax(tensor[index])
    range = cmax - cmin

    cb.set_ticks([cmin + (0.1*range), cmax - (0.1*range)])
    cb.set_ticklabels(["{:.2f}".format(cmin + (0.1*range)), "{:.2f}".format(cmax - (0.1*range))])
    cb.set_label("V", labelpad = -10)

def show_timeslices(tensor, indices, absolute = True):
    size = len(indices)

    if size >= 3:
        ncols = 3
        nrows = size//3
    else:
        ncols = size
        nrows = 1
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols,
     figsize = (4*ncols, 4*nrows))
    for i in range(size):
        plot_timeslice(tensor, indices[i], axes.flat[i])
    return fig

def show_timeseries(tensor, start, stop):
    size = stop - start
    nrows = size//20+1
    ncols = 20
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (16, 0.8*nrows))
    for i in range(start, stop):
        im = axes.flat[i-start].imshow(tensor[i], cmap = 'viridis', interpolation = 'nearest')
    for ax in axes.flat:
        ax.axis('off')
    # fig.subplots_adjust(right = 0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax = cbar_ax, label = 'Network Point Voltage (V)')
    return fig

def plot_loglog(data, ax = None):
    dfig = matplotlib.figure.Figure()
    dax = matplotlib.axes.Axes(dfig, (0, 0, 0, 0))
    n, b, p = dax.hist(np.abs(data), bins = 20)
    del dax, dfig
    if not(ax):
        fig, ax = plt.subplots(figsize = (3, 3))
    # ax.loglog(b[1:], n, 'ro')

    # plt.show()
    n = n/len(data)
    x = []
    y = []
    for i in range(len(n)):
        if n[i] != 0:
            x.append(b[i])
            y.append(n[i])
    x = np.array(x)
    y = np.array(y)
    lx = np.log(x)
    ly = np.log(y)
    def powerlaw(x, a, c):
        return np.exp(c)*x**a
    def linear(x, m, c):
        return m*x+c
    p, dp = opt.curve_fit(linear, lx, ly)
    # print(p, dp)
    ax.loglog(x, y, 'ro')
    ax.plot(x, powerlaw(x, *p), 'k-', label = "$\\alpha =${:.2f}$\pm${:.2f}".format(p[0], dp[0, 0]))
    ax.legend()

    # plt.legend()
