#!/usr/bin/python3
import numpy as np
import unittest
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib
import glob
import scipy.optimize as opt
import re
import sys
import scipy.constants as const





def format_primary_axis(ax,xlabel,ylabel,title,font=15,legloc='best'):
    ax.set_xlabel(xlabel, fontsize=font)
    ax.set_ylabel(ylabel, fontsize=font)
    ax.tick_params(axis='both', which='major', labelsize=font)
    ax.grid(True,which="Major")
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_size(font)
    ax.set_title(title,fontsize=font,y=1.05)
    ax.legend(loc=legloc, fancybox=True, framealpha=0.5,fontsize=font,ncol=2)

def format_second_axis(ax,ylabel,font=15,color='b'):
    ax.set_ylabel(ylabel,fontsize=font,color='b')
    ax.tick_params(axis='both', which='major', labelsize=font)
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_size(font)
    offset_text.set_color('b')
    for tl in ax.get_yticklabels():
        tl.set_color('b')


def linear(x,m,c):
    return m*x+c

class Multisweep(object):
    """contains the data, figures and key metriks for a
    single pa run collecting 2 terminal IV data in a 2 column format
    When initialized the Multisweep object has the full data
    """
    def __init__(self, path, sweepnum, updown=True):
        super(Multisweep, self).__init__()
        #argument variables
        self.path = path
        self.sweepnum = sweepnum
        self.updown = updown

        #variables outomatically generated from arguments
        self.data = np.genfromtxt(fname=self.path,skip_header=1,delimiter=',',dtype=float)
        self.step=self.data[1,0]-self.data[0,0]
        try:
            self.runs = self.splitdata()
        except Exception as e:
            print(e)
            self.runs=[]
            print(self.path,"will only plot as a single dataset")
        self.timestamp = self.get_timestamp()
        self.maxV=max(self.data[:,0])

        # variables empty unless specifically populated
        self.fig= matplotlib.figure.Figure()
        self.gradR=np.empty(0)
        self.finalR=np.empty(0)


    def __enter__(self):
        return self
    def __str__(self):
        return "Multisweep of "+self.path

    def splitdata(self):
        """
        splits the data set into individual runs, where each direction
        is a seperate run. eg a single 0 to 1V updown would be split into 2
        """
        if self.updown:
            return np.array(np.split(self.data,self.sweepnum*2))
        else:
            return np.array(np.split(self.data, self.sweepnum))

    def simpleplot(self,ax):
        """plots the data as one whole lump on ax"""
        ax.plot(self.data[:,0],self.data[:,1],linewidth=2.0,label="")

    def multiplot(self,ax):
        """plots the data onto ax,
        where each sweep of the data is labeled and colored seperately
        """
        for i in range(int(len(self.runs)/2)):
            ax.plot(np.concatenate((self.runs[i*2,:,0],self.runs[i*2+1,:,0])),
                     np.concatenate((self.runs[i*2,:,1],self.runs[i*2+1,:,1])),"-",linewidth=2.0,label=str(i+1))
    def singleplot(self, ax, runs):
        """plots an arbitrary list of data sweeps onto ax,
        sweeps are defined by runs which is a list of indices
        NEEDS FURTHER WORK
        """
        for i in runs:
            ax.plot(np.concatenate((self.runs[i*2,:,0],self.runs[i*2+1,:,0])),
                    np.concatenate((self.runs[i*2,:,1],self.runs[i*2+1,:,1]))
                    ,"-",linewidth=2.0,label=str(i+1)+" of " + str(sweepnum))

    def sawplotI(self,ax):
        ax.plot(self.data[:,1],linewidth=2)
        ax2=ax.twinx()
        ax2.plot(self.data[:,0],'b--',linewidth=1)
        return ax2
    def sawplotG(self,ax):

        g0data=self.data[:,1]/self.data[:,0] / const.physical_constants["conductance quantum"][0]
        ax.plot(g0data,linewidth=2)
        ax2=ax.twinx()
        ax2.plot(self.data[:,0],'b--',linewidth=1)
        return ax2





    def make_singleaxis_fig(self,plottype="simple",runs=[0],v=False):
        col_width =10 #in cm

        #make the figure
        fig=plt.figure(self.path,figsize=(col_width,col_width*0.6), facecolor="white")
        ax1=plt.subplot(111)
        #plot the graph
        if plottype == "simple":
            self.simpleplot(ax1)
        elif plottype == "multi":
            self.multiplot(ax1)
        elif plottype == "single":
            self.singleplot(ax1,runs)
        #format axis
        format_primary_axis(ax1,"V","I",os.path.basename(self.path))
        self.fig=fig
        if v:
            print(plottype,"graph made from ",self.path)
    def make_sawtoothIV_fig(self,v=False):
        #make figure
        col_width=10
        fig=plt.figure(self.path,figsize=(col_width,col_width*0.6), facecolor="white")
        ax1=plt.subplot(111)
        ax2=self.sawplotI(ax1)
        format_primary_axis(ax1, "index", "I (A)", os.path.basename(self.path))
        format_second_axis(ax2, "V")
        self.fig=fig
        if v:
            print("sawtooth graph made from ",self.path)

    def make_sawtoothGV_fig(self,v=False):
        #make figure
        col_width=10
        fig=plt.figure(self.path,figsize=(col_width,col_width*0.6), facecolor="white")
        ax1=plt.subplot(111)
        ax2=self.sawplotG(ax1)
        format_primary_axis(ax1, "index", "G $(G_0)$", os.path.basename(self.path))
        format_second_axis(ax2, "V")
        self.fig=fig
        if v:
            print("sawtooth graph made from ",self.path)



    def savefig(self,savedir="plots/",tag='',v=False):
        if os.path.isdir(savedir) == False:
            os.system("mkdir " +savedir)
        plt.figure(self.path) #sets the current figure as labeled by path
        plt.savefig(os.path.join(savedir,os.path.basename(self.path)[:-4]+tag+".png"))
        if v:
            print("saved: ",self.path)

    def get_timestamp(self):
        return self.path[-19:-4]

    def calc_gradR(self,data):
        """returns R,dR from a linear fit to the first 100mV of data
        """
        xdata=data[:int(0.1/self.step),0]
        ydata=data[:int(0.1/self.step),1]
        var,covar=opt.curve_fit(linear, xdata, ydata)
        return [1/var[0],covar[0,0]/var[0]**2]

    def make_gradR(self):
        """returns a 2d array of gradR for the 0-100mv region of each
        run where the resistance is the first column and its uncertainty
        is the second"""
        res=[]
        for data in self.runs:
            res.append(self.calc_gradR(self.data))
        self.gradR=np.array(res)
        return self.gradR
    def make_finalR(self):
        """returns a 1D array of the resistances where voltage is at a max """
        res=[]
        for data in self.runs[::2]:
            res.append(data[-1,0]/data[-1,1])
        self.finalR=np.array(res)
        return self.finalR

    def __exit__(self,*err):
        plt.close(self.fig)


class sequentialMeasurements(object):
        """
        makes a full list of the sequential datasets of all of the data in a folder by initializing them as Multisweep objects
        """
        def __init__(self, samplefolder):
            super(sequentialMeasurements, self).__init__()
            self.dir=samplefolder
            self.datasets=[]
            self.initlog = self.make_MultisweepData()
            self.paths= [x.path for x in self.datasets]
            self.timestamps = [ x.timestamp for x in self.datasets]
            self.make_finalRplot = matplotlib.figure.Figure()

        def __enter__(self):
            return self

        def make_MultisweepData(self):
            log=[]
            for path in sorted(glob.glob(self.dir+"data/*.csv")):
                if "x" in path:
                    try:
                        self.datasets.append(Multisweep(path,int(re.search("x(\d{1,3})",path).group(1))))
                    except ValueError as e:
                        log.append([path,e])
                        print(path)
                        print('line {}'.format(sys.exc_info()[-1].tb_lineno), type(e), e)
                        print("incomplete dataset probably causeing uneven splitting. plot manually if necessary")

                    except Exception as e:
                        log.append([path,e])

                        print(path)
                        print('line {}'.format(sys.exc_info()[-1].tb_lineno), type(e), e)


            return log
        def make_multiplots(self,v=False):
            """calls the multiplot function on every dataset in the measurement
            sequence"""
            for i in self.datasets:
                if i.runs==[]:
                    i.make_singleaxis_fig("simple",v=v)
                else:
                    i.make_singleaxis_fig("multi",v=v)
        def make_sawtoothIVplots(self,v=False):
            for i in self.datasets:
                i.make_sawtoothIV_fig(v=v)
        def make_sawtoothGVplots(self,v=False):
            for i in self.datasets:
                i.make_sawtoothGV_fig(v=v)
        def save_plots(self,tag='',v=False):
            if v:
                print ("beginning plotting sequence")
            if os.path.isdir("plots") == False:
                os.system("mkdir " +self.dir+ "plots/")
            for i in self.datasets:
                i.savefig(self.dir+"plots/",tag=tag,v=v)
        def make_R(self):
            for i in self.datasets:
                i.make_gradR()
                i.make_finalR()

        def make_finalRplot(self):
            col_width =7 #in cm
            #note, first argument is the label of the figure for later saving
            fig=plt.figure(figsize=(col_width,col_width*0.8), facecolor="white")
            ax1=plt.subplot(111)

            for i in snt82.datasets:
                runnum = int(i.path[i.path.find("run")+3:i.path.find("run")+6])
                sweeparray =np.array([runnum]*len(i.finalR))

                ax1.semilogy(sweeparray,i.finalR,'o')
            self.make_finalRplot = fig



        def __exit__(self,*err):
            pass







##############################################################################
###                              UNITTEST CODE                             ###
##############################################################################


class MyTest(unittest.TestCase):
    def setUp(self):
        self.testdataset="SNT080/data/SNT080_device1_21minsulfur_0.1Vx5_activation1_2016_10_19_1000.csv"
        self.testdir="SNT080/"


    def test_split(self):
        with Multisweep(self.testdataset,5) as testdata:
            self.assertEqual(testdata.splitdata().shape, (10,51,2))

    def test_gradR(self):
        with Multisweep(self.testdataset, 5) as testdata:
            self.assertEqual(testdata.calc_gradR(testdata.runs[0]),
                    [2451394.4249741249, 1.9146863336543544e-05])
            self.assertEqual(testdata.make_gradR().shape,(10,2))
    def test_finalR(self):
        with Multisweep(self.testdataset,5) as testdata:
            self.assertEqual(testdata.make_finalR().shape,(5,))
    def test_timestamp(self):
        with Multisweep(self.testdataset, 5) as testdata:
            self.assertEqual(testdata.timestamp, "2016_10_19_1000")


    def test_make_MultisweepData(self):
        with sequentialMeasurements(self.testdir) as testfolder:
            self.assertEqual(type(testfolder.datasets[0]), Multisweep)
            self.assertEqual(len(testfolder.datasets), 28)
            self.assertEqual(testfolder.initlog, [])



if __name__ == "__main__":
    #ensures that test cases are not run when importing the module.
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    suite = unittest.TestLoader().loadTestsFromTestCase(MyTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
