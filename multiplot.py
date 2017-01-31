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





def formatmultiplotaxis(ax,xlabel,ylabel,font=15,legloc=3):
    ax.set_xlabel(xlabel, fontsize=font)
    ax.set_ylabel(ylabel, fontsize=font)
    ax.tick_params(axis='both', which='major', labelsize=font)
    ax.grid(True,which="Major")
    ax.legend(loc='best', fancybox=True, framealpha=0.5,fontsize=font,ncol=2)
def linear(x,m,c):
    return m*x+c

class multisweep(object):
    """contains the data, figures and key metriks for a
    single pa run collecting 2 terminal IV data in a 2 column format
    When initialized the multisweep object has the full data
    """
    def __init__(self, path, sweepnum, updown=True):
        super(multisweep, self).__init__()
        #argument variables
        self.path = path
        self.sweepnum = sweepnum
        self.updown = updown

        #variables outomatically generated from arguments
        self.data = np.genfromtxt(fname=self.path,skip_header=1,delimiter=',',dtype=float)
        self.step=self.data[1,0]-self.data[0,0]
        self.runs = self.splitdata()
        self.timestamp = self.get_timestamp()
        self.maxV=max(self.data[:,0])

        # variables empty unless specifically populated
        self.fig= matplotlib.figure.Figure()
        self.gradR=np.empty(0)
        self.finalR=np.empty(0)


    def __enter__(self):
        return self
    def __str__(self):
        return "multisweep of "+self.path

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



    def makefig(self,type="simple",runs=[0],v=False):
        col_width =7 #in cm
        #note, first argument is the label of the figure for later saving
        fig=plt.figure(self.path,figsize=(col_width,col_width*0.8), facecolor="white")
        ax1=plt.subplot(111)
        if type == "simple":
            self.simpleplot(ax1)
        elif type == "multi":
            self.multiplot(ax1)
        elif type == "single":
            self.singleplot(ax1,runs)
        formatmultiplotaxis(ax1,"$V$","$I$")
        ax1.set_title(os.path.basename(self.path),fontsize=10,y=1.05)
        ax1.legend(loc='best', fancybox=True, framealpha=0.5,fontsize=15,ncol=2)
        self.fig=fig
        if v:
            print(type,"graph made from ",self.path)


    def savefig(self,savedir="plots/",v=False):
        if os.path.isdir(savedir) == False:
            os.system("mkdir " +savedir)
        plt.figure(self.path) #sets the current figure as labeled by path
        plt.savefig(os.path.join(savedir,os.path.basename(self.path)[:-4]+".png"))
        if v:
            print("saved: ",self.path)

    def get_timestamp(self):
        return self.path[-19:-4]



    def __exit__(self,*err):
        plt.close(self.fig)


class sequentialMeasurements(object):
        """
        makes a full list of the sequential datasets of all of the data in a folder by initializing them as multisweep objects
        """
        def __init__(self, samplefolder):
            super(sequentialMeasurements, self).__init__()
            self.dir=samplefolder
            self.datasets=[]
            self.initlog = self.make_multisweepData()
            self.paths= [x.path for x in self.datasets]
            self.timestamps = [ x.timestamp for x in self.datasets]
            self.make_finalRplot = matplotlib.figure.Figure()

        def __enter__(self):
            return self

        def make_multisweepData(self):
            log=[]
            for path in sorted(glob.glob(self.dir+"data/*.csv")):
                if "x" in path:
                    try:
                        self.datasets.append(multisweep(path,int(re.search("x(\d{1,3})",path).group(1))))
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
                i.makefig("multi",v=v)
        def save_plots(self,v=False):
            if v:
                print ("beginning plotting sequence")
            if os.path.isdir("plots") == False:
                os.system("mkdir " +self.dir+ "plots/")
            for i in self.datasets:
                i.savefig(self.dir+"plots/",v)
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
        with multisweep(self.testdataset,5) as testdata:
            self.assertEqual(testdata.splitdata().shape, (10,51,2))

    def test_gradR(self):
        with multisweep(self.testdataset, 5) as testdata:
            self.assertEqual(testdata.calc_gradR(testdata.runs[0]),
                    [2451394.4249741249, 1.9146863336543544e-05])
            self.assertEqual(testdata.make_gradR().shape,(10,2))
    def test_finalR(self):
        with multisweep(self.testdataset,5) as testdata:
            self.assertEqual(testdata.make_finalR().shape,(5,))
    def test_timestamp(self):
        with multisweep(self.testdataset, 5) as testdata:
            self.assertEqual(testdata.timestamp, "2016_10_19_1000")


    def test_make_multisweepData(self):
        with sequentialMeasurements(self.testdir) as testfolder:
            self.assertEqual(type(testfolder.datasets[0]), multisweep)
            self.assertEqual(len(testfolder.datasets), 28)
            self.assertEqual(testfolder.initlog, [])



if __name__ == "__main__":
    #ensures that test cases are not run when importing the module.
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    suite = unittest.TestLoader().loadTestsFromTestCase(MyTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
