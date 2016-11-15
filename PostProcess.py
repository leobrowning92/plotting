import modules.PAplot as PAplot
import os
import modules.cleandata as cleandata
import traceback
import sys
import time
#Plotter.plot_two(os.getcwd()+"\\10umgap0_10V.csv",skip=26,title="",xlabel="Gate Voltage (V)",ylabel="Current (A) x$10^{-8}$",data1="$I_G$",data2="$I_D$")
#Plotter.plot_from_file("pa/10umgap-10_10Vdiode.csv",title="",xlabel="Voltage (V)",ylabel="Current (I) x$10^{-12}$")


# allows the Run script to be used within
os.chdir(os.path.dirname(os.path.realpath(__file__)))
print ("Running post process script\n"+time.strftime('%Y_%m_%d_%H%M'))
print ("\n=============================================================\n")
filenames = os.listdir("data")
paths = []

for name in filenames:
    paths.append("data/" + name)
# uncomment for fet measurements#

for path in paths:
    try:
        cleandata.check(path)
    except Exception as e:
        print e
for path in paths:
    try:
        if "FET" in path:
            try:
                PAplot.plot_two_yscales(path, skip=1, title="", show=False, log=True,xlabel="Gate Voltage (VG)", y1label="ID (A)", y2label="IG (A)")
                print path + "\nplotted"
                print "----------"
            except Exception as e:
                print('error while plotting : \n ' + path)
                print(e)

        elif "diode" in path:
            try:
                PAplot.plot_IV(path, skip=2, title="", show=False, xlabel="Gate Voltage (VG)", ylabel="ID (A)",log=False)
                print path + "\nplotted"
                print "----------"
            except Exception as e:
                print('error while plotting : \n ' + path)
                print(e)
        else:
            print name + "unprocessed, uspecified plot type Try including 'FET' or 'diode' in the filename"
    except Exception as e:
        print e
