#!/usr/bin/env python
from __future__ import print_function, division

import sys
import os
import glob
import inspect
import optparse
import scipy as sp
import numpy as np
import pylab as pl
import pandas as pd

from scipy import optimize
from scipy.interpolate import interp1d
from mpmath import polyroots
import time
import random
import pprint
import pickle

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path


from sort2vectors import sort2vectors

# set up plot parameters
pl.rc('axes', linewidth=2)
mpl.rcParams['font.size'] = 20
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.labelsize'] = 21
mpl.rcParams['xtick.labelsize'] = 21.
mpl.rcParams['ytick.labelsize'] = 21.

# add CfA lib folder to path
CFALIB = False
if os.getenv("SESNCFAlib"):
    CFALIB = True    
    cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))

    if cmd_folder not in sys.path:
        sys.path.insert(0, cmd_folder)

try:
    raw_input
except NameError:
    #  Python 3
    raw_input = input

# setup variables
NITMAX = 20  # max number iterations
EPS = 1.0E-20  # tollerance
LN10x2p5 = 5.75646273249  # algebra done ahead of time
minNboot = 20

# use CUT[LATE] = True if you want to bootstrap the edges of the subsample choice for the determination of peak
CUT = True  # False # randomely cutting early data points if there are enough
CUTLATE = True  # False # randomely cutting late data points if there are enough
SPEEDITUP = True  # looser convergence criteria but faster computing
DEBUG = False
GRAPHIC = False

bands = ['U', 'B', 'V', 'R', 'I', 'r', 'i', 'H', 'J', 'K', "w1", "w2", "m2"]  # CfA survey bands, this list is the default "all bands" choice
mycolors = {'U': 'k', 
            'B': '#0066cc', 
            'V': '#47b56c', 
            'R': '#b20000', 
            'I': 'm', 
            'r': '#b20000', 
            'i': 'm', 
            'J': '#9999EE', 
            'H': '#FF77AA', 
            'K': '#6BB5FF',
            'w1': '#6BB5FF',
            'w2': '#6BB5FF',
            'm2': '#6BB5FF'}



flagbadfit, flagmissmax, flagmiss15 = 0, 0, 0



def print2log(message):
    logoutput.write(str(message))


class SelectFromCollection: 
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()



class SelectFromCollection2(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool highlights
    selected points by fading them out (i.e., reducing their alpha values).
    If your collection has alpha < 1, this tool will permanently alter them.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)
        print(self.Npts)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, self.Npts).reshape(self.Npts, -1)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero([path.contains_point(xy) for xy in self.xys])[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        

def checkfit(y, yfit, closenough):
    ''' checks if a fit is sufficiently close
    '''
    bfac = 6.0
    dev = y - yfit
    # following idl robust_sigma

    # First, the median absolute deviation MAD about the median:
    mad = np.median(np.abs(dev)) / 0.6745

    # If the MAD = 0, try the MEAN absolute deviation:
    if mad < EPS:
        mad = np.mean(np.abs(dev)) / 0.8
    if mad < EPS: 
        print2log("fit already good enough")
        return (0, 0, 0)
    elif SPEEDITUP:
        # return anyways to speed things up
        return (0, 0, 0)

    # we actually never  used to go below this line in v1.0
    # but I am leaving the rest of the function for record
    
    # Now the biweighted value:
    u = (dev / 6. * mad)**2

    try:
        q = np.where(u < 1.0)[0]
        count = len(q)
    except:
        print2log("something strange about the distribution, " +
                  "robustcheck failed")
        return (-1, 0, 0)
    if count < 3:
        print2log("ROBUST_SIGMA: This distribution is TOO WEIRD! " +
                  "Returning -1")
        return (-1, 0, 0)
    else:
        n = np.nansum(dev)
        sig = n*np.sum((dev[q]**2) * ((1.0 - u[q])**4))
        den = np.sum((1.0 - u[q])*(1.0 - 5.0 * u[q]))
        sig = np.sqrt(sig / (den * (den - 1)))
      
    # If the standard deviation = 0 then we're done:
    if sig < EPS: 
        print2log("standard deviation is tiny: fit is fine")
        return (0, sig, 0)
    if closenough > 0:
        try:
            q = np.where(np.abs(yfit) > EPS)[0]
            if len(q) < 3:
                fracdev = 0.0
            else:
                fracdev = np.median(np.abs(dev[q] / yfit[q]))
                if fracdev < closenough: 
                    print2log("fit good enough")
                    return (0, sig, 0)
        except:  # this should have an explicit exception
            return (0, sig, 0)

    # Calculate the (bi)weights:
    b = np.abs(dev) / (6 * sig)
    try:
        s = np.where(b > 1.0)[0]
        if len(s) > 0:
            b[s] = 1
    except:
        s = []
    ngood = len(y) - len(s)
    if ngood < 10:
        print2log("too few good points left")
        return (2, sig, 0)
    w = (1.0 - b * b)
    w /= sum(w)
    return (1, sig, w)


def myrobustpolyfit(x, y, deg, weights):
    '''robust polyfit for weighted fits'''
    pars = np.polynomial.polynomial.polyfit(x, y, deg=deg, w=weights)
    try:
        # this should have only specific exceptions
        # but I forgot whagt they would be
        closenough = np.max(0.3 * np.sqrt(0.5 / (len(y) - 1)), EPS)
    except:
        closenough = EPS

    (cf, sig, w) = checkfit(y, np.poly1d(np.array(pars[::-1]))(x), 
                            closenough)

    if cf == 1:  # fit did not converge, redoit till it improves enough
        diff = 1.0e10
        sig1 = min((100. * sig), EPS)
        nit = 0
        while (diff > closenough) and (nit < NITMAX):
            print2log("iteration ", nit)
            nit = nit + 1
            sig2 = sig1
            sig1 = sig
            # We use the "obsolete" POLYFITW routine because it allows us to input weights
            # rather than measure errors
            g = np.where(w > 0)[0]
            try:
                ng = len(g)
            except:
                ng = 0
            if ng < len(w):
                # ;Throw out points with zero weight
                x = x[g]
                y = y[g]
                w = w[g]
            
            pars = np.polynomial.polynomial.polyfit(x, y, deg=deg, w=weights)
            (cf, sig, w) = checkfit(y, 
                                    np.poly1d(np.array(pars[::-1]))(x), 
                                    closenough)

            if cf == 0:
                nit = NITMAX
                print2log("fit converged!")
            if cf == 2:
                print2log("too few good point for robust iterations")
                nit = NITMAX

            diff = min((np.abs(sig1 - sig) / sig), (abs(sig2 - sig) / sig))
      
    if cf == 2:
        print2log("too few good point for robust iterations")
    if cf == 0: 
        print2log("fit converged!")

    return pars


def mypolynomial(x, pars):
    """calculates polynomial for a given x and set of coefficients"""
    y = 0.0
    for i, c in enumerate(pars):
        y += c * x**i
    return y


def mytemplate(x, pars):
    """calculates templateiven parameters and template form"""
    return pars[0] * (template.tempfuncy()(x-pars[1]))+pars[2]  


def myresiduals(pars, x, y, e, functype):
    """Calculates residuals between data and template for polynomial or fuctional template"""
    if functype == 'poly':
        resids = (y - (mypolynomial(x, pars))) / e  
    elif functype == 'template': 
        resids = (y - (mytemplate(x, pars))) / e
        nonanindx = np.where(~np.isnan(resids))[0]
        resids = resids[nonanindx]  
    else: 
        print ("function can only be 'template' or 'poly'")
        sys.exit()
        #  pl.show()
        #  print sum(resids**2)
    m = np.ma.masked_array(resids, np.isnan(resids))  
    return resids


def sumsqres(pars, x, y, e, functype):
    """sum resuduals squared"""
    return sum(myresiduals(pars, x, y, e, functype)**2)


def fitit(x, y, dy, deg=2):
    """fits polynomial to data"""

    # keeps trying if redo is 1 and iterationshere<100
    # initialize redo iterationshere
    redo = 1
    iterationshere = 0
    while redo and iterationshere < 100:
        iterationshere += 1
        pars = myrobustpolyfit(x, y, deg, 1.0 / dy)
        pars = np.polynomial.polynomial.polyfit(x, y, deg=deg, w=1.0/dy)
        polysol = np.poly1d(np.array(pars[::-1]))
        xp = np.linspace(min(x), max(x), 1000)
        #       myfig = pl.figure(figcounter)             
        minyall = max(y)  # faintest magnitude 
        maxyall = min(y)  # brightest magnitude
        
        #  fitting with polynomial
        optimize.leastsq(myresiduals, pars, args=(x, y, dy, 'poly'),
                         full_output=1, epsfcn=0.00001,
                         ftol=1.49012e-08)   
        #        sys.exit()
        try:
            all = optimize.leastsq(myresiduals, pars, args=(x, y, dy, 'poly'),
                                   full_output=1, epsfcn=1.0e-5,
                                   ftol=1.49012e-08)
        except:  # missing explicit exception
            redo = 0
            print2log("ERRRRRRRRRRRRRRRRRRRRRRRRROR: cant optimize")
            continue
    return all, pars, xp

if __name__ == '__main__':

    parser = optparse.OptionParser(usage="python getmaxdate_boot.py " +
                                   "[snname or filename if not --loadlc] " +
                                   "-t # -m # -d # ",
                                   conflict_handler="resolve")
    parser.add_option('-t', '--timecol', default=0, type="int", 
                      help='time column (start w 0th)')
    parser.add_option('-m', '--magcol', default=2, type="int", 
                      help='mag column')
    parser.add_option('-d', '--dmagcol', default=3, type="int", 
                      help='dmag column')
    parser.add_option('-n', '--np', default=0, type='int', 
                      help='number of datapoints (n=-1 for all)')
    parser.add_option('-s', '--sp', default=0, type='int', 
                      help='number of datapoints to skip')
    parser.add_option('-g', '--graphic', default=False,
                      action="store_true", 
                      help='selecting the portion of the lightcurve ' + 
                      'to use geraphically with python widgets')    
    parser.add_option('-l', '--loadlc', default=False, action="store_true", 
                      help='load from cfa photometry file ' + 
                      '(you can omit file name)')
    parser.add_option('--lit', default=False, action="store_true", 
                      help='load from literature cfa formatted ' + 
                      "photometry file " + 
                      "(you can omit file name)")
    parser.add_option('-b', '--band', default='all', type='string', 
                      help='photometric band, needed with loadlc')
    parser.add_option('--bandname', default=None, type='string', 
                      help='photometric name of the photometric band in thte file')
    parser.add_option('--snname', default=None, type='string', 
                      help='SN name')
    parser.add_option('--bandcolumn', default=0, type="int", help='band column')
    parser.add_option('--separator', default=',', type='string', 
                       help='column separator, default is ","')
    parser.add_option('--verbose', default=False, action="store_true",
                       help='verbose mode')

    

    options, args = parser.parse_args()
    if options.verbose:
        DEBUG = True

    if DEBUG:
        print(options)
    if len(args) != 1 and len(args) != 2: # not enough arguments
        sys.argv.append('--help')  
        options, args = parser.parse_args()
        sys.exit(0)

    #load a file in standard text format (space separated values)
    if not options.loadlc:
        f = args[0]

    # automatically load files from the CfA supernova sample with the CfA supernova library
    if options.loadlc:
        if not CFALIB: # https://github.com/nyusngroup/SESNCfAlib
            print("cannot use this option without the SESNcfalib")
            sys.exit()
        from snclasses import *
        f = args[0]  # +"*[cf]")

        if not options.lit:
            f = glob.glob(os.environ["SESNPATH"] +
                          "/finalphot/*"+args[0] + "*.[cf]")[0]
            print (f)
        else: 
            f = glob.glob(os.environ["SESNPATH"] +
                          "/templateLitData/phot/*" +
                          args[0] + "*.[cf]")
            if len(f) == 0:
                print (glob.glob(os.environ["SESNPATH"] +
                                 "/literaturedata/phot/*" +
                                 args[0] + "*.[cf]"))
                f = glob.glob(os.environ["SESNPATH"] +
                              "/literaturedata/phot/*" +
                              args[0] + "*.[cf]")
            if len(f) == 0:
                print ("no files available")
            f = f[0]
    
        bandcounter = 1

    # selects the points with graphic interaction
    if options.graphic:
        GRAPHIC = True
    if 'all' not in options.band:      
        bands = options.band

    if not os.path.isdir('logs'):
        os.mkdir('logs')
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')        


    print ("BANDS: %s" % options.band)
    if isinstance(bands, str):
        bands = [bands]

    for b in bands:
        if not options.lit:
            output = f + "_" + b + ".log"
        else:
            output = f + "_" + b + "_lit.log"
        
        if DEBUG: print("output left in", output)

        global logoutput 
        logoutput = open(os.path.join('logs', output), 'a')


        print2log("########################################## ")
        print2log("TIME NOW %f" % time.time())    
        if options.loadlc:
            # uses SESNCfA lib for reading in lcvs
            thissn = mysn(f)
            lc, flux, dflux, snname = thissn.loadsn(f)
            print2log("\n\nSN NAME: %s\n\n" % snname)
            print2log("input file: %s" % f)
            print ("\n\nSN NAME: \n\n", snname, "\n\n")
            print ("band:", b)
            thissn.setphot()
            thissn.getphot()
      
            lc = thissn.photometry[b]
            thissn.printsn()
            lc = pd.DataFrame(lc)
        else:

            usecols = np.array([options.bandcolumn,
                                options.timecol, 
                                options.magcol, 
                                options.dmagcol])

            indx = np.argsort(usecols)
            names = np.array(['band', 'mjd', 'mag', 'dmag'])
            formats = np.array(['str', 'float', 'float', 'float'])
            
            # reads in an ascii file with lcv data
            snname = options.snname
            
            if snname is None:
                snname = f.split("/")[-1].split(".")[0]
                snname.replace('sn0', 'SN 200')
            print2log("\n\nSN NAME: %s\n\n" % snname)
            print2log("input file: %s" % f)
            print ("\n\nSN NAME: \n\n", snname, "\n\n")            
            
            if DEBUG:
                print("input file: %s" % f)                

            
            """
            lc = np.genfromtxt(f, usecols=(options.bandcolumn,
                                           options.timecol, 
                                           options.magcol, 
                                           options.dmagcol), 
                               dtype={'names': ('band', 'mjd', 'dmag', 'mag'), 
                                        'formats': ('<S5', 'f8', 'f8', 'f8')},
                               delimiter=options.separator)
            """
            
            lc = pd.read_csv(f, header=None,
                             sep=options.separator,                             
                             names=names[indx],
                             dtype={names[indx][i]:formats[indx][i]
                                    #'band':'str', 'mjd':float, 'mag':float, 'dmag':float
                                    for i in range(4)},
                             usecols=usecols[indx])
            if DEBUG:
                print(lc, "bands", bands)
                
            #snname = f.split("/")[-1].split('.')[0]
            
            if len(bands) == 1:
                if options.bandname is None:
                    options.bandname = bands[0]
                lc = lc[lc['band'] == options.bandname]

        if DEBUG:
            print(lc)
        # set negative uncertainties to 0 - should not happen but some placeholders are used
        #lc['dmag'][lc['dmag'] == 0] = min(lc['dmag'][lc['dmag'] >= 0])
        lc.loc[lc['dmag'] == 0, 'dmag'] = min(lc.loc[lc['dmag'] > 0, 'dmag'])

        # sorting by date if not already
        #indx = np.argsort(lc['mjd'])
        #lc['mjd'] = lc['mjd'][indx]
        #lc['mag'] = lc['mag'][indx]
        #lc['dmag'] = lc['dmag'][indx]
        lc.sort_values(by="mjd")
        lc.reset_index(inplace=True)

        if DEBUG:
            print(lc)
        if len(lc['mag']) == 0:
            continue
        mindate = maxdate = 0
        
        fboot = os.path.join('outputs', snname + '.boot')
        fboot = fboot.replace('.boot', '_' + b + '.boot')
      
        if options.np == 0:
            # ask the user to look at the LC and select the number of points to use and skip
            fig = pl.figure(figsize=(9, 9))
            mjdrange = max(lc['mjd']) - min(lc['mjd'])
            ax = fig.add_subplot(
                xlim=(min(lc['mjd']) - mjdrange * 0.05, max(lc['mjd']) + mjdrange * 0.05),
                ylim=(max(lc['mag']) + 0.2, min(lc['mag']) - 0.2),
                title='Lasso points using left mouse button')

            pts = ax.scatter(lc['mjd'], lc['mag'], s=20, ###alpha=0.5,
                             color=mycolors[b])
            
 
            if not GRAPHIC:
                ax.errorbar(lc['mjd'], lc['mag'], yerr=lc['dmag'],
                            color=mycolors[b], fmt='.',
                            alpha=0.3)
 
                pl.draw()
                pl.show()

                try:
                    options.np = int(raw_input('how many datapoints should we use?\n'))
                except ValueError:
                    print ("Not a number")
                    sys.exit()

            else:
                done = []
                selector = SelectFromCollection(ax, pts)

                #Make sure you all points are within a range:
                #partial selection and exclusion of individual datapoints
                #is not allowed (and weird things may happen).
                print("Select points in the figure by enclosing them within a polygon. Then press any key to accept selected points, then close the window.")
                ax.set_title("Select points in the figure by enclosing them within a polygon\n Close this window to accept selected points")
           

                def accept(event):
                    if event.key == "enter":
                        print("Selected points:")
                        print(selector.xys[selector.ind])
                        selector.disconnect()
                        ax.set_title("")
                        fig.canvas.draw()

                fig.canvas.mpl_connect("key_press_event", accept)
                pl.show()
                selector.disconnect()
                
                xys = selector.xys[selector.ind]

                # After figure is closed print the coordinates of the selected points
                print('\nSelected points:')
                                    
                if DEBUG:
                    print("Selected points:")
                    print(xys)

                lcv = np.array(list(zip(lc['mjd'], lc['mag'])))
                jj = 0
                while not (lcv[jj] == xys[0]).all():
                    jj += 1
                options.sp = jj
                options.np = len(xys)
        
            if options.np == 0: # no points
                continue
            if options.np < len(lc['mag']) and not GRAPHIC:
        
                try:
                    options.sp = int(raw_input('how many datapoints should we skip?\n'))
                except ValueError:
                    print ("Not a number")
                    options.sp = 0
        mindate = options.sp
        mynp = options.np if options.np >= 0 else len(lc['mjd'])
        
        print ("")
        print ("using", mynp, "skipping", options.sp)
    
        output =  os.path.join('outputs', output.replace('.log', '_s%d_n%d.dat' %
                                                           (options.sp, options.np)))
        finaloutput = open(output, 'w')
        
        fboot = open(fboot, "w")
        
        maxjd = 0
        minyall = 0
        maxyall = 100
        deg = 2  # polynomial degrees (always should be 2)

        #  these lists will host the fit values
        maxs = []
        mjdmaxs = []
        d15s = []
        allchisqs = []
        alldegs = []
        indx0 = range(mynp)
        ndata = len(lc['mjd'])

        # testing date format (trivially by comparing w 6e4)
        if lc.loc[0, 'mjd'] > 60_000:
            #lc['mjd'][0] > 60000: 
            print2log("CAREFUL: date in JD instead of MJD")
            # resetting date to MJD
            lc['mjd'] = lc['mjd'] - 2400000.5

        # subtracting 5e4 to simplify calculations
        subtract = 50000.00000
        mjd = lc['mjd'] - subtract

        # setup figure
        myfig = pl.figure()
        ax = myfig.add_subplot(1, 1, 1)
        ax.minorticks_on()
        majorFormatter = FormatStrFormatter('%d')
        minorLocator = MultipleLocator(0.2)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_formatter(majorFormatter)

        # number of bootstrap iterations:
        # number of datapoints * ln(number datapoints)**2 or 200 if it is less than that
        Nboot = mynp
        print2log("Nboot %d" % Nboot)
        Nboot = int(Nboot*((np.log(Nboot))**2)+0.5)

        # if you are using the whole lightcurve just fit it (does this need a better criterion??)
        if mynp == len(lc['mag']) :
            x = mjd.astype(np.float64) 
            y = lc['mag'].values.astype(np.float64)
            dy = lc['dmag'].values.astype(np.float64)
            all, pars, xp = fitit(mjd, y, dy, deg=deg)
            lsq = all[0]
            covar = all[1]
            success = all[4]
        
            polysol = np.poly1d(np.array(lsq[::-1]))
            
            # deviation from model
            dev = y - polysol(x)
            # mad = np.median(np.abs(dev)) / 0.6745
            chisq = sum((dev**2) / lc['dmag'][indx])
            rchisq = chisq / (len(dev) - deg - 2)  # reduced chi sq
            lsq = all[0]  # /[0.1, 10.0, 1000., 10000]
            covar = all[1]
            success = all[4]
            solution = {'sol': polysol,
                        'deg': deg,
                        'pars': pars,
                        'covar': covar}
            try:
                root = polyroots(solution['pars'])[0].real
            except:  # what is this exception...
                continue
            
            mjdindex = np.where(solution['sol'](xp) == 
                                min(solution['sol'](xp)))[0]  # index of max
      
            maxjd = np.mean(xp[mjdindex])  # if there are more than one identical maxima - this should make you suspicious anyways
            maxflux = solution['sol'](maxjd)  # mag at max
            print ("maxjd", maxjd+50000, "maxflux", maxflux)

            ax.plot(x + 0.5, y, 'o', alpha=0.1, color='k')
            ax.plot(xp + 0.5, solution['sol'](xp + 0.5), '-',
                    color="k", alpha=0.1)
            # why + 0.5? I forgot
            pl.show()

        
        else:
            # you have extra datapoints at the edges of the region you selected
            nmc = max(Nboot, minNboot)
            nmc = 20
            for nb in range(nmc):
                try:
                    print ('{0}/{1}\r'.format(nb, nmc), end='\r', flush=True)
                except TypeError:  # python 3.x
                    sys.stdout.write('{0}/{1}\r'.format(nb, nmc))
                    sys.stdout.flush()
                    
                print2log("mindate %f, maxdate %f " % (mindate, maxdate))
                if CUT:
                    # randomely cut earliest two datapoints
                    newmindate = random.randint(mindate,
                                                mindate + min(2, int(ndata / 10)))
                    newmaxdate = random.randint(maxdate,
                                                maxdate + max(int(ndata / 10), 2))
                elif CUTLATE:
                    # randomely cut latest 10% datapoints
                    newmindate = mindate
                    newmaxdate = random.randint(maxdate, maxdate +
                                                max(int(ndata / 10), 1))
                else:
                    newmindate = mindate
                    newmaxdate = maxdate
                print2log("new min %f and max %f %f: " % (newmindate, 
                                                          newmaxdate, mynp))
                # reset minimum date
                indx = np.array(range(newmindate, mynp + newmindate))
                #print(indx)
                #needed if all points are included after peak                
                indx = indx[indx <= mynp] 
                if len(indx) - newmaxdate >= 3 and not newmaxdate == 0:
                    indx = indx[:-newmaxdate]
                ndata = len(indx)
                if ndata < 3:
                    print2log("ndata too short")
                    print ("ndata too short")
                    continue

                x = mjd[indx].values
                # randomely sample form each photometry measurement and errorbar in a gaussian
                y = np.random.normal(lc['mag'][indx], lc['dmag'][indx])
                x = x.astype(np.float64) 
                y = y.astype(np.float64)
                dy = lc['dmag'][indx].values.astype(np.float64)

                itercount = 1
                solutions = []
                rchis = []

                # fit this realization
                all, pars, xp = fitit(x, y, dy, deg=deg)
                lsq = all[0]   # /[0.1, 10.0, 1000., 10000]
                covar = all[1]
                success = all[4]
        
                print2log(all[3])
                print2log("SUCCESS: %d" % success)
                print2log('''
                

                ''')

                # get the fit values
                polysol = np.poly1d(np.array(lsq[::-1]))
                # measure deviation from data and chisq
                dev = y - polysol(x)
                chisq = sum((dev**2) / lc['dmag'][indx])
                rchisq = chisq / (len(dev) - deg - 2)
                solutions.append({'sol': polysol,
                                  'deg': deg,
                                  'pars': pars,
                                  'covar': covar})
                rchis.append(rchisq)
        
                print2log("rchisq, deg, iteration count: %f %d %d " % (rchisq, deg, itercount))

                # collapsing array of solution to the one that has reduced chisq closest to 1
                sol = np.where(abs(np.array(rchis) - 1) == np.min(abs(np.array(rchis) - 1)))[0]
                
                if len(sol) > 1: 
                    sol = sol[0]
                    print ("WARNING: some fits are equivalent. very suspicious")
                    print2log("WARNING: some fits are equivalent. very suspicious")
                else:
                    sol = sol[0]
                    
                solution = solutions[sol]
                rchisq = rchis[sol]
                print2log("final chisq: %f" % rchisq)
                resids = y - solution['sol'](x)
                errors = np.sqrt(np.diag(solution['covar'] * rchisq))  # *solution['pars']
                
                print2log("parameters and errors")
                print2log(pars)
                print2log(errors)
                
                try:
                    root = polyroots(solution['pars'][::-1])[0].real
                except:
                    continue
                if DEBUG:
                    print(root)

                print2log(root)
                print2log(xp[np.where(solution['sol'](xp) == 
                                      min(solution['sol'](xp)))[0]])
        
                mjdindex = np.where(solution['sol'](xp) == 
                                    min(solution['sol'](xp)))[0]
                print2log(xp[mjdindex])
        
                maxjd = np.mean(xp[mjdindex])
                if DEBUG:
                    print ("MaxMJD %f" % maxjd)
                maxflux = solution['sol'](maxjd)
                if DEBUG:
                    print ("MaxFlux %f" % maxflux)
                try:
                    dm15 = maxflux - solution['sol'](maxjd + 15.0)
                    if DEBUG:
                        print ("dm15 %f" % dm15)
                except:
                    maxjd = -nan
                    dm15 = -nan
                    print ("########### could not find maximum. moving on ##########")
                    print2log("dm15 FAILED!!")
                    continue
                print2log("\n\nSummary %f %f %f %f \n\n" %
                          (maxflux, maxjd,
                           np.sum(resids*resids) / (ndata-solution['deg']),
                           rchisq))
                redo = 0
                
                maxs.append(maxflux)
                mjdmaxs.append(maxjd)
                d15s.append(maxflux - solution['sol'](maxjd + 15.0))
                allchisqs.append(rchisq)
                alldegs.append(solution['deg'])
                
                ax.plot(x+0.5, y, 'o', alpha=0.1, color='k')
                ax.plot(xp+0.5, solution['sol'](xp+0.5), '-', color = "k", alpha = 0.1)


        print (f + " max mjd in band " + b + ": %f %.2f " %
               (maxjd + 50000.0000, maxflux))

        dm15 = np.median(d15s)  # maxflux-solution['sol'](maxjd+15.0)    
        medianmjdmax = np.median(mjdmaxs)
        mjdpercentiles = np.percentile(mjdmaxs, [25,75])
        medianmaxs = np.median(maxs)
        maxspercentiles = np.percentile(maxs, [25,75])        
        print(dm15, medianmjdmax, mjdpercentiles, medianmaxs, maxspercentiles)


        ax.plot(lc['mjd'].values + 0.5 - subtract, lc['mag'].values, '.',
                color=mycolors[b])
        ax.errorbar(lc['mjd'].values + 0.5 - subtract, lc['mag'].values,
                    yerr=lc['dmag'].values, 
                    fmt = 'none', ecolor=mycolors[b])

        pl.ylim(max(lc['mag'].values) + 0.2, min(lc['mag'].values) - 0.2)            
        pl.xlim(min(lc['mjd'].values) - 2 - subtract, max(lc['mjd'].values) +
                3 - subtract)
        ax.locator_params(tight = True, nbins = 4)
        # myplot_setlabel
        ax.set_xlabel('MJD - 50000.00')
        ax.set_ylabel(b + ' mag')  # , title = snname+' '+b, ax = ax)


        fboot.write(" max mjd in band " + b + ": %f (%f, %f) %.2f (%.2f, %.2f) \n" %
                    (maxjd,
                     mjdpercentiles[0] + subtract,
                     mjdpercentiles[1] + subtract,
                     maxflux, maxspercentiles[0], maxspercentiles[1]))
        
        
        if not isinstance(medianmjdmax, float):
            medianmjdmax = medianmjdmax[0]

        ax.arrow(medianmjdmax, np.median(maxs) + 0.5, 0,
                 -0.2, head_width=0.5, head_length=0.05,
                 fc='r', ec='r')
        ax.plot(mjdpercentiles, [np.median(maxs) + 0.5, np.median(maxs) + 0.5],
                'r-')
        
        pl.title(" %s skip = %d use = %d" % (snname, options.sp, options.np))
        
        if 1: #try: 
            if isinstance(medianmjdmax, float):
                pl.text(0.50, 0.70, (r'$JD_\mathrm{max}$: 24%.2f (%.2f)' %
                                     (medianmjdmax + subtract + .5,
                                      np.std(mjdmaxs))),
                                     ha='left', fontsize=17, 
                                     transform=myfig.transFigure)
        if 1:#except:   # missing exception :-( what was I afraid of??
            pl.text(0.50, 0.70, 
                    (r'$MJD_\mathrm{max}$: %.2f (%.2f, %.2f)' % (medianmjdmax,
                                                                 mjdpercentiles[0],
                                                                 mjdpercentiles[1])),
                    ha = 'left', fontsize = 17, transform = myfig.transFigure)

        finaloutput.write("skipped %d used %d\n" % (options.sp, options.np))
        print ("\n\nskipped %d used %d" % (options.sp, options.np))
        if 1: #try:
            finaloutput.write('JD_max, 24%.2f (24%.2f 24%.2f)\n' % ((medianmjdmax +
                                                                 subtract + .5), 
                                                                mjdpercentiles[0] + subtract + .5,
                                                                mjdpercentiles[1] + subtract + .5))
            finaloutput.write('MJD_max: 5%.2f (5%.2f 5%.2f) \n' % (medianmjdmax,
                                                                   mjdpercentiles[0],
                                                                mjdpercentiles[1]))            
            finaloutput.write('M_max: %.2f (%.2f %.2f)\n' % (medianmaxs,
                                                             maxspercentiles[0],
                                                             maxspercentiles[1]))

            print ('JD_max, 24%.2f (%.2f %.2f)' % ((medianmjdmax +
                                               subtract + .5),
                                               mjdpercentiles[0] + subtract + .5,
                                              mjdpercentiles[1] + subtract + .5))
            print ('MJD_max: 5%.2f %.2f %.2f' % (medianmjdmax, 
                                              mjdpercentiles[0],  mjdpercentiles[1]))
            print ('M_max: %.2f %.2f %.2f' % (medianmaxs, maxspercentiles[0],
                                                             maxspercentiles[1]))
                                                                         
        if 1: #except:
            finaloutput.write('JD_max: 24%.2f %.2f\n' % (medianmjdmax +
                                                        subtract + .5, 
                                                        np.std(mjdmaxs)))
            finaloutput.write('JD precentiles 25th: 24{0:.2f} 75th: 24{1:.2f}\n'.\
                              format(*mjdpercentiles + subtract + .5))

            finaloutput.write('MJD_max: 5%.2f %.2f\n' % (medianmjdmax, 
                                                np.std(mjdmaxs)))
            finaloutput.write('MJD precentiles 25th: {0:.2f} 75th:{1:.2f}\n'.\
                              format(*mjdpercentiles))

            
            finaloutput.write('M_max: %.2f %.2f\n' % (np.median(maxs), 
                                                     np.std(maxs)))

            
            finaloutput.write('M_max percentiles 25th: {0:.2f} 75th:{1:.2f}\n'.\
                              format(*np.percentile(maxs,
                                            [25,75])))
            
            print ('JD_max: 24%.2f %.2f ' % (medianmjdmax +
                                             subtract + .5, 
                                             np.std(mjdmaxs)))
            print ('MJD_max: 5%.2f %.2f ' % (medianmjdmax, 
                                             np.std(mjdmaxs)))
            
            print ('M_max: %.2f %.2f ' % (np.median(maxs), 
                                          np.std(maxs)))
            print('JD precentiles 25th: 24{0:.2f} 75th: 24{1:.2f}'.\
                              format(*mjdpercentiles +
                                     subtract + .5))
            
            print('MJD_max: 5%.2f %.2f ' % (medianmjdmax, 
                                            np.std(mjdmaxs)))
            print('MJD precentiles 25th: {0:.2f} 75th:{1:.2f}'.\
                              format(*mjdpercentiles))
            
            
            print('M_max: %.2f %.2f ' % (np.median(maxs), 
                                                     np.std(maxs)))
            
            
            print('M percentiles 25th: {0:.2f} 75th:{1:.2f}'.\
                              format(*np.percentile(maxs,
                                [25,75])))
            pl.text(0.50, 0.75, (r'$M_\mathrm{max}$: %.2f (%.2f)' %
                                 (np.median(maxs), 
                                  np.std(maxs))), 
                    ha = 'left', fontsize = 17, transform = myfig.transFigure)
            pl.text(0.50, 0.65, (r'$\Delta m_{15}$: %.2f (%.2f)' %
                                 (-dm15, np.std(d15s))), 
                    ha = 'left', fontsize = 17, transform = myfig.transFigure)
            
            try:
                finaloutput.write('dm15 : %.2f %.2f\n' % (-dm15, np.std(d15s)))
                print('dm15 : %.2f %.2f\n\n' % (-dm15, np.std(d15s)))
            except:
                pass
            
            print ("\n\noutput saved in ", output)
            if b == 'r':
                b = 'rp'
            elif b == 'i':
                b = 'ip'
                
                pl.text(0.5, 0.15, snname+' '+b, ha = 'center', fontsize = 17,
                        transform = myfig.transFigure)
                if not os.path.isdir("bootstrap"):
                    os.system("mkdir bootstrap")
                if not options.lit:
                    print ("bootstrap/%s_%s_boot%03d_s%d_n%d.pdf" %
                           (snname.replace('SN ', 'sn'),
                            b, nb, options.sp, options.np))
                    myfig.savefig("bootstrap/%s_%s_boot%03d_s%d_n%d.pdf" %
                                  (snname.replace('SN ', 'sn'),
                                   b, nb, options.sp, options.np))
                if CFALIB:
                    print(CFALIB)
                    os.system("perl %s/"%os.environ['SESNPATH'] +
                                 "pdfcrop.pl " +
                           "bootstrap/%s_%s_boot%03d_s%d_n%d.pdf" %
                           (snname.replace('SN ', 'sn'),
                            b, nb, options.sp, options.np))
            else:
                if not os.path.isdir("bootstrapPhotLit"):
                    os.system("mkdir bootstrapPhotLit")                
                print ("bootstrapPhotLit/%s_%s_boot%03d.png" %
                       (snname.replace('SN ', 'sn'), b, nb))
                myfig.savefig("bootstrapPhotLit/%s_%s_boot%03d_s%d_n%d.pdf" %
                              (snname.replace('SN ', 'sn'),
                               b, nb, options.sp, options.np))
                if CFALIB:
                    os.system("perl %s/pdfcrop.pl %s" %
                          (os.environ['SESNPATH'],
                           "bootstrapPhotLit/%s_%s_boot%03d.pdf" %
                           (snname.replace('SN ', 'sn'), b, nb)))     
            pl.show()
            if GRAPHIC:
                raw_input("press any key to kill\n")
