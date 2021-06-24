# IMPORTING RELEVANT PACKAGES
import math, re
import matplotlib.pyplot as plt
from decimal import *
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np

k = 3
d = 100
eps = 1.5
n = 50000
dta = 0.25
width = 0.35
parset = ['t', 'k', 'm', 'd', 'eps', 'n']
limit = 10

# THE X-AXIS, TICKET AND TITLE ARE INDIVIDUALLY TAILORED FOR EACH PARAMETER AND WHETHER DISCRETE FOURIER TRANSFORM IS USED
def custom(index, dft):

    # VARYING THE NUMBER OF COORDINATES T RETAINED
    if index == 0:
        plt.xticks(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']) 
        plt.xlabel('Number of coordinates retained', labelpad = 8)

        # A SINGLE EXPERIMENTAL ERROR IS PLOTTED IN THE BASIC CASE
        if dft == 0:
            plt.title('Experimental error by number of coordinates retained')

        # RATIO BETWEEN EXPERIMENTAL ERRORS IS PLOTTED IN THE FOURIER CASE
        else:
            plt.title('Ratio between experimental errors by number of coordinates retained')

    # VARYING THE NUMBER OF BUCKETS K USED
    elif index == 1:
        plt.xticks(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        plt.xlabel('Number of buckets used', labelpad = 8)

        if dft == 0:
            plt.title('Experimental error by number of buckets used')
        else:
            plt.title('Ratio between experimental errors by number of buckets used')

    # VARYING THE NUMBER OF FOURIER COEFFICIENTS M
    elif index == 2:
        plt.xticks(['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
        plt.xlabel('% of Fourier coefficients retained', labelpad = 8)

        if dft == 0:
            plt.title('Experimental error by % of Fourier coefficients retained')
        else:
            plt.title('Ratio between experimental errors by % of Fourier coefficients retained')

    # VARYING THE VECTOR DIMENSION D
    elif index == 3:
        plt.xticks(['60', '70', '80', '90', '100', '110', '120', '130', '140', '150'])
        plt.xlabel('Vector dimension', labelpad = 8)

        if dft == 0:
            plt.title('Experimental error by vector dimension')
        else:
            plt.title('Ratio between experimental errors by vector dimension')
    
    # VARYING THE VALUE OF EPSILON
    elif index == 4:
        plt.xticks(['0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.5', '2.0', '2.5', '3.0'])
        plt.xlabel('Value of epsilon', labelpad = 8)

        if dft == 0:
            plt.title('Experimental error by value of epsilon')
        else:
            plt.title('Ratio between experimental errors by value of epsilon')

    # VARYING THE NUMBER OF VECTORS N USED
    else:
        plt.xticks(['6', '7', '9', '12', '15', '20', '30', '40', '50', '60'])
        plt.xlabel('Number of vectors used' + ' ' + 'x' + ' ' + '$10^{3}$', labelpad = 8)

        if dft == 0:
            plt.title('Experimental error by number of vectors used')
        else:
            plt.title('Ratio between experimental errors by number of vectors used')

# THE SKELETON DRAWING FUNCTION IN THE BASIC CASE
def drawBasic(index):
    labels = list()
    totalErrors = list()
    totalStandardDeviation = list()
    gammas = list()
    seeds = list()
    rowCount = 0

    # PUTTING THE DATA ON THE AXES
    with open("errorvary" + str(index) + "%s.txt" % parset[index]) as reader:
        for line in reader:
            tab = line.split()
            
            if index == 4:
                labels.append(f'{float(tab[0])}')
                seeds.append(float(tab[0]))
            else:
                labels.append(f'{int(tab[0])}')
                seeds.append(int(tab[0]))

            totalErrors.append(Decimal(tab[1]))
            totalStandardDeviation.append(Decimal(tab[2]))
            gammas.append(float(tab[3]))
            rowCount += 1

            if rowCount >= limit:
                break

    # THE BARS PLOTTED IS THE SAME FOR EACH PARAMETER
    plt.bar(labels, totalErrors, width, alpha = 0.6, color = 'm', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

    # PLOTTING COMPARISON LINE GRAPHS TO VERIFY DEPENDENCIES WITH D, EPSILON AND N
    plotTuple = tuple(zip(seeds, gammas))
    plotTupleEps1 = tuple(zip(seeds[:6], gammas[:6]))
    plotTupleEps2 = tuple(zip(seeds[5:], gammas[5:]))
    x = np.array(labels)
    xEps1 = x[:6]
    xEps2 = x[5:]

    if index == 3:
        p1 = [0.0000005*((s**(29/12))/((1-g))**2) for s, g in plotTuple]
        y1 = np.array(p1)
        plt.plot(x, y1, label = 'Best fit curve', alpha = 0.6, color = 'k')
    elif index == 4:
        p2 = [0.04*((1/(s**(7/6)))/((1-g))**2) for s, g in plotTupleEps1]
        y2 = np.array(p2)
        plt.plot(xEps1, y2, label = 'Best fit curve part 1', alpha = 0.6, color = 'b')
        p3 = [0.04*((1/(s**(1/3)))/((1-g))**2) for s, g in plotTupleEps2]
        y3 = np.array(p3)
        plt.plot(xEps2, y3, label = 'Best fit curve part 2', alpha = 0.6, color = 'r')
    elif index == 5:
        p4 = [(0.8*((1/(s**(7/6)))/((1-g))**2))+0.025 for s, g in plotTuple]
        y4 = np.array(p4)
        plt.plot(x, y4, label = 'Best fit curve', alpha = 0.6, color = 'k')

    # THE Y-AXIS IS THE SAME FOR EACH PARAMETER
    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.ylabel('Total experimental MSE')

    # CREATING A LOGARITHMIC Y-AXIS FOR THE EPSILON AND N DEPENDENCIES
    if index == 4:
        plt.yscale('log')
        plt.ylim(0.02, 1.2)
        selectiveFormatter = FixedFormatter(["0.01", "0.1", "1"])
        selectiveLocator = FixedLocator([0.1, 1])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)
    elif index == 5:
        plt.yscale('log')
        plt.ylim(0.02, 15)
        selectiveFormatter = FixedFormatter(["0.01", "0.1", "1", "10"])
        selectiveLocator = FixedLocator([0.1, 1, 10])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

# THE SKELETON SAVING FUNCTION IN THE BASIC CASE
def saveBasic(index):
    if index >= 3:
        plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.savefig("errorchartvary" + str(index) + "%s.png" % parset[index])
    plt.clf()
    plt.cla()

# MAIN PLOTTING FUNCTION IN THE BASIC CASE: COMBINING THE ABOVE
def plotBasic():

    # LEAVING OUT THE PARAMETER M AS IT IS NOT USED HERE
    for index in range(6):
        if index == 2:
            continue

        drawBasic(index)
        custom(index, 0)
        saveBasic(index)

# THE SKELETON DRAWING FUNCTION IN THE FOURIER CASE
def drawDft(index):
    labels = list()
    perErrors = list()
    recErrors = list()
    totalErrors = list()
    totalStandardDeviation = list()
    rowCount = 0

    # PUTTING THE DATA ON THE AXES
    with open("dfterrorvary" + str(index) + "%s.txt" % parset[index]) as reader:
        for line in reader:
            tab = line.split()

            if index == 4:
                labels.append(f'{float(tab[0])}')
            else:
                labels.append(f'{int(tab[0])}')

            perErrors.append((Decimal(tab[1])))
            recErrors.append((Decimal(tab[2])))
            totalErrors.append((Decimal(tab[3])))
            totalStandardDeviation.append((Decimal(tab[4])))
            rowCount += 1

            if rowCount >= limit:
                break

    # THE BARS PLOTTED AND Y-AXIS ARE THE SAME FOR EACH PARAMETER
    plt.bar(labels, recErrors, width, label = 'Reconstruction error', alpha = 0.6, color = 'r', edgecolor = 'k')
    plt.bar(labels, perErrors, width, bottom = recErrors, label = 'Perturbation error', alpha = 0.6, color = 'c', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')
    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.ylabel('Total experimental MSE')

# THE SKELETON SAVING FUNCTION IN THE FOURIER CASE
def saveDft(index):
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    plt.tight_layout()
    plt.draw()
    plt.savefig("dfterrorchartvary" + str(index) + "%s.png" % parset[index])
    plt.clf()
    plt.cla()

def fitCurveDft(index):
    labels = list()
    perErrors = list()
    perStandardDeviation = list()
    gammas = list()
    seeds = list()
    rowCount = 0

    # PUTTING THE DATA ON THE AXES
    with open("dfterrorvary" + str(index) + "%s.txt" % parset[index]) as reader:
        for line in reader:
            tab = line.split()

            if index == 4:
                labels.append(f'{float(tab[0])}')
                seeds.append(float(tab[0]))
            else:
                labels.append(f'{int(tab[0])}')
                seeds.append(int(tab[0]))
            
            perErrors.append((Decimal(tab[1])))
            perStandardDeviation.append((Decimal(tab[5])))
            gammas.append(float(tab[6]))
            rowCount += 1

            if rowCount >= limit:
                break

    # THE BARS PLOTTED ARE THE SAME FOR EACH PARAMETER EXCEPT M
    plt.bar(labels, perErrors, width, alpha = 0.6, color = 'c', edgecolor = 'k')
    plt.errorbar(labels, perErrors, perStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

    # PLOTTING COMPARISON LINE GRAPHS TO VERIFY DEPENDENCIES WITH D, EPSILON AND N
    plotTuple = tuple(zip(seeds, gammas))
    plotTupleEps1 = tuple(zip(seeds[:6], gammas[:6]))
    plotTupleEps2 = tuple(zip(seeds[5:], gammas[5:]))
    x = np.array(labels)
    xEps1 = x[:6]
    xEps2 = x[5:]

    if index == 2:
        p1 = [(0.00000015*((s**(29/12))/((1-g))**2))+0.0005 for s, g in plotTuple]
        y1 = np.array(p1)
        plt.plot(x, y1, label = 'Best fit curve', alpha = 0.6, color = 'k')
    elif index == 4:
        p2 = [(0.002*((1/(s**(7/6)))/((1-g))**2))+0.005 for s, g in plotTupleEps1]
        y2 = np.array(p2)
        plt.plot(xEps1, y2, label = 'Best fit curve part 1', alpha = 0.6, color = 'b')
        p3 = [(0.004*((1/(s**(2/3)))/((1-g))**2))+0.0023 for s, g in plotTupleEps2]
        y3 = np.array(p3)
        plt.plot(xEps2, y3, label = 'Best fit curve part 2', alpha = 0.6, color = 'm')
    elif index == 5:
        p4 = [(0.06*((1/(s**(5/3)))/((1-g))**2))+0.005 for s, g in plotTuple]
        y4 = np.array(p4)
        plt.plot(x, y4, label = 'Best fit curve', alpha = 0.6, color = 'k')

    # THE Y-AXIS IS THE SAME FOR EACH PARAMETER
    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.ylabel('Perturbation error')

def saveCurveDft(index):
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.savefig("dftcurvechartvary" + str(index) + "%s.png" % parset[index])
    plt.clf()
    plt.cla()

# MAIN PLOTTING FUNCTION IN THE FOURIER CASE: COMBINING THE ABOVE
def plotDft():

    # LEAVING OUT THE PARAMETER D AS IT IS NOT USED HERE
    for index in range(6):
        if index == 3:
            continue

        drawDft(index)
        custom(index, 1)
        saveDft(index)

        if index == 2 or index >= 4:
            fitCurveDft(index)
            custom(index, 1)
            saveCurveDft(index)

# CALLING ALL THE ABOVE FUNCTIONS: SOME ARE NESTED
plotBasic()
plotDft()