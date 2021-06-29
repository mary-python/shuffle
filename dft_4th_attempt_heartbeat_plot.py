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
parset = ['t', 'k', 'm', 'd', 'eps', 'eps', 'n']
limit = 10

# THE X-AXIS, TICKET AND TITLE ARE INDIVIDUALLY TAILORED FOR EACH PARAMETER AND WHETHER DISCRETE FOURIER TRANSFORM IS USED
def custom(index, dft):

    # VARYING THE NUMBER OF COORDINATES T RETAINED
    if index == 0:
        plt.xticks(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']) 
        plt.xlabel('Number of coordinates retained (t)', labelpad = 8)

        # A SINGLE EXPERIMENTAL ERROR IS PLOTTED IN THE BASIC CASE
        if dft == 0:
            plt.title('Experimental error by number of coordinates retained (t)')

        # RATIO BETWEEN EXPERIMENTAL ERRORS IS PLOTTED IN THE FOURIER CASE
        else:
            plt.title('Ratio between errors by number of coordinates retained (t)')

    # VARYING THE NUMBER OF BUCKETS K USED
    elif index == 1:
        plt.xticks(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        plt.xlabel('Number of buckets used (k)', labelpad = 8)

        if dft == 0:
            plt.title('Experimental error by number of buckets used (k)')
        else:
            plt.title('Ratio between errors by number of buckets used (k)')

    # VARYING THE NUMBER OF FOURIER COEFFICIENTS M
    elif index == 2:
        plt.xticks(['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
        plt.xlabel('% of Fourier coefficients retained (m)', labelpad = 8)

        if dft == 0:
            plt.title('Experimental error by % of Fourier coefficients retained (m)')
        elif dft == 1:
            plt.title('Ratio between errors by % of Fourier coefficients retained (m)')

        # CHANGE THE TITLE WHEN PERTURBATION ERROR IS ISOLATED
        else:
            plt.title('Perturbation error by % of Fourier coefficients retained (m)')

    # VARYING THE VECTOR DIMENSION D
    elif index == 3:
        plt.xticks(['60', '70', '80', '90', '100', '110', '120', '130', '140', '150'])
        plt.xlabel('Vector dimension (d)', labelpad = 8)

        if dft == 0:
            plt.title('Experimental error by vector dimension (d)')
        else:
            plt.title('Ratio between errors by vector dimension (d)')
    
    # VARYING THE VALUE OF EPSILON
    elif index == 4:
        plt.xticks(['0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95'])
        plt.xlabel('Value of epsilon', labelpad = 8)

        if dft == 0:
            plt.title('Experimental error by value of epsilon')
        elif dft == 1:
            plt.title('Ratio between errors by value of epsilon')
        else:
            plt.title('Perturbation error by value of epsilon')

    elif index == 5:
        plt.xticks(['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5'])
        plt.xlabel('Value of epsilon', labelpad = 8)

        if dft == 0:
            plt.title('Experimental error by value of epsilon')
        elif dft == 1:
            plt.title('Ratio between errors by value of epsilon')
        else:
            plt.title('Perturbation error by value of epsilon')

    # VARYING THE NUMBER OF VECTORS N USED
    else:
        plt.xticks(['6', '7', '9', '12', '15', '20', '30', '40', '50', '60'])
        plt.xlabel('Number of vectors used (n)' + ' ' + 'x' + ' ' + '$10^{3}$', labelpad = 8)

        if dft == 0:
            plt.title('Experimental error by number of vectors used (n)')
        elif dft == 1:
            plt.title('Ratio between errors by number of vectors used (n)')
        else:
            plt.title('Perturbation error by number of vectors used (n)')

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
            
            if index == 4 or index == 5:
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
    x = np.array(labels)

    if index >= 3:

        if index == 3:
            p = [0.0000005*((s**(29/12))/((1-g))**2) for s, g in plotTuple]
        elif index == 4:
            p = [0.04*((1/(s**(7/6)))/((1-g))**2) for s, g in plotTuple]
        elif index == 5:
            p = [0.04*((1/(s**(1/3)))/((1-g))**2) for s, g in plotTuple]
        else:
            p = [(0.8*((1/(s**(7/6)))/((1-g))**2))+0.025 for s, g in plotTuple]

        y = np.array(p)
        plt.plot(x, y, alpha = 0.6, color = 'k')
    
    # THE Y-AXIS IS THE SAME FOR EACH PARAMETER
    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.ylabel('Total experimental MSE')

    # CREATING A LOGARITHMIC Y-AXIS FOR THE EPSILON AND N DEPENDENCIES
    if index == 4:
        plt.yscale('log')
        plt.ylim(0.05, 1.2)
        selectiveFormatter = FixedFormatter(["0.1", "1"])
        selectiveLocator = FixedLocator([0.1, 1])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)
    elif index == 5:
        plt.yscale('log')
        plt.ylim(0.02, 0.08)
        selectiveFormatter = FixedFormatter(["0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08"])
        selectiveLocator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)
    elif index == 6:
        plt.yscale('log')
        plt.ylim(0.02, 15)
        selectiveFormatter = FixedFormatter(["0.1", "1", "10"])
        selectiveLocator = FixedLocator([0.1, 1, 10])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

# THE SKELETON SAVING FUNCTION IN THE BASIC CASE
def saveBasic(index):
    plt.tight_layout()
    plt.draw()
    plt.savefig("errorchartvary" + str(index) + "%s.png" % parset[index])
    plt.clf()
    plt.cla()

# MAIN PLOTTING FUNCTION IN THE BASIC CASE: COMBINING THE ABOVE
def plotBasic():

    # LEAVING OUT THE PARAMETER M AS IT IS NOT USED HERE
    for index in range(7):
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
    gammas = list()
    seeds = list()
    rowCount = 0

    # PUTTING THE DATA ON THE AXES
    with open("dfterrorvary" + str(index) + "%s.txt" % parset[index]) as reader:
        for line in reader:
            tab = line.split()

            if index == 4 or index == 5:
                labels.append(f'{float(tab[0])}')
                seeds.append(float(tab[0]))
            else:
                labels.append(f'{int(tab[0])}')
                seeds.append(int(tab[0]))

            perErrors.append((Decimal(tab[1])))
            recErrors.append((Decimal(tab[2])))
            totalErrors.append((Decimal(tab[3])))
            totalStandardDeviation.append((Decimal(tab[4])))
            gammas.append(float(tab[6]))
            rowCount += 1

            if rowCount >= limit:
                break

    # THE BARS PLOTTED AND Y-AXIS ARE THE SAME FOR EACH PARAMETER
    plt.bar(labels, recErrors, width, label = 'Reconstruction error', alpha = 0.6, color = 'r', edgecolor = 'k')
    plt.bar(labels, perErrors, width, bottom = recErrors, label = 'Perturbation error', alpha = 0.6, color = 'c', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

    # PLOTTING COMPARISON LINE GRAPHS TO VERIFY DEPENDENCIES WITH D, EPSILON AND N
    plotTuple = tuple(zip(seeds, gammas))
    x = np.array(labels)

    if index == 2 or index >= 4:

        if index == 2:
            p = [(4*((1/(s**(19/24)))/((1-g))**2))-0.125 for s, g in plotTuple]
        elif index == 4:
            p = [(0.0035*((1/(s**(7/6)))/((1-g))**2))+0.016 for s, g in plotTuple]
        elif index == 5:
            p = [(0.0037*((1/(s**(2/3)))/((1-g))**2))+0.017 for s, g in plotTuple]
        else:
            p = [(0.05*((1/(s**(3/2)))/((1-g))**2))+0.017 for s, g in plotTuple]

        y = np.array(p)
        plt.plot(x, y, alpha = 0.6, color = 'k')

    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.ylabel('Total experimental MSE')

    # CREATING A LOGARITHMIC Y-AXIS FOR THE EPSILON AND N DEPENDENCIES
    if index == 4:
        plt.yscale('log')
        plt.ylim(0.01, 0.3)
        selectiveFormatter = FixedFormatter(["0.01", "0.1"])
        selectiveLocator = FixedLocator([0.01, 0.1])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)
    elif index == 5:
        plt.yscale('log')
        plt.ylim(0.01, 0.04)
        selectiveFormatter = FixedFormatter(["0.01", "0.015", "0.02", "0.025", "0.03"])
        selectiveLocator = FixedLocator([0.01, 0.015, 0.02, 0.025, 0.03])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)
    elif index == 6:
        plt.yscale('log')
        plt.ylim(0.008, 1)
        selectiveFormatter = FixedFormatter(["0.01", "0.1", "1"])
        selectiveLocator = FixedLocator([0.01, 0.1, 1])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

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

# A SKELETON FUNCTION ISOLATING THE PERTURBATION ERROR
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

            if index == 4 or index == 5:
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

    # NEED TO ISOLATE PERTURBATION ERRORS TO VERIFY DEPENDENCIES
    plt.bar(labels, perErrors, width, alpha = 0.6, color = 'c', edgecolor = 'k')
    plt.errorbar(labels, perErrors, perStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

    # PLOTTING COMPARISON LINE GRAPHS TO VERIFY DEPENDENCIES WITH D, EPSILON AND N
    plotTuple = tuple(zip(seeds, gammas))
    x = np.array(labels)

    if index == 2:
        p = [(0.00000015*((s**(29/12))/((1-g))**2))+0.0005 for s, g in plotTuple]
    elif index == 4:
        p = [(0.0035*((1/(s**(7/6)))/((1-g))**2))+0.005 for s, g in plotTuple]
    elif index == 5:
        p = [(0.0037*((1/(s**(2/3)))/((1-g))**2))+0.006 for s, g in plotTuple]
    else:
        p = [(0.05*((1/(s**(3/2)))/((1-g))**2))+0.006 for s, g in plotTuple]
    
    y = np.array(p)
    plt.plot(x, y, alpha = 0.6, color = 'k')

    # THE Y-AXIS IS THE SAME FOR EACH PARAMETER
    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.ylabel('Perturbation error')

    # CREATING A LOGARITHMIC Y-AXIS FOR THE EPS AND N DEPENDENCIES
    if index == 4:
        plt.yscale('log')
        plt.ylim(0.005, 0.25)
        selectiveFormatter = FixedFormatter(["0.01", "0.1"])
        selectiveLocator = FixedLocator([0.01, 0.1])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)
    elif index == 5:
        plt.yscale('log')
        plt.ylim(0.002, 0.025)
        selectiveFormatter = FixedFormatter(["0.002", "0.003", "0.004", "0.006", "0.01", "0.02"])
        selectiveLocator = FixedLocator([0.002, 0.003, 0.004, 0.006, 0.01, 0.02])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)
    elif index == 6:
        plt.yscale('log')
        plt.ylim(0.002, 1)
        selectiveFormatter = FixedFormatter(["0.01", "0.1", "1"])
        selectiveLocator = FixedLocator([0.01, 0.1, 1])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

def saveCurveDft(index):
    plt.tight_layout()
    plt.draw()
    plt.savefig("dftcurvechartvary" + str(index) + "%s.png" % parset[index])
    plt.clf()
    plt.cla()

# MAIN PLOTTING FUNCTION IN THE FOURIER CASE: COMBINING THE ABOVE
def plotDft():

    # LEAVING OUT THE PARAMETER D AS IT IS NOT USED HERE
    for index in range(7):
        if index == 3:
            continue

        drawDft(index)
        custom(index, 1)
        saveDft(index)

        if index == 2 or index >= 4:
            fitCurveDft(index)
            custom(index, 2)
            saveCurveDft(index)

# CALLING ALL THE ABOVE FUNCTIONS: SOME ARE NESTED
plotBasic()
plotDft()