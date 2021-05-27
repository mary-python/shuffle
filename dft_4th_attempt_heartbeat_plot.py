# IMPORTING RELEVANT PACKAGES
import math, re
import matplotlib.pyplot as plt
from decimal import *
from matplotlib.ticker import FixedFormatter, FixedLocator
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
        plt.xticks(['50', '55', '60', '65', '70', '75', '80', '85', '90', '95'])
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
        plt.xticks(['0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0', '2.25', '2.5', '2.75'])
        plt.xlabel('Value of epsilon', labelpad = 8)

        if dft == 0:
            plt.title('Experimental error by value of epsilon')
        else:
            plt.title('Ratio between experimental errors by value of epsilon')

    # VARYING THE NUMBER OF VECTORS N USED
    else:
        plt.xticks(['3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
        plt.xlabel('Number of vectors used' + ' ' + 'x' + ' ' + '$10^{4}$', labelpad = 8)

        if dft == 0:
            plt.title('Experimental error by number of vectors used')
        else:
            plt.title('Ratio between experimental errors by number of vectors used')

# THE SKELETON DRAWING FUNCTION IN THE BASIC CASE
def drawBasic(index):
    labels = list()
    totalErrors = list()
    totalStandardDeviation = list()
    rowCount = 0

    # PUTTING THE DATA ON THE AXES
    with open("errorvary" + str(index) + "%s.txt" % parset[index]) as reader:
        for line in reader:
            tab = line.split()
            
            if index == 4:
                labels.append(f'{float(tab[0])}')
            else:
                labels.append(f'{int(tab[0])}')

            totalErrors.append((Decimal(tab[1])))
            totalStandardDeviation.append((Decimal(tab[2])))

            rowCount += 1
            if rowCount >= limit:
                break

    # THE BARS PLOTTED AND THE Y-AXIS ARE THE SAME FOR EACH PARAMETER
    plt.bar(labels, totalErrors, width, alpha = 0.6, color = 'm', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')
    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.ylabel('Total experimental MSE')

    # CREATING A LOGARITHMIC Y-AXIS FOR THE EPSILON DEPENDENCY
    if index == 4:
        plt.yscale('log')
        plt.ylim(0.02, 1.2)
        selectiveFormatter = FixedFormatter(["0.01", "0.1", "1"])
        selectiveLocator = FixedLocator([0.01, 0.1, 1])
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

    # THE BARS PLOTTED AND THE Y-AXIS ARE THE SAME FOR EACH PARAMETER
    plt.bar(labels, recErrors, width, label = 'Reconstruction error', alpha = 0.6, color = 'r', edgecolor = 'k')
    plt.bar(labels, perErrors, width, bottom = recErrors, label = 'Perturbation error', alpha = 0.6, color = 'c', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')
    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.ylabel('Total experimental MSE')

# THE SKELETON SAVING FUNCTION IN THE FOURIER CASE
def saveDft(index):
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.savefig("dfterrorchartvary" + str(index) + "%s.png" % parset[index])
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

# CALLING ALL THE ABOVE FUNCTIONS: SOME ARE NESTED
plotBasic()
plotDft()