# IMPORTING RELEVANT PACKAGES
import math, re
import matplotlib.pyplot as plt
from decimal import *
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np

# INITIALISING PARAMETERS/CONSTANTS
width = 0.35
parset = ['t', 'k', 'm', 'd', 'eps', 'eps', 'n']
limit = 10

# SETTING FONTSIZES FOR GRAPHS
plt.rc('font', size = 16)
plt.rc('axes', titlesize = 16, labelsize = 16)
plt.rc('xtick', labelsize = 16)
plt.rc('ytick', labelsize = 16)
plt.rc('legend', fontsize = 14)
plt.rc('figure', titlesize = 16)

# THE X-AXIS, TICKET AND TITLE ARE INDIVIDUALLY TAILORED FOR EACH PARAMETER AND WHETHER DISCRETE FOURIER TRANSFORM IS USED
def custom(index, dft):

    # VARYING THE NUMBER OF COORDINATES T RETAINED
    if index == 0:
        plt.xticks(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']) 
        plt.xlabel('Number of coordinates ' + '$\mathit{t}$ ' + 'retained', labelpad = 8)

    # VARYING THE NUMBER OF BUCKETS K USED
    elif index == 1:
        plt.xticks(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        plt.xlabel('Number of buckets ' + '$\mathit{k}$ ' + 'used', labelpad = 8)

    # VARYING THE NUMBER OF FOURIER COEFFICIENTS M
    elif index == 2:
        plt.xticks(['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
        plt.xlabel('% of Fourier coefficients ' + '$\mathit{m}$ ' + 'retained', labelpad = 8)

    # VARYING THE VECTOR DIMENSION D
    elif index == 3:
        plt.xticks(['60', '70', '80', '90', '100', '110', '120', '130', '140', '150'])
        plt.xlabel('Vector dimension ' + '$\mathit{d}$', labelpad = 8)
    
    # VARYING THE VALUE OF EPSILON: LESS THAN OR EQUAL TO 1
    elif index == 4:
        plt.xticks(['0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95'])
        plt.xlabel('Value of ' + '$\mathit{\u03b5}$', labelpad = 8)

    # VARYING THE VALUE OF EPSILON: GREATER THAN 1
    elif index == 5:
        plt.xticks(['1.0','1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5'])
        plt.xlabel('Value of ' + '$\mathit{\u03b5}$', labelpad = 8)

    # VARYING THE NUMBER OF VECTORS N USED
    else:
        plt.xticks(['10', '11', '14', '17', '20', '30', '40', '50', '60', '70'])
        plt.xlabel('Number of vectors ' + '$\mathit{n}$ ' + 'used ' + 'x ' + '$10^{3}$', labelpad = 8)

# THE SKELETON DRAWING FUNCTION IN THE BASIC CASE
def drawBasic(index):
    labels = list()
    totalErrors = list()
    totalStandardDeviation = list()
    gammas = list()
    seeds = list()
    rowCount = 0

    # PUTTING THE DATA ON THE AXES
    with open("errordatabasic" + str(index) + "%s.txt" % parset[index]) as reader:
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
    plt.bar(labels, totalErrors, width, label = 'Total experimental error',  alpha = 0.6, color = 'm', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, label = 'Total standard deviation',  linestyle = 'None', capsize = 2, color = 'g')

    # PLOTTING COMPARISON LINE GRAPHS TO VERIFY DEPENDENCIES WITH D, EPSILON AND N
    plotTuple = tuple(zip(seeds, gammas))
    x = np.array(labels)

    if index >= 3:

        if index == 3:
            p = [0.00000018*((s**(8/3))/((1-g))**2)+0.002 for s, g in plotTuple]
        elif index == 4:
            p = [0.025*((1/(s**(4/3)))/((1-g))**2)+0.016 for s, g in plotTuple]
        elif index == 5:
            p = [0.08*((1/(s**(4/3)))/((1-g))**2)+0.017 for s, g in plotTuple]
        else:
            p = [1.1*(((1/(s**(7/6)))/((1-g))**2))+0.024 for s, g in plotTuple]

        y = np.array(p)
        plt.plot(x, y, label = 'Best fit curve', alpha = 0.6, color = 'k')
    
    # THE Y-AXIS IS THE SAME FOR EACH PARAMETER
    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.ylabel('Total experimental $\widehat{MSE}$')

    # CREATING A LOGARITHMIC Y-AXIS FOR THE EPSILON AND N DEPENDENCIES
    if index == 4:
        plt.yscale('log')
        plt.ylim(0.04, 0.3)
        selectiveFormatter = FixedFormatter(["0.04", "0.06", "0.1", "0.2", "0.3"])
        selectiveLocator = FixedLocator([0.04, 0.06, 0.1, 0.2, 0.3])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

    elif index == 5:
        plt.yscale('log')
        plt.ylim(0.015, 0.9)
        selectiveFormatter = FixedFormatter(["0.015", "0.1", "0.9"])
        selectiveLocator = FixedLocator([0.015, 0.1, 0.9])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

    elif index == 6:
        plt.yscale('log')
        plt.ylim(0.03, 5)
        selectiveFormatter = FixedFormatter(["0.03", "0.1", "0.3", "1", "5"])
        selectiveLocator = FixedLocator([0.03, 0.1, 0.3, 1, 5])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

# THE SKELETON SAVING FUNCTION IN THE BASIC CASE
def saveBasic(index):
    plt.legend()

    if index >= 3:
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1, 0, 2]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    plt.tight_layout()
    plt.draw()
    plt.savefig("errorchartbasic" + str(index) + "%s.png" % parset[index])
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

# FUNCTION TO READ EACH DATAFILE: DEFINED OUTSIDE MAIN DRAWING FUNCTION AS REFERENCED MULTIPLE TIMES
def readDft(reader, index, labels, seeds, perErrors, recErrors, totalErrors, totalStandardDeviation, gammas, rowCount):

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

# THE SKELETON DRAWING FUNCTION IN THE FOURIER CASE
def drawDft(baseline, heartOrSynth, index):
    labels = list()
    perErrors = list()
    recErrors = list()
    totalErrors = list()
    totalStandardDeviation = list()
    gammas = list()
    seeds = list()
    rowCount = 0

    # PUTTING THE DATA ON THE AXES
    if index == 2:
        if heartOrSynth == 0:
            if baseline == 0:
                with open("errordatafourier" + str(index) + "%sheart.txt" % parset[index]) as reader:
                    readDft(reader, index, labels, seeds, perErrors, recErrors, totalErrors, totalStandardDeviation, gammas, rowCount)
            else:
                with open("errordatanofourier" + str(index) + "%sheart.txt" % parset[index]) as reader:
                    readDft(reader, index, labels, seeds, perErrors, recErrors, totalErrors, totalStandardDeviation, gammas, rowCount)
        else:
            if baseline == 0:
                with open("errordatafourier" + str(index) + "%ssynth.txt" % parset[index]) as reader:
                    readDft(reader, index, labels, seeds, perErrors, recErrors, totalErrors, totalStandardDeviation, gammas, rowCount)
            else:
                with open("errordatanofourier" + str(index) + "%ssynth.txt" % parset[index]) as reader:
                    readDft(reader, index, labels, seeds, perErrors, recErrors, totalErrors, totalStandardDeviation, gammas, rowCount)
    else:
        if baseline == 0:
            with open("errordatafourier" + str(index) + "%s.txt" % parset[index]) as reader:
                readDft(reader, index, labels, seeds, perErrors, recErrors, totalErrors, totalStandardDeviation, gammas, rowCount)
        else:
            with open("errordatanofourier" + str(index) + "%s.txt" % parset[index]) as reader:
                readDft(reader, index, labels, seeds, perErrors, recErrors, totalErrors, totalStandardDeviation, gammas, rowCount)

    # THE BARS PLOTTED AND Y-AXIS ARE THE SAME FOR EACH PARAMETER
    plt.bar(labels, recErrors, width, label = 'Reconstruction error', alpha = 0.6, color = 'r', edgecolor = 'k')
    plt.bar(labels, perErrors, width, bottom = recErrors, label = 'Perturbation error', alpha = 0.6, color = 'c', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, label = 'Total standard deviation', linestyle = 'None', capsize = 2, color = 'g')
    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.ylabel('Total experimental $\widehat{MSE}$')


# THE SKELETON SAVING FUNCTION IN THE FOURIER CASE
def saveDft(baseline, heartOrSynth, index):
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    plt.tight_layout()
    plt.draw()

    if index == 2:
        if heartOrSynth == 0:
            if baseline == 0:
                plt.savefig("errorchartfourier" + str(index) + "%sheart.png" % parset[index])
            else:
                plt.savefig("errorchartnofourier" + str(index) + "%sheart.png" % parset[index])
        else:
            if baseline == 0:
                plt.savefig("errorchartfourier" + str(index) + "%ssynth.png" % parset[index])
            else:
                plt.savefig("errorchartnofourier" + str(index) + "%ssynth.png" % parset[index])
    else:
        if baseline == 0:
            plt.savefig("errorchartfourier" + str(index) + "%s.png" % parset[index])
        else:
            plt.savefig("errorchartnofourier" + str(index) + "%s.png" % parset[index])

    plt.clf()
    plt.cla()

# FUNCTION TO READ EACH DATAFILE: DEFINED OUTSIDE MAIN DRAWING FUNCTION AS REFERENCED MULTIPLE TIMES
def readPerDft(reader, index, labels, seeds, perErrors, perStandardDeviation, gammas, rowCount):
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

# A SKELETON FUNCTION ISOLATING THE PERTURBATION ERROR
def fitPerDft(baseline, index):
    labels = list()
    perErrors = list()
    perStandardDeviation = list()
    gammas = list()
    seeds = list()
    rowCount = 0

    # PUTTING THE DATA ON THE AXES
    if index == 2:
        if baseline == 0:
            with open("errordatafourier" + str(index) + "%sheart.txt" % parset[index]) as reader:
                readPerDft(reader, index, labels, seeds, perErrors, perStandardDeviation, gammas, rowCount)
        else:
            with open("errordatanofourier" + str(index) + "%sheart.txt" % parset[index]) as reader:
                readPerDft(reader, index, labels, seeds, perErrors, perStandardDeviation, gammas, rowCount)
    else:
        if baseline == 0:
            with open("errordatafourier" + str(index) + "%s.txt" % parset[index]) as reader:
                readPerDft(reader, index, labels, seeds, perErrors, perStandardDeviation, gammas, rowCount)
        else:
            with open("errordatanofourier" + str(index) + "%s.txt" % parset[index]) as reader:
                readPerDft(reader, index, labels, seeds, perErrors, perStandardDeviation, gammas, rowCount)

    # NEED TO ISOLATE PERTURBATION ERRORS TO VERIFY DEPENDENCIES
    plt.bar(labels, perErrors, width, label = 'Perturbation error',  alpha = 0.6, color = 'c', edgecolor = 'k')
    plt.errorbar(labels, perErrors, perStandardDeviation, label = 'Standard deviation', linestyle = 'None', capsize = 2, color = 'g')

    # PLOTTING COMPARISON LINE GRAPHS TO VERIFY DEPENDENCIES WITH D, EPSILON AND N
    plotTuple = tuple(zip(seeds, gammas))
    x = np.array(labels)

    if baseline == 0:
        if index == 2:
            p = [(0.000000055*((s**(8/3))/((1-g))**2))+0.0005 for s, g in plotTuple]
        elif index == 4:
            p = [(0.0027*((1/(s**(4/3)))/((1-g))**2))+0.0055 for s, g in plotTuple]
        elif index == 5:
            p = [(0.01*((1/(s**(4/3)))/((1-g))**2))+0.0055 for s, g in plotTuple]
        else:
            p = [(0.32*((1/(s**(5/3)))/((1-g))**2))+0.0065 for s, g in plotTuple]
    else:
        if index == 2:
            p = [(0.0000002*((s**(8/3))/((1-g))**2)) for s, g in plotTuple]
        elif index == 4:
            p = [(0.02*((1/(s**(4/3)))/((1-g))**2))+0.017 for s, g in plotTuple]
        elif index == 5:
            p = [(0.075*((1/(s**(4/3)))/((1-g))**2))+0.015 for s, g in plotTuple]
        else:
            p = [(3.4*((1/(s**(5/3)))/((1-g))**2))+0.028 for s, g in plotTuple]
    
    y = np.array(p)
    plt.plot(x, y, label = 'Best fit curve', alpha = 0.6, color = 'k')

    # THE Y-AXIS IS THE SAME FOR EACH PARAMETER
    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.ylabel('Perturbation error')

    # CREATING A LOGARITHMIC Y-AXIS FOR THE EPS AND N DEPENDENCIES
    if baseline == 0:
        if index == 4:
            plt.yscale('log')
            plt.ylim(0.002, 0.07)
            selectiveFormatter = FixedFormatter(["0.002", "0.01", "0.03", "0.07"])
            selectiveLocator = FixedLocator([0.002, 0.01, 0.03, 0.07])
            plt.gca().yaxis.set_major_formatter(selectiveFormatter)
            plt.gca().yaxis.set_major_locator(selectiveLocator)

        elif index == 5:
            plt.yscale('log')
            plt.ylim(0.003, 0.09)
            selectiveFormatter = FixedFormatter(["0.003", "0.01", "0.03", "0.09"])
            selectiveLocator = FixedLocator([0.003, 0.01, 0.03, 0.09])
            plt.gca().yaxis.set_major_formatter(selectiveFormatter)
            plt.gca().yaxis.set_major_locator(selectiveLocator)

        elif index == 6:
            plt.yscale('log')
            plt.ylim(0.002, 0.13)
            selectiveFormatter = FixedFormatter(["0.001", "0.01", "0.1"])
            selectiveLocator = FixedLocator([0.002, 0.01, 0.1])
            plt.gca().yaxis.set_major_formatter(selectiveFormatter)
            plt.gca().yaxis.set_major_locator(selectiveLocator)

    else:
        if index == 4:
            plt.yscale('log')
            plt.ylim(0.035, 0.25)
            selectiveFormatter = FixedFormatter(["0.04", "0.06", "0.1", "0.2", "0.25"])
            selectiveLocator = FixedLocator([0.04, 0.06, 0.1, 0.2, 0.25])
            plt.gca().yaxis.set_major_formatter(selectiveFormatter)
            plt.gca().yaxis.set_major_locator(selectiveLocator)

        elif index == 5:
            plt.yscale('log')
            plt.ylim(0.015, 0.8)
            selectiveFormatter = FixedFormatter(["0.02", "0.1", "0.8"])
            selectiveLocator = FixedLocator([0.02, 0.1, 0.8])
            plt.gca().yaxis.set_major_formatter(selectiveFormatter)
            plt.gca().yaxis.set_major_locator(selectiveLocator)

        elif index == 6:
            plt.yscale('log')
            plt.ylim(0.02, 1)
            selectiveFormatter = FixedFormatter(["0.02", "0.1", "0.3", "1"])
            selectiveLocator = FixedLocator([0.02, 0.1, 0.3, 1])
            plt.gca().yaxis.set_major_formatter(selectiveFormatter)
            plt.gca().yaxis.set_major_locator(selectiveLocator)

def savePerDft(baseline, index):
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    plt.tight_layout()
    plt.draw()
    
    if baseline == 0:
        plt.savefig("errorchartfourierperturb" + str(index) + "%s.png" % parset[index])
    else:
        plt.savefig("errorchartnofourierperturb" + str(index) + "%s.png" % parset[index])

    plt.clf()
    plt.cla()

# MAIN PLOTTING FUNCTION IN THE FOURIER CASE: COMBINING THE ABOVE
def plotDft():

    # PLOTTING THE GRAPHS WITH DFT THEN THE BASELINE GRAPHS
    for base in range(2):

        # LOOPING THROUGH THE PARAMETER INDICES
        for index in range(7):

            # LEAVING OUT THE PARAMETER D AS IT IS NOT USED HERE
            if index == 3:
                continue
        
            if index <= 2:
                drawDft(base, 0, index)
                custom(index, 1)
                saveDft(base, 0, index)

            if index == 2:
                drawDft(base, 1, index)
                custom(index, 1)
                saveDft(base, 1, index)

            if index == 2 or index >= 4:
                fitPerDft(base, index)
                custom(index, 2)
                savePerDft(base, index)

# CALLING ALL THE ABOVE FUNCTIONS: SOME ARE NESTED
plotBasic()
plotDft()