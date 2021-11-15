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
plt.rc('legend', fontsize = 12)
plt.rc('figure', titlesize = 16)

# THE X-AXIS LABEL IS INDIVIDUALLY TAILORED FOR EACH PARAMETER
def custom(index, dft):

    # VARYING THE NUMBER OF COORDINATES T RETAINED
    if index == 0:
        plt.xlabel('Number of coordinates ' + '$\mathit{t}$ ' + 'retained', labelpad = 8)

    # VARYING THE NUMBER OF BUCKETS K USED
    elif index == 1:
        plt.xlabel('Number of buckets ' + '$\mathit{k}$ ' + 'used', labelpad = 8)

    # VARYING THE NUMBER OF FOURIER COEFFICIENTS M
    elif index == 2:
        plt.xlabel('% of Fourier coefficients ' + '$\mathit{m}$ ' + 'retained', labelpad = 8)

    # VARYING THE VECTOR DIMENSION D
    elif index == 3:
        plt.xlabel('Vector dimension ' + '$\mathit{d}$', labelpad = 8)
    
    # VARYING THE VALUE OF EPSILON: LESS THAN OR EQUAL TO 1
    elif index == 4:
        plt.xlabel('Value of ' + '$\mathit{\u03b5}$', labelpad = 8)

    # VARYING THE VALUE OF EPSILON: GREATER THAN 1
    elif index == 5:
        plt.xlabel('Value of ' + '$\mathit{\u03b5}$', labelpad = 8)

    # VARYING THE NUMBER OF VECTORS N USED
    else:
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
    plt.xticks(labels)

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
            labels[rowCount] = f'{(float(tab[0]))}'
            seeds.append(float(tab[0]))
        else:
            labels[rowCount] = f'{(int(tab[0]))}'
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
def drawDft(heartOrSynth, index):
    labels = [0]*limit
    perErrorsA = list()
    perErrorsB = list()
    recErrorsA = list()
    recErrorsB = list()
    totalErrorsA = list()
    totalErrorsB = list()
    totalStandardDeviationA = list()
    totalStandardDeviationB = list()
    gammasA = list()
    gammasB = list()
    seedsA = list()
    seedsB = list()
    rowCount = 0

    # PUTTING THE DATA ON THE AXES
    if index == 2:
        if heartOrSynth == 0:
            with open("errordatafourier" + str(index) + "%sheart.txt" % parset[index]) as reader:
                readDft(reader, index, labels, seedsA, perErrorsA, recErrorsA, totalErrorsA, totalStandardDeviationA, gammasA, rowCount)
            with open("errordatanofourier" + str(index) + "%sheart.txt" % parset[index]) as reader:
                readDft(reader, index, labels, seedsB, perErrorsB, recErrorsB, totalErrorsB, totalStandardDeviationB, gammasB, rowCount)
        else:
            with open("errordatafourier" + str(index) + "%ssynth.txt" % parset[index]) as reader:
                readDft(reader, index, labels, seedsA, perErrorsA, recErrorsA, totalErrorsA, totalStandardDeviationA, gammasA, rowCount)
            with open("errordatanofourier" + str(index) + "%ssynth.txt" % parset[index]) as reader:
                readDft(reader, index, labels, seedsB, perErrorsB, recErrorsB, totalErrorsB, totalStandardDeviationB, gammasB, rowCount)
    else:
        with open("errordatafourier" + str(index) + "%s.txt" % parset[index]) as reader:
            readDft(reader, index, labels, seedsA, perErrorsA, recErrorsA, totalErrorsA, totalStandardDeviationA, gammasA, rowCount)
        with open("errordatanofourier" + str(index) + "%s.txt" % parset[index]) as reader:
            readDft(reader, index, labels, seedsB, perErrorsB, recErrorsB, totalErrorsB, totalStandardDeviationB, gammasB, rowCount)

    # THE BARS PLOTTED AND Y-AXIS ARE THE SAME FOR EACH PARAMETER
    xLabels = np.arange(len(labels))
    plt.bar(xLabels - width/2, recErrorsA, width, label = 'Reconstruction err.', alpha = 0.6, color = 'tab:red', edgecolor = 'k') 
    plt.bar(xLabels - width/2, perErrorsA, width, bottom = recErrorsA, label = 'Perturbation err.', alpha = 0.6, color = 'tab:green', edgecolor = 'k')
    plt.errorbar(xLabels - width/2, totalErrorsA, totalStandardDeviationA, label = 'Total s.d.', linestyle = 'None', capsize = 2, color = 'tab:blue')
    
    plt.bar(xLabels + width/2, recErrorsB, width, label = 'RE for baseline', alpha = 0.6, color = 'tab:cyan', edgecolor = 'k')
    plt.bar(xLabels + width/2, perErrorsB, width, bottom = recErrorsB, label = 'PE for baseline', alpha = 0.6, color = 'tab:orange', edgecolor = 'k')
    plt.errorbar(xLabels + width/2, totalErrorsB, totalStandardDeviationB, linestyle = 'None', capsize = 2, color = 'tab:blue')
    
    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.ylabel('Total experimental $\widehat{MSE}$')
    plt.xticks(xLabels, labels)

    if index == 0:
        plt.yscale('log')
        plt.ylim(0.1, 50)
        selectiveFormatter = FixedFormatter(["0.1", "1", "10", "50"])
        selectiveLocator = FixedLocator([0.1, 1, 10, 50])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

    elif index == 1:
        plt.yscale('log')
        plt.ylim(0.1, 50)
        selectiveFormatter = FixedFormatter(["0.1", "1", "10", "50"])
        selectiveLocator = FixedLocator([0.1, 1, 10, 50])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

    elif index == 2:
        if heartOrSynth == 0:
            plt.yscale('log')
            plt.ylim(0.004, 100)
            selectiveFormatter = FixedFormatter(["0.004", "0.01", "0.1", "1", "10", "100"])
            selectiveLocator = FixedLocator([0.004, 0.01, 0.1, 1, 10, 100])
            plt.gca().yaxis.set_major_formatter(selectiveFormatter)
            plt.gca().yaxis.set_major_locator(selectiveLocator)
        else:
            plt.yscale('log')
            plt.ylim(0.0002, 10)
            selectiveFormatter = FixedFormatter(["0.0002", "0.001", "0.01", "0.1", "1", "10"])
            selectiveLocator = FixedLocator([0.0002, 0.001, 0.01, 0.1, 1, 10])
            plt.gca().yaxis.set_major_formatter(selectiveFormatter)
            plt.gca().yaxis.set_major_locator(selectiveLocator)

# THE SKELETON SAVING FUNCTION IN THE FOURIER CASE
def saveDft(heartOrSynth, index):
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0, 4, 3, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol = 2)
    plt.tight_layout()
    plt.draw()

    if index == 2:
        if heartOrSynth == 0:
            plt.savefig("errorchartfourier" + str(index) + "%sheart.png" % parset[index])
        else:
            plt.savefig("errorchartfourier" + str(index) + "%ssynth.png" % parset[index])
    else:
        plt.savefig("errorchartfourier" + str(index) + "%s.png" % parset[index])

    plt.clf()
    plt.cla()

# FUNCTION TO READ EACH DATAFILE: DEFINED OUTSIDE MAIN DRAWING FUNCTION AS REFERENCED MULTIPLE TIMES
def readPerDft(reader, index, labels, seeds, perErrors, perStandardDeviation, gammas, rowCount):
    for line in reader:
        tab = line.split()

        if index == 4 or index == 5:
            labels[rowCount] = f'{(float(tab[0]))}'
            seeds.append(float(tab[0]))
        else:
            labels[rowCount] = f'{(int(tab[0]))}'
            seeds.append(int(tab[0]))
            
        perErrors.append((Decimal(tab[1])))
        perStandardDeviation.append((Decimal(tab[5])))
        gammas.append(float(tab[6]))
        rowCount += 1

        if rowCount >= limit:
            break

# A SKELETON FUNCTION ISOLATING THE PERTURBATION ERROR
def fitPerDft(index):
    labels = [0]*limit
    perErrorsA = list()
    perErrorsB = list()
    perStandardDeviationA = list()
    perStandardDeviationB = list()
    gammasA = list()
    gammasB = list()
    seedsA = list()
    seedsB = list()
    rowCount = 0

    # PUTTING THE DATA ON THE AXES
    if index == 2:
        with open("errordatafourier" + str(index) + "%sheart.txt" % parset[index]) as reader:
            readPerDft(reader, index, labels, seedsA, perErrorsA, perStandardDeviationA, gammasA, rowCount)
        with open("errordatanofourier" + str(index) + "%sheart.txt" % parset[index]) as reader:
            readPerDft(reader, index, labels, seedsB, perErrorsB, perStandardDeviationB, gammasB, rowCount)
    else:
        with open("errordatafourier" + str(index) + "%s.txt" % parset[index]) as reader:
            readPerDft(reader, index, labels, seedsA, perErrorsA, perStandardDeviationA, gammasA, rowCount)
        with open("errordatanofourier" + str(index) + "%s.txt" % parset[index]) as reader:
            readPerDft(reader, index, labels, seedsB, perErrorsB, perStandardDeviationB, gammasB, rowCount)

    # NEED TO ISOLATE PERTURBATION ERRORS TO VERIFY DEPENDENCIES
    xLabels = np.arange(len(labels))
    plt.bar(xLabels - width/2, perErrorsA, width, label = 'Perturbation error',  alpha = 0.6, color = 'tab:red', edgecolor = 'k')
    plt.errorbar(xLabels - width/2, perErrorsA, perStandardDeviationA, label = 'Standard deviation', linestyle = 'None', capsize = 2, color = 'tab:blue')
    plt.bar(xLabels + width/2, perErrorsB, width, label = 'PE for baseline',  alpha = 0.6, color = 'tab:orange', edgecolor = 'k')
    plt.errorbar(xLabels + width/2, perErrorsB, perStandardDeviationB, linestyle = 'None', capsize = 2, color = 'tab:blue')

    # PLOTTING COMPARISON LINE GRAPHS TO VERIFY DEPENDENCIES WITH D, EPSILON AND N
    plotTupleA = tuple(zip(seedsA, gammasA))
    plotTupleB = tuple(zip(seedsB, gammasB))
    x = np.arange(len(np.array(labels)))

    if index == 2:
        pA = [(0.000000055*((s**(8/3))/((1-g))**2))+0.0005 for s, g in plotTupleA]
        pB = [(0.0000002*((s**(8/3))/((1-g))**2)) for s, g in plotTupleB]
    elif index == 4:
        pA = [(0.00006*((1/(s**(4/3)))/((1-g))**2))+0.0001 for s, g in plotTupleA]
        pB = [(0.00001*((1/(s**(4/3)))/((1-g))**2))+0.00003 for s, g in plotTupleB]
    elif index == 5:
        pA = [(0.0002*((1/(s**(4/3)))/((1-g))**2))+0.00014 for s, g in plotTupleA]
        pB = [(0.000034*((1/(s**(4/3)))/((1-g))**2))+0.00004 for s, g in plotTupleB]
    else:
        pA = [(0.0035*((1/(s**(5/3)))/((1-g))**2))+0.00015 for s, g in plotTupleA]
        pB = [(0.0015*((1/(s**(5/3)))/((1-g))**2))+0.000027 for s, g in plotTupleB]

    yA = np.array(pA)
    yB = np.array(pB)
    plt.plot(x - width/2, yA, label = 'Best fit curves', alpha = 0.6, color = 'k')
    plt.plot(x + width/2, yB, alpha = 0.6, color = 'k')

    # THE Y-AXIS IS THE SAME FOR EACH PARAMETER
    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.ylabel('Perturbation error')
    plt.xticks(x, labels)

    if index == 2:
        plt.ylim(0, 0.075)

    # CREATING A LOGARITHMIC Y-AXIS FOR THE EPS AND N DEPENDENCIES
    elif index == 4:
        plt.yscale('log')
        plt.ylim(0.00001, 0.005)
        selectiveFormatter = FixedFormatter(["0.00001", "0.0001", "0.001", "0.005"])
        selectiveLocator = FixedLocator([0.00001, 0.0001, 0.001, 0.005])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

    elif index == 5:
        plt.yscale('log')
        plt.ylim(0.00001, 0.005)
        selectiveFormatter = FixedFormatter(["0.00001", "0.0001", "0.001", "0.005"])
        selectiveLocator = FixedLocator([0.00001, 0.0001, 0.001, 0.005])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

    elif index == 6:
        plt.yscale('log')
        plt.ylim(0.00001, 0.005)
        selectiveFormatter = FixedFormatter(["0.00001", "0.0001", "0.001", "0.005"])
        selectiveLocator = FixedLocator([0.00001, 0.0001, 0.001, 0.005])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

def savePerDft(index):
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 3, 0, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    plt.tight_layout()
    plt.draw()
    plt.savefig("errorchartfourierperturb" + str(index) + "%s.png" % parset[index])
    plt.clf()
    plt.cla()

# MAIN PLOTTING FUNCTION IN THE FOURIER CASE: COMBINING THE ABOVE
def plotDft():

    # LOOPING THROUGH THE PARAMETER INDICES
    for index in range(7):

        # LEAVING OUT THE PARAMETER D AS IT IS NOT USED HERE
        if index == 3:
            continue
        
        if index <= 2:
            drawDft(0, index)
            custom(index, 1)
            saveDft(0, index)

        if index == 2:
            drawDft(1, index)
            custom(index, 1)
            saveDft(1, index)

        if index == 2 or index >= 4:
            fitPerDft(index)
            custom(index, 2)
            savePerDft(index)

# CALLING ALL THE ABOVE FUNCTIONS: SOME ARE NESTED
plotBasic()
plotDft()