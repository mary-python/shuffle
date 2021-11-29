# IMPORTING RELEVANT PACKAGES
import matplotlib.pyplot as plt
from decimal import *
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np

# INITIALISING PARAMETERS/CONSTANTS
width = 0.35
mset2 = [5, 20, 40, 55, 75, 95]
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
def custom(index):

    # VARYING THE NUMBER OF COORDINATES T RETAINED
    if index == 0:
        plt.xlabel('Number of coordinates ' + '$\mathit{t}$ ' + 'retained', labelpad = 8)

    # VARYING THE NUMBER OF BUCKETS K USED
    elif index == 1:
        plt.xlabel('Number of buckets ' + '$\mathit{k}$ ' + 'used', labelpad = 8)

    # VARYING THE NUMBER OF FOURIER COEFFICIENTS M
    elif index == 2:
        plt.xlabel('% of (Fourier) coefficients ' + '$\mathit{m}$ ' + 'retained', labelpad = 8)

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
    
    # THE BARS PLOTTED ARE THE SAME FOR EACH PARAMETER
    plt.bar(labels, totalErrors, width, label = 'Total experimental err.',  alpha = 0.6, color = 'm', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, label = 'Total s.d.',  linestyle = 'None', capsize = 2, color = 'g')

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
    plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.15), ncol = 2)

    if index >= 3:
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1, 0, 2]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc = "upper center", bbox_to_anchor = (0.5, 1.2), ncol = 2, shadow = True)

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
        custom(index)
        saveBasic(index)

# FUNCTION TO READ EACH DATAFILE: DEFINED OUTSIDE MAIN DRAWING FUNCTION AS REFERENCED MULTIPLE TIMES
def readDft(reader, index, labels, perErrors, recErrors, totalErrors, totalStandardDeviation, rowCount):

    for line in reader:
        tab = line.split()

        if index == 4 or index == 5:
            labels[rowCount] = f'{(float(tab[0]))}'
        else:
            labels[rowCount] = f'{(int(tab[0]))}'

        perErrors.append((Decimal(tab[1])))
        recErrors.append((Decimal(tab[2])))
        totalErrors.append((Decimal(tab[3])))
        totalStandardDeviation.append((Decimal(tab[4])))
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
    rowCount = 0

    # PUTTING THE DATA ON THE AXES
    if index == 2:
        if heartOrSynth == 0:
            with open("errordatafourier" + str(index) + "%sheart.txt" % parset[index]) as reader:
                readDft(reader, index, labels, perErrorsA, recErrorsA, totalErrorsA, totalStandardDeviationA, rowCount)
            with open("errordatanofourier" + str(index) + "%sheart.txt" % parset[index]) as reader:
                readDft(reader, index, labels, perErrorsB, recErrorsB, totalErrorsB, totalStandardDeviationB, rowCount)
        else:
            with open("errordatafourier" + str(index) + "%ssynth.txt" % parset[index]) as reader:
                readDft(reader, index, labels, perErrorsA, recErrorsA, totalErrorsA, totalStandardDeviationA, rowCount)
            with open("errordatanofourier" + str(index) + "%ssynth.txt" % parset[index]) as reader:
                readDft(reader, index, labels, perErrorsB, recErrorsB, totalErrorsB, totalStandardDeviationB, rowCount)
    else:
        with open("errordatafourier" + str(index) + "%s.txt" % parset[index]) as reader:
            readDft(reader, index, labels, perErrorsA, recErrorsA, totalErrorsA, totalStandardDeviationA, rowCount)
        with open("errordatanofourier" + str(index) + "%s.txt" % parset[index]) as reader:
            readDft(reader, index, labels, perErrorsB, recErrorsB, totalErrorsB, totalStandardDeviationB, rowCount)

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

    # CREATING A LOGARITHMIC Y-AXIS FOR THE T, K AND M DEPENDENCIES
    if index == 0:
        plt.yscale('log')
        plt.ylim(0.005, 3)
        selectiveFormatter = FixedFormatter(["0.005", "0.01", "0.1", "1"])
        selectiveLocator = FixedLocator([0.005, 0.01, 0.1, 1])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

    elif index == 1:
        plt.yscale('log')
        plt.ylim(0.005, 1)
        selectiveFormatter = FixedFormatter(["0.005", "0.01", "0.1", "1"])
        selectiveLocator = FixedLocator([0.005, 0.01, 0.1, 1])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

    elif index == 2:
        if heartOrSynth == 0:
            plt.yscale('log')
            plt.ylim(0.004, 20)
            selectiveFormatter = FixedFormatter(["0.004", "0.01", "0.1", "1", "10"])
            selectiveLocator = FixedLocator([0.004, 0.01, 0.1, 1, 10])
            plt.gca().yaxis.set_major_formatter(selectiveFormatter)
            plt.gca().yaxis.set_major_locator(selectiveLocator)
        else:
            plt.yscale('log')
            plt.ylim(0.0002, 2)
            selectiveFormatter = FixedFormatter(["0.0002", "0.001", "0.01", "0.1", "1", "10"])
            selectiveLocator = FixedLocator([0.0002, 0.001, 0.01, 0.1, 1, 10])
            plt.gca().yaxis.set_major_formatter(selectiveFormatter)
            plt.gca().yaxis.set_major_locator(selectiveLocator)

# THE SKELETON SAVING FUNCTION IN THE FOURIER CASE
def saveDft(heartOrSynth, index):
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0, 4, 3, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc = "upper center", bbox_to_anchor = (0.5, 1.2), ncol = 2, shadow = True)
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

# FUNCTION TO READ EACH DATAFILE: ISOLATING THE PERTURBATION ERROR
def readPerDft(reader, index, labels, seeds, perErrors, perStandardDeviation, gammas, rowCount):
    for line in reader:
        tab = line.split()

        if index == 4 or index == 5:
            labels[rowCount] = f'{(float(tab[0]))}'
            seeds[rowCount] = float(tab[0])
        else:
            labels[rowCount] = f'{(int(tab[0]))}'
            seeds[rowCount] = int(tab[0])
            
        perErrors.append((Decimal(tab[1])))
        perStandardDeviation.append((Decimal(tab[5])))
        gammas[rowCount] = float(tab[6])
        rowCount += 1

        if rowCount >= limit:
            break

# A SKELETON FUNCTION ISOLATING THE PERTURBATION ERROR
def fitPerDft(heartOrSynth, index, m):
    labels = [0]*limit
    perErrorsA = list()
    perErrorsB = list()
    perStandardDeviationA = list()
    perStandardDeviationB = list()
    gammas = [0]*limit
    seeds = [0]*limit
    rowCount = 0

    # PUTTING THE DATA ON THE AXES: SEPARATED BY INDEX AND WHETHER BASELINE DATA IS USED OR NOT
    if index == 2:
        if heartOrSynth == 0:
            with open("errordatafourier" + str(index) + "%sheart.txt" % parset[index]) as reader:
                readPerDft(reader, index, labels, seeds, perErrorsA, perStandardDeviationA, gammas, rowCount)
            with open("errordatanofourier" + str(index) + "%sheart.txt" % parset[index]) as reader:
                readPerDft(reader, index, labels, seeds, perErrorsB, perStandardDeviationB, gammas, rowCount)
        else:
            with open("errordatafourier" + str(index) + "%ssynth.txt" % parset[index]) as reader:
                readPerDft(reader, index, labels, seeds, perErrorsA, perStandardDeviationA, gammas, rowCount)
            with open("errordatanofourier" + str(index) + "%ssynth.txt" % parset[index]) as reader:
                readPerDft(reader, index, labels, seeds, perErrorsB, perStandardDeviationB, gammas, rowCount)
    
    # THE EPSILON AND N DEPENDENCIES ARE SEPARATED FURTHER BY THE NUMBER OF FOURIER COEFFICIENTS M SELECTED
    elif index >= 4:
        if m == 5:
            with open("errordatafourier" + str(index) + "%s" % parset[index] + str(0) + str(m) + ".txt") as reader:
                readPerDft(reader, index, labels, seeds, perErrorsA, perStandardDeviationA, gammas, rowCount)
            with open("errordatanofourier" + str(index) + "%s" % parset[index] + str(0) + str(m) + ".txt") as reader:
                readPerDft(reader, index, labels, seeds, perErrorsB, perStandardDeviationB, gammas, rowCount)
        else:
            with open("errordatafourier" + str(index) + "%s" % parset[index] + str(m) + ".txt") as reader:
                readPerDft(reader, index, labels, seeds, perErrorsA, perStandardDeviationA, gammas, rowCount)
            with open("errordatanofourier" + str(index) + "%s" % parset[index] + str(m) + ".txt") as reader:
                readPerDft(reader, index, labels, seeds, perErrorsB, perStandardDeviationB, gammas, rowCount)
    else:
        with open("errordatafourier" + str(index) + "%s.txt" % parset[index]) as reader:
            readPerDft(reader, index, labels, seeds, perErrorsA, perStandardDeviationA, gammas, rowCount)
        with open("errordatanofourier" + str(index) + "%s.txt" % parset[index]) as reader:
            readPerDft(reader, index, labels, seeds, perErrorsB, perStandardDeviationB, gammas, rowCount)

    # NEED TO ISOLATE PERTURBATION ERRORS TO VERIFY DEPENDENCIES
    xLabels = np.arange(len(labels))
    plt.bar(xLabels - width/2, perErrorsA, width, label = 'Perturbation error',  alpha = 0.6, color = 'tab:green', edgecolor = 'k')
    plt.errorbar(xLabels - width/2, perErrorsA, perStandardDeviationA, label = 'Standard deviation', linestyle = 'None', capsize = 2, color = 'tab:blue')
    plt.bar(xLabels + width/2, perErrorsB, width, label = 'PE for baseline',  alpha = 0.6, color = 'tab:orange', edgecolor = 'k')
    plt.errorbar(xLabels + width/2, perErrorsB, perStandardDeviationB, linestyle = 'None', capsize = 2, color = 'tab:blue')

    # PLOTTING COMPARISON LINE GRAPHS TO VERIFY DEPENDENCIES WITH D, EPSILON AND N
    plotTuple = tuple(zip(seeds, gammas))
    x = np.arange(len(np.array(labels)))

    # CHANGING D
    if index == 2:
        if heartOrSynth == 0:
            pA = [(0.000000038*((s**(8/3))/((1-g))**2))+0.001 for s, g in plotTuple]
            pB = [(0.0000002*((s**(8/3))/((1-g))**2))+0.00026 for s, g in plotTuple]
        else:
            pA = [(0.000000009*((s**(8/3))/((1-g))**2))+0.00008 for s, g in plotTuple]
            pB = [(0.00000018*((s**(8/3))/((1-g))**2))+0.00036 for s, g in plotTuple]
    
    # EPSILON LESS THAN 1: SEPARATED BY M
    elif index == 4:
        if m == 5:
            pA = [(0.00012*((1/(s**(4/3)))/((1-g))**2))+0.00003 for s, g in plotTuple]
            pB = [(0.00001*((1/(s**(4/3)))/((1-g))**2))+0.00003 for s, g in plotTuple]

        elif m == 20:
            pA = [(0.00016*((1/(s**(4/3)))/((1-g))**2))+0.0012 for s, g in plotTuple]
            pB = [(0.00015*((1/(s**(4/3)))/((1-g))**2))+0.00085 for s, g in plotTuple]

        elif m == 40:
            pA = [(0.0009*((1/(s**(4/3)))/((1-g))**2))+0.003 for s, g in plotTuple]
            pB = [(0.0014*((1/(s**(4/3)))/((1-g))**2))+0.0036 for s, g in plotTuple]
    
        elif m == 55:
            pA = [(0.002*((1/(s**(4/3)))/((1-g))**2))+0.0026 for s, g in plotTuple]
            pB = [(0.0046*((1/(s**(4/3)))/((1-g))**2))+0.005 for s, g in plotTuple]

        elif m == 75:
            pA = [(0.0023*((1/(s**(4/3)))/((1-g))**2))+0.006 for s, g in plotTuple]
            pB = [(0.012*((1/(s**(4/3)))/((1-g))**2))+0.0075 for s, g in plotTuple]

        else:
            pA = [(0.0027*((1/(s**(4/3)))/((1-g))**2))+0.008 for s, g in plotTuple]
            pB = [(0.019*((1/(s**(4/3)))/((1-g))**2))+0.022 for s, g in plotTuple]

    # EPSILON EQUAL OR GREATER THAN 1: SEPARATED BY M
    elif index == 5:
        if m == 5:
            pA = [(0.00015*((1/(s**(4/3)))/((1-g))**2))+0.00015 for s, g in plotTuple]
            pB = [(0.000042*((1/(s**(4/3)))/((1-g))**2))+0.000037 for s, g in plotTuple]

        elif m == 20:
            pA = [(0.0013*((1/(s**(4/3)))/((1-g))**2))+0.001 for s, g in plotTuple]
            pB = [(0.0009*((1/(s**(4/3)))/((1-g))**2))+0.00075 for s, g in plotTuple]

        elif m == 40:
            pA = [(0.0035*((1/(s**(4/3)))/((1-g))**2))+0.0021 for s, g in plotTuple]
            pB = [(0.0068*((1/(s**(4/3)))/((1-g))**2))+0.0028 for s, g in plotTuple]
    
        elif m == 55:
            pA = [(0.0045*((1/(s**(4/3)))/((1-g))**2))+0.003 for s, g in plotTuple]
            pB = [(0.014*((1/(s**(4/3)))/((1-g))**2))+0.0053 for s, g in plotTuple]

        elif m == 75:
            pA = [(0.0096*((1/(s**(4/3)))/((1-g))**2))+0.0045 for s, g in plotTuple]
            pB = [(0.035*((1/(s**(4/3)))/((1-g))**2))+0.0096 for s, g in plotTuple]

        else:
            pA = [(0.013*((1/(s**(4/3)))/((1-g))**2))+0.0053 for s, g in plotTuple]
            pB = [(0.065*((1/(s**(4/3)))/((1-g))**2))+0.017 for s, g in plotTuple]

    # CHANGING N: SEPARATED BY M
    else:
        if m == 5:
            pA = [(0.0045*((1/(s**(5/3)))/((1-g))**2))+0.00015 for s, g in plotTuple]
            pB = [(0.00002*((1/(s**(5/3)))/((1-g))**2))+0.000038 for s, g in plotTuple]

        elif m == 20:
            pA = [(0.05*((1/(s**(5/3)))/((1-g))**2))+0.001 for s, g in plotTuple]
            pB = [(0.038*((1/(s**(5/3)))/((1-g))**2))+0.00065 for s, g in plotTuple]

        elif m == 40:
            pA = [(0.1*((1/(s**(5/3)))/((1-g))**2))+0.0022 for s, g in plotTuple]
            pB = [(0.28*((1/(s**(5/3)))/((1-g))**2))+0.0032 for s, g in plotTuple]
    
        elif m == 55:
            pA = [(0.2*((1/(s**(5/3)))/((1-g))**2))+0.0026 for s, g in plotTuple]
            pB = [(0.8*((1/(s**(5/3)))/((1-g))**2))+0.0065 for s, g in plotTuple]

        elif m == 75:
            pA = [(0.135*((1/(s**(5/3)))/((1-g))**2))+0.0075 for s, g in plotTuple]
            pB = [(1.85*((1/(s**(5/3)))/((1-g))**2))+0.016 for s, g in plotTuple]

        else:
            pA = [(0.35*((1/(s**(5/3)))/((1-g))**2))+0.0072 for s, g in plotTuple]
            pB = [(4.0*((1/(s**(5/3)))/((1-g))**2))+0.028 for s, g in plotTuple]

    yA = np.array(pA)
    yB = np.array(pB)
    plt.plot(x - width/2, yA, label = 'Best fit curves', alpha = 0.6, color = 'k')
    plt.plot(x + width/2, yB, alpha = 0.6, color = 'k')

    # THE Y-AXIS IS THE SAME FOR EACH PARAMETER
    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.ylabel('Perturbation error')
    plt.xticks(x, labels)

    if index == 2:
        if heartOrSynth == 0:
            plt.ylim(0.0001, 0.075)
            selectiveFormatter = FixedFormatter(["0.0001", "0.001", "0.01", "0.075"])
            selectiveLocator = FixedLocator([0.0001, 0.001, 0.01, 0.075])
        else:
            plt.ylim(0.00001, 0.06)
            selectiveFormatter = FixedFormatter(["0.00001", "0.0001", "0.001", "0.01", "0.06"])
            selectiveLocator = FixedLocator([0.00001, 0.0001, 0.001, 0.01, 0.06])

    # CREATING A LOGARITHMIC Y-AXIS FOR EPSILON LESS THAN 1: SEPARATED BY M
    elif index == 4:
        plt.yscale('log')

        if m == 5:
            plt.ylim(0.00001, 0.002)
            selectiveFormatter = FixedFormatter(["0.00001", "0.0001", "0.001", "0.002"])
            selectiveLocator = FixedLocator([0.00001, 0.0001, 0.001, 0.002])

        elif m == 20:
            plt.ylim(0.0003, 0.007)
            selectiveFormatter = FixedFormatter(["0.0001", "0.001", "0.007"])
            selectiveLocator = FixedLocator([0.0003, 0.001, 0.007])

        elif m == 40:
            plt.ylim(0.001, 0.03)
            selectiveFormatter = FixedFormatter(["0.001", "0.01", "0.03"])
            selectiveLocator = FixedLocator([0.001, 0.01, 0.03])
    
        elif m == 55:
            plt.ylim(0.001, 0.05)
            selectiveFormatter = FixedFormatter(["0.001", "0.01", "0.05"])
            selectiveLocator = FixedLocator([0.001, 0.01, 0.05])

        elif m == 75:
            plt.ylim(0.002, 0.2)
            selectiveFormatter = FixedFormatter(["0.002", "0.01", "0.1", "0.2"])
            selectiveLocator = FixedLocator([0.002, 0.01, 0.1, 0.2])

        else:
            plt.ylim(0.002, 0.3)
            selectiveFormatter = FixedFormatter(["0.002", "0.01", "0.1", "0.3"])
            selectiveLocator = FixedLocator([0.002, 0.01, 0.1, 0.3])

        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

    # EPSILON EQUAL OR GREATER THAN 1: SEPARATED BY M
    elif index == 5:
        plt.yscale('log')

        if m == 5:
            plt.ylim(0.00001, 0.001)
            selectiveFormatter = FixedFormatter(["0.00001", "0.0001", "0.001"])
            selectiveLocator = FixedLocator([0.00001, 0.0001, 0.001])

        elif m == 20:
            plt.ylim(0.0003, 0.006)
            selectiveFormatter = FixedFormatter(["0.0003", "0.001", "0.006"])
            selectiveLocator = FixedLocator([0.0003, 0.001, 0.006])

        elif m == 40:
            plt.ylim(0.0005, 0.03)
            selectiveFormatter = FixedFormatter(["0.0005", "0.001", "0.01", "0.03"])
            selectiveLocator = FixedLocator([0.0005, 0.001, 0.01, 0.03])
    
        elif m == 55:
            plt.ylim(0.001, 0.06)
            selectiveFormatter = FixedFormatter(["0.001", "0.01", "0.06"])
            selectiveLocator = FixedLocator([0.001, 0.01, 0.06])

        elif m == 75:
            plt.ylim(0.002, 0.3)
            selectiveFormatter = FixedFormatter(["0.002", "0.01", "0.1", "0.3"])
            selectiveLocator = FixedLocator([0.002, 0.01, 0.1, 0.3])

        else:
            plt.ylim(0.002, 0.9)
            selectiveFormatter = FixedFormatter(["0.002", "0.01", "0.1", "0.9"])
            selectiveLocator = FixedLocator([0.002, 0.01, 0.1, 0.9])
          
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

    # CHANGING N: SEPARATED BY M
    elif index == 6:
        plt.yscale('log')
        
        if m == 5:
            plt.ylim(0.00001, 0.001)
            selectiveFormatter = FixedFormatter(["0.00001", "0.0001", "0.001"])
            selectiveLocator = FixedLocator([0.00001, 0.0001, 0.001])

        elif m == 20:
            plt.ylim(0.0002, 0.004)
            selectiveFormatter = FixedFormatter(["0.0002", "0.001", "0.004"])
            selectiveLocator = FixedLocator([0.0002, 0.001, 0.004])

        elif m == 40:
            plt.ylim(0.0005, 0.03)
            selectiveFormatter = FixedFormatter(["0.0005", "0.001", "0.01", "0.03"])
            selectiveLocator = FixedLocator([0.0005, 0.001, 0.01, 0.03])
    
        elif m == 55:
            plt.ylim(0.0005, 0.07)
            selectiveFormatter = FixedFormatter(["0.0005", "0.001", "0.01", "0.07"])
            selectiveLocator = FixedLocator([0.0005, 0.001, 0.01, 0.07])

        elif m == 75:
            plt.ylim(0.001, 0.4)
            selectiveFormatter = FixedFormatter(["0.001", "0.01", "0.1", "0.4"])
            selectiveLocator = FixedLocator([0.001, 0.01, 0.1, 0.4])

        else:
            plt.ylim(0.002, 1.7)
            selectiveFormatter = FixedFormatter(["0.001", "0.01", "0.1", "1", "1.7"])
            selectiveLocator = FixedLocator([0.002, 0.01, 0.1, 1, 1.7])
 
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

# THE SKELETON SAVING FUNCTION FOR THE ISOLATED PERTURBATION ERROR
def savePerDft(heartOrSynth, index, m):
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 3, 0, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc = "upper center", bbox_to_anchor = (0.5, 1.2), ncol = 2, shadow = True)
    plt.tight_layout()
    plt.draw()
    
    if index == 2:
        if heartOrSynth == 0:
            plt.savefig("errorchartfourierperturb" + str(index) + "%sheart.png" % parset[index])
        else:
            plt.savefig("errorchartfourierperturb" + str(index) + "%ssynth.png" % parset[index])

    elif index >= 4:
        if m == 5:
            plt.savefig("errorchartfourierperturb" + str(index) + "%s" % parset[index] + str(0) + str(m) + ".png")
        else:
            plt.savefig("errorchartfourierperturb" + str(index) + "%s" % parset[index] + str(m) + ".png")
    else:
        plt.savefig("errorchartfourierperturb" + str(index) + "%s.png" % parset[index])

    plt.clf()
    plt.cla()

# A FINAL FUNCTION TO GRAB ALL THE BEST FIT LINES FROM ABOVE AND PLOT THEM TOGETHER
def drawDftLines(index):
    labels = [0]*limit
    gammas = [0]*limit
    seeds = [0]*limit
    rowCount = 0

    # CAN GRAB THE LABELS, SEEDS AND GAMMAS FROM ANY FOURIER DATA FILE
    with open("errordatafourier" + str(index) + "%s" % parset[index] + str(0) + str(5) + ".txt") as reader:
        for line in reader:
            tab = line.split()
            
            if index == 4 or index == 5:
                labels[rowCount] = f'{(float(tab[0]))}'
                seeds[rowCount] = float(tab[0])
            else:
                labels[rowCount] = f'{(int(tab[0]))}'
                seeds[rowCount] = int(tab[0])

            gammas[rowCount] = float(tab[6])
            rowCount += 1

            if rowCount >= limit:
                break

    # PLOTTING ALL LINE GRAPHS FOR DEPENDENCIES WITH EPSILON AND N
    plotTuple = tuple(zip(seeds, gammas))
    x = np.arange(len(np.array(labels)))

    # EPSILON LESS THAN 1: SEPARATED BY M
    if index == 4:
        p1A = [(0.00006*((1/(s**(4/3)))/((1-g))**2))+0.0001 for s, g in plotTuple]
        p1B = [(0.00001*((1/(s**(4/3)))/((1-g))**2))+0.00003 for s, g in plotTuple]

        p2A = [(0.00037*((1/(s**(4/3)))/((1-g))**2))+0.001 for s, g in plotTuple]
        p2B = [(0.00023*((1/(s**(4/3)))/((1-g))**2))+0.0008 for s, g in plotTuple]

        p3A = [(0.0012*((1/(s**(4/3)))/((1-g))**2))+0.0019 for s, g in plotTuple]
        p3B = [(0.0022*((1/(s**(4/3)))/((1-g))**2))+0.0021 for s, g in plotTuple]
    
        p4A = [(0.0025*((1/(s**(4/3)))/((1-g))**2))+0.001 for s, g in plotTuple]
        p4B = [(0.0041*((1/(s**(4/3)))/((1-g))**2))+0.0064 for s, g in plotTuple]

        p5A = [(0.0047*((1/(s**(4/3)))/((1-g))**2)) for s, g in plotTuple]
        p5B = [(0.011*((1/(s**(4/3)))/((1-g))**2))+0.01 for s, g in plotTuple]

        p6A = [(0.0027*((1/(s**(4/3)))/((1-g))**2))+0.0055 for s, g in plotTuple]
        p6B = [(0.02*((1/(s**(4/3)))/((1-g))**2))+0.017 for s, g in plotTuple]

    # EPSILON EQUAL OR GREATER THAN 1: SEPARATED BY M
    elif index == 5:
        p1A = [(0.0002*((1/(s**(4/3)))/((1-g))**2))+0.00014 for s, g in plotTuple]
        p1B = [(0.000034*((1/(s**(4/3)))/((1-g))**2))+0.00004 for s, g in plotTuple]

        p2A = [(0.0012*((1/(s**(4/3)))/((1-g))**2))+0.001 for s, g in plotTuple]
        p2B = [(0.001*((1/(s**(4/3)))/((1-g))**2))+0.0007 for s, g in plotTuple]

        p3A = [(0.0055*((1/(s**(4/3)))/((1-g))**2))+0.0021 for s, g in plotTuple]
        p3B = [(0.0072*((1/(s**(4/3)))/((1-g))**2))+0.0029 for s, g in plotTuple]
    
        p4A = [(0.0025*((1/(s**(4/3)))/((1-g))**2))+0.0037 for s, g in plotTuple]
        p4B = [(0.013*((1/(s**(4/3)))/((1-g))**2))+0.0055 for s, g in plotTuple]

        p5A = [(0.0035*((1/(s**(4/3)))/((1-g))**2))+0.0065 for s, g in plotTuple]
        p5B = [(0.037*((1/(s**(4/3)))/((1-g))**2))+0.0084 for s, g in plotTuple]

        p6A = [(0.01*((1/(s**(4/3)))/((1-g))**2))+0.0055 for s, g in plotTuple]
        p6B = [(0.075*((1/(s**(4/3)))/((1-g))**2))+0.015 for s, g in plotTuple]

    # CHANGING N: SEPARATED BY M
    else:
        p1A = [(0.0035*((1/(s**(5/3)))/((1-g))**2))+0.00015 for s, g in plotTuple]
        p1B = [(0.0015*((1/(s**(5/3)))/((1-g))**2))+0.000027 for s, g in plotTuple]

        p2A = [(0.015*((1/(s**(5/3)))/((1-g))**2))+0.0012 for s, g in plotTuple]
        p2B = [(0.025*((1/(s**(5/3)))/((1-g))**2))+0.0008 for s, g in plotTuple]

        p3A = [(0.05*((1/(s**(5/3)))/((1-g))**2))+0.0023 for s, g in plotTuple]
        p3B = [(0.3*((1/(s**(5/3)))/((1-g))**2))+0.0033 for s, g in plotTuple]
    
        p4A = [(0.26*((1/(s**(5/3)))/((1-g))**2))+0.0025 for s, g in plotTuple]
        p4B = [(0.74*((1/(s**(5/3)))/((1-g))**2))+0.0062 for s, g in plotTuple]

        p5A = [(0.25*((1/(s**(5/3)))/((1-g))**2))+0.006 for s, g in plotTuple]
        p5B = [(1.6*((1/(s**(5/3)))/((1-g))**2))+0.016 for s, g in plotTuple]

        p6A = [(0.32*((1/(s**(5/3)))/((1-g))**2))+0.0065 for s, g in plotTuple]
        p6B = [(3.4*((1/(s**(5/3)))/((1-g))**2))+0.028 for s, g in plotTuple]

    y1A = np.array(p1A)
    y1B = np.array(p1B)

    y2A = np.array(p2A)
    y2B = np.array(p2B)

    y3A = np.array(p3A)
    y3B = np.array(p3B)

    y4A = np.array(p4A)
    y4B = np.array(p4B)

    y5A = np.array(p5A)
    y5B = np.array(p5B)

    y6A = np.array(p6A)
    y6B = np.array(p6B)

    plt.plot(x, y1A, label = '$\mathit{m} = 5$', alpha = 0.6, color = 'darkorange')
    plt.plot(x, y1B, label = '$\mathit{m} = 5$ (baseline)', alpha = 0.6, color = 'chocolate')

    plt.plot(x, y2A, label = '$\mathit{m} = 20$', alpha = 0.6, color = 'limegreen')
    plt.plot(x, y2B, label = '$\mathit{m} = 20$ (baseline)', alpha = 0.6, color = 'darkgreen')

    plt.plot(x, y3A, label = '$\mathit{m} = 40$', alpha = 0.6, color = 'darkolivegreen')
    plt.plot(x, y3B, label = '$\mathit{m} = 40$ (baseline)', alpha = 0.6, color = 'deepskyblue')

    plt.plot(x, y4A, label = '$\mathit{m} = 55$', alpha = 0.6, color = 'royalblue')
    plt.plot(x, y4B, label = '$\mathit{m} = 55$ (baseline)', alpha = 0.6, color = 'navy')

    plt.plot(x, y5A, label = '$\mathit{m} = 75$', alpha = 0.6, color = 'deeppink')
    plt.plot(x, y5B, label = '$\mathit{m} = 75$ (baseline)', alpha = 0.6, color = 'darkmagenta')

    plt.plot(x, y6A, label = '$\mathit{m} = 95$', alpha = 0.6, color = 'crimson')
    plt.plot(x, y6B, label = '$\mathit{m} = 95$ (baseline)', alpha = 0.6, color = 'darkred')

    # THE Y-AXIS IS THE SAME FOR EACH PARAMETER
    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.ylabel('Perturbation error')
    plt.xticks(x, labels)

    # CREATING A LOGARITHMIC Y-AXIS FOR THE EPSILON AND N DEPENDENCIES
    if index == 4:
        plt.yscale('log')
        plt.ylim(0.00001, 1)
        selectiveFormatter = FixedFormatter(["0.00001", "0.0001", "0.001", "0.01", "0.1", "1"])
        selectiveLocator = FixedLocator([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

    elif index == 5:
        plt.yscale('log')
        plt.ylim(0.00001, 1)
        selectiveFormatter = FixedFormatter(["0.00001", "0.0001", "0.001", "0.01", "0.1", "1"])
        selectiveLocator = FixedLocator([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

    elif index == 6:
        plt.yscale('log')
        plt.ylim(0.00001, 2)
        selectiveFormatter = FixedFormatter(["0.00001", "0.0001", "0.001", "0.01", "0.1", "1"])
        selectiveLocator = FixedLocator([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
        plt.gca().yaxis.set_major_formatter(selectiveFormatter)
        plt.gca().yaxis.set_major_locator(selectiveLocator)

# SAVING THE FINAL GRAPH
def saveDftLines(index):
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc = "upper center", bbox_to_anchor = (0.5, 1.3), ncol = 2, shadow = True, fontsize = 9)
    plt.tight_layout()
    plt.draw()
    plt.savefig("errorchartfourierplines" + str(index) + "%s.png" % parset[index])
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
            custom(index)
            saveDft(0, index)

        if index == 2:
            drawDft(1, index)
            custom(index)
            saveDft(1, index)

            for i in range(2):
                fitPerDft(i, index, mset2[5])
                custom(index)
                savePerDft(i, index, mset2[5])

        if index >= 4:
            for m in mset2:
                fitPerDft(0, index, m)
                custom(index)
                savePerDft(0, index, m)
            
            drawDftLines(index)
            custom(index)
            saveDftLines(index)

# CALLING ALL THE ABOVE FUNCTIONS: SOME ARE NESTED
plotBasic()
plotDft()