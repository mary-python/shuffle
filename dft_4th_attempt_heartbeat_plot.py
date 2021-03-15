import math, re
import matplotlib.pyplot as plt
from decimal import *
width = 0.35

def plotBasicVaryT():

    labels = list()
    totalErrors = list()
    totalStandardDeviation = list()

    with open("errorvaryt.txt") as reader:
        for line in reader:
            tab = line.split()
            labels.append(f'{int(tab[0])}')
            totalErrors.append((Decimal(tab[1])))
            totalStandardDeviation.append((Decimal(tab[2])))

    plt.bar(labels, totalErrors, width, alpha = 0.6, color = 'm', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.xticks(['1', '2', '3', '4', '5'])
    plt.ylabel('Total experimental MSE')
    plt.xlabel('Number of coordinates retained', labelpad = 8)
    plt.title('Experimental error by number of coordinates retained')

    plt.tight_layout()
    plt.draw()
    plt.savefig("errorchartvaryt.png")
    plt.clf()
    plt.cla()

def plotBasicVaryK():
    
    labels = list()
    totalErrors = list()
    totalStandardDeviation = list()

    with open("errorvaryk.txt") as reader:
        for line in reader:
            tab = line.split()
            labels.append(f'{int(tab[0])}')
            totalErrors.append((Decimal(tab[1])))
            totalStandardDeviation.append((Decimal(tab[2])))

    plt.bar(labels, totalErrors, width, alpha = 0.6, color = 'm', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.xticks(['4', '5', '6', '7', '8', '9', '10'])
    plt.ylabel('Total experimental MSE')
    plt.xlabel('Number of buckets used', labelpad = 8)
    plt.title('Experimental error by number of buckets used')

    plt.tight_layout()
    plt.draw()
    plt.savefig("errorchartvaryk.png")
    plt.clf()
    plt.cla()

def plotBasicVaryD():

    labels = list()
    totalErrors = list()
    totalStandardDeviation = list()

    with open("errorvaryd.txt") as reader:
        for line in reader:
            tab = line.split()
            labels.append(f'{int(tab[0])}')
            totalErrors.append((Decimal(tab[1])))
            totalStandardDeviation.append((Decimal(tab[2])))

    plt.bar(labels, totalErrors, width, alpha = 0.6, color = 'm', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.xticks(['80', '90', '100', '110', '120', '130', '140', '150'])
    plt.ylabel('Total experimental MSE')
    plt.xlabel('Dimension of vector', labelpad = 8)
    plt.title('Experimental error by dimension of vector')

    plt.tight_layout()
    plt.draw()
    plt.savefig("errorchartvaryd.png")
    plt.clf()
    plt.cla()

def plotBasicVaryEps():

    labels = list()
    totalErrors = list()
    totalStandardDeviation = list()

    with open("errorvaryeps.txt") as reader:
        for line in reader:
            tab = line.split()
            labels.append(f'{float(tab[0])}')
            totalErrors.append((Decimal(tab[1])))
            totalStandardDeviation.append((Decimal(tab[2])))

    plt.bar(labels, totalErrors, width, alpha = 0.6, color = 'm', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.xticks(['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0'])
    plt.ylabel('Total experimental MSE')
    plt.xlabel('Value of epsilon', labelpad = 8)
    plt.title('Experimental error by value of epsilon')

    plt.tight_layout()
    plt.draw()
    plt.savefig("errorchartvaryeps.png")
    plt.clf()
    plt.cla()

def plotBasicVaryN():

    labels = list()
    totalErrors = list()
    totalStandardDeviation = list()

    with open("errorvaryn.txt") as reader:
        for line in reader:
            tab = line.split()
            labels.append(f'{int(tab[0])}')
            totalErrors.append((Decimal(tab[1])))
            totalStandardDeviation.append((Decimal(tab[2])))

    plt.bar(labels, totalErrors, width, alpha = 0.6, color = 'm', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.xticks(['3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    plt.ylabel('Total experimental MSE')
    plt.xlabel('Number of vectors used' + ' ' + 'x' + ' ' + '$10^{4}$', labelpad = 8)
    plt.title('Experimental error by number of vectors used')

    plt.tight_layout()
    plt.draw()
    plt.savefig("errorchartvaryn.png")
    plt.clf()
    plt.cla()

def plotDftVaryT():

    labels = list()
    perErrors = list()
    recErrors = list()
    totalErrors = list()
    totalStandardDeviation = list()

    with open("dfterrorvaryt.txt") as reader:
        for line in reader:
            tab = line.split()
            labels.append(f'{int(tab[0])}')
            perErrors.append((Decimal(tab[1])))
            recErrors.append((Decimal(tab[2])))
            totalErrors.append((Decimal(tab[3])))
            totalStandardDeviation.append((Decimal(tab[4])))

    plt.bar(labels, perErrors, width, label = 'Perturbation error', alpha = 0.6, color = 'r', edgecolor = 'k')
    plt.bar(labels, recErrors, width, bottom = perErrors, label = 'Reconstruction error', alpha = 0.6, color = 'c', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.xticks(['1', '2', '3', '4', '5'])
    plt.ylabel('Total experimental MSE')
    plt.xlabel('Number of coordinates retained', labelpad = 8)
    plt.title('Ratio between experimental errors by number of coordinates retained')

    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.savefig("dfterrorchartvaryt.png")
    plt.clf()
    plt.cla()

def plotDftVaryK():

    labels = list()
    perErrors = list()
    recErrors = list()
    totalErrors = list()
    totalStandardDeviation = list()


    with open("dfterrorvaryk.txt") as reader:
        for line in reader:
            tab = line.split()
            labels.append(f'{int(tab[0])}')
            perErrors.append((Decimal(tab[1])))
            recErrors.append((Decimal(tab[2])))
            totalErrors.append((Decimal(tab[3])))
            totalStandardDeviation.append((Decimal(tab[4])))

    plt.bar(labels, perErrors, width, label = 'Perturbation error', alpha = 0.6, color = 'r', edgecolor = 'k')
    plt.bar(labels, recErrors, width, bottom = perErrors, label = 'Reconstruction error', alpha = 0.6, color = 'c', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.xticks(['4', '5', '6', '7', '8', '9', '10'])
    plt.ylabel('Total experimental MSE')
    plt.xlabel('Number of buckets used', labelpad = 8)
    plt.title('Ratio between experimental errors by number of buckets used')

    plt.legend()
    plt.tight_layout() 
    plt.draw()
    plt.savefig("dfterrorchartvaryk.png")
    plt.clf()
    plt.cla()

def plotDftVaryM():

    labels = list()
    perErrors = list()
    recErrors = list()
    totalErrors = list()
    totalStandardDeviation = list()

    with open("dfterrorvarym.txt") as reader:
        for line in reader:
            tab = line.split()
            labels.append(f'{int(tab[0])}%')
            perErrors.append((Decimal(tab[1])))
            recErrors.append((Decimal(tab[2])))
            totalErrors.append((Decimal(tab[3])))
            totalStandardDeviation.append((Decimal(tab[4])))

    plt.bar(labels, perErrors, width, label = 'Perturbation error', alpha = 0.6, color = 'r', edgecolor = 'k')
    plt.bar(labels, recErrors, width, bottom = perErrors, label = 'Reconstruction error', alpha = 0.6, color = 'c', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.xticks(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    plt.ylabel('Total experimental MSE')
    plt.xlabel('% of Fourier coefficients retained', labelpad = 8)
    plt.title('Ratio between experimental errors by % of Fourier coefficients retained')

    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.savefig("dfterrorchartvarym.png")
    plt.clf()
    plt.cla()

def plotDftVaryD():

    labels = list()
    perErrors = list()
    recErrors = list()
    totalErrors = list()
    totalStandardDeviation = list()

    with open("dfterrorvaryd.txt") as reader:
        for line in reader:
            tab = line.split()
            labels.append(f'{int(tab[0])}')
            perErrors.append((Decimal(tab[1])))
            recErrors.append((Decimal(tab[2])))
            totalErrors.append((Decimal(tab[3])))
            totalStandardDeviation.append((Decimal(tab[4])))

    plt.bar(labels, perErrors, width, label = 'Perturbation error', alpha = 0.6, color = 'r', edgecolor = 'k')
    plt.bar(labels, recErrors, width, bottom = perErrors, label = 'Reconstruction error', alpha = 0.6, color = 'c', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.xticks(['80', '90', '100', '110', '120', '130', '140', '150'])
    plt.ylabel('Total experimental MSE')
    plt.xlabel('Dimension of vector', labelpad = 8)
    plt.title('Ratio between experimental errors by dimension of vector')

    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.savefig("dfterrorchartvaryd.png")
    plt.clf()
    plt.cla()

def plotDftVaryEps():

    labels = list()
    perErrors = list()
    recErrors = list()
    totalErrors = list()
    totalStandardDeviation = list()

    with open("dfterrorvaryeps.txt") as reader:
        for line in reader:
            tab = line.split()
            labels.append(f'{float(tab[0])}')
            perErrors.append((Decimal(tab[1])))
            recErrors.append((Decimal(tab[2])))
            totalErrors.append((Decimal(tab[3])))
            totalStandardDeviation.append((Decimal(tab[4])))

    plt.bar(labels, perErrors, width, label = 'Perturbation error', alpha = 0.6, color = 'r', edgecolor = 'k')
    plt.bar(labels, recErrors, width, bottom = perErrors, label = 'Reconstruction error', alpha = 0.6, color = 'c', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.xticks(['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0'])
    plt.ylabel('Total experimental MSE')
    plt.xlabel('Value of epsilon', labelpad = 8)
    plt.title('Ratio between experimental errors by value of epsilon')

    plt.legend()
    plt.tight_layout() 
    plt.draw()
    plt.savefig("dfterrorchartvaryeps.png")
    plt.clf()
    plt.cla()

def plotDftVaryN():

    labels = list()
    perErrors = list()
    recErrors = list()
    totalErrors = list()
    totalStandardDeviation = list()

    with open("dfterrorvaryn.txt") as reader:
        for line in reader:
            tab = line.split()
            labels.append(f'{int(tab[0])}')
            perErrors.append((Decimal(tab[1])))
            recErrors.append((Decimal(tab[2])))
            totalErrors.append((Decimal(tab[3])))
            totalStandardDeviation.append((Decimal(tab[4])))

    plt.bar(labels, perErrors, width, label = 'Perturbation error', alpha = 0.6, color = 'r', edgecolor = 'k')
    plt.bar(labels, recErrors, width, bottom = perErrors, label = 'Reconstruction error', alpha = 0.6, color = 'c', edgecolor = 'k')
    plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

    plt.ticklabel_format(axis = 'y', style = 'plain')
    plt.xticks(['3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    plt.ylabel('Total experimental MSE')
    plt.xlabel('Number of vectors used' + ' ' + 'x' + ' ' + '$10^{4}$', labelpad = 8)
    plt.title('Ratio between experimental errors by number of vectors used')

    plt.legend()
    plt.tight_layout
    plt.draw()
    plt.savefig("dfterrorchartvaryn.png")
    plt.clf()
    plt.cla()

plotBasicVaryT()
plotBasicVaryK()
plotBasicVaryD()
plotBasicVaryEps()
plotBasicVaryN()

plotDftVaryT()
plotDftVaryK()
plotDftVaryM()
plotDftVaryD()
plotDftVaryEps()
plotDftVaryN()