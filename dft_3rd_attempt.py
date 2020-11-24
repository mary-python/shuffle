import random, math, time; import numpy as np
from scipy.fftpack import rfft, irfft
import matplotlib.pyplot as plt; from matplotlib.ticker import PercentFormatter
from brokenaxes import brokenaxes

startTime = time.perf_counter()

d = 1000; k = 6; n = 100000; eps = 0.1; dta = 0.185; V = 10; s = 15; v = 5
gamma = max((((14*k*(math.log(2/dta))))/((n-1)*(eps**2))), (27*k)/((n-1)*eps))

loopTotal = list(); perErrors = list(); recErrors = list(); labels = list()
randomVector = [0]*d; dftRandomVector = [0]*d

sampledList = list(); debiasedList = list()
indexTracker = [0]*d; submittedTotal = [0]*d; totalVector = [0]*d

print(f"\n Processing the basic optimal summation result.")

from progress.bar import FillingSquaresBar
bar = FillingSquaresBar(max=n, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

for i in range(0, n):

    for a in range(0, d):
        randomCoord = -(math.log(1 - (1 - math.exp(-s))*(random.random())))/s
        randomVector[a] = randomCoord
        totalVector[a] += randomCoord

    randomIndex = random.randint(0, d-1)
    sampledPair = (randomIndex, randomVector[randomIndex])
    sampledCoord = sampledPair[1]
    sampledList.append(sampledCoord)

    roundedPair = (randomIndex, (math.floor(sampledCoord*k)\
        + np.random.binomial(1, sampledCoord*k - math.floor(sampledCoord*k))))
    b = np.random.binomial(1, gamma)

    if b == 0:
        submittedPair = roundedPair
    else:
        submittedPair = (randomIndex, (np.random.randint(0, k+1)))

    submittedCoord = submittedPair[1]
    submittedTotal[randomIndex] += submittedCoord
    indexTracker[randomIndex] += 1

    descaledCoord = submittedCoord/k
    debiasedCoord = (descaledCoord - (gamma/2))/(1 - gamma)
    debiasedList.append(debiasedCoord)

    bar.next()
bar.finish()

descaledTotal = [idx/k for idx in submittedTotal]
mergedTracker = tuple(zip(indexTracker, descaledTotal))
debiasedTotal = [(z - ((gamma/2)*count))/(1 - gamma) for count, z in mergedTracker]

averageVector = [idx/n for idx in totalVector]
errorTuple = tuple(zip(debiasedTotal, averageVector))
meanSquaredError = [(a - b)**2 for a, b in errorTuple]
totalMeanSquaredError = sum(meanSquaredError)

for value in range(0, V):

    loopTime = time.perf_counter(); m = (value + 1)*(10)
    dftSampledList = list(); dftDebiasedList = list()
    dftIndexTracker = [0]*m; dftSubmittedTotal = [0]*m; dftTotalVector = [0]*d
    sampledError = 0; returnedError = 0

    print(f"\n Processing the optimal summation result with DFT for the value m = {m}.")

    from progress.bar import FillingSquaresBar
    bar = FillingSquaresBar(max=n, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

    for i in range(0, n):
            
        for a in range(0, d):
            dftRandomCoord = -(math.log(1 - (1 - math.exp(-s))*(random.random())))/s
            dftRandomVector[a] = dftRandomCoord
            dftTotalVector[a] += dftRandomCoord

        dftVector = (rfft(dftRandomVector)).tolist()
        slicedDftVector = dftVector[0:m]

        dftRandomIndex = random.randint(0, m-1)
        dftSampledPair = (dftRandomIndex, slicedDftVector[dftRandomIndex])

        dftSampledCoord = dftSampledPair[1]
        dftSampledList.append(dftSampledCoord)

        dftRoundedPair = (dftRandomIndex, (math.floor(dftSampledCoord*k)\
            + np.random.binomial(1, dftSampledCoord*k - math.floor(dftSampledCoord*k))))
        b = np.random.binomial(1, gamma)

        if b == 0:
            dftSubmittedPair = dftRoundedPair
        else:
            dftSubmittedPair = (dftRandomIndex, (np.random.randint(0, k+1)))

        dftSubmittedCoord = dftSubmittedPair[1]
        dftSubmittedTotal[dftRandomIndex] += dftSubmittedCoord
        dftIndexTracker[dftRandomIndex] += 1

        dftDescaledCoord = dftSubmittedCoord/k
        dftDebiasedCoord = (dftDescaledCoord - (gamma/2))/(1 - gamma)
        dftDebiasedList.append(dftDebiasedCoord)
    
        bar.next()
    bar.finish()

    dftDescaledTotal = [idx/k for idx in dftSubmittedTotal]
    dftMergedTracker = tuple(zip(dftIndexTracker, dftDescaledTotal))
    dftDebiasedTotal = [(z - ((gamma/2)*count))/(1 - gamma) for count, z in dftMergedTracker]
    paddedTotal = dftDebiasedTotal + [0]*(d-m)
    finalTotal = (irfft(paddedTotal)).tolist()

    dftAverageVector = [idx/n for idx in dftTotalVector]        
    dftErrorTuple = tuple(zip(finalTotal, dftAverageVector))
    dftMeanSquaredError = [(a - b)**2 for a, b in dftErrorTuple]
    totalDftMeanSquaredError = sum(dftMeanSquaredError)

    reconstructionTuple = tuple(zip(debiasedTotal, dftAverageVector))
    reconstructionError = [(a - b)**2 for a, b in reconstructionTuple]
    totalReconstructionError = sum(reconstructionError)
    
    if s == 5:
        datafile = open("fourier" + str(m) + "factor0" + str(s) + ".txt", "w")
    else:
        datafile = open("fourier" + str(m) + "factor" + str(s) + ".txt", "w")

    datafile.write(f"Number of Fourier coefficients m: {m} \n")
    datafile.write(f"Case 1: Optimal Summation in the Shuffle Model \n")

    comparison = (2*(14**(2/3))*(d**(2/3))*(n**(1/3))*(np.log(1/dta))*(np.log(2/dta)))/(((1-gamma)**2)*(eps**(4/3)))
    datafile.write(f"Theoretical Upper Bound for MSE: {round(comparison)} \n")
    datafile.write(f"Experimental MSE: {round(totalMeanSquaredError)} \n")
    error1 = round((100)*((totalMeanSquaredError)/comparison))
    datafile.write(f"Experimental MSE was {error1}% of the theoretical upper bound for MSE. \n\n")

    datafile.write(f"Case 2: Fourier Summation Algorithm \n")

    dftComparison = (2*(14**(2/3))*(m**(2/3))*(n**(1/3))*(np.log(1/dta))*(np.log(2/dta)))/(((1-gamma)**2)*(eps**(4/3)))
    datafile.write(f"Theoretical upper bound for perturbation error: {round(dftComparison)} \n")
    datafile.write(f"Experimental perturbation error: {round(totalDftMeanSquaredError)} \n")
    error2 = round((100)*((totalDftMeanSquaredError)/dftComparison))
    datafile.write(f"Experimental perturbation error was {error2}% of the theoretical upper bound for perturbation error. \n")

    datafile.write(f"Experimental reconstruction error: {round(totalReconstructionError)} \n")
    
    perErrors.append(totalDftMeanSquaredError)
    recErrors.append(totalReconstructionError)
    labels.append(f'{round(m/10)}%')

    datafile.write(f"Total experimental MSE: {round((totalDftMeanSquaredError) + (totalReconstructionError))} \n")
    error3 = round((100)*((totalReconstructionError)/((totalDftMeanSquaredError) + (totalReconstructionError))))
    datafile.write(f"Reconstruction error was {error3}% of the total experimental MSE. \n\n")

    plt.style.use('seaborn-white'); plt.tight_layout()
    plt.subplot(2, 2, 1); plt.subplot(2, 2, 2); plt.subplot(2, 2, 3); plt.subplot(2, 2, 4)
    mng = plt.get_current_fig_manager(); mng.window.state('zoomed')
    plt.draw()
    
    if s == 5:
        plt.savefig("fourier" + str(m) + "factor0" + str(s) + ".png")
    else:
        plt.savefig("fourier" + str(m) + "factor" + str(s) + ".png")
    
    plt.clf(); plt.cla()

    plt.subplot(2, 2, 1)
    (freq1, bins1, patches) = plt.hist(sampledList, weights = np.ones(len(sampledList)) / len(sampledList),\
        bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],\
            alpha = 0.3, histtype = 'bar', color = 'g', edgecolor = 'k')

    plt.xlim(0, 1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set(title = 'Histogram of sampled coordinates in the original domain', xlabel = 'Value', ylabel = 'Frequency')

    listFreq1 = freq1.tolist(); formattedFreq1 = list()
    for item in listFreq1:
        formattedFreq1.append(int(float(item*(len(sampledList)))))

    datafile.write(f"Frequencies of sampled coordinates in the original domain: \n")
    datafile.write(f"{str(formattedFreq1)[1:-1]} \n")
    datafile.write(f"Total: {sum(formattedFreq1)} \n")
    datafile.write(f"Percentage of sampled coordinates between 0 and 1: {round((100)*(sum(formattedFreq1))/(sum(indexTracker)))}% \n\n")

    plt.subplot(2, 2, 2)
    (freq2, bins2, patches) = plt.hist(debiasedList, weights = np.ones(len(debiasedList)) / len(debiasedList),\
        bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],\
            alpha = 0.3, histtype = 'bar', color = 'b', edgecolor = 'k')
    
    plt.xlim(0, 1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set(title = 'Histogram of returned coordinates in the original domain', xlabel = 'Value', ylabel = 'Frequency')

    listFreq2 = freq2.tolist(); formattedFreq2 = list()
    for item in listFreq2:
        formattedFreq2.append(int(float(item*(len(debiasedList)))))

    datafile.write(f"Frequencies of returned coordinates in the original domain: \n")
    datafile.write(f"{str(formattedFreq2)[1:-1]} \n")
    datafile.write(f"Total: {sum(formattedFreq2)} \n")
    datafile.write(f"Percentage of returned coordinates between 0 and 1: {round((100)*(sum(formattedFreq2))/(sum(indexTracker)))}% \n\n")

    plt.subplot(2, 2, 3)
    (freq3, bins3, patches) = plt.hist(dftSampledList, weights = np.ones(len(dftSampledList)) / len(dftSampledList),\
        bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],\
            alpha = 0.5, histtype = 'bar', color = 'g', edgecolor = 'k')

    plt.xlim(0, 1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set(title = 'Histogram of sampled coordinates in the Fourier domain', xlabel = 'Value', ylabel = 'Frequency')

    listFreq3 = freq3.tolist(); formattedFreq3 = list()
    for item in listFreq3:
        formattedFreq3.append(int(float(item*(len(dftSampledList)))))
    
    datafile.write(f"Frequencies of sampled coordinates in the Fourier domain: \n")
    datafile.write(f"{str(formattedFreq3)[1:-1]} \n")
    datafile.write(f"Total: {sum(formattedFreq3)} \n")
    perc1 = round((100)*(sum(formattedFreq3))/(sum(dftIndexTracker)), 1)
    datafile.write(f"Percentage of sampled coordinates between 0 and 1: {perc1}% \n\n")

    plt.subplot(2, 2, 4)
    (freq4, bins4, patches) = plt.hist(dftDebiasedList, weights = np.ones(len(dftDebiasedList)) / len(dftDebiasedList),\
        bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],\
            alpha = 0.5, histtype = 'bar', color = 'b', edgecolor = 'k')

    plt.xlim(0, 1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set(title = 'Histogram of returned coordinates in the Fourier domain', xlabel = 'Value', ylabel = 'Frequency')

    listFreq4 = freq4.tolist(); formattedFreq4 = list()
    for item in listFreq4:
        formattedFreq4.append(int(float(item*(len(dftDebiasedList)))))

    datafile.write(f"Frequencies of returned coordinates in the Fourier domain: \n")
    datafile.write(f"{str(formattedFreq4)[1:-1]} \n")
    datafile.write(f"Total: {sum(formattedFreq4)} \n")
    perc2 = round((100)*(sum(formattedFreq4))/(sum(dftIndexTracker)), 1)
    datafile.write(f"Percentage of returned coordinates between 0 and 1: {perc2}% \n\n")

    for a, b in zip(formattedFreq1, formattedFreq3):
        sampledError += abs(a - b)
    datafile.write(f"Total difference between frequencies of sampled coordinates: {round(sampledError)} \n")
    datafile.write(f"Percentage difference: {round((100)*(sampledError)/(max(sum(formattedFreq1), sum(formattedFreq3))))}% \n")

    for a, b in zip(formattedFreq2, formattedFreq4):
        returnedError += abs(a - b)
    datafile.write(f"Total difference between frequencies of returned coordinates: {round(returnedError)} \n")
    datafile.write(f"Percentage difference: {round((100)*(returnedError)/(max(sum(formattedFreq2), sum(formattedFreq4))))}% \n\n")

    plt.tight_layout(); mng = plt.get_current_fig_manager(); mng.window.state('zoomed')
    plt.draw()
    
    if s == 5:
        plt.savefig("fourier" + str(m) + "factor0" + str(s) + ".png")
    else:
        plt.savefig("fourier" + str(m) + "factor" + str(s) + ".png")
    
    plt.clf(); plt.cla()

    loopTotal.append(time.perf_counter() - loopTime)
    casetime = round(loopTotal[value]); casemins = math.floor(casetime/60); casehrs = math.floor(casemins/60)
    datafile.write(f"Total time for case m = {m}: {casehrs}h {casemins - (casehrs*60)}m {casetime - (casemins*60)}s")

width = 0.35

plotPerErrors = [a/(10**5) for a in perErrors]
plotRecErrors = [b/(10**5) for b in recErrors]

if s == 5:
    limit1 = 16
    limit2 = math.floor((plotPerErrors[2] + plotRecErrors[2])*10)/10 - 0.2
    limit3 = limit2 + 0.5
    limit4 = math.floor((plotPerErrors[1] + plotRecErrors[1])*10)/10 - 0.1
    limit5 = limit4 + 0.5
    limit6 = math.floor((plotPerErrors[0] + plotRecErrors[0])*10)/10 - 0.2
    limit7 = limit6 + 0.5

    fig = plt.figure()
    bax = brokenaxes(ylims = ((0, limit1), (limit2, limit3), (limit4, limit5), (limit6, limit7)), hspace = .05)

elif s == 10:
    limit1 = 4
    limit2 = math.floor((plotPerErrors[2] + plotRecErrors[2])*10)/10 - 0.2
    limit3 = limit2 + 0.8
    limit4 = math.floor((plotPerErrors[1] + plotRecErrors[1])*10)/10 - 0.1
    limit5 = limit4 + 0.8
    limit6 = math.floor((plotPerErrors[0] + plotRecErrors[0])*10)/10 - 0.2
    limit7 = limit6 + 0.8

    fig = plt.figure()
    bax = brokenaxes(ylims = ((0, limit1), (limit2, limit3), (limit4, limit5), (limit6, limit7)), hspace = .05)

elif s == 15:
    limit1 = math.ceil((plotPerErrors[2] + plotRecErrors[2])*10)/10
    limit2 = math.floor((plotPerErrors[1] + plotRecErrors[1])*10)/10 - 0.1
    limit3 = limit2 + 0.4
    limit4 = math.floor((plotPerErrors[0] + plotRecErrors[0])*10)/10 - 0.2
    limit5 = limit4 + 0.4

    fig = plt.figure()
    bax = brokenaxes(ylims = ((0, limit1), (limit2, limit3), (limit4, limit5)), hspace = .05)

elif s == 20:
    limit1 = 1.3
    limit2 = math.floor((plotPerErrors[1] + plotRecErrors[1])*10)/10 - 0.1
    limit3 = limit2 + 0.3
    limit4 = math.floor((plotPerErrors[0] + plotRecErrors[0])*10)/10 - 0.2
    limit5 = limit4 + 0.3

    fig = plt.figure()
    bax = brokenaxes(ylims = ((0, limit1), (limit2, limit3), (limit4, limit5)), hspace = .05)

else:
    limit1 = 1.1
    limit2 = math.floor((plotPerErrors[1] + plotRecErrors[1])*10)/10 - 0.1
    limit3 = limit2 + 0.4
    limit4 = math.floor((plotPerErrors[0] + plotRecErrors[0])*10)/10 - 0.2
    limit5 = limit4 + 0.4

    fig = plt.figure()
    bax = brokenaxes(ylims = ((0, limit1), (limit2, limit3), (limit4, limit5)), hspace = .05) 

bax.bar(labels, plotPerErrors, width, label = 'Perturbation error', alpha = 0.6, color = 'r', edgecolor = 'k')
bax.bar(labels, plotRecErrors, width, bottom = plotPerErrors, label = 'Reconstruction error', alpha = 0.6, color = 'c', edgecolor = 'k')

bax.ticklabel_format(axis = 'y', style = 'plain')
bax.set_xticks(['1%', '2%', '3%', '4%', '5%', '6%', '7%', '8%', '9%', '10%'])
bax.set_ylabel('Total experimental MSE' + ' ' + 'x' + ' ' + '$10^5$')
bax.set_xlabel('% of Fourier coefficients retained', labelpad = 20)

bax.set_title('Ratio between experimental errors by % of Fourier coefficients retained')
bax.legend()
plt.draw()

if s == 5:
    plt.savefig("errorchartfactor0" + str(s) + ".png")
else:
    plt.savefig("errorchartfactor" + str(s) + ".png")

avgtime = round((sum(loopTotal))/(V)); avgmins = math.floor(avgtime/60); avghrs = math.floor(avgmins/60)
datafile.write(f"\nAverage time for each case: {avghrs}h {avgmins - (avghrs*60)}m {avgtime - (avgmins*60)}s \n")

totaltime = round(time.perf_counter() - startTime); totalmins = math.floor(totaltime/60); totalhrs = math.floor(totalmins/60)
datafile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")

datafile.close()

print("Thank you for using the Shuffle Model for Vectors.")