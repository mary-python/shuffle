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
debiasedTotal = [((z/(max(1, count))) - (gamma/2))/(1 - gamma) for count, z in mergedTracker]

averageVector = [idx/n for idx in totalVector]
errorTuple = tuple(zip(debiasedTotal, averageVector))
meanSquaredError = [(a - b)**2 for a, b in errorTuple]
totalMeanSquaredError = sum(meanSquaredError)

averageSquares = [idx**2 for idx in averageVector]
sumOfSquares = sum(averageSquares)

if s == 5:
    datafile = open("basicfactor0" + str(s) + ".txt", "w")
else:
    datafile = open("basicfactor" + str(s) + ".txt", "w")

datafile.write(f"Case 1: Optimal Summation in the Shuffle Model \n")

comparison = (2*(14**(2/3))*(d**(2/3))*(n**(1/3))*(np.log(1/dta))*(np.log(2/dta)))/(((1-gamma)**2)*(eps**(4/3)))/n
datafile.write(f"Theoretical Upper Bound for MSE: {round(comparison, 1)} \n")
datafile.write(f"Experimental MSE: {round(totalMeanSquaredError, 2)} \n")
error1 = round((100)*((totalMeanSquaredError)/comparison), 2)
datafile.write(f"Experimental MSE was {error1}% of the theoretical upper bound for MSE. \n")
datafile.write(f"Sum of squares of average vector: {round(sumOfSquares, 2)} \n\n")

plt.style.use('seaborn-white'); plt.tight_layout()
plt.subplot(1, 2, 1); plt.subplot(1, 2, 2)
mng = plt.get_current_fig_manager(); mng.window.state('zoomed'); plt.draw()
    
if s == 5:
    plt.savefig("basicfactor0" + str(s) + ".png")
else:
    plt.savefig("basicfactor" + str(s) + ".png")
    
plt.clf(); plt.cla()

plt.subplot(1, 2, 1)
(freq1, bins1, patches) = plt.hist(sampledList, weights = np.ones(len(sampledList)) / len(sampledList),\
    bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],\
        alpha = 0.4, histtype = 'bar', color = 'g', edgecolor = 'k')

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

plt.subplot(1, 2, 2)
(freq2, bins2, patches) = plt.hist(debiasedList, weights = np.ones(len(debiasedList)) / len(debiasedList),\
    bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],\
        alpha = 0.4, histtype = 'bar', color = 'b', edgecolor = 'k')
    
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

plt.tight_layout(); mng = plt.get_current_fig_manager(); mng.window.state('zoomed'); plt.draw()
    
if s == 5:
    plt.savefig("basicfactor0" + str(s) + ".png")
else:
    plt.savefig("basicfactor" + str(s) + ".png")
    
plt.clf(); plt.cla()

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

        dftVectorSum = sum(dftRandomVector)
        dftNormalisedVector = [coord/dftVectorSum for coord in dftRandomVector]

        for a in range(0, d):
            dftTotalVector[a] += dftNormalisedVector[a]

        dftVector = (rfft(dftNormalisedVector)).tolist()
        slicedDftVector = dftVector[0:m]

        dftRandomIndex = random.randint(0, m-1)
        dftSampledPair = (dftRandomIndex, (1.0 + slicedDftVector[dftRandomIndex])/2.0)

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
    dftDebiasedTotal = [2.0*((z - ((gamma/2)*count))/(1 - gamma)/max(count, 1))-1.0 for count, z in dftMergedTracker]
    paddedTotal = dftDebiasedTotal + [0]*(d-m)
    paddedTotal[0] = 1.0
    finalTotal = (irfft(paddedTotal)).tolist()

    dftAverageVector = [idx/n for idx in dftTotalVector]        
    dftErrorTuple = tuple(zip(finalTotal, dftAverageVector))
    dftMeanSquaredError = [(a - b)**2 for a, b in dftErrorTuple]
    totalDftMeanSquaredError = sum(dftMeanSquaredError)

    dftAverageSquares = [idx**2 for idx in dftAverageVector]
    dftSumOfSquares = sum(dftAverageSquares)

    exactVector = irfft(rfft(dftAverageVector).tolist()[0:m] + [0]*(d-m)).tolist()
    reconstructionTuple = tuple(zip(exactVector, dftAverageVector))
    reconstructionError = [(a - b)**2 for a, b in reconstructionTuple]
    totalReconstructionError = sum(reconstructionError)
    
    if s == 5:
        datafile = open("fourier" + str(m) + "factor0" + str(s) + ".txt", "w")
    else:
        datafile = open("fourier" + str(m) + "factor" + str(s) + ".txt", "w")

    datafile.write(f"Number of Fourier coefficients m: {m} \n")
    datafile.write(f"Case 2: Fourier Summation Algorithm \n")

    dftComparison = (2*(14**(2/3))*(m**(2/3))*(n**(1/3))*(np.log(1/dta))*(np.log(2/dta)))/(((1-gamma)**2)*(eps**(4/3)))/n
    datafile.write(f"Theoretical upper bound for perturbation error: {round(dftComparison, 2)} \n")
    datafile.write(f"Experimental perturbation error: {round(totalDftMeanSquaredError, 6)} \n")
    error2 = round((100)*((totalDftMeanSquaredError)/dftComparison), 6)
    datafile.write(f"Experimental perturbation error was {error2}% of the theoretical upper bound for perturbation error. \n")
    datafile.write(f"Experimental reconstruction error: {round(totalReconstructionError, 6)} \n")
    
    perErrors.append(totalDftMeanSquaredError)
    recErrors.append(totalReconstructionError)
    labels.append(f'{round(m/10)}%')

    datafile.write(f"Total experimental MSE: {round((totalDftMeanSquaredError) + (totalReconstructionError), 6)} \n")
    error3 = round((100)*((totalReconstructionError)/((totalDftMeanSquaredError) + (totalReconstructionError))), 2)
    datafile.write(f"Reconstruction error was {error3}% of the total experimental MSE. \n")
    datafile.write(f"Sum of squares of average vector: {round(dftSumOfSquares, 6)} \n\n")

    plt.style.use('seaborn-white'); plt.tight_layout()
    plt.subplot(1, 2, 1); plt.subplot(1, 2, 2)
    mng = plt.get_current_fig_manager(); mng.window.state('zoomed'); plt.draw()
    
    if s == 5:
        plt.savefig("fourier" + str(m) + "factor0" + str(s) + ".png")
    else:
        plt.savefig("fourier" + str(m) + "factor" + str(s) + ".png")
    
    plt.clf(); plt.cla()

    plt.subplot(1, 2, 1)
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
    perc1 = round((100)*(sum(formattedFreq3))/(sum(dftIndexTracker)))
    datafile.write(f"Percentage of sampled coordinates between 0 and 1: {perc1}% \n\n")

    plt.subplot(1, 2, 2)
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
    perc2 = round((100)*(sum(formattedFreq4))/(sum(dftIndexTracker)))
    datafile.write(f"Percentage of returned coordinates between 0 and 1: {perc2}% \n\n")

    plt.tight_layout(); mng = plt.get_current_fig_manager(); mng.window.state('zoomed'); plt.draw()
    
    if s == 5:
        plt.savefig("fourier" + str(m) + "factor0" + str(s) + ".png")
    else:
        plt.savefig("fourier" + str(m) + "factor" + str(s) + ".png")
    
    plt.clf(); plt.cla()

    loopTotal.append(time.perf_counter() - loopTime)
    casetime = round(loopTotal[value]); casemins = math.floor(casetime/60)
    datafile.write(f"Total time for case m = {m}: {casemins}m {casetime - (casemins*60)}s")

width = 0.35

limit1 = math.ceil((perErrors[2] + recErrors[2])*10)/10
limit2 = math.floor((perErrors[1] + recErrors[1])*10)/10
limit3 = limit2 + 0.1
limit4 = math.floor((perErrors[0] + recErrors[0])*10)/10
limit5 = limit4 + 0.1

fig = plt.figure()
bax = brokenaxes(ylims = ((0, limit1), (limit2, limit3), (limit4, limit5)), hspace = .05)

bax.bar(labels, perErrors, width, label = 'Perturbation error', alpha = 0.6, color = 'r', edgecolor = 'k')
bax.bar(labels, recErrors, width, bottom = perErrors, label = 'Reconstruction error', alpha = 0.6, color = 'c', edgecolor = 'k')

bax.ticklabel_format(axis = 'y', style = 'plain')
bax.set_xticks(['1%', '2%', '3%', '4%', '5%', '6%', '7%', '8%', '9%', '10%'])
bax.set_ylabel('Total experimental MSE')
bax.set_xlabel('% of Fourier coefficients retained', labelpad = 20)

bax.set_title('Ratio between experimental errors by % of Fourier coefficients retained')
bax.legend()
plt.draw()

if s == 5:
    plt.savefig("errorchartfactor0" + str(s) + ".png")
else:
    plt.savefig("errorchartfactor" + str(s) + ".png")

avgtime = round((sum(loopTotal))/(V)); avgmins = math.floor(avgtime/60)
datafile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
totaltime = round(time.perf_counter() - startTime); totalmins = math.floor(totaltime/60)
datafile.write(f"Total time elapsed: {totalmins}m {totaltime - (totalmins*60)}s")
datafile.close()
print("Thank you for using the Shuffle Model for Vectors.")