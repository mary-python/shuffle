import random, math, time; import numpy as np; from decimal import *
from scipy.fftpack import rfft, irfft
import matplotlib.pyplot as plt; from matplotlib.ticker import PercentFormatter

startTime = time.perf_counter()
d = 100; k = 6; n = 400000; eps = 0.1; dta = 0.9832; V = 10; R = 3; t = 1

if t == 1:
    gamma = max((((14*k*(math.log(2/dta))))/((n-1)*(eps**2))), (27*k)/((n-1)*eps))
else:
    gamma = (((56*d*k*(math.log(1/dta))*(math.log((2*t)/dta))))/((n-1)*(eps**2)))

loopTotal = list(); perErrors = list(); recErrors = list()
perStandardDeviation = list(); recStandardDeviation = list()
randomVector = [0]*d; dftRandomVector = [0]*d
sampledList = list(); debiasedList = list()
indexTracker = [0]*d; submittedTotal = [0]*d; totalVector = [0]*d
totalMeanSquaredError = 0; sumOfSquares = 0

for r in range(0, R):

    print(f"\n Processing the basic optimal summation result, repeat {r+1}.")
    from progress.bar import FillingSquaresBar
    bar = FillingSquaresBar(max=n, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

    with open("glove.6B.300d.txt", encoding = "utf8") as reader:
        for line in reader:
            tab = line.split()
            offset = len(tab) - d

            for a in range(0, d):
                randomCoord = float(tab[a + offset])
                randomVector[a] = randomCoord

            vectorSum = sum(randomVector)
            normalisedVector = [coord/vectorSum for coord in randomVector]

            for a in range(0, d):
                totalVector[a] += normalisedVector[a]

            for a in range(0, t):
                randomIndex = random.randint(0, d-1)
                sampledPair = (randomIndex, (1.0 + randomVector[randomIndex])/2.0)
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
                debiasedCoord = 2.0*((descaledCoord - (gamma/2))/(1 - gamma))-1.0
                debiasedList.append(debiasedCoord)

            bar.next()
        bar.finish()

    descaledTotal = [idx/k for idx in submittedTotal]
    mergedTracker = tuple(zip(indexTracker, descaledTotal))
    debiasedTotal = [2.0*((z - ((gamma/2)*count))/(1 - gamma)/max(count, 1))-1.0 for count, z in mergedTracker]

    averageVector = [idx/n for idx in totalVector]
    errorTuple = tuple(zip(debiasedTotal, averageVector))
    meanSquaredError = [(a - b)**2 for a, b in errorTuple]
    totalMeanSquaredError += sum(meanSquaredError)

    averageSquares = [idx**2 for idx in averageVector]
    sumOfSquares += sum(averageSquares)

averageMeanSquaredError = totalMeanSquaredError/R
averageSumofSquares = sumOfSquares/R

datafile = open("basic.txt", "w")
datafile.write(f"Case 1: Optimal Summation in the Shuffle Model \n")

comparison = (2*(14**(2/3))*(d**(2/3))*(n**(1/3))*t*(np.log(1/dta))*(np.log(2/dta)))/(((1-gamma)**2)*(eps**(4/3)))/n
datafile.write(f"Theoretical Upper Bound for MSE: {round(comparison, 1)} \n")
datafile.write(f"Experimental MSE: {round(averageMeanSquaredError, 2)} \n")
error1 = round((100)*((averageMeanSquaredError)/comparison), 2)
datafile.write(f"Experimental MSE was {error1}% of the theoretical upper bound for MSE. \n")
datafile.write(f"Sum of squares of average vector: {round(averageSumofSquares, 2)} \n\n")

plt.style.use('seaborn-white'); plt.tight_layout()
plt.subplot(1, 2, 1); plt.subplot(1, 2, 2)
mng = plt.get_current_fig_manager(); mng.window.state('zoomed'); plt.draw()
plt.savefig("basic.png"); plt.clf(); plt.cla()

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
datafile.write(f"Percentage of returned coordinates between 0 and 1: {round((100)*(sum(formattedFreq2))/(sum(indexTracker)))}%")

plt.tight_layout(); mng = plt.get_current_fig_manager(); mng.window.state('zoomed'); plt.draw()
plt.savefig("basic.png"); plt.clf(); plt.cla()

for value in range(0, V):

    loopTime = time.perf_counter(); m = (value + 1)*(int(d/25))
    dftSampledList = list(); dftDebiasedList = list()
    dftIndexTracker = [0]*m; dftSubmittedTotal = [0]*m; dftTotalVector = [0]*d
    totalDftMeanSquaredError = list(); dftSumOfSquares = 0; totalReconstructionError = list()
    sampledError = 0; returnedError = 0

    for r in range(0, R):

        print(f"\n Processing the optimal summation result with DFT for the value m = {m}, repeat {r+1}.")

        from progress.bar import FillingSquaresBar
        bar = FillingSquaresBar(max=n, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

        with open("glove.6B.300d.txt", encoding = "utf8") as reader:
            for line in reader:
                tab = line.split()
                offset = len(tab) - d

                for a in range(0, d):
                    dftRandomCoord = float(tab[a + offset])
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
                dftDebiasedCoord = 2.0*((dftDescaledCoord - (gamma/2))/(1 - gamma))-1.0
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
        totalDftMeanSquaredError.append(sum(dftMeanSquaredError))

        dftAverageSquares = [idx**2 for idx in dftAverageVector]
        dftSumOfSquares += sum(dftAverageSquares)

        exactVector = irfft(rfft(dftAverageVector).tolist()[0:m] + [0]*(d-m)).tolist()
        reconstructionTuple = tuple(zip(exactVector, dftAverageVector))
        reconstructionError = [(a - b)**2 for a, b in reconstructionTuple]
        totalReconstructionError.append(sum(reconstructionError))
    
    averageDftMeanSquaredError = (sum(totalDftMeanSquaredError))/R
    averageDftSumofSquares = dftSumOfSquares/R
    averageReconstructionError = (sum(totalReconstructionError))/R

    differencesMeanSquaredError = [(value - averageDftMeanSquaredError)**2 for value in totalDftMeanSquaredError]
    differencesReconstructionError = [(value - averageReconstructionError)**2 for value in totalReconstructionError]
    standardDeviationMeanSquaredError = math.sqrt((sum(differencesMeanSquaredError))/R)
    standardDeviationReconstructionError = math.sqrt((sum(differencesReconstructionError))/R)

    datafile = open("fourier" + str(m) + ".txt", "w")
    datafile.write(f"Number of Fourier coefficients m: {m} \n")
    datafile.write(f"Case 2: Fourier Summation Algorithm \n")

    dftComparison = (2*(14**(2/3))*(m**(2/3))*(n**(1/3))*t*(np.log(1/dta))*(np.log(2/dta)))/(((1-gamma)**2)*(eps**(4/3)))/n
    datafile.write(f"Theoretical upper bound for perturbation error: {round(dftComparison, 2)} \n")
    datafile.write(f"Experimental perturbation error: {round(averageDftMeanSquaredError, 2)} \n")
    error2 = round((100)*((averageDftMeanSquaredError)/dftComparison))
    datafile.write(f"Experimental perturbation error was {error2}% of the theoretical upper bound for perturbation error. \n")
    datafile.write(f"Standard deviation of perturbation error: {round(standardDeviationMeanSquaredError, 2)} \n")
    datafile.write(f"Experimental reconstruction error: {round(averageReconstructionError, 2)} \n")

    perErrors.append(Decimal(averageDftMeanSquaredError))
    recErrors.append(Decimal(averageReconstructionError))
    perStandardDeviation.append(Decimal(standardDeviationMeanSquaredError))
    recStandardDeviation.append(Decimal(standardDeviationReconstructionError))

    datafile.write(f"Total experimental MSE: {round((averageDftMeanSquaredError) + (averageReconstructionError), 2)} \n")
    error3 = round((100)*((averageReconstructionError)/((averageDftMeanSquaredError) + (averageReconstructionError))), 1)
    datafile.write(f"Reconstruction error was {error3}% of the total experimental MSE. \n")
    datafile.write(f"Standard deviation of reconstruction error: {round(standardDeviationReconstructionError, 2)} \n")
    datafile.write(f"Sum of squares of average vector: {round(averageDftSumofSquares, 2)} \n\n")

    plt.style.use('seaborn-white'); plt.tight_layout()
    plt.subplot(1, 2, 1); plt.subplot(1, 2, 2)
    mng = plt.get_current_fig_manager(); mng.window.state('zoomed'); plt.draw()
    plt.savefig("fourier" + str(m) + ".png"); plt.clf(); plt.cla()

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
    plt.savefig("fourier" + str(m) + ".png")
    
    plt.clf(); plt.cla()

    loopTotal.append(time.perf_counter() - loopTime)
    casetime = round(loopTotal[value]); casemins = math.floor(casetime/60)
    datafile.write(f"Total time for case m = {m}: {casemins}m {casetime - (casemins*60)}s")

errorfile = open("errortemp.txt", "w")

for value in range(0, V):
    if value != (V - 1):
        errorfile.write(f"{4*(value + 1)} {perErrors[value]} {recErrors[value]} {perStandardDeviation[value]} {recStandardDeviation[value]} \n")
    else:
        errorfile.write(f"{4*(value + 1)} {perErrors[value]} {recErrors[value]} {perStandardDeviation[value]} {recStandardDeviation[value]}")

errorfile.close()

avgtime = round((sum(loopTotal))/(V)); avgmins = math.floor(avgtime/60)
datafile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
totaltime = round(time.perf_counter() - startTime); totalmins = math.floor(totaltime/60)
datafile.write(f"Total time elapsed: {totalmins}m {totaltime - (totalmins*60)}s")
datafile.close()
print("Thank you for using the Shuffle Model for Vectors.")