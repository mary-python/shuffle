import random, math, time; import numpy as np
from scipy.fftpack import rfft, irfft
import matplotlib.pyplot as plt; from matplotlib.ticker import PercentFormatter
from brokenaxes import brokenaxes

startTime = time.perf_counter()

d = 1000; k = 6; n = 100000; eps = 0.1; dta = 0.185; R = 1; V = 10; s = 15; v = 5
gamma = max((((14*k*(math.log(2/dta))))/((n-1)*(eps**2))), (27*k)/((n-1)*eps))

loopTotal = list(); perErrors = list(); recErrors = list(); labels = list()
randomVector = [0]*d; dftRandomVector = [0]*d

for j in range(0, R):

    sampledList = list(); debiasedList = list()
    indexTracker = [0]*d; submittedTotal = [0]*d; totalVector = [0]*d

    print(f"\n Processing repeat {j+1} for the basic optimal summation result.")

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
    print(round(sum(meanSquaredError)))

    for value in range(0, V):

        loopTime = time.perf_counter(); m = (value + 1)*(10)
        dftSampledList = list(); dftDebiasedList = list()
        dftIndexTracker = [0]*m; dftSubmittedTotal = [0]*m; dftTotalVector = [0]*d

        print(f"\n Processing repeat {j+1} for the value m = {m}.")

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
        print(round(sum(dftMeanSquaredError)))

        dftComparisonTuple = tuple(zip(debiasedTotal, dftAverageVector))
        dftComparisonError = [(a - b)**2 for a, b in dftComparisonTuple]
        print(round(sum(dftComparisonError)))

print("Thank you for using the Shuffle Model for Vectors.")