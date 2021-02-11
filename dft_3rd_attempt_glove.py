import random, math, time; import numpy as np; from decimal import *
from scipy.fftpack import rfft, irfft
import matplotlib.pyplot as plt; from matplotlib.ticker import PercentFormatter

random.seed(2196018)
np.random.seed(2196018)
startTime = time.perf_counter()
d = 200; k = 7; n = 400000; eps = 0.1; dta = 0.9832; V = 10; R = 3; t = 2

if t == 1:
    gamma = max((((14*k*(math.log(2/dta))))/((n-1)*(eps**2))), (27*k)/((n-1)*eps))
else:
    gamma = (((56*d*k*(math.log(1/dta))*(math.log((2*t)/dta))))/((n-1)*(eps**2)))

loopTotal = list(); perErrors = list(); recErrors = list(); totalErrors = list()
totalStandardDeviation = list(); sampledVector = [0]*40; outputVector = [0]*56
randomVector = [0]*d; clippedVector = [0]*d; normalisedDebiasedVector = [0]*d; normalisedFinalVector = [0]*d
indexTracker = [0]*d; submittedVector = [0]*d; totalVector = [0]*d
totalMeanSquaredError = 0; sumOfSquares = 0

for r in range(0, R):

    print(f"\n Processing the basic optimal summation result, repeat {r+1}.")
    from progress.bar import FillingSquaresBar
    bar = FillingSquaresBar(max=n, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

    randomIndices = list(); randomisedResponse = list(); submittedCoords = list(); outputList = list()

    with open("glove.6B.300d.txt", encoding = "utf8") as reader:
        for line in reader:
            tab = line.split()
            offset = len(tab) - 300

            for a in range(0, d):
                randomCoord = float(tab[a + offset])
                randomVector[a] = (2*randomCoord)/3

            for a in range(0, d):
                if randomVector[a] > 1:
                    clippedVector[a] = 1
                elif randomVector[a] < -1:
                    clippedVector[a] = -1
                else:
                    clippedVector[a] = randomVector[a]

                totalVector[a] += clippedVector[a]
                
            positiveVector = [(1 + coord)/2 for coord in clippedVector]

            for a in range(0, t):
                randomIndex = random.randint(0, d-1)

                if len(randomIndices) < 10:
                    randomIndices.append(randomIndex)

                sampledPair = (randomIndex, positiveVector[randomIndex])
                sampledCoord = sampledPair[1]
                adjustedSampledCoord = 2*(sampledCoord)-1

                for coord in range(0, 40):
                    if adjustedSampledCoord >= ((0.05*coord)-1) and adjustedSampledCoord < ((0.05*coord)-0.95):
                        sampledVector[coord] += 1
                    if coord == 39 and adjustedSampledCoord == ((0.05*coord)-0.95):
                        sampledVector[coord] += 1

                roundedPair = (randomIndex, (math.floor(sampledCoord*k)\
                    + np.random.binomial(1, sampledCoord*k - math.floor(sampledCoord*k))))
                b = np.random.binomial(1, gamma)

                if len(randomisedResponse) < 10:
                    randomisedResponse.append(b)

                if b == 0:
                    submittedPair = roundedPair
                else:
                    submittedPair = (randomIndex, (np.random.randint(0, k+1)))

                submittedCoord = submittedPair[1]
                
                if len(submittedCoords) < 10:
                    submittedCoords.append(submittedCoord)

                submittedVector[randomIndex] += submittedCoord
                indexTracker[randomIndex] += 1

            bar.next()
        bar.finish()

    print(f"\n{randomIndices}")
    print(f"{randomisedResponse}")
    print(f"{submittedCoords}")
    
    maxInput = max(sampledVector)
    minInput = min(sampledVector) 
    print(f"{maxInput}")
    print(f"{minInput}")  

    descaledVector = [idx/k for idx in submittedVector]
    mergedTracker = tuple(zip(indexTracker, descaledVector))
    debiasedVector = [2*((z - ((gamma/2)*count))/(1 - gamma)/max(count, 1))-1 for count, z in mergedTracker]
    print(f"\n{len(debiasedVector)}")

    maxOutput = max(debiasedVector)
    minOutput = min(debiasedVector) 
    print(f"{maxOutput}")
    print(f"{minOutput}")

    for vector in debiasedVector:
        for coord in range(0, 56):
            if vector >= ((0.0125*coord)-0.3) and vector < ((0.0125*coord)-0.2875):
                outputVector[coord] += 1
            if coord == 55 and vector == ((0.0125*coord)-0.2875):
                outputVector[coord] += 1
    print(f"\n{outputVector}")

    averageVector = [idx/n for idx in totalVector]
    errorTuple = tuple(zip(debiasedVector, averageVector))
    meanSquaredError = [(a - b)**2 for a, b in errorTuple]
    totalMeanSquaredError += sum(meanSquaredError)

    averageSquares = [idx**2 for idx in averageVector]
    sumOfSquares += sum(averageSquares)

averageMeanSquaredError = totalMeanSquaredError/R
averageSumOfSquares = sumOfSquares/R

datafile = open("basic.txt", "w")
datafile.write(f"Case 1: Optimal Summation in the Shuffle Model \n")

comparison = (2*(14**(2/3))*(d**(2/3))*(n**(1/3))*t*(np.log(1/dta))*(np.log(2/dta)))/(((1-gamma)**2)*(eps**(4/3)))/n
datafile.write(f"Theoretical Upper Bound for MSE: {round(comparison, 3)} \n")
datafile.write(f"Experimental MSE: {round(averageMeanSquaredError, 4)} \n")
error1 = round((100)*((averageMeanSquaredError)/comparison), 1)
datafile.write(f"Experimental MSE was {error1}% of the theoretical upper bound for MSE. \n")
datafile.write(f"Sum of squares of average vector: {round(averageSumOfSquares, 5)} \n\n")

plt.style.use('seaborn-white'); plt.tight_layout()
plt.subplot(1, 2, 1); plt.subplot(1, 2, 2)
mng = plt.get_current_fig_manager(); mng.window.state('zoomed'); plt.draw()
plt.savefig("basic.png"); plt.clf(); plt.cla()

plt.subplot(1, 2, 1)
sampledBarIntervals = ['-1 to -0.95', '-0.95 to -0.9', '-0.9 to -0.85', '-0.85 to -0.8', '-0.8 to -0.75', '-0.75 to -0.7', '-0.7 to -0.65', '-0.65 to -0.6', '-0.6 to -0.55', '-0.55 to -0.5', '-0.5 to -0.45', '-0.45 to -0.4', '-0.4 to -0.35', '-0.35 to -0.3', '-0.3 to -0.25', '-0.25 to -0.2', '-0.2 to -0.15', '-0.15 to -0.1', '-0.1 to -0.05', '-0.05 to 0', '0 to 0.05', '0.05 to 0.1', '0.1 to 0.15', '0.15 to 0.2', '0.2 to 0.25', '0.25 to 0.3', '0.3 to 0.35', '0.35 to 0.4', '0.4 to 0.45', '0.45 to 0.5', '0.5 to 0.55', '0.55 to 0.6', '0.6 to 0.65', '0.65 to 0.7', '0.7 to 0.75', '0.75 to 0.8', '0.8 to 0.85', '0.85 to 0.9', '0.9 to 0.95', '0.95 to 1']
sampledVectorSum = sum(sampledVector)
percentageSampledVector = [coord/sampledVectorSum for coord in sampledVector]
print(f"\n{len(sampledBarIntervals)}")
print(f"{len(percentageSampledVector)}")
plt.bar(sampledBarIntervals, percentageSampledVector, width = 1, align = 'edge', alpha = 0.4, color = 'g', edgecolor = 'k')
plt.xticks(rotation = 45)

print(f"\n{percentageSampledVector}")

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().set(title = 'Histogram of sampled coordinates in the original domain', xlabel = 'Value', ylabel = 'Frequency')

datafile.write(f"Frequencies of sampled coordinates in the original domain: \n")
datafile.write(f"{str(sampledVector)[1:-1]} \n")
datafile.write(f"Total: {sampledVectorSum} \n")

plt.subplot(1, 2, 2)
outputBarIntervals = ['-0.3 to -0.2875', '-0.2875 to -0.275', '-0.275 to -0.2625', '-0.2625 to -0.25', '-0.25 to -0.2375', '-0.2375 to -0.225', '-0.225 to -0.2125', '-0.2125 to -0.2', '-0.2 to -0.1875', '-0.1875 to -0.175', '-0.175 to -0.1625', '-0.1625 to -0.15', '-0.15 to -0.1375', '-0.1375 to -0.125', '-0.125 to -0.1125', '-0.1125 to -0.1', '-0.1 to -0.0875', '-0.0875 to -0.075', '-0.075 to -0.0625', '-0.0625 to -0.05', '-0.05 to -0.375', '-0.375 to -0.25', '-0.25 to -0.125', '-0.125 to 0', '0 to 0.0125', '0.0125 to 0.025', '0.025 to 0.0375', '0.0375 to 0.05', '0.05 to 0.0625', '0.0625 to 0.075', '0.075 to 0.0875', '0.0875 to 0.1', '0.1 to 0.1125', '0.1125 to 0.125', '0.125 to 0.1375', '0.1375 to 0.15', '0.15 to 0.1625', '0.1625 to 0.175', '0.175 to 0.1875', '0.1875 to 0.2', '0.2 to 0.2125', '0.2125 to 0.225', '0.225 to 0.2375', '0.2375 to 0.25', '0.25 to 0.2625', '0.2625 to 0.275', '0.275 to 0.2875', '0.2875 to 0.3', '0.3 to 0.3125', '0.3125 to 0.325', '0.325 to 0.3375', '0.3375 to 0.35', '0.35 to 0.3625', '0.3625 to 0.375', '0.375 to 0.3875', '0.3875 to 0.4']
outputVectorSum = sum(outputVector)
percentageOutputVector = [coord/outputVectorSum for coord in outputVector]
print(f"\n{len(outputBarIntervals)}")
print(f"{len(percentageOutputVector)}")
plt.bar(outputBarIntervals, percentageOutputVector, width = 1, align = 'edge', alpha = 0.4, color = 'b', edgecolor = 'k')
plt.xticks(rotation = 45)

print(f"\n{percentageOutputVector}")

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().set(title = 'Histogram of returned coordinates in the original domain', xlabel = 'Value', ylabel = 'Frequency')

datafile.write(f"Frequencies of returned coordinates in the original domain: \n")
datafile.write(f"{str(outputVector)[1:-1]} \n")
datafile.write(f"Total: {outputVectorSum} \n")

plt.tight_layout(); mng = plt.get_current_fig_manager(); mng.window.state('zoomed'); plt.draw()
plt.savefig("basic.png"); plt.clf(); plt.cla()

dftRandomVector = [0]*d; dftNormalisedVector = [0]*d; dftClippedVector = [0]*d

for value in range(0, V):

    loopTime = time.perf_counter(); m = (value + 1)*(int(d/25))
    dftSampledVector = [0]*40; dftDebiasedVector = list(); dftOutputVector = [0]*47
    totalDftMeanSquaredError = list(); dftSumOfSquares = 0; totalReconstructionError = list(); totalPerturbationError = list()
    dftIndexTracker = [0]*m; dftSubmittedVector = [0]*m; dftTotalVector = [0]*d
    sampledError = 0; returnedError = 0

    for r in range(0, R):

        print(f"\n Processing the optimal summation result with DFT for the value m = {m}, repeat {r+1}.")
        from progress.bar import FillingSquaresBar
        bar = FillingSquaresBar(max=n, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

        dftRandomIndices = list(); dftRandomisedResponse = list(); dftSubmittedCoords = list(); outOfBounds = list()

        with open("glove.6B.300d.txt", encoding = "utf8") as reader:
            for line in reader:
                tab = line.split()
                offset = len(tab) - 300

                for a in range(0, d):
                    dftRandomCoord = float(tab[a + offset])
                    dftRandomVector[a] = dftRandomCoord/15

                for a in range(0, d):
                    if dftRandomVector[a] > 1:
                        dftClippedVector[a] = 1
                    elif dftRandomVector[a] < -1:
                        dftClippedVector[a] = -1
                    else:
                        dftClippedVector[a] = dftRandomVector[a]

                    dftTotalVector[a] += dftClippedVector[a]
 
                dftVector = (rfft(dftClippedVector)).tolist()
                dftPositiveVector = [(1 + coord)/2 for coord in dftVector]
                dftSlicedVector = dftPositiveVector[0:m]
                
                for a in range(0, t):
                    dftRandomIndex = random.randint(0, m-1)

                    if len(dftRandomIndices) < 10:
                        dftRandomIndices.append(dftRandomIndex)

                    dftSampledPair = (dftRandomIndex, dftSlicedVector[dftRandomIndex])
                    dftSampledCoord = dftSampledPair[1]
                    dftAdjustedVector = 2*(dftSampledCoord)-1

                    for coord in range(0, 40):
                        if dftAdjustedVector >= ((0.05*coord)-1) and dftAdjustedVector < ((0.05*coord)-0.95):
                            dftSampledVector[coord] += 1
                        if coord == 39 and dftAdjustedVector == ((0.05*coord)-0.95):
                            dftSampledVector[coord] += 1

                    dftRoundedPair = (dftRandomIndex, (math.floor(dftSampledCoord*k)\
                        + np.random.binomial(1, dftSampledCoord*k - math.floor(dftSampledCoord*k))))
                    b = np.random.binomial(1, gamma)

                    if len(dftRandomisedResponse) < 10:
                        dftRandomisedResponse.append(b)

                    if b == 0:
                        dftSubmittedPair = dftRoundedPair
                    else:
                        dftSubmittedPair = (dftRandomIndex, (np.random.randint(0, k+1)))

                    dftSubmittedCoord = dftSubmittedPair[1]

                    if len(dftSubmittedCoords) < 10:
                        dftSubmittedCoords.append(dftSubmittedCoord)

                    if dftSubmittedCoord > 7 or dftSubmittedCoord < 0:
                        if len(outOfBounds) == 0:
                            outOfBounds.append(dftRandomCoord)
                            outOfBounds.append(dftSlicedVector[dftRandomIndex])
                            outOfBounds.append(dftSampledCoord)
                            outOfBounds.append(dftSampledCoord*k - math.floor(dftSampledCoord*k))
                            outOfBounds.append(dftRoundedPair[1])

                    dftSubmittedVector[dftRandomIndex] += dftSubmittedCoord
                    dftIndexTracker[dftRandomIndex] += 1
    
                bar.next()
            bar.finish()

        print(f"\n{dftRandomIndices}")
        print(f"{dftRandomisedResponse}")
        print(f"{dftSubmittedCoords}")
        print(f"{outOfBounds}")

        dftMaxInput = max(dftSampledVector)
        dftMinInput = min(dftSampledVector) 
        print(f"{dftMaxInput}")
        print(f"{dftMinInput}")  

        dftDescaledVector = [idx/k for idx in dftSubmittedVector]
        dftMergedTracker = tuple(zip(dftIndexTracker, dftDescaledVector))
        dftDebiasedVector = [2*((z - ((gamma/2)*count))/(1 - gamma)/max(count, 1))-1 for count, z in dftMergedTracker]
        paddedVector = dftDebiasedVector + [0]*(d-m)
        paddedVector[0] = 1
        finalVector = (irfft(paddedVector, d)).tolist()
        print(f"\n{len(finalVector)}")

        dftMaxOutput = max(finalVector)
        dftMinOutput = min(finalVector) 
        print(f"{dftMaxOutput}")
        print(f"{dftMinOutput}")

        for vector in finalVector:
            for coord in range(0, 47):
                if vector >= ((0.0005*coord)-0.0055) and vector < ((0.0005*coord)-0.005):
                    dftOutputVector[coord] += 1
                if coord == 46 and vector == ((0.0005*coord)-0.005):
                    dftOutputVector[coord] += 1
        print(f"{dftOutputVector}")

        dftAverageVector = [idx/n for idx in dftTotalVector]        
        dftErrorTuple = tuple(zip(finalVector, dftAverageVector))
        dftMeanSquaredError = [(a - b)**2 for a, b in dftErrorTuple]
        totalDftMeanSquaredError.append(sum(dftMeanSquaredError))

        dftAverageSquares = [idx**2 for idx in dftAverageVector]
        dftSumOfSquares += sum(dftAverageSquares)

        exactVector = irfft(rfft(dftAverageVector).tolist()[0:m] + [0]*(d-m)).tolist()
        reconstructionTuple = tuple(zip(exactVector, dftAverageVector))
        reconstructionError = [(a - b)**2 for a, b in reconstructionTuple]
        totalReconstructionError.append(sum(reconstructionError))
        totalPerturbationError.append((sum(dftMeanSquaredError)) - (sum(reconstructionError)))
    
    averageDftMeanSquaredError = (sum(totalDftMeanSquaredError))/R
    averageDftSumOfSquares = dftSumOfSquares/R
    averageReconstructionError = (sum(totalReconstructionError))/R
    averagePerturbationError = (sum(totalPerturbationError))/R

    differencesMeanSquaredError = [(value - averageDftMeanSquaredError)**2 for value in totalDftMeanSquaredError] 
    differencesReconstructionError = [(value - averageReconstructionError)**2 for value in totalReconstructionError]
    differencesPerturbationError = [(value - averagePerturbationError)**2 for value in totalPerturbationError]
    standardDeviationMeanSquaredError = math.sqrt((sum(differencesMeanSquaredError))/R)
    standardDeviationReconstructionError = math.sqrt((sum(differencesReconstructionError))/R)
    standardDeviationPerturbationError = math.sqrt((sum(differencesPerturbationError))/R)
    
    datafile = open("fourier" + str(m) + ".txt", "w")
    datafile.write(f"Number of Fourier coefficients m: {m} \n")
    datafile.write(f"Case 2: Fourier Summation Algorithm \n")

    dftComparison = (2*(14**(2/3))*(m**(2/3))*(n**(1/3))*t*(np.log(1/dta))*(np.log(2/dta)))/(((1-gamma)**2)*(eps**(4/3)))/n
    datafile.write(f"Theoretical upper bound for perturbation error: {round(dftComparison, 4)} \n")
    datafile.write(f"Experimental perturbation error: {round(averagePerturbationError, 4)} \n")
    error2 = round((100)*((averagePerturbationError)/dftComparison), 1)
    datafile.write(f"Experimental perturbation error was {error2}% of the theoretical upper bound for perturbation error. \n")
    datafile.write(f"Standard deviation of perturbation error: {round(standardDeviationPerturbationError, 5)} \n")
    datafile.write(f"Experimental reconstruction error: {round(averageReconstructionError, 5)} \n")

    perErrors.append(Decimal(averagePerturbationError))
    recErrors.append(Decimal(averageReconstructionError))
    totalErrors.append(Decimal(averageDftMeanSquaredError))
    totalStandardDeviation.append(Decimal(standardDeviationMeanSquaredError))

    datafile.write(f"Total experimental MSE: {round(averageDftMeanSquaredError, 4)} \n")
    error3 = round((100)*((averageReconstructionError)/(averageDftMeanSquaredError)), 1)
    datafile.write(f"Reconstruction error was {error3}% of the total experimental MSE. \n")
    datafile.write(f"Standard deviation of reconstruction error: {round(standardDeviationReconstructionError, 5)} \n")
    datafile.write(f"Sum of squares of average vector: {round(averageDftSumOfSquares, 5)} \n\n")

    plt.style.use('seaborn-white'); plt.tight_layout()
    plt.subplot(1, 2, 1); plt.subplot(1, 2, 2)
    mng = plt.get_current_fig_manager(); mng.window.state('zoomed'); plt.draw()
    plt.savefig("fourier" + str(m) + ".png"); plt.clf(); plt.cla()

    plt.subplot(1, 2, 1)
    dftSampledBarIntervals = ['-1 to -0.95', '-0.95 to -0.9', '-0.9 to -0.85', '-0.85 to -0.8', '-0.8 to -0.75', '-0.75 to -0.7', '-0.7 to -0.65', '-0.65 to -0.6', '-0.6 to -0.55', '-0.55 to -0.5', '-0.5 to -0.45', '-0.45 to -0.4', '-0.4 to -0.35', '-0.35 to -0.3', '-0.3 to -0.25', '-0.25 to -0.2', '-0.2 to -0.15', '-0.15 to -0.1', '-0.1 to -0.05', '-0.05 to 0', '0 to 0.05', '0.05 to 0.1', '0.1 to 0.15', '0.15 to 0.2', '0.2 to 0.25', '0.25 to 0.3', '0.3 to 0.35', '0.35 to 0.4', '0.4 to 0.45', '0.45 to 0.5', '0.5 to 0.55', '0.55 to 0.6', '0.6 to 0.65', '0.65 to 0.7', '0.7 to 0.75', '0.75 to 0.8', '0.8 to 0.85', '0.85 to 0.9', '0.9 to 0.95', '0.95 to 1']
    dftSampledVectorSum = sum(dftSampledVector)
    dftPercentageSampledVector = [coord/dftSampledVectorSum for coord in dftSampledVector]
    print(f"\n{len(dftSampledBarIntervals)}")
    print(f"{len(dftPercentageSampledVector)}")
    plt.bar(dftSampledBarIntervals, dftPercentageSampledVector, width = 1, align = 'edge', alpha = 0.4, color = 'g', edgecolor = 'k')
    plt.xticks(rotation = 45)

    print(f"\n{dftPercentageSampledVector}")

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set(title = 'Histogram of sampled coordinates in the Fourier domain', xlabel = 'Value', ylabel = 'Frequency')

    datafile.write(f"Frequencies of sampled coordinates in the Fourier domain: \n")
    datafile.write(f"{str(dftSampledVector)[1:-1]} \n")
    datafile.write(f"Total: {dftSampledVectorSum} \n")

    plt.subplot(1, 2, 2)
    dftOutputBarIntervals = ['-0.0055 to -0.005', '-0.005 to -0.0045', '-0.0045 to -0.004', '-0.004 to -0.0035', '-0.0035 to -0.003', '-0.003 to -0.0025', '-0.0025 to -0.002', '-0.002 to -0.0015', '-0.0015 to -0.001', '-0.001 to -0.0005', '-0.0005 to 0', '0 to 0.0005', '0.0005 to 0.001', '0.001 to 0.0015', '0.0015 to 0.002', '0.002 to 0.0025', '0.0025 to 0.003', '0.003 to 0.0035', '0.0035 to 0.004', '0.004 to 0.0045', '0.0045 to 0.005', '0.005 to 0.0055', '0.0055 to 0.006', '0.006 to 0.0065', '0.0065 to 0.007', '0.007 to 0.0075', '0.0075 to 0.008', '0.008 to 0.0085', '0.0085 to 0.009', '0.009 to 0.0095', '0.0095 to 0.01', '0.01 to 0.0105', '0.0105 to 0.011', '0.011 to 0.0115', '0.0115 to 0.012', '0.012 to 0.0125', '0.0125 to 0.013', '0.013 to 0.0135', '0.0135 to 0.014', '0.014 to 0.0145', '0.0145 to 0.015', '0.015 to 0.0155', '0.0155 to 0.016', '0.016 to 0.0165', '0.0165 to 0.017', '0.017 to 0.0175', '0.0175 to 0.018']
    dftOutputVectorSum = sum(dftOutputVector)
    dftPercentageOutputVector = [coord/dftOutputVectorSum for coord in dftOutputVector]
    print(f"\n{len(dftOutputBarIntervals)}")
    print(f"{len(dftPercentageOutputVector)}")
    plt.bar(dftOutputBarIntervals, dftPercentageOutputVector, width = 1, align = 'edge', alpha = 0.4, color = 'b', edgecolor = 'k')
    plt.xticks(rotation = 45)

    print(f"\n{dftPercentageOutputVector}")

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set(title = 'Histogram of returned coordinates in the Fourier domain', xlabel = 'Value', ylabel = 'Frequency')

    datafile.write(f"Frequencies of returned coordinates in the Fourier domain: \n")
    datafile.write(f"{str(dftOutputVector)[1:-1]} \n")
    datafile.write(f"Total: {dftOutputVectorSum} \n")

    plt.tight_layout(); mng = plt.get_current_fig_manager(); mng.window.state('zoomed'); plt.draw()
    plt.savefig("fourier" + str(m) + ".png")
    
    plt.clf(); plt.cla()

    loopTotal.append(time.perf_counter() - loopTime)
    casetime = round(loopTotal[value]); casemins = math.floor(casetime/60)
    datafile.write(f"Total time for case m = {m}: {casemins}m {casetime - (casemins*60)}s")

errorfile = open("errortemp.txt", "w")

for value in range(0, V):
    if value != (V - 1):
        errorfile.write(f"{4*(value + 1)} {perErrors[value]} {recErrors[value]} {totalErrors[value]} {totalStandardDeviation[value]} \n")
    else:
        errorfile.write(f"{4*(value + 1)} {perErrors[value]} {recErrors[value]} {totalErrors[value]} {totalStandardDeviation[value]}")

errorfile.close()

avgtime = round((sum(loopTotal))/(V)); avgmins = math.floor(avgtime/60)
datafile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
totaltime = round(time.perf_counter() - startTime); totalmins = math.floor(totaltime/60); totalhrs = math.floor(totalmins/60)
datafile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
datafile.close()
print("Thank you for using the Shuffle Model for Vectors.")