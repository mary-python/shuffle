import random, math, time
import numpy as np
from decimal import *
from scipy.fftpack import rfft, irfft
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FixedLocator, FixedFormatter

# initialization
random.seed(2196018)
np.random.seed(2196018)
startTime = time.perf_counter()

d = 100
k = 7
n = 400000
eps = 0.1
dta = 0.9832
V = 10
R = 3
t = 2

if t == 1:
    gamma = max((((14*k*(math.log(2/dta))))/((n-1)*(eps**2))), (27*k)/((n-1)*eps))
else:
    gamma = (((56*d*k*(math.log(1/dta))*(math.log((2*t)/dta))))/((n-1)*(eps**2)))

# set up some global variables
gloveData = np.zeros((n,d))
totalVector = np.zeros(d)
dftGloveData = np.zeros((n,d))
dftTotalVector = np.zeros(d)

def readBasicData():

    global gloveData
    global totalVector
    clippedVector = np.zeros(d)
    gloveDimension = 300
    scalingFactor = 1.5
    rowCount = 0

    with open("glove.6B.300d.txt", encoding = "utf8") as reader:

        print(f"\n Reading in the data file...")
        from progress.bar import FillingSquaresBar
        bar = FillingSquaresBar(max=n-1, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

        for line in reader:
            tab = line.split()
            offset = len(tab) - gloveDimension

            for a in range(0, d):
                newCoord = (float(tab[a + offset]))/scalingFactor
                
                if newCoord > 1:
                    clippedVector[a] = 1
                elif newCoord < -1:
                    clippedVector[a] = -1
                else:
                    clippedVector[a] = newCoord

            totalVector += clippedVector
            gloveData[rowCount] = clippedVector
            rowCount += 1

            if rowCount >= n:
                break

            bar.next()
        bar.finish()

def readDftData():

    global dftGloveData
    global dftTotalVector
    dftClippedVector = np.zeros(d)
    gloveDimension = 300
    dftScalingFactor = 15
    rowCount = 0

    with open("glove.6B.300d.txt", encoding = "utf8") as reader:

        print(f"\n Reading in the data file...")
        from progress.bar import FillingSquaresBar
        bar = FillingSquaresBar(max=n-1, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

        for line in reader:
            tab = line.split()
            offset = len(tab) - gloveDimension

            for a in range(0, d):
                # different scaling to basic case
                newCoord = (float(tab[a + offset]))/dftScalingFactor
                
                if newCoord > 1:
                    dftClippedVector[a] = 1
                elif newCoord < -1:
                    dftClippedVector[a] = -1
                else:
                    dftClippedVector[a] = newCoord

            dftTotalVector += dftClippedVector
            dftGloveData[rowCount] = dftClippedVector
            rowCount += 1

            if rowCount >= n:
                break

            bar.next()
        bar.finish()

def runBasic(R):

    # global glove data
    global totalVector
    numBuckets = 48
    inputVector = [0]*(numBuckets)
    outputVector = [0]*(numBuckets)
    indexTracker = [0]*d
    submittedVector = [0]*d
    checkLength = 10
    totalMeanSquaredError = 0
    sumOfSquares = 0

    for r in range(0, R):

        print(f"\n Processing the basic optimal summation result, repeat {r+1}.")
        from progress.bar import FillingSquaresBar
        bar = FillingSquaresBar(max=n, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

        randomIndices = list()
        randomisedResponse = list()
        submittedCoords = list()
        outputList = list()

        for clippedVector in gloveData:
    
            for a in range(0, t):
                randomIndex = random.randint(0, d-1)

                if len(randomIndices) < checkLength:
                    randomIndices.append(randomIndex)

                sampledCoord = (1 + clippedVector[randomIndex])/2
                roundedCoord = math.floor(sampledCoord*k) + np.random.binomial(1, sampledCoord*k - math.floor(sampledCoord*k))
                b = np.random.binomial(1, gamma)

                if b == 0:
                    submittedCoord = roundedCoord
                else:
                    submittedCoord = np.random.randint(0, k+1)

                submittedVector[randomIndex] += submittedCoord
                indexTracker[randomIndex] += 1
                
                if len(randomisedResponse) < checkLength:
                    randomisedResponse.append(b)
                if len(submittedCoords) < checkLength:
                    submittedCoords.append(submittedCoord)

            bar.next()
        bar.finish()

        print(f"\n{randomIndices}")
        print(f"{randomisedResponse}")
        print(f"{submittedCoords}")
    
        averageVector = [idx/n for idx in totalVector]

        maxInput = max(averageVector)
        minInput = min(averageVector) 
        print(f"{maxInput}")
        print(f"{minInput}")

        # generating statistics for the true average vectors
        for vector in averageVector:
            inputBucketCoord = int((numBuckets/4) + math.floor((5/3)*numBuckets*vector))
            inputVector[min(inputBucketCoord, numBuckets - 1)] += 1
            print(f"{vector}")
            print(f"{inputBucketCoord}")
            print(f"{inputVector}")

        descaledVector = [idx/k for idx in submittedVector]
        mergedTracker = tuple(zip(indexTracker, descaledVector))
        debiasedVector = [2*((z - ((gamma/2)*count))/(1 - gamma)/max(count, 1))-1 for count, z in mergedTracker]

        maxOutput = max(debiasedVector)
        minOutput = min(debiasedVector) 
        print(f"{maxOutput}")
        print(f"{minOutput}")

        # generating statistics for the reconstructed unbiased vectors
        for vector in debiasedVector:
            outputBucketCoord = int((numBuckets/4) + math.floor((5/3)*numBuckets*vector))
            outputVector[min(outputBucketCoord, numBuckets - 1)] += 1

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
    datafile.write(f"Sum of squares of average vector: {round(averageSumOfSquares, 3)} \n\n")

    plt.style.use('seaborn-white')
    plt.tight_layout()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 2)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.draw()
    plt.savefig("basic.png")
    plt.clf()
    plt.cla()

    plt.subplot(1, 2, 1)
    inputBarIntervals = ['-0.15 to -0.1375', '-0.1375 to -0.125', '-0.125 to -0.1125', '-0.1125 to -0.1', '-0.1 to -0.0875', '-0.0875 to -0.075', '-0.075 to -0.0625', '-0.0625 to -0.05', '-0.05 to -0.0375', '-0.0375 to -0.025', '-0.025 to -0.0125', '-0.0125 to 0', '0 to 0.0125', '0.0125 to 0.025', '0.025 to 0.0375', '0.0375 to 0.05', '0.05 to 0.0625', '0.0625 to 0.075', '0.075 to 0.0875', '0.0875 to 0.1', '0.1 to 0.1125', '0.1125 to 0.125', '0.125 to 0.1375', '0.1375 to 0.15', '0.15 to 0.1625', '0.1625 to 0.175', '0.175 to 0.1875', '0.1875 to 0.2', '0.2 to 0.2125', '0.2125 to 0.225', '0.225 to 0.2375', '0.2375 to 0.25', '0.25 to 0.2625', '0.2625 to 0.275', '0.275 to 0.2875', '0.2875 to 0.3', '0.3 to 0.3125', '0.3125 to 0.325', '0.325 to 0.3375', '0.3375 to 0.35', '0.35 to 0.3625', '0.3625 to 0.375', '0.375 to 0.3875', '0.3875 to 0.4', '0.4 to 0.4125', '0.4125 to 0.425', '0.425 to 0.4375', '0.4375 to 0.45']
    inputVectorSum = sum(inputVector)
    percentageInputVector = [coord/inputVectorSum for coord in inputVector]
    plt.bar(inputBarIntervals, percentageInputVector, width = 1, align = 'edge', alpha = 0.4, color = 'g', edgecolor = 'k')
    plt.tick_params(length = 3)

    selectiveInputFormatter = FixedFormatter(["-0.15", "-0.1", "-0.05", "0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45"])
    selectiveInputLocator = FixedLocator([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48])
    plt.gca().xaxis.set_major_formatter(selectiveInputFormatter)
    plt.gca().xaxis.set_major_locator(selectiveInputLocator)

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set(title = 'Histogram of true average vectors in the basic case', xlabel = 'Value', ylabel = 'Frequency')

    datafile.write(f"Frequencies of true average vectors in the basic case: \n")
    datafile.write(f"{str(inputVector)[1:-1]} \n")
    datafile.write(f"Total: {inputVectorSum} \n\n")

    plt.subplot(1, 2, 2)
    outputBarIntervals = ['-0.15 to -0.1375', '-0.1375 to -0.125', '-0.125 to -0.1125', '-0.1125 to -0.1', '-0.1 to -0.0875', '-0.0875 to -0.075', '-0.075 to -0.0625', '-0.0625 to -0.05', '-0.05 to -0.0375', '-0.0375 to -0.025', '-0.025 to -0.0125', '-0.0125 to 0', '0 to 0.0125', '0.0125 to 0.025', '0.025 to 0.0375', '0.0375 to 0.05', '0.05 to 0.0625', '0.0625 to 0.075', '0.075 to 0.0875', '0.0875 to 0.1', '0.1 to 0.1125', '0.1125 to 0.125', '0.125 to 0.1375', '0.1375 to 0.15', '0.15 to 0.1625', '0.1625 to 0.175', '0.175 to 0.1875', '0.1875 to 0.2', '0.2 to 0.2125', '0.2125 to 0.225', '0.225 to 0.2375', '0.2375 to 0.25', '0.25 to 0.2625', '0.2625 to 0.275', '0.275 to 0.2875', '0.2875 to 0.3', '0.3 to 0.3125', '0.3125 to 0.325', '0.325 to 0.3375', '0.3375 to 0.35', '0.35 to 0.3625', '0.3625 to 0.375', '0.375 to 0.3875', '0.3875 to 0.4', '0.4 to 0.4125', '0.4125 to 0.425', '0.425 to 0.4375', '0.4375 to 0.45']
    outputVectorSum = sum(outputVector)
    percentageOutputVector = [coord/outputVectorSum for coord in outputVector]
    plt.bar(outputBarIntervals, percentageOutputVector, width = 1, align = 'edge', alpha = 0.4, color = 'b', edgecolor = 'k')
    plt.tick_params(length = 3)

    selectiveOutputFormatter = FixedFormatter(["-0.15", "-0.1", "-0.05", "0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45"])
    selectiveOutputLocator = FixedLocator([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48])
    plt.gca().xaxis.set_major_formatter(selectiveOutputFormatter)
    plt.gca().xaxis.set_major_locator(selectiveOutputLocator)

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set(title = 'Histogram of reconstructed unbiased vectors in the basic case', xlabel = 'Value', ylabel = 'Frequency')

    datafile.write(f"Frequencies of reconstructed unbiased vectors in the basic case: \n")
    datafile.write(f"{str(outputVector)[1:-1]} \n")
    datafile.write(f"Total: {outputVectorSum} \n")

    plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.draw()
    plt.savefig("basic.png")
    plt.clf()
    plt.cla()


def runDft(R,V):  
    perErrors = list()
    recErrors = list()
    totalErrors = list()
    totalStandardDeviation = list()
    loopTotal = list()

    for value in range(0, V):

        loopTime = time.perf_counter()
        m = (value + 1)*(int(d/25))
        numBuckets = 48
        dftInputVector = [0]*(numBuckets)
        dftOutputVector = [0]*(numBuckets)
        dftDebiasedVector = list()
        totalReconstructionError = list()
        totalPerturbationError = list()
        totalDftMeanSquaredError = list()
        dftIndexTracker = [0]*m
        dftSubmittedVector = [0]*m
        checkLength = 10
        dftSumOfSquares = 0
        sampledError = 0
        returnedError = 0

        for r in range(0, R):

            print(f"\n Processing the optimal summation result with DFT for the value m = {m}, repeat {r+1}.")
            from progress.bar import FillingSquaresBar
            bar = FillingSquaresBar(max=n, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

            dftRandomIndices = list()
            dftRandomisedResponse = list()
            dftSubmittedCoords = list()

            for dftClippedVector in dftGloveData:
            
                dftVector = (rfft(dftClippedVector)).tolist()
                
                for a in range(0, t):
                    dftRandomIndex = random.randint(0, m-1)

                    if len(dftRandomIndices) < checkLength:
                        dftRandomIndices.append(dftRandomIndex)

                    dftSampledCoord = (1 + dftVector[dftRandomIndex])/2
                    dftRoundedCoord = math.floor(dftSampledCoord*k) + np.random.binomial(1, dftSampledCoord*k - math.floor(dftSampledCoord*k))
                    b = np.random.binomial(1, gamma)

                    if b == 0:
                        dftSubmittedCoord = dftRoundedCoord
                    else:
                        dftSubmittedCoord = np.random.randint(0, k+1)

                    dftSubmittedVector[dftRandomIndex] += dftSubmittedCoord
                    dftIndexTracker[dftRandomIndex] += 1

                    if len(dftRandomisedResponse) < checkLength:
                        dftRandomisedResponse.append(b)
                    if len(dftSubmittedCoords) < checkLength:
                        dftSubmittedCoords.append(dftSubmittedCoord)
    
                bar.next()
            bar.finish()

            print(f"\n{dftRandomIndices}")
            print(f"{dftRandomisedResponse}")
            print(f"{dftSubmittedCoords}")

            dftAverageVector = [idx/n for idx in dftTotalVector]

            dftMaxInput = max(dftAverageVector)
            dftMinInput = min(dftAverageVector) 
            print(f"{dftMaxInput}")
            print(f"{dftMinInput}")  

            for vector in dftAverageVector:
                dftInputBucketCoord = int((numBuckets/4) + math.floor(10*(5/3)*numBuckets*vector))
                dftInputVector[min(dftInputBucketCoord, numBuckets - 1)] += 1
                print(f"{vector}")
                print(f"{dftInputBucketCoord}")
                print(f"{dftInputVector}")

            dftDescaledVector = [idx/k for idx in dftSubmittedVector]
            dftMergedTracker = tuple(zip(dftIndexTracker, dftDescaledVector))
            dftDebiasedVector = [2*((z - ((gamma/2)*count))/(1 - gamma)/max(count, 1))-1 for count, z in dftMergedTracker]
            paddedVector = dftDebiasedVector + [0]*(d-m)
            finalVector = (irfft(paddedVector, d)).tolist()

            dftMaxOutput = max(finalVector)
            dftMinOutput = min(finalVector) 
            print(f"{dftMaxOutput}")
            print(f"{dftMinOutput}")

            for vector in finalVector:
                dftOutputBucketCoord = int(((numBuckets/2) - 5) + math.floor(25*(5/3)*numBuckets*vector))
                dftOutputVector[min(dftOutputBucketCoord, numBuckets - 1)] += 1
      
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

        plt.style.use('seaborn-white')
        plt.tight_layout()
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 2)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.draw()
        plt.savefig("fourier" + str(m) + ".png")
        plt.clf()
        plt.cla()

        plt.subplot(1, 2, 1)
        dftInputBarIntervals = ['-0.015 to -0.01375', '-0.01375 to -0.0125', '-0.0125 to -0.01125', '-0.01125 to -0.01', '-0.01 to -0.00875', '-0.00875 to -0.0075', '-0.0075 to -0.00625', '-0.00625 to -0.005', '-0.005 to -0.00375', '-0.00375 to -0.0025', '-0.0025 to -0.00125', '-0.00125 to 0', '0 to 0.00125', '0.00125 to 0.0025', '0.0025 to 0.00375', '0.00375 to 0.005', '0.005 to 0.00625', '0.00625 to 0.0075', '0.0075 to 0.00875', '0.00875 to 0.01', '0.01 to 0.01125', '0.01125 to 0.0125', '0.0125 to 0.01375', '0.01375 to 0.015', '0.015 to 0.01625', '0.01625 to 0.0175', '0.0175 to 0.01875', '0.01875 to 0.02', '0.02 to 0.02125', '0.02125 to 0.0225', '0.0225 to 0.02375', '0.02375 to 0.025', '0.025 to 0.02625', '0.02625 to 0.0275', '0.0275 to 0.02875', '0.02875 to 0.03', '0.03 to 0.03125', '0.03125 to 0.0325', '0.0325 to 0.03375', '0.03375 to 0.035', '0.035 to 0.03625', '0.03625 to 0.0375', '0.0375 to 0.03875', '0.03875 to 0.04', '0.04 to 0.04125', '0.04125 to 0.0425', '0.0425 to 0.04375', '0.04375 to 0.045']
        dftInputVectorSum = sum(dftInputVector)
        dftPercentageInputVector = [coord/dftInputVectorSum for coord in dftInputVector]
        plt.bar(dftInputBarIntervals, dftPercentageInputVector, width = 1, align = 'edge', alpha = 0.4, color = 'g', edgecolor = 'k')
        plt.tick_params(length = 3)

        selectiveDftInputFormatter = FixedFormatter(["-0.015", "-0.01", "-0.005", "0", "0.005", "0.01", "0.015", "0.02", "0.025", "0.03", "0.035", "0.04", "0.045"])
        selectiveDftInputLocator = FixedLocator([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48])
        plt.gca().xaxis.set_major_formatter(selectiveDftInputFormatter)
        plt.gca().xaxis.set_major_locator(selectiveDftInputLocator)

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.gca().set(title = 'Histogram of true average vectors in the Fourier case', xlabel = 'Value', ylabel = 'Frequency')

        datafile.write(f"Frequencies of true average vectors in the Fourier case: \n")
        datafile.write(f"{str(dftInputVector)[1:-1]} \n")
        datafile.write(f"Total: {dftInputVectorSum} \n\n")

        plt.subplot(1, 2, 2)
        dftOutputBarIntervals = ['-0.0095 to -0.009', '-0.009 to -0.0085', '-0.0085 to -0.008', '-0.008 to -0.0075', '-0.0075 to -0.007', '-0.007 to -0.0065', '-0.0065 to -0.006', '-0.006 to -0.0055', '-0.0055 to -0.005', '-0.005 to -0.0045', '-0.0045 to -0.004', '-0.004 to -0.0035', '-0.0035 to -0.003', '-0.003 to -0.0025', '-0.0025 to -0.002', '-0.002 to -0.0015', '-0.0015 to -0.001', '-0.001 to -0.0005', '-0.0005 to 0', '0 to 0.0005', '0.0005 to 0.001', '0.001 to 0.0015', '0.0015 to 0.002', '0.002 to 0.0025', '0.0025 to 0.003', '0.003 to 0.0035', '0.0035 to 0.004', '0.004 to 0.0045', '0.0045 to 0.005', '0.005 to 0.0055', '0.0055 to 0.006', '0.006 to 0.0065', '0.0065 to 0.007', '0.007 to 0.0075', '0.0075 to 0.008', '0.008 to 0.0085', '0.0085 to 0.009', '0.009 to 0.0095', '0.0095 to 0.01', '0.01 to 0.0105', '0.0105 to 0.011', '0.011 to 0.0115', '0.0115 to 0.012', '0.012 to 0.0125', '0.0125 to 0.013', '0.013 to 0.0135', '0.0135 to 0.014', '0.014 to 0.0145']
        dftOutputVectorSum = sum(dftOutputVector)
        dftPercentageOutputVector = [coord/dftOutputVectorSum for coord in dftOutputVector]
        plt.bar(dftOutputBarIntervals, dftPercentageOutputVector, width = 1, align = 'edge', alpha = 0.4, color = 'b', edgecolor = 'k')
        plt.tick_params(length = 3)

        selectiveDftOutputFormatter = FixedFormatter(["-0.0095", "-0.0075", "-0.0055", "-0.0035", "-0.0015", "0.0005", "0.0025", "0.0045", "0.0065", "0.0085", "0.0105", "0.0125", "0.0145"])
        selectiveDftOutputLocator = FixedLocator([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48])
        plt.gca().xaxis.set_major_formatter(selectiveDftOutputFormatter)
        plt.gca().xaxis.set_major_locator(selectiveDftOutputLocator)

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.gca().set(title = 'Histogram of reconstructed unbiased vectors in the Fourier case', xlabel = 'Value', ylabel = 'Frequency')

        datafile.write(f"Frequencies of reconstructed unbiased vectors in the Fourier case: \n")
        datafile.write(f"{str(dftOutputVector)[1:-1]} \n")
        datafile.write(f"Total: {dftOutputVectorSum} \n\n")

        plt.tight_layout()
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.draw()
        plt.savefig("fourier" + str(m) + ".png")
        plt.clf()
        plt.cla()

        loopTotal.append(time.perf_counter() - loopTime)
        casetime = round(loopTotal[value])
        casemins = math.floor(casetime/60)
        datafile.write(f"Total time for case m = {m}: {casemins}m {casetime - (casemins*60)}s")

    errorfile = open("errortemp.txt", "w")

    for value in range(0, V):
        if value != (V - 1):
            errorfile.write(f"{4*(value + 1)} {perErrors[value]} {recErrors[value]} {totalErrors[value]} {totalStandardDeviation[value]} \n")
        else:
            errorfile.write(f"{4*(value + 1)} {perErrors[value]} {recErrors[value]} {totalErrors[value]} {totalStandardDeviation[value]}")

    errorfile.close()

    avgtime = round((sum(loopTotal))/(V))
    avgmins = math.floor(avgtime/60)
    datafile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
    totaltime = round(time.perf_counter() - startTime)
    totalmins = math.floor(totaltime/60)
    totalhrs = math.floor(totalmins/60)
    datafile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
    datafile.close()

# gamma is probability of reporting a false value
print(f"gamma = {gamma}") 
readBasicData()
runBasic(R)
readDftData()
runDft(R,V)
print("Thank you for using the Shuffle Model for Vectors.")