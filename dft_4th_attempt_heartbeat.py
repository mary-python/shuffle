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

d = 150
k = 7
n = 123998
n1 = 21892
n2 = 87554
n3 = 10506
n4 = 4046
eps = 0.1
dta = 0.16
V = 10
R = 3
t = 1

if t == 1:
    gamma = max((((14*k*(math.log(2/dta))))/((n-1)*(eps**2))), (27*k)/((n-1)*eps))
else:
    gamma = (((56*d*k*(math.log(1/dta))*(math.log((2*t)/dta))))/((n-1)*(eps**2)))

# set up some global variables
heartbeatData = np.zeros((n,d))
totalVector = np.zeros(d)

def readTestingData():

    global heartbeatData
    global totalVector
    newVector = np.zeros(d)
    coordCount = 0
    rowCount = 0

    with open("mitbih_test.csv", encoding = "utf8") as reader:

        print(f"\n Reading in the testing data file...")
        from progress.bar import FillingSquaresBar
        bar = FillingSquaresBar(max=n1-1, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

        for line in reader:
            coordCount = 0

            for numString in line.split(","):    
                newCoord = float(numString)

                if newCoord > 1:
                    clippedCoord = 1
                elif newCoord < -1:
                    clippedCoord = -1
                else:
                    clippedCoord = newCoord

                newVector[coordCount] = clippedCoord
                coordCount += 1

                if coordCount >= d:
                    break

            totalVector += newVector
            heartbeatData[rowCount] = newVector
            rowCount += 1

            if rowCount >= n1:
                break

            bar.next()
        bar.finish()

def readTrainingData():

    global heartbeatData
    global totalVector
    newVector = np.zeros(d)
    coordCount = 0
    rowCount = n1

    with open("mitbih_train.csv", encoding = "utf8") as reader:

        print(f"\n Reading in the training data file...")
        from progress.bar import FillingSquaresBar
        bar = FillingSquaresBar(max=n2-1, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

        for line in reader:
            coordCount = 0

            for numString in line.split(","):    
                newCoord = float(numString)

                if newCoord > 1:
                    clippedCoord = 1
                elif newCoord < -1:
                    clippedCoord = -1
                else:
                    clippedCoord = newCoord

                newVector[coordCount] = clippedCoord
                coordCount += 1

                if coordCount >= d:
                    break

            totalVector += newVector
            heartbeatData[rowCount] = newVector
            rowCount += 1

            if rowCount >= n1 + n2:
                break

            bar.next()
        bar.finish()

def readAbnormalData():

    global heartbeatData
    global totalVector
    newVector = np.zeros(d)
    coordCount = 0
    rowCount = n1 + n2

    with open("mitbih_train.csv", encoding = "utf8") as reader:

        print(f"\n Reading in the abnormal data file...")
        from progress.bar import FillingSquaresBar
        bar = FillingSquaresBar(max=n3-1, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

        for line in reader:
            coordCount = 0

            for numString in line.split(","):    
                newCoord = float(numString)

                if newCoord > 1:
                    clippedCoord = 1
                elif newCoord < -1:
                    clippedCoord = -1
                else:
                    clippedCoord = newCoord

                newVector[coordCount] = clippedCoord
                coordCount += 1

                if coordCount >= d:
                    break

            totalVector += newVector
            heartbeatData[rowCount] = newVector
            rowCount += 1

            if rowCount >= n1 + n2 + n3:
                break

            bar.next()
        bar.finish()

def readNormalData():

    global heartbeatData
    global totalVector
    newVector = np.zeros(d)
    coordCount = 0
    rowCount = n1 + n2 + n3

    with open("mitbih_train.csv", encoding = "utf8") as reader:

        print(f"\n Reading in the normal data file...")
        from progress.bar import FillingSquaresBar
        bar = FillingSquaresBar(max=n4-1, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

        for line in reader:
            coordCount = 0

            for numString in line.split(","):    
                newCoord = float(numString)

                if newCoord > 1:
                    clippedCoord = 1
                elif newCoord < -1:
                    clippedCoord = -1
                else:
                    clippedCoord = newCoord

                newVector[coordCount] = clippedCoord
                coordCount += 1

                if coordCount >= d:
                    break

            totalVector += newVector
            heartbeatData[rowCount] = newVector
            rowCount += 1

            if rowCount >= n:
                break

            bar.next()
        bar.finish()

def runBasic(R):

    # global heartbeat data
    global totalVector
    numBuckets = 40
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

        for newVector in heartbeatData:
    
            for a in range(0, t):
                randomIndex = random.randint(0, d-1)

                if len(randomIndices) < checkLength:
                    randomIndices.append(randomIndex)

                sampledCoord = (1 + newVector[randomIndex])/2
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
            inputBucketCoord = math.floor(numBuckets*vector)
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
            outputBucketCoord = math.floor(numBuckets*vector)
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
    datafile.write(f"Sum of squares of the average vector: {round(averageSumOfSquares, 2)} \n")
    error2 = round((100)*((averageMeanSquaredError)/(averageSumOfSquares)), 2)
    datafile.write(f"Total experimental MSE was {error2}% of the sum of squares of the average vector. \n\n")

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
    inputBarIntervals = ['0 to 0.025', '0.025 to 0.05', '0.05 to 0.075', '0.075 to 0.1', '0.1 to 0.125', '0.125 to 0.15', '0.15 to 0.175', '0.175 to 0.2', '0.2 to 0.225', '0.225 to 0.25', '0.25 to 0.275', '0.275 to 0.3', '0.3 to 0.325', '0.325 to 0.35', '0.35 to 0.375', '0.375 to 0.4', '0.4 to 0.425', '0.425 to 0.45', '0.45 to 0.475', '0.475 to 0.5', '0.5 to 0.525', '0.525 to 0.55', '0.55 to 0.575', '0.575 to 0.6', '0.6 to 0.625', '0.625 to 0.65', '0.65 to 0.675', '0.675 to 0.7', '0.7 to 0.725', '0.725 to 0.75', '0.75 to 0.775', '0.775 to 0.8', '0.8 to 0.825', '0.825 to 0.85', '0.85 to 0.875', '0.875 to 0.9', '0.9 to 0.925', '0.925 to 0.95', '0.95 to 0.975', '0.975 to 1']
    inputVectorSum = sum(inputVector)
    percentageInputVector = [coord/inputVectorSum for coord in inputVector]
    plt.bar(inputBarIntervals, percentageInputVector, width = 1, align = 'edge', alpha = 0.4, color = 'g', edgecolor = 'k')
    plt.tick_params(length = 3)

    selectiveInputFormatter = FixedFormatter(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"])
    selectiveInputLocator = FixedLocator([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    plt.gca().xaxis.set_major_formatter(selectiveInputFormatter)
    plt.gca().xaxis.set_major_locator(selectiveInputLocator)

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set(title = 'Histogram of true average vectors in the basic case', xlabel = 'Value', ylabel = 'Frequency')

    datafile.write(f"Frequencies of true average vectors in the basic case: \n")
    datafile.write(f"{str(inputVector)[1:-1]} \n")
    datafile.write(f"Total: {inputVectorSum} \n\n")

    plt.subplot(1, 2, 2)
    outputBarIntervals = ['0 to 0.025', '0.025 to 0.05', '0.05 to 0.075', '0.075 to 0.1', '0.1 to 0.125', '0.125 to 0.15', '0.15 to 0.175', '0.175 to 0.2', '0.2 to 0.225', '0.225 to 0.25', '0.25 to 0.275', '0.275 to 0.3', '0.3 to 0.325', '0.325 to 0.35', '0.35 to 0.375', '0.375 to 0.4', '0.4 to 0.425', '0.425 to 0.45', '0.45 to 0.475', '0.475 to 0.5', '0.5 to 0.525', '0.525 to 0.55', '0.55 to 0.575', '0.575 to 0.6', '0.6 to 0.625', '0.625 to 0.65', '0.65 to 0.675', '0.675 to 0.7', '0.7 to 0.725', '0.725 to 0.75', '0.75 to 0.775', '0.775 to 0.8', '0.8 to 0.825', '0.825 to 0.85', '0.85 to 0.875', '0.875 to 0.9', '0.9 to 0.925', '0.925 to 0.95', '0.95 to 0.975', '0.975 to 1']
    outputVectorSum = sum(outputVector)
    percentageOutputVector = [coord/outputVectorSum for coord in outputVector]
    plt.bar(outputBarIntervals, percentageOutputVector, width = 1, align = 'edge', alpha = 0.4, color = 'b', edgecolor = 'k')
    plt.tick_params(length = 3)

    selectiveOutputFormatter = FixedFormatter(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"])
    selectiveOutputLocator = FixedLocator([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
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
        m = (value + 1)*(int(d/10))
        numBuckets = 40
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

            for newVector in heartbeatData:
            
                dftVector = (rfft(newVector)).tolist()
                
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

            dftAverageVector = [idx/n for idx in totalVector]

            dftMaxInput = max(dftAverageVector)
            dftMinInput = min(dftAverageVector) 
            print(f"{dftMaxInput}")
            print(f"{dftMinInput}")  

            for vector in dftAverageVector:
                dftInputBucketCoord = math.floor(numBuckets*vector)
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
                dftOutputBucketCoord = math.floor(numBuckets*vector)
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
        error3 = round((100)*((averagePerturbationError)/dftComparison), 1)
        datafile.write(f"Experimental perturbation error was {error3}% of the theoretical upper bound for perturbation error. \n")
        datafile.write(f"Standard deviation of perturbation error: {round(standardDeviationPerturbationError, 5)} \n")
        datafile.write(f"Experimental reconstruction error: {round(averageReconstructionError, 5)} \n")

        perErrors.append(Decimal(averagePerturbationError))
        recErrors.append(Decimal(averageReconstructionError))
        totalErrors.append(Decimal(averageDftMeanSquaredError))
        totalStandardDeviation.append(Decimal(standardDeviationMeanSquaredError))

        datafile.write(f"Total experimental MSE: {round(averageDftMeanSquaredError, 4)} \n")
        error4 = round((100)*((averageReconstructionError)/(averageDftMeanSquaredError)), 1)
        datafile.write(f"Reconstruction error was {error4}% of the total experimental MSE. \n")
        datafile.write(f"Standard deviation of reconstruction error: {round(standardDeviationReconstructionError, 5)} \n")
        datafile.write(f"Sum of squares of the average vector: {round(averageDftSumOfSquares, 2)} \n")
        error5 = round((100)*((averageDftMeanSquaredError)/(averageDftSumOfSquares)), 2)
        datafile.write(f"Total experimental MSE was {error5}% of the sum of squares of the average vector. \n\n")

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
        dftInputBarIntervals = ['0 to 0.025', '0.025 to 0.05', '0.05 to 0.075', '0.075 to 0.1', '0.1 to 0.125', '0.125 to 0.15', '0.15 to 0.175', '0.175 to 0.2', '0.2 to 0.225', '0.225 to 0.25', '0.25 to 0.275', '0.275 to 0.3', '0.3 to 0.325', '0.325 to 0.35', '0.35 to 0.375', '0.375 to 0.4', '0.4 to 0.425', '0.425 to 0.45', '0.45 to 0.475', '0.475 to 0.5', '0.5 to 0.525', '0.525 to 0.55', '0.55 to 0.575', '0.575 to 0.6', '0.6 to 0.625', '0.625 to 0.65', '0.65 to 0.675', '0.675 to 0.7', '0.7 to 0.725', '0.725 to 0.75', '0.75 to 0.775', '0.775 to 0.8', '0.8 to 0.825', '0.825 to 0.85', '0.85 to 0.875', '0.875 to 0.9', '0.9 to 0.925', '0.925 to 0.95', '0.95 to 0.975', '0.975 to 1']
        dftInputVectorSum = sum(dftInputVector)
        dftPercentageInputVector = [coord/dftInputVectorSum for coord in dftInputVector]
        plt.bar(dftInputBarIntervals, dftPercentageInputVector, width = 1, align = 'edge', alpha = 0.4, color = 'g', edgecolor = 'k')
        plt.tick_params(length = 3)

        selectiveDftInputFormatter = FixedFormatter(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"])
        selectiveDftInputLocator = FixedLocator([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
        plt.gca().xaxis.set_major_formatter(selectiveDftInputFormatter)
        plt.gca().xaxis.set_major_locator(selectiveDftInputLocator)

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.gca().set(title = 'Histogram of true average vectors in the Fourier case', xlabel = 'Value', ylabel = 'Frequency')

        datafile.write(f"Frequencies of true average vectors in the Fourier case: \n")
        datafile.write(f"{str(dftInputVector)[1:-1]} \n")
        datafile.write(f"Total: {dftInputVectorSum} \n\n")

        plt.subplot(1, 2, 2)
        dftOutputBarIntervals = ['0 to 0.025', '0.025 to 0.05', '0.05 to 0.075', '0.075 to 0.1', '0.1 to 0.125', '0.125 to 0.15', '0.15 to 0.175', '0.175 to 0.2', '0.2 to 0.225', '0.225 to 0.25', '0.25 to 0.275', '0.275 to 0.3', '0.3 to 0.325', '0.325 to 0.35', '0.35 to 0.375', '0.375 to 0.4', '0.4 to 0.425', '0.425 to 0.45', '0.45 to 0.475', '0.475 to 0.5', '0.5 to 0.525', '0.525 to 0.55', '0.55 to 0.575', '0.575 to 0.6', '0.6 to 0.625', '0.625 to 0.65', '0.65 to 0.675', '0.675 to 0.7', '0.7 to 0.725', '0.725 to 0.75', '0.75 to 0.775', '0.775 to 0.8', '0.8 to 0.825', '0.825 to 0.85', '0.85 to 0.875', '0.875 to 0.9', '0.9 to 0.925', '0.925 to 0.95', '0.95 to 0.975', '0.975 to 1']
        dftOutputVectorSum = sum(dftOutputVector)
        dftPercentageOutputVector = [coord/dftOutputVectorSum for coord in dftOutputVector]
        plt.bar(dftOutputBarIntervals, dftPercentageOutputVector, width = 1, align = 'edge', alpha = 0.4, color = 'b', edgecolor = 'k')
        plt.tick_params(length = 3)

        selectiveDftOutputFormatter = FixedFormatter(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"])
        selectiveDftOutputLocator = FixedLocator([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
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
            errorfile.write(f"{10*(value + 1)} {perErrors[value]} {recErrors[value]} {totalErrors[value]} {totalStandardDeviation[value]} \n")
        else:
            errorfile.write(f"{10*(value + 1)} {perErrors[value]} {recErrors[value]} {totalErrors[value]} {totalStandardDeviation[value]}")

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
readTestingData()
readTrainingData()
readAbnormalData()
readNormalData()
runBasic(R)
runDft(R,V)
print("Thank you for using the Shuffle Model for Vectors.")