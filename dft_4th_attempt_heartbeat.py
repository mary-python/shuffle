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

tconst = 1
kconst = 7
mconst = 80
dconst = 100
dmax = 150
epsconst = 2.5
dta = 0.25
nconst = 40000
nmax = 120000
n1 = 21892
n2const = 18108
n2vary = 87554
n3 = 10506
n4 = 48

tset = [1, 2, 3, 4, 5]
kset = [4, 5, 6, 7, 8, 9, 10]
mset = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
dset = [50, 100, 150]
epsset = [1, 1.5, 2, 2.5, 3, 3.5, 4]
nset = [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000]
rset = [1, 2, 3, 4, 5]
R = len(rset)

# set up some global variables
heartbeatDataConstDConstN = np.zeros((nconst, dconst))
heartbeatDataVaryDConstN = np.zeros((nconst, dmax))
heartbeatDataConstDVaryN = np.zeros((nmax, dconst))
totalVectorConstDConstN = np.zeros(dconst)
totalVectorVaryDConstN = np.zeros(dmax)
totalVectorConstDVaryN = np.zeros(dconst)

def readTestingDataConstDConstN():

    global heartbeatDataConstDConstN
    global totalVectorConstDConstN
    newVector = np.zeros(dconst)
    coordCount = 0
    rowCount = 0

    with open("mitbih_test.csv", encoding = "utf8") as reader:

        print(f"\n Reading in the testing data file for constant n and constant d...")
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

                if coordCount >= dconst:
                    break

            totalVectorConstDConstN += newVector
            heartbeatDataConstDConstN[rowCount] = newVector
            rowCount += 1

            if rowCount >= n1:
                break

            bar.next()
        bar.finish()

def readTrainingDataConstDConstN():

    global heartbeatDataConstDConstN
    global totalVectorConstDConstN
    newVector = np.zeros(dconst)
    coordCount = 0
    rowCount = n1

    with open("mitbih_train.csv", encoding = "utf8") as reader:

        print(f"\n Reading in the training data file for constant n and constant d...")
        from progress.bar import FillingSquaresBar
        bar = FillingSquaresBar(max=n2const-1, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

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

                if coordCount >= dconst:
                    break

            totalVectorConstDConstN += newVector
            heartbeatDataConstDConstN[rowCount] = newVector
            rowCount += 1

            if rowCount >= nconst:
                break

            bar.next()
        bar.finish()

def readTestingDataVaryDConstN():

    global heartbeatDataVaryDConstN
    global totalVectorVaryDConstN
    newVector = np.zeros(dmax)
    coordCount = 0
    rowCount = 0

    with open("mitbih_test.csv", encoding = "utf8") as reader:

        print(f"\n Reading in the testing data file for constant n and varying d...")
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

                if coordCount >= dmax:
                    break

            totalVectorVaryDConstN += newVector
            heartbeatDataVaryDConstN[rowCount] = newVector
            rowCount += 1

            if rowCount >= n1:
                break

            bar.next()
        bar.finish()

def readTrainingDataVaryDConstN():

    global heartbeatDataVaryDConstN
    global totalVectorVaryDConstN
    newVector = np.zeros(dmax)
    coordCount = 0
    rowCount = n1

    with open("mitbih_train.csv", encoding = "utf8") as reader:

        print(f"\n Reading in the training data file for constant n and varying d...")
        from progress.bar import FillingSquaresBar
        bar = FillingSquaresBar(max=n2const-1, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

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

                if coordCount >= dmax:
                    break

            totalVectorVaryDConstN += newVector
            heartbeatDataVaryDConstN[rowCount] = newVector
            rowCount += 1

            if rowCount >= nconst:
                break

            bar.next()
        bar.finish()

def readTestingDataConstDVaryN():

    global heartbeatDataConstDVaryN
    global totalVectorConstDVaryN
    newVector = np.zeros(dconst)
    coordCount = 0
    rowCount = 0

    with open("mitbih_test.csv", encoding = "utf8") as reader:

        print(f"\n Reading in the testing data file for varying n and constant d...")
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

                if coordCount >= dconst:
                    break

            totalVectorConstDVaryN += newVector
            heartbeatDataConstDVaryN[rowCount] = newVector
            rowCount += 1

            if rowCount >= n1:
                break

            bar.next()
        bar.finish()

def readTrainingDataConstDVaryN():

    global heartbeatDataConstDVaryN
    global totalVectorConstDVaryN
    newVector = np.zeros(dconst)
    coordCount = 0
    rowCount = n1

    with open("mitbih_train.csv", encoding = "utf8") as reader:

        print(f"\n Reading in the training data file for varying n and constant d...")
        from progress.bar import FillingSquaresBar
        bar = FillingSquaresBar(max=n2vary-1, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

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

                if coordCount >= dconst:
                    break

            totalVectorConstDVaryN += newVector
            heartbeatDataConstDVaryN[rowCount] = newVector
            rowCount += 1

            if rowCount >= n1 + n2vary:
                break

            bar.next()
        bar.finish()

def readAbnormalDataConstDVaryN():

    global heartbeatDataConstDVaryN
    global totalVectorConstDVaryN
    newVector = np.zeros(dconst)
    coordCount = 0
    rowCount = n1 + n2vary

    with open("mitbih_train.csv", encoding = "utf8") as reader:

        print(f"\n Reading in the abnormal data file for varying n and constant d...")
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

                if coordCount >= dconst:
                    break

            totalVectorConstDVaryN += newVector
            heartbeatDataConstDVaryN[rowCount] = newVector
            rowCount += 1

            if rowCount >= n1 + n2vary + n3:
                break

            bar.next()
        bar.finish()

def readNormalDataConstDVaryN():

    global heartbeatDataConstDVaryN
    global totalVectorConstDVaryN
    newVector = np.zeros(dconst)
    coordCount = 0
    rowCount = n1 + n2vary + n3

    with open("mitbih_train.csv", encoding = "utf8") as reader:

        print(f"\n Reading in the normal data file for varying n and constant d...")
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

                if coordCount >= dconst:
                    break

            totalVectorConstDVaryN += newVector
            heartbeatDataConstDVaryN[rowCount] = newVector
            rowCount += 1

            if rowCount >= nmax:
                break

            bar.next()
        bar.finish()

def runBasicVaryT(rset, tset):

    totalErrors = list()
    totalStandardDeviation = list()
    loopTotal = list()

    for t in tset:

        loopTime = time.perf_counter()
        numBuckets = 40
        inputVector = [0]*(numBuckets)
        outputVector = [0]*(numBuckets)
        indexTracker = [0]*dconst
        submittedVector = [0]*dconst
        checkLength = 10
        totalMeanSquaredError = list()
        sumOfSquares = 0

        # gamma is probability of reporting a false value
        if t == 1:
            gamma = max((((14*dconst*kconst*(math.log(2/dta))))/((nconst-1)*(epsconst**2))), (27*dconst*kconst)/((nconst-1)*epsconst))
        else:
            gamma = (((56*dconst*kconst*(math.log(1/dta))*(math.log((2*t)/dta))))/((nconst-1)*(epsconst**2)))
   
        print(f"gamma = {gamma}")

        for r in rset:

            print(f"\n Processing the basic optimal summation result for the value t = {t}, repeat {r}.")
            from progress.bar import FillingSquaresBar
            bar = FillingSquaresBar(max=nconst, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

            randomIndices = list()
            randomisedResponse = list()
            submittedCoords = list()
            outputList = list()

            for newVector in heartbeatDataConstDConstN:
    
                for a in range(0, t):
                    randomIndex = random.randint(0, dconst - 1)

                    if len(randomIndices) < checkLength:
                        randomIndices.append(randomIndex)

                    sampledCoord = (1 + newVector[randomIndex])/2
                    roundedCoord = math.floor(sampledCoord*kconst) + np.random.binomial(1, sampledCoord*kconst - math.floor(sampledCoord*kconst))
                    b = np.random.binomial(1, gamma)

                    if b == 0:
                        submittedCoord = roundedCoord
                    else:
                        submittedCoord = np.random.randint(0, kconst + 1)

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
    
            averageVector = [idx/nconst for idx in totalVectorConstDConstN]

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

            descaledVector = [idx/kconst for idx in submittedVector]
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
            totalMeanSquaredError.append(sum(meanSquaredError))

            averageSquares = [idx**2 for idx in averageVector]
            sumOfSquares += sum(averageSquares)

        averageMeanSquaredError = (sum(totalMeanSquaredError))/R
        averageSumOfSquares = sumOfSquares/R
        differencesMeanSquaredError = [(value - averageMeanSquaredError)**2 for value in totalMeanSquaredError] 
        standardDeviationMeanSquaredError = math.sqrt((sum(differencesMeanSquaredError))/R)
        totalErrors.append(Decimal(averageMeanSquaredError))
        totalStandardDeviation.append(Decimal(standardDeviationMeanSquaredError))

        datafile = open("basic" + str(t) + "t.txt", "w")
        datafile.write(f"Case 1: Optimal Summation in the Shuffle Model \n")

        if t == 1:
            comparison = max((((98*(1/3))*(dconst**(2/3))*(nconst**(1/3))*(np.log(2/dta)))/(((1-gamma)**2)*(epsconst**(4/3)))), (18*(dconst**(2/3))*(nconst**(1/3)))/(((1-gamma)**2)*((4*epsconst)**(2/3))))
        else:
            comparison = (2*(14**(2/3))*(dconst**(2/3))*(nconst**(1/3))*t*(np.log(1/dta))*(np.log(2/dta)))/(((1-gamma)**2)*(epsconst**(4/3)))/nconst

        datafile.write(f"Theoretical Upper Bound for MSE: {round(comparison)} \n")
        datafile.write(f"Experimental MSE: {round(averageMeanSquaredError, 4)} \n")
        error1 = round((100)*((averageMeanSquaredError)/comparison), 1)
        datafile.write(f"Experimental MSE was {error1}% of the theoretical upper bound for MSE. \n")
        datafile.write(f"Sum of squares of the average vector: {round(averageSumOfSquares, 2)} \n")
        error2 = round((100)*((averageMeanSquaredError)/(averageSumOfSquares)), 3)
        datafile.write(f"Total experimental MSE was {error2}% of the sum of squares of the average vector. \n\n")

        plt.style.use('seaborn-white')
        plt.tight_layout()
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 2)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.draw()
        plt.savefig("basic" + str(t) + "t.png")
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
        plt.savefig("basic" + str(t) + "t.png")
        plt.clf()
        plt.cla()

        loopTotal.append(time.perf_counter() - loopTime)
        casetime = round(loopTotal[t-1])
        casemins = math.floor(casetime/60)
        datafile.write(f"Total time for case t = {t}: {casemins}m {casetime - (casemins*60)}s")

    errorfile = open("errorvaryt.txt", "w")

    for t in tset:
        if t != 5:
            errorfile.write(f"{t} {totalErrors[t-1]} {totalStandardDeviation[t-1]} \n")
        else:
            errorfile.write(f"{t} {totalErrors[t-1]} {totalStandardDeviation[t-1]}")

    errorfile.close()

    avgtime = round((sum(loopTotal))/(len(loopTotal)))
    avgmins = math.floor(avgtime/60)
    datafile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
    totaltime = round(time.perf_counter() - startTime)
    totalmins = math.floor(totaltime/60)
    totalhrs = math.floor(totalmins/60)
    datafile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
    datafile.close()

def runBasicVaryK(rset, kset):

    totalErrors = list()
    totalStandardDeviation = list()
    loopTotal = list()

    for k in kset:

        loopTime = time.perf_counter()
        numBuckets = 40
        inputVector = [0]*(numBuckets)
        outputVector = [0]*(numBuckets)
        indexTracker = [0]*dconst
        submittedVector = [0]*dconst
        checkLength = 10
        totalMeanSquaredError = list()
        sumOfSquares = 0

        gamma = max((((14*dconst*k*(math.log(2/dta))))/((nconst-1)*(epsconst**2))), (27*dconst*k)/((nconst-1)*epsconst))
        print(f"gamma = {gamma}")

        for r in rset:

            print(f"\n Processing the basic optimal summation result for the value k = {k}, repeat {r}.")
            from progress.bar import FillingSquaresBar
            bar = FillingSquaresBar(max=nconst, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

            randomIndices = list()
            randomisedResponse = list()
            submittedCoords = list()
            outputList = list()

            for newVector in heartbeatDataConstDConstN:
    
                randomIndex = random.randint(0, dconst - 1)

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
    
            averageVector = [idx/nconst for idx in totalVectorConstDConstN]

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
            totalMeanSquaredError.append(sum(meanSquaredError))

            averageSquares = [idx**2 for idx in averageVector]
            sumOfSquares += sum(averageSquares)

        averageMeanSquaredError = (sum(totalMeanSquaredError))/R
        averageSumOfSquares = sumOfSquares/R
        differencesMeanSquaredError = [(value - averageMeanSquaredError)**2 for value in totalMeanSquaredError] 
        standardDeviationMeanSquaredError = math.sqrt((sum(differencesMeanSquaredError))/R)
        totalErrors.append(Decimal(averageMeanSquaredError))
        totalStandardDeviation.append(Decimal(standardDeviationMeanSquaredError))

        datafile = open("basic" + str(k) + "k.txt", "w")
        datafile.write(f"Case 1: Optimal Summation in the Shuffle Model \n")

        comparison = max((((98*(1/3))*(dconst**(2/3))*(nconst**(1/3))*(np.log(2/dta)))/(((1-gamma)**2)*(epsconst**(4/3)))), (18*(dconst**(2/3))*(nconst**(1/3)))/(((1-gamma)**2)*((4*epsconst)**(2/3))))

        datafile.write(f"Theoretical Upper Bound for MSE: {round(comparison)} \n")
        datafile.write(f"Experimental MSE: {round(averageMeanSquaredError, 4)} \n")
        error1 = round((100)*((averageMeanSquaredError)/comparison), 1)
        datafile.write(f"Experimental MSE was {error1}% of the theoretical upper bound for MSE. \n")
        datafile.write(f"Sum of squares of the average vector: {round(averageSumOfSquares, 2)} \n")
        error2 = round((100)*((averageMeanSquaredError)/(averageSumOfSquares)), 3)
        datafile.write(f"Total experimental MSE was {error2}% of the sum of squares of the average vector. \n\n")

        plt.style.use('seaborn-white')
        plt.tight_layout()
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 2)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.draw()
        plt.savefig("basic" + str(k) + "k.png")
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
        plt.savefig("basic" + str(k) + "k.png")
        plt.clf()
        plt.cla()

        loopTotal.append(time.perf_counter() - loopTime)
        casetime = round(loopTotal[k-4])
        casemins = math.floor(casetime/60)
        datafile.write(f"Total time for case k = {k}: {casemins}m {casetime - (casemins*60)}s")

    errorfile = open("errorvaryk.txt", "w")

    for k in kset:
        if k != 10:
            errorfile.write(f"{k} {totalErrors[k-4]} {totalStandardDeviation[k-4]} \n")
        else:
            errorfile.write(f"{k} {totalErrors[k-4]} {totalStandardDeviation[k-4]}")

    errorfile.close()

    avgtime = round((sum(loopTotal))/(len(loopTotal)))
    avgmins = math.floor(avgtime/60)
    datafile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
    totaltime = round(time.perf_counter() - startTime)
    totalmins = math.floor(totaltime/60)
    totalhrs = math.floor(totalmins/60)
    datafile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
    datafile.close()

def runBasicVaryD(rset, dset):

    totalErrors = list()
    totalStandardDeviation = list()
    loopTotal = list()

    for d in dset:

        loopTime = time.perf_counter()
        numBuckets = 40
        inputVector = [0]*(numBuckets)
        outputVector = [0]*(numBuckets)
        indexTracker = [0]*d
        submittedVector = [0]*d
        checkLength = 10
        totalMeanSquaredError = list()
        sumOfSquares = 0

        gamma = max((((14*d*kconst*(math.log(2/dta))))/((nconst-1)*(epsconst**2))), (27*d*kconst)/((nconst-1)*epsconst))
        print(f"gamma = {gamma}")

        for r in rset:

            print(f"\n Processing the basic optimal summation result for the value d = {d}, repeat {r}.")
            from progress.bar import FillingSquaresBar
            bar = FillingSquaresBar(max=nconst, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

            randomIndices = list()
            randomisedResponse = list()
            submittedCoords = list()
            outputList = list()

            for newVector in heartbeatDataVaryDConstN:
    
                randomIndex = random.randint(0, d-1)

                if len(randomIndices) < checkLength:
                    randomIndices.append(randomIndex)

                sampledCoord = (1 + newVector[randomIndex])/2
                roundedCoord = math.floor(sampledCoord*kconst) + np.random.binomial(1, sampledCoord*kconst - math.floor(sampledCoord*kconst))
                b = np.random.binomial(1, gamma)

                if b == 0:
                    submittedCoord = roundedCoord
                else:
                    submittedCoord = np.random.randint(0, kconst + 1)

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
    
            averageVector = [idx/nconst for idx in totalVectorVaryDConstN]

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

            descaledVector = [idx/kconst for idx in submittedVector]
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
            totalMeanSquaredError.append(sum(meanSquaredError))

            averageSquares = [idx**2 for idx in averageVector]
            sumOfSquares += sum(averageSquares)

        averageMeanSquaredError = (sum(totalMeanSquaredError))/R
        averageSumOfSquares = sumOfSquares/R
        differencesMeanSquaredError = [(value - averageMeanSquaredError)**2 for value in totalMeanSquaredError] 
        standardDeviationMeanSquaredError = math.sqrt((sum(differencesMeanSquaredError))/R)
        totalErrors.append(Decimal(averageMeanSquaredError))
        totalStandardDeviation.append(Decimal(standardDeviationMeanSquaredError))

        datafile = open("basic" + str(d) + "d.txt", "w")
        datafile.write(f"Case 1: Optimal Summation in the Shuffle Model \n")

        comparison = max((((98*(1/3))*(d**(2/3))*(nconst**(1/3))*(np.log(2/dta)))/(((1-gamma)**2)*(epsconst**(4/3)))), (18*(d**(2/3))*(nconst**(1/3)))/(((1-gamma)**2)*((4*epsconst)**(2/3))))

        datafile.write(f"Theoretical Upper Bound for MSE: {round(comparison)} \n")
        datafile.write(f"Experimental MSE: {round(averageMeanSquaredError, 4)} \n")
        error1 = round((100)*((averageMeanSquaredError)/comparison), 1)
        datafile.write(f"Experimental MSE was {error1}% of the theoretical upper bound for MSE. \n")
        datafile.write(f"Sum of squares of the average vector: {round(averageSumOfSquares, 2)} \n")
        error2 = round((100)*((averageMeanSquaredError)/(averageSumOfSquares)), 3)
        datafile.write(f"Total experimental MSE was {error2}% of the sum of squares of the average vector. \n\n")

        plt.style.use('seaborn-white')
        plt.tight_layout()
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 2)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.draw()
        plt.savefig("basic" + str(d) + "d.png")
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
        plt.savefig("basic" + str(d) + "d.png")
        plt.clf()
        plt.cla()

        loopTotal.append(time.perf_counter() - loopTime)
        casetime = round(loopTotal[int((d/50)-1)])
        casemins = math.floor(casetime/60)
        datafile.write(f"Total time for case d = {d}: {casemins}m {casetime - (casemins*60)}s")

    errorfile = open("errorvaryd.txt", "w")

    for d in dset:
        if d != dmax:
            errorfile.write(f"{d} {totalErrors[int((d/50)-1)]} {totalStandardDeviation[int((d/50)-1)]} \n")
        else:
            errorfile.write(f"{d} {totalErrors[int((d/50)-1)]} {totalStandardDeviation[int((d/50)-1)]}")

    errorfile.close()

    avgtime = round((sum(loopTotal))/(len(loopTotal)))
    avgmins = math.floor(avgtime/60)
    datafile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
    totaltime = round(time.perf_counter() - startTime)
    totalmins = math.floor(totaltime/60)
    totalhrs = math.floor(totalmins/60)
    datafile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
    datafile.close()

def runBasicVaryEps(rset, epsset):

    totalErrors = list()
    totalStandardDeviation = list()
    loopTotal = list()

    for eps in epsset:

        loopTime = time.perf_counter()
        numBuckets = 40
        inputVector = [0]*(numBuckets)
        outputVector = [0]*(numBuckets)
        indexTracker = [0]*dconst
        submittedVector = [0]*dconst
        checkLength = 10
        totalMeanSquaredError = list()
        sumOfSquares = 0

        gamma = max((((14*dconst*kconst*(math.log(2/dta))))/((nconst-1)*(eps**2))), (27*dconst*kconst)/((nconst-1)*eps))
        print(f"gamma = {gamma}")

        for r in rset:

            print(f"\n Processing the basic optimal summation result for the value eps = {eps}, repeat {r}.")
            from progress.bar import FillingSquaresBar
            bar = FillingSquaresBar(max=nconst, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

            randomIndices = list()
            randomisedResponse = list()
            submittedCoords = list()
            outputList = list()

            for newVector in heartbeatDataConstDConstN:
    
                randomIndex = random.randint(0, dconst - 1)

                if len(randomIndices) < checkLength:
                    randomIndices.append(randomIndex)

                sampledCoord = (1 + newVector[randomIndex])/2
                roundedCoord = math.floor(sampledCoord*kconst) + np.random.binomial(1, sampledCoord*kconst - math.floor(sampledCoord*kconst))
                b = np.random.binomial(1, gamma)

                if b == 0:
                    submittedCoord = roundedCoord
                else:
                    submittedCoord = np.random.randint(0, kconst + 1)

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
    
            averageVector = [idx/nconst for idx in totalVectorConstDConstN]

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

            descaledVector = [idx/kconst for idx in submittedVector]
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
            totalMeanSquaredError.append(sum(meanSquaredError))

            averageSquares = [idx**2 for idx in averageVector]
            sumOfSquares += sum(averageSquares)

        averageMeanSquaredError = (sum(totalMeanSquaredError))/R
        averageSumOfSquares = sumOfSquares/R
        differencesMeanSquaredError = [(value - averageMeanSquaredError)**2 for value in totalMeanSquaredError] 
        standardDeviationMeanSquaredError = math.sqrt((sum(differencesMeanSquaredError))/R)
        totalErrors.append(Decimal(averageMeanSquaredError))
        totalStandardDeviation.append(Decimal(standardDeviationMeanSquaredError))

        datafile = open("basic" + str(eps) + "eps.txt", "w")
        datafile.write(f"Case 1: Optimal Summation in the Shuffle Model \n")

        comparison = max((((98*(1/3))*(dconst**(2/3))*(nconst**(1/3))*(np.log(2/dta)))/(((1-gamma)**2)*(eps**(4/3)))), (18*(dconst**(2/3))*(nconst**(1/3)))/(((1-gamma)**2)*((4*eps)**(2/3))))

        datafile.write(f"Theoretical Upper Bound for MSE: {round(comparison)} \n")
        datafile.write(f"Experimental MSE: {round(averageMeanSquaredError, 4)} \n")
        error1 = round((100)*((averageMeanSquaredError)/comparison), 1)
        datafile.write(f"Experimental MSE was {error1}% of the theoretical upper bound for MSE. \n")
        datafile.write(f"Sum of squares of the average vector: {round(averageSumOfSquares, 2)} \n")
        error2 = round((100)*((averageMeanSquaredError)/(averageSumOfSquares)), 3)
        datafile.write(f"Total experimental MSE was {error2}% of the sum of squares of the average vector. \n\n")

        plt.style.use('seaborn-white')
        plt.tight_layout()
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 2)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.draw()
        plt.savefig("basic" + str(eps) + "eps.png")
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
        plt.savefig("basic" + str(eps) + "eps.png")
        plt.clf()
        plt.cla()

        loopTotal.append(time.perf_counter() - loopTime)
        casetime = round(loopTotal[int((2*eps)-2)])
        casemins = math.floor(casetime/60)
        datafile.write(f"Total time for case eps = {eps}: {casemins}m {casetime - (casemins*60)}s")

    errorfile = open("errorvaryeps.txt", "w")

    for eps in epsset:
        if eps != 2.5:
            errorfile.write(f"{eps} {totalErrors[int((2*eps)-2)]} {totalStandardDeviation[int((2*eps)-2)]} \n")
        else:
            errorfile.write(f"{eps} {totalErrors[int((2*eps)-2)]} {totalStandardDeviation[int((2*eps)-2)]}")

    errorfile.close()

    avgtime = round((sum(loopTotal))/(len(loopTotal)))
    avgmins = math.floor(avgtime/60)
    datafile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
    totaltime = round(time.perf_counter() - startTime)
    totalmins = math.floor(totaltime/60)
    totalhrs = math.floor(totalmins/60)
    datafile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
    datafile.close()

def runBasicVaryN(rset, nset):

    totalErrors = list()
    totalStandardDeviation = list()
    loopTotal = list()

    for n in nset:
    
        loopTime = time.perf_counter()
        numBuckets = 40
        inputVector = [0]*(numBuckets)
        outputVector = [0]*(numBuckets)
        indexTracker = [0]*dconst
        submittedVector = [0]*dconst
        checkLength = 10
        totalMeanSquaredError = list()
        sumOfSquares = 0

        gamma = max((((14*dconst*kconst*(math.log(2/dta))))/((n-1)*(epsconst**2))), (27*dconst*kconst)/((n-1)*epsconst))
        print(f"gamma = {gamma}")

        for r in rset:

            print(f"\n Processing the basic optimal summation result for the value n = {n}, repeat {r}.")
            from progress.bar import FillingSquaresBar
            bar = FillingSquaresBar(max=n, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

            randomIndices = list()
            randomisedResponse = list()
            submittedCoords = list()
            outputList = list()
            
            for newVector in heartbeatDataConstDVaryN[0:n]:
    
                randomIndex = random.randint(0, dconst - 1)

                if len(randomIndices) < checkLength:
                    randomIndices.append(randomIndex)

                sampledCoord = (1 + newVector[randomIndex])/2
                roundedCoord = math.floor(sampledCoord*kconst) + np.random.binomial(1, sampledCoord*kconst - math.floor(sampledCoord*kconst))
                b = np.random.binomial(1, gamma)

                if b == 0:
                    submittedCoord = roundedCoord
                else:
                    submittedCoord = np.random.randint(0, kconst + 1)

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
    
            averageVector = [idx/n for idx in totalVectorConstDVaryN]

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

            descaledVector = [idx/kconst for idx in submittedVector]
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
            totalMeanSquaredError.append(sum(meanSquaredError))

            averageSquares = [idx**2 for idx in averageVector]
            sumOfSquares += sum(averageSquares)

        averageMeanSquaredError = (sum(totalMeanSquaredError))/R
        averageSumOfSquares = sumOfSquares/R
        differencesMeanSquaredError = [(value - averageMeanSquaredError)**2 for value in totalMeanSquaredError] 
        standardDeviationMeanSquaredError = math.sqrt((sum(differencesMeanSquaredError))/R)
        totalErrors.append(Decimal(averageMeanSquaredError))
        totalStandardDeviation.append(Decimal(standardDeviationMeanSquaredError))

        datafile = open("basic" + str(n) + "n.txt", "w")
        datafile.write(f"Case 1: Optimal Summation in the Shuffle Model \n")

        comparison = max((((98*(1/3))*(dconst**(2/3))*(n**(1/3))*(np.log(2/dta)))/(((1-gamma)**2)*(epsconst**(4/3)))), (18*(dconst**(2/3))*(n**(1/3)))/(((1-gamma)**2)*((4*epsconst)**(2/3))))

        datafile.write(f"Theoretical Upper Bound for MSE: {round(comparison)} \n")
        datafile.write(f"Experimental MSE: {round(averageMeanSquaredError, 4)} \n")
        error1 = round((100)*((averageMeanSquaredError)/comparison), 1)
        datafile.write(f"Experimental MSE was {error1}% of the theoretical upper bound for MSE. \n")
        datafile.write(f"Sum of squares of the average vector: {round(averageSumOfSquares, 2)} \n")
        error2 = round((100)*((averageMeanSquaredError)/(averageSumOfSquares)), 3)
        datafile.write(f"Total experimental MSE was {error2}% of the sum of squares of the average vector. \n\n")

        plt.style.use('seaborn-white')
        plt.tight_layout()
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 2)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.draw()
        plt.savefig("basic" + str(n) + "n.png")
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
        plt.savefig("basic" + str(n) + "n.png")
        plt.clf()
        plt.cla()

        loopTotal.append(time.perf_counter() - loopTime)
        casetime = round(loopTotal[int((n/10000)-2)])
        casemins = math.floor(casetime/60)
        datafile.write(f"Total time for case n = {n}: {casemins}m {casetime - (casemins*60)}s")

    errorfile = open("errorvaryn.txt", "w")

    for n in nset:
        if n != nmax:
            errorfile.write(f"{n} {totalErrors[int((n/10000)-2)]} {totalStandardDeviation[int((n/10000)-2)]} \n")
        else:
            errorfile.write(f"{n} {totalErrors[int((n/10000)-2)]} {totalStandardDeviation[int((n/10000)-2)]}")

    errorfile.close()

    avgtime = round((sum(loopTotal))/(len(loopTotal)))
    avgmins = math.floor(avgtime/60)
    datafile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
    totaltime = round(time.perf_counter() - startTime)
    totalmins = math.floor(totaltime/60)
    totalhrs = math.floor(totalmins/60)
    datafile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
    datafile.close()

def runDftVaryT(rset, tset):
    perErrors = list()
    recErrors = list()
    totalDftErrors = list()
    totalDftStandardDeviation = list()
    loopTotal = list()

    for t in tset:

        loopTime = time.perf_counter()
        numBuckets = 40
        dftInputVector = [0]*(numBuckets)
        dftOutputVector = [0]*(numBuckets)
        dftDebiasedVector = list()
        totalReconstructionError = list()
        totalPerturbationError = list()
        totalDftMeanSquaredError = list()
        dftIndexTracker = [0]*mconst
        dftSubmittedVector = [0]*mconst
        checkLength = 10
        dftSumOfSquares = 0
        sampledError = 0
        returnedError = 0

        if t == 1:
            gamma = max((((14*dconst*kconst*(math.log(2/dta))))/((nconst-1)*(epsconst**2))), (27*dconst*kconst)/((nconst-1)*epsconst))
        else:
            gamma = (((56*dconst*kconst*(math.log(1/dta))*(math.log((2*t)/dta))))/((nconst-1)*(epsconst**2)))
   
        print(f"gamma = {gamma}")

        for r in rset:

            print(f"\n Processing the optimal summation result with DFT for the value t = {t}, repeat {r}.")
            from progress.bar import FillingSquaresBar
            bar = FillingSquaresBar(max=nconst, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

            dftRandomIndices = list()
            dftRandomisedResponse = list()
            dftSubmittedCoords = list()

            for newVector in heartbeatDataConstDConstN:
            
                dftVector = (rfft(newVector)).tolist()
                
                for a in range(0, t):
                    dftRandomIndex = random.randint(0, mconst - 1)

                    if len(dftRandomIndices) < checkLength:
                        dftRandomIndices.append(dftRandomIndex)

                    dftSampledCoord = (1 + dftVector[dftRandomIndex])/2
                    dftRoundedCoord = math.floor(dftSampledCoord*kconst) + np.random.binomial(1, dftSampledCoord*kconst - math.floor(dftSampledCoord*kconst))
                    b = np.random.binomial(1, gamma)

                    if b == 0:
                        dftSubmittedCoord = dftRoundedCoord
                    else:
                        dftSubmittedCoord = np.random.randint(0, kconst + 1)

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

            dftAverageVector = [idx/nconst for idx in totalVectorConstDConstN]

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

            dftDescaledVector = [idx/kconst for idx in dftSubmittedVector]
            dftMergedTracker = tuple(zip(dftIndexTracker, dftDescaledVector))
            dftDebiasedVector = [2*((z - ((gamma/2)*count))/(1 - gamma)/max(count, 1))-1 for count, z in dftMergedTracker]
            paddedVector = dftDebiasedVector + [0]*(dconst -mconst)
            finalVector = (irfft(paddedVector, dconst)).tolist()

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

            exactVector = irfft(rfft(dftAverageVector).tolist()[0:mconst] + [0]*(dconst - mconst)).tolist()
            reconstructionTuple = tuple(zip(exactVector, dftAverageVector))
            reconstructionError = [(a - b)**2 for a, b in reconstructionTuple]
            totalReconstructionError.append(sum(reconstructionError))
            totalPerturbationError.append((sum(dftMeanSquaredError)) - (sum(reconstructionError)))
    
        averageDftMeanSquaredError = (sum(totalDftMeanSquaredError))/R
        averageDftSumOfSquares = dftSumOfSquares/R
        averageReconstructionError = (sum(totalReconstructionError))/R
        averagePerturbationError = (sum(totalPerturbationError))/R

        differencesDftMeanSquaredError = [(value - averageDftMeanSquaredError)**2 for value in totalDftMeanSquaredError] 
        differencesReconstructionError = [(value - averageReconstructionError)**2 for value in totalReconstructionError]
        differencesPerturbationError = [(value - averagePerturbationError)**2 for value in totalPerturbationError]
        standardDeviationDftMeanSquaredError = math.sqrt((sum(differencesDftMeanSquaredError))/R)
        standardDeviationReconstructionError = math.sqrt((sum(differencesReconstructionError))/R)
        standardDeviationPerturbationError = math.sqrt((sum(differencesPerturbationError))/R)
    
        datafile = open("fourier" + str(t) + "t.txt", "w")
        datafile.write(f"Number of coordinates t retained: {t} \n")
        datafile.write(f"Case 2: Fourier Summation Algorithm \n")

        if t == 1:
            dftComparison = max((((98*(1/3))*(mconst**(2/3))*(nconst**(1/3))*(np.log(2/dta)))/(((1-gamma)**2)*(epsconst**(4/3)))), (18*(mconst**(2/3))*(nconst**(1/3)))/(((1-gamma)**2)*((4*epsconst)**(2/3))))
        else:
            dftComparison = (2*(14**(2/3))*(mconst**(2/3))*(nconst**(1/3))*t*(np.log(1/dta))*(np.log(2/dta)))/(((1-gamma)**2)*(epsconst**(4/3)))/nconst

        datafile.write(f"Theoretical upper bound for perturbation error: {round(dftComparison, 4)} \n")
        datafile.write(f"Experimental perturbation error: {round(averagePerturbationError, 4)} \n")
        error3 = round((100)*((averagePerturbationError)/dftComparison), 1)
        datafile.write(f"Experimental perturbation error was {error3}% of the theoretical upper bound for perturbation error. \n")
        datafile.write(f"Standard deviation of perturbation error: {round(standardDeviationPerturbationError, 5)} \n")
        datafile.write(f"Experimental reconstruction error: {round(averageReconstructionError, 5)} \n")

        perErrors.append(Decimal(averagePerturbationError))
        recErrors.append(Decimal(averageReconstructionError))
        totalDftErrors.append(Decimal(averageDftMeanSquaredError))
        totalDftStandardDeviation.append(Decimal(standardDeviationDftMeanSquaredError))

        datafile.write(f"Total experimental MSE: {round(averageDftMeanSquaredError, 4)} \n")
        error4 = round((100)*((averageReconstructionError)/(averageDftMeanSquaredError)), 1)
        datafile.write(f"Reconstruction error was {error4}% of the total experimental MSE. \n")
        datafile.write(f"Standard deviation of reconstruction error: {round(standardDeviationReconstructionError, 5)} \n")
        datafile.write(f"Sum of squares of the average vector: {round(averageDftSumOfSquares, 2)} \n")
        error5 = round((100)*((averageDftMeanSquaredError)/(averageDftSumOfSquares)), 3)
        datafile.write(f"Total experimental MSE was {error5}% of the sum of squares of the average vector. \n\n")

        plt.style.use('seaborn-white')
        plt.tight_layout()
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 2)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.draw()
        plt.savefig("fourier" + str(t) + "t.png")
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
        plt.savefig("fourier" + str(t) + "t.png")
        plt.clf()
        plt.cla()

        loopTotal.append(time.perf_counter() - loopTime)
        casetime = round(loopTotal[t-1])
        casemins = math.floor(casetime/60)
        datafile.write(f"Total time for case t = {t}: {casemins}m {casetime - (casemins*60)}s")

    errorfile = open("dfterrorvaryt.txt", "w")

    for t in tset:
        if t != 5:
            errorfile.write(f"{t} {perErrors[t-1]} {recErrors[t-1]} {totalDftErrors[t-1]} {totalDftStandardDeviation[t-1]} \n")
        else:
            errorfile.write(f"{t} {perErrors[t-1]} {recErrors[t-1]} {totalDftErrors[t-1]} {totalDftStandardDeviation[t-1]}")

    errorfile.close()

    avgtime = round((sum(loopTotal))/(len(loopTotal)))
    avgmins = math.floor(avgtime/60)
    datafile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
    totaltime = round(time.perf_counter() - startTime)
    totalmins = math.floor(totaltime/60)
    totalhrs = math.floor(totalmins/60)
    datafile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
    datafile.close()

def runDftVaryK(rset, kset):
    perErrors = list()
    recErrors = list()
    totalDftErrors = list()
    totalDftStandardDeviation = list()
    loopTotal = list()

    for k in kset:

        loopTime = time.perf_counter()
        numBuckets = 40
        dftInputVector = [0]*(numBuckets)
        dftOutputVector = [0]*(numBuckets)
        dftDebiasedVector = list()
        totalReconstructionError = list()
        totalPerturbationError = list()
        totalDftMeanSquaredError = list()
        dftIndexTracker = [0]*mconst
        dftSubmittedVector = [0]*mconst
        checkLength = 10
        dftSumOfSquares = 0
        sampledError = 0
        returnedError = 0

        gamma = max((((14*dconst*kconst*(math.log(2/dta))))/((nconst-1)*(epsconst**2))), (27*dconst*kconst)/((nconst-1)*epsconst))
        print(f"gamma = {gamma}")

        for r in rset:

            print(f"\n Processing the optimal summation result with DFT for the value k = {k}, repeat {r}.")
            from progress.bar import FillingSquaresBar
            bar = FillingSquaresBar(max=nconst, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

            dftRandomIndices = list()
            dftRandomisedResponse = list()
            dftSubmittedCoords = list()

            for newVector in heartbeatDataConstDConstN:
            
                dftVector = (rfft(newVector)).tolist()
                dftRandomIndex = random.randint(0, mconst - 1)

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

            dftAverageVector = [idx/nconst for idx in totalVectorConstDConstN]

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
            paddedVector = dftDebiasedVector + [0]*(dconst - mconst)
            finalVector = (irfft(paddedVector, dconst)).tolist()

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

            exactVector = irfft(rfft(dftAverageVector).tolist()[0:mconst] + [0]*(dconst - mconst)).tolist()
            reconstructionTuple = tuple(zip(exactVector, dftAverageVector))
            reconstructionError = [(a - b)**2 for a, b in reconstructionTuple]
            totalReconstructionError.append(sum(reconstructionError))
            totalPerturbationError.append((sum(dftMeanSquaredError)) - (sum(reconstructionError)))
    
        averageDftMeanSquaredError = (sum(totalDftMeanSquaredError))/R
        averageDftSumOfSquares = dftSumOfSquares/R
        averageReconstructionError = (sum(totalReconstructionError))/R
        averagePerturbationError = (sum(totalPerturbationError))/R

        differencesDftMeanSquaredError = [(value - averageDftMeanSquaredError)**2 for value in totalDftMeanSquaredError] 
        differencesReconstructionError = [(value - averageReconstructionError)**2 for value in totalReconstructionError]
        differencesPerturbationError = [(value - averagePerturbationError)**2 for value in totalPerturbationError]
        standardDeviationDftMeanSquaredError = math.sqrt((sum(differencesDftMeanSquaredError))/R)
        standardDeviationReconstructionError = math.sqrt((sum(differencesReconstructionError))/R)
        standardDeviationPerturbationError = math.sqrt((sum(differencesPerturbationError))/R)
    
        datafile = open("fourier" + str(k) + "k.txt", "w")
        datafile.write(f"Number of buckets k used: {k} \n")
        datafile.write(f"Case 2: Fourier Summation Algorithm \n")

        dftComparison = max((((98*(1/3))*(mconst**(2/3))*(nconst**(1/3))*(np.log(2/dta)))/(((1-gamma)**2)*(epsconst**(4/3)))), (18*(mconst**(2/3))*(nconst**(1/3)))/(((1-gamma)**2)*((4*epsconst)**(2/3))))

        datafile.write(f"Theoretical upper bound for perturbation error: {round(dftComparison, 4)} \n")
        datafile.write(f"Experimental perturbation error: {round(averagePerturbationError, 4)} \n")
        error3 = round((100)*((averagePerturbationError)/dftComparison), 1)
        datafile.write(f"Experimental perturbation error was {error3}% of the theoretical upper bound for perturbation error. \n")
        datafile.write(f"Standard deviation of perturbation error: {round(standardDeviationPerturbationError, 5)} \n")
        datafile.write(f"Experimental reconstruction error: {round(averageReconstructionError, 5)} \n")

        perErrors.append(Decimal(averagePerturbationError))
        recErrors.append(Decimal(averageReconstructionError))
        totalDftErrors.append(Decimal(averageDftMeanSquaredError))
        totalDftStandardDeviation.append(Decimal(standardDeviationDftMeanSquaredError))

        datafile.write(f"Total experimental MSE: {round(averageDftMeanSquaredError, 4)} \n")
        error4 = round((100)*((averageReconstructionError)/(averageDftMeanSquaredError)), 1)
        datafile.write(f"Reconstruction error was {error4}% of the total experimental MSE. \n")
        datafile.write(f"Standard deviation of reconstruction error: {round(standardDeviationReconstructionError, 5)} \n")
        datafile.write(f"Sum of squares of the average vector: {round(averageDftSumOfSquares, 2)} \n")
        error5 = round((100)*((averageDftMeanSquaredError)/(averageDftSumOfSquares)), 3)
        datafile.write(f"Total experimental MSE was {error5}% of the sum of squares of the average vector. \n\n")

        plt.style.use('seaborn-white')
        plt.tight_layout()
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 2)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.draw()
        plt.savefig("fourier" + str(k) + "k.png")
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
        plt.savefig("fourier" + str(k) + "k.png")
        plt.clf()
        plt.cla()

        loopTotal.append(time.perf_counter() - loopTime)
        casetime = round(loopTotal[k-4])
        casemins = math.floor(casetime/60)
        datafile.write(f"Total time for case k = {k}: {casemins}m {casetime - (casemins*60)}s")

    errorfile = open("dfterrorvaryk.txt", "w")

    for k in kset:
        if k != 10:
            errorfile.write(f"{k} {perErrors[k-4]} {recErrors[k-4]} {totalDftErrors[k-4]} {totalDftStandardDeviation[k-4]} \n")
        else:
            errorfile.write(f"{k} {perErrors[k-4]} {recErrors[k-4]} {totalDftErrors[k-4]} {totalDftStandardDeviation[k-4]}")

    errorfile.close()

    avgtime = round((sum(loopTotal))/(len(loopTotal)))
    avgmins = math.floor(avgtime/60)
    datafile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
    totaltime = round(time.perf_counter() - startTime)
    totalmins = math.floor(totaltime/60)
    totalhrs = math.floor(totalmins/60)
    datafile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
    datafile.close()

def runDftVaryM(rset, mset):  
    perErrors = list()
    recErrors = list()
    totalDftErrors = list()
    totalDftStandardDeviation = list()
    loopTotal = list()

    for m in mset:

        loopTime = time.perf_counter()
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

        gamma = max((((14*dconst*kconst*(math.log(2/dta))))/((nconst-1)*(epsconst**2))), (27*dconst*kconst)/((nconst-1)*epsconst))
        print(f"gamma = {gamma}")

        for r in rset:

            print(f"\n Processing the optimal summation result with DFT for the value m = {m}, repeat {r}.")
            from progress.bar import FillingSquaresBar
            bar = FillingSquaresBar(max=nconst, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

            dftRandomIndices = list()
            dftRandomisedResponse = list()
            dftSubmittedCoords = list()

            for newVector in heartbeatDataConstDConstN:
            
                dftVector = (rfft(newVector)).tolist()
                dftRandomIndex = random.randint(0, m-1)

                if len(dftRandomIndices) < checkLength:
                    dftRandomIndices.append(dftRandomIndex)

                dftSampledCoord = (1 + dftVector[dftRandomIndex])/2
                dftRoundedCoord = math.floor(dftSampledCoord*kconst) + np.random.binomial(1, dftSampledCoord*kconst - math.floor(dftSampledCoord*kconst))
                b = np.random.binomial(1, gamma)

                if b == 0:
                    dftSubmittedCoord = dftRoundedCoord
                else:
                    dftSubmittedCoord = np.random.randint(0, kconst + 1)

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

            dftAverageVector = [idx/nconst for idx in totalVectorConstDConstN]

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

            dftDescaledVector = [idx/kconst for idx in dftSubmittedVector]
            dftMergedTracker = tuple(zip(dftIndexTracker, dftDescaledVector))
            dftDebiasedVector = [2*((z - ((gamma/2)*count))/(1 - gamma)/max(count, 1))-1 for count, z in dftMergedTracker]
            paddedVector = dftDebiasedVector + [0]*(dconst - m)
            finalVector = (irfft(paddedVector, dconst)).tolist()

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

            exactVector = irfft(rfft(dftAverageVector).tolist()[0:m] + [0]*(dconst - m)).tolist()
            reconstructionTuple = tuple(zip(exactVector, dftAverageVector))
            reconstructionError = [(a - b)**2 for a, b in reconstructionTuple]
            totalReconstructionError.append(sum(reconstructionError))
            totalPerturbationError.append((sum(dftMeanSquaredError)) - (sum(reconstructionError)))
    
        averageDftMeanSquaredError = (sum(totalDftMeanSquaredError))/R
        averageDftSumOfSquares = dftSumOfSquares/R
        averageReconstructionError = (sum(totalReconstructionError))/R
        averagePerturbationError = (sum(totalPerturbationError))/R

        differencesDftMeanSquaredError = [(value - averageDftMeanSquaredError)**2 for value in totalDftMeanSquaredError] 
        differencesReconstructionError = [(value - averageReconstructionError)**2 for value in totalReconstructionError]
        differencesPerturbationError = [(value - averagePerturbationError)**2 for value in totalPerturbationError]
        standardDeviationDftMeanSquaredError = math.sqrt((sum(differencesDftMeanSquaredError))/R)
        standardDeviationReconstructionError = math.sqrt((sum(differencesReconstructionError))/R)
        standardDeviationPerturbationError = math.sqrt((sum(differencesPerturbationError))/R)
    
        datafile = open("fourier" + str(m) + "m.txt", "w")
        datafile.write(f"Number of Fourier coefficients m: {m} \n")
        datafile.write(f"Case 2: Fourier Summation Algorithm \n")

        dftComparison = max((((98*(1/3))*(m**(2/3))*(nconst**(1/3))*(np.log(2/dta)))/(((1-gamma)**2)*(epsconst**(4/3)))), (18*(m**(2/3))*(nconst**(1/3)))/(((1-gamma)**2)*((4*epsconst)**(2/3))))

        datafile.write(f"Theoretical upper bound for perturbation error: {round(dftComparison, 4)} \n")
        datafile.write(f"Experimental perturbation error: {round(averagePerturbationError, 4)} \n")
        error3 = round((100)*((averagePerturbationError)/dftComparison), 1)
        datafile.write(f"Experimental perturbation error was {error3}% of the theoretical upper bound for perturbation error. \n")
        datafile.write(f"Standard deviation of perturbation error: {round(standardDeviationPerturbationError, 5)} \n")
        datafile.write(f"Experimental reconstruction error: {round(averageReconstructionError, 5)} \n")

        perErrors.append(Decimal(averagePerturbationError))
        recErrors.append(Decimal(averageReconstructionError))
        totalDftErrors.append(Decimal(averageDftMeanSquaredError))
        totalDftStandardDeviation.append(Decimal(standardDeviationDftMeanSquaredError))

        datafile.write(f"Total experimental MSE: {round(averageDftMeanSquaredError, 4)} \n")
        error4 = round((100)*((averageReconstructionError)/(averageDftMeanSquaredError)), 1)
        datafile.write(f"Reconstruction error was {error4}% of the total experimental MSE. \n")
        datafile.write(f"Standard deviation of reconstruction error: {round(standardDeviationReconstructionError, 5)} \n")
        datafile.write(f"Sum of squares of the average vector: {round(averageDftSumOfSquares, 2)} \n")
        error5 = round((100)*((averageDftMeanSquaredError)/(averageDftSumOfSquares)), 3)
        datafile.write(f"Total experimental MSE was {error5}% of the sum of squares of the average vector. \n\n")

        plt.style.use('seaborn-white')
        plt.tight_layout()
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 2)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.draw()
        plt.savefig("fourier" + str(m) + "m.png")
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
        plt.savefig("fourier" + str(m) + "m.png")
        plt.clf()
        plt.cla()

        loopTotal.append(time.perf_counter() - loopTime)
        casetime = round(loopTotal[int((m/10)-1)])
        casemins = math.floor(casetime/60)
        datafile.write(f"Total time for case m = {m}: {casemins}m {casetime - (casemins*60)}s")

    errorfile = open("dfterrorvarym.txt", "w")

    for m in mset:
        if m != 100:
            errorfile.write(f"{m} {perErrors[int((m/10)-1)]} {recErrors[int((m/10)-1)]} {totalDftErrors[int((m/10)-1)]} {totalDftStandardDeviation[int((m/10)-1)]} \n")
        else:
            errorfile.write(f"{m} {perErrors[int((m/10)-1)]} {recErrors[int((m/10)-1)]} {totalDftErrors[int((m/10)-1)]} {totalDftStandardDeviation[int((m/10)-1)]}")

    errorfile.close()

    avgtime = round((sum(loopTotal))/(len(loopTotal)))
    avgmins = math.floor(avgtime/60)
    datafile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
    totaltime = round(time.perf_counter() - startTime)
    totalmins = math.floor(totaltime/60)
    totalhrs = math.floor(totalmins/60)
    datafile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
    datafile.close()

def runDftVaryD(rset, dset):
    perErrors = list()
    recErrors = list()
    totalDftErrors = list()
    totalDftStandardDeviation = list()
    loopTotal = list()

    for d in dset:

        loopTime = time.perf_counter()
        numBuckets = 40
        dftInputVector = [0]*(numBuckets)
        dftOutputVector = [0]*(numBuckets)
        dftDebiasedVector = list()
        totalReconstructionError = list()
        totalPerturbationError = list()
        totalDftMeanSquaredError = list()
        dftIndexTracker = [0]*mconst
        dftSubmittedVector = [0]*mconst
        checkLength = 10
        dftSumOfSquares = 0
        sampledError = 0
        returnedError = 0

        gamma = max((((14*dconst*kconst*(math.log(2/dta))))/((nconst-1)*(epsconst**2))), (27*dconst*kconst)/((nconst-1)*epsconst))
        print(f"gamma = {gamma}")

        for r in rset:

            print(f"\n Processing the optimal summation result with DFT for the value d = {d}, repeat {r}.")
            from progress.bar import FillingSquaresBar
            bar = FillingSquaresBar(max=nconst, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

            dftRandomIndices = list()
            dftRandomisedResponse = list()
            dftSubmittedCoords = list()

            for newVector in heartbeatDataVaryDConstN:
            
                dftVector = (rfft(newVector)).tolist()
                dftRandomIndex = random.randint(0, mconst - 1)

                if len(dftRandomIndices) < checkLength:
                    dftRandomIndices.append(dftRandomIndex)

                dftSampledCoord = (1 + dftVector[dftRandomIndex])/2
                dftRoundedCoord = math.floor(dftSampledCoord*kconst) + np.random.binomial(1, dftSampledCoord*kconst - math.floor(dftSampledCoord*kconst))
                b = np.random.binomial(1, gamma)

                if b == 0:
                    dftSubmittedCoord = dftRoundedCoord
                else:
                    dftSubmittedCoord = np.random.randint(0, kconst + 1)

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

            dftAverageVector = [idx/nconst for idx in totalVectorVaryDConstN]

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

            dftDescaledVector = [idx/kconst for idx in dftSubmittedVector]
            dftMergedTracker = tuple(zip(dftIndexTracker, dftDescaledVector))
            dftDebiasedVector = [2*((z - ((gamma/2)*count))/(1 - gamma)/max(count, 1))-1 for count, z in dftMergedTracker]
            paddedVector = dftDebiasedVector + [0]*(d-mconst)
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

            exactVector = irfft(rfft(dftAverageVector).tolist()[0:mconst] + [0]*(d-mconst)).tolist()
            reconstructionTuple = tuple(zip(exactVector, dftAverageVector))
            reconstructionError = [(a - b)**2 for a, b in reconstructionTuple]
            totalReconstructionError.append(sum(reconstructionError))
            totalPerturbationError.append((sum(dftMeanSquaredError)) - (sum(reconstructionError)))
    
        averageDftMeanSquaredError = (sum(totalDftMeanSquaredError))/R
        averageDftSumOfSquares = dftSumOfSquares/R
        averageReconstructionError = (sum(totalReconstructionError))/R
        averagePerturbationError = (sum(totalPerturbationError))/R

        differencesDftMeanSquaredError = [(value - averageDftMeanSquaredError)**2 for value in totalDftMeanSquaredError] 
        differencesReconstructionError = [(value - averageReconstructionError)**2 for value in totalReconstructionError]
        differencesPerturbationError = [(value - averagePerturbationError)**2 for value in totalPerturbationError]
        standardDeviationDftMeanSquaredError = math.sqrt((sum(differencesDftMeanSquaredError))/R)
        standardDeviationReconstructionError = math.sqrt((sum(differencesReconstructionError))/R)
        standardDeviationPerturbationError = math.sqrt((sum(differencesPerturbationError))/R)
    
        datafile = open("fourier" + str(d) + "d.txt", "w")
        datafile.write(f"Dimension of vector d: {d} \n")
        datafile.write(f"Case 2: Fourier Summation Algorithm \n")

        dftComparison = max((((98*(1/3))*(mconst**(2/3))*(nconst**(1/3))*(np.log(2/dta)))/(((1-gamma)**2)*(epsconst**(4/3)))), (18*(mconst**(2/3))*(nconst**(1/3)))/(((1-gamma)**2)*((4*epsconst)**(2/3))))

        datafile.write(f"Theoretical upper bound for perturbation error: {round(dftComparison, 4)} \n")
        datafile.write(f"Experimental perturbation error: {round(averagePerturbationError, 4)} \n")
        error3 = round((100)*((averagePerturbationError)/dftComparison), 1)
        datafile.write(f"Experimental perturbation error was {error3}% of the theoretical upper bound for perturbation error. \n")
        datafile.write(f"Standard deviation of perturbation error: {round(standardDeviationPerturbationError, 5)} \n")
        datafile.write(f"Experimental reconstruction error: {round(averageReconstructionError, 5)} \n")

        perErrors.append(Decimal(averagePerturbationError))
        recErrors.append(Decimal(averageReconstructionError))
        totalDftErrors.append(Decimal(averageDftMeanSquaredError))
        totalDftStandardDeviation.append(Decimal(standardDeviationDftMeanSquaredError))

        datafile.write(f"Total experimental MSE: {round(averageDftMeanSquaredError, 4)} \n")
        error4 = round((100)*((averageReconstructionError)/(averageDftMeanSquaredError)), 1)
        datafile.write(f"Reconstruction error was {error4}% of the total experimental MSE. \n")
        datafile.write(f"Standard deviation of reconstruction error: {round(standardDeviationReconstructionError, 5)} \n")
        datafile.write(f"Sum of squares of the average vector: {round(averageDftSumOfSquares, 2)} \n")
        error5 = round((100)*((averageDftMeanSquaredError)/(averageDftSumOfSquares)), 3)
        datafile.write(f"Total experimental MSE was {error5}% of the sum of squares of the average vector. \n\n")

        plt.style.use('seaborn-white')
        plt.tight_layout()
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 2)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.draw()
        plt.savefig("fourier" + str(d) + "d.png")
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
        plt.savefig("fourier" + str(d) + "d.png")
        plt.clf()
        plt.cla()

        loopTotal.append(time.perf_counter() - loopTime)
        casetime = round(loopTotal[int((d/50)-1)])
        casemins = math.floor(casetime/60)
        datafile.write(f"Total time for case d = {d}: {casemins}m {casetime - (casemins*60)}s")

    errorfile = open("dfterrorvaryd.txt", "w")

    for d in dset:
        if d != dmax:
            errorfile.write(f"{d} {perErrors[int((d/50)-1)]} {recErrors[int((d/50)-1)]} {totalDftErrors[int((d/50)-1)]} {totalDftStandardDeviation[int((d/50)-1)]} \n")
        else:
            errorfile.write(f"{d} {perErrors[int((d/50)-1)]} {recErrors[int((d/50)-1)]} {totalDftErrors[int((d/50)-1)]} {totalDftStandardDeviation[int((d/50)-1)]}")

    errorfile.close()

    avgtime = round((sum(loopTotal))/(len(loopTotal)))
    avgmins = math.floor(avgtime/60)
    datafile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
    totaltime = round(time.perf_counter() - startTime)
    totalmins = math.floor(totaltime/60)
    totalhrs = math.floor(totalmins/60)
    datafile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
    datafile.close()

def runDftVaryEps(rset, epsset):
    perErrors = list()
    recErrors = list()
    totalDftErrors = list()
    totalDftStandardDeviation = list()
    loopTotal = list()

    for eps in epsset:

        loopTime = time.perf_counter()
        numBuckets = 40
        dftInputVector = [0]*(numBuckets)
        dftOutputVector = [0]*(numBuckets)
        dftDebiasedVector = list()
        totalReconstructionError = list()
        totalPerturbationError = list()
        totalDftMeanSquaredError = list()
        dftIndexTracker = [0]*mconst
        dftSubmittedVector = [0]*mconst
        checkLength = 10
        dftSumOfSquares = 0
        sampledError = 0
        returnedError = 0

        gamma = max((((14*dconst*kconst*(math.log(2/dta))))/((nconst-1)*(epsconst**2))), (27*dconst*kconst)/((nconst-1)*epsconst))
        print(f"gamma = {gamma}")

        for r in range(0, R):

            print(f"\n Processing the optimal summation result with DFT for the value eps = {eps}, repeat {r}.")
            from progress.bar import FillingSquaresBar
            bar = FillingSquaresBar(max=nconst, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

            dftRandomIndices = list()
            dftRandomisedResponse = list()
            dftSubmittedCoords = list()

            for newVector in heartbeatDataConstDConstN:
            
                dftVector = (rfft(newVector)).tolist()
                dftRandomIndex = random.randint(0, mconst - 1)

                if len(dftRandomIndices) < checkLength:
                    dftRandomIndices.append(dftRandomIndex)

                dftSampledCoord = (1 + dftVector[dftRandomIndex])/2
                dftRoundedCoord = math.floor(dftSampledCoord*kconst) + np.random.binomial(1, dftSampledCoord*kconst - math.floor(dftSampledCoord*kconst))
                b = np.random.binomial(1, gamma)

                if b == 0:
                    dftSubmittedCoord = dftRoundedCoord
                else:
                    dftSubmittedCoord = np.random.randint(0, kconst + 1)

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

            dftAverageVector = [idx/nconst for idx in totalVectorConstDConstN]

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

            dftDescaledVector = [idx/kconst for idx in dftSubmittedVector]
            dftMergedTracker = tuple(zip(dftIndexTracker, dftDescaledVector))
            dftDebiasedVector = [2*((z - ((gamma/2)*count))/(1 - gamma)/max(count, 1))-1 for count, z in dftMergedTracker]
            paddedVector = dftDebiasedVector + [0]*(dconst - mconst)
            finalVector = (irfft(paddedVector, dconst)).tolist()

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

            exactVector = irfft(rfft(dftAverageVector).tolist()[0:mconst] + [0]*(dconst - mconst)).tolist()
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
    
        datafile = open("fourier" + str(eps) + "eps.txt", "w")
        datafile.write(f"Value of epsilon: {eps} \n")
        datafile.write(f"Case 2: Fourier Summation Algorithm \n")

        dftComparison = max((((98*(1/3))*(mconst**(2/3))*(nconst**(1/3))*(np.log(2/dta)))/(((1-gamma)**2)*(eps**(4/3)))), (18*(mconst**(2/3))*(nconst**(1/3)))/(((1-gamma)**2)*((4*eps)**(2/3))))

        datafile.write(f"Theoretical upper bound for perturbation error: {round(dftComparison, 4)} \n")
        datafile.write(f"Experimental perturbation error: {round(averagePerturbationError, 4)} \n")
        error3 = round((100)*((averagePerturbationError)/dftComparison), 1)
        datafile.write(f"Experimental perturbation error was {error3}% of the theoretical upper bound for perturbation error. \n")
        datafile.write(f"Standard deviation of perturbation error: {round(standardDeviationPerturbationError, 5)} \n")
        datafile.write(f"Experimental reconstruction error: {round(averageReconstructionError, 5)} \n")

        perErrors.append(Decimal(averagePerturbationError))
        recErrors.append(Decimal(averageReconstructionError))
        totalDftErrors.append(Decimal(averageDftMeanSquaredError))
        totalDftStandardDeviation.append(Decimal(standardDeviationMeanSquaredError))

        datafile.write(f"Total experimental MSE: {round(averageDftMeanSquaredError, 4)} \n")
        error4 = round((100)*((averageReconstructionError)/(averageDftMeanSquaredError)), 1)
        datafile.write(f"Reconstruction error was {error4}% of the total experimental MSE. \n")
        datafile.write(f"Standard deviation of reconstruction error: {round(standardDeviationReconstructionError, 5)} \n")
        datafile.write(f"Sum of squares of the average vector: {round(averageDftSumOfSquares, 2)} \n")
        error5 = round((100)*((averageDftMeanSquaredError)/(averageDftSumOfSquares)), 3)
        datafile.write(f"Total experimental MSE was {error5}% of the sum of squares of the average vector. \n\n")

        plt.style.use('seaborn-white')
        plt.tight_layout()
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 2)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.draw()
        plt.savefig("fourier" + str(eps) + "eps.png")
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
        plt.savefig("fourier" + str(eps) + "eps.png")
        plt.clf()
        plt.cla()

        loopTotal.append(time.perf_counter() - loopTime)
        casetime = round(loopTotal[int((2*eps)-2)])
        casemins = math.floor(casetime/60)
        datafile.write(f"Total time for case eps = {eps}: {casemins}m {casetime - (casemins*60)}s")

    errorfile = open("dfterrorvaryeps.txt", "w")

    for eps in epsset:
        if eps != 4:
            errorfile.write(f"{eps} {perErrors[int((2*eps)-2)]} {recErrors[int((2*eps)-2)]} {totalDftErrors[int((2*eps)-2)]} {totalDftStandardDeviation[int((2*eps)-2)]} \n")
        else:
            errorfile.write(f"{eps} {perErrors[int((2*eps)-2)]} {recErrors[int((2*eps)-2)]} {totalDftErrors[int((2*eps)-2)]} {totalDftStandardDeviation[int((2*eps)-2)]}")

    errorfile.close()

    avgtime = round((sum(loopTotal))/(len(loopTotal)))
    avgmins = math.floor(avgtime/60)
    datafile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
    totaltime = round(time.perf_counter() - startTime)
    totalmins = math.floor(totaltime/60)
    totalhrs = math.floor(totalmins/60)
    datafile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
    datafile.close()  

def runDftVaryN(rset, nset):
    perErrors = list()
    recErrors = list()
    totalDftErrors = list()
    totalDftStandardDeviation = list()
    loopTotal = list()

    for n in nset:

        loopTime = time.perf_counter()
        numBuckets = 40
        dftInputVector = [0]*(numBuckets)
        dftOutputVector = [0]*(numBuckets)
        dftDebiasedVector = list()
        totalReconstructionError = list()
        totalPerturbationError = list()
        totalDftMeanSquaredError = list()
        dftIndexTracker = [0]*mconst
        dftSubmittedVector = [0]*mconst
        checkLength = 10
        dftSumOfSquares = 0
        sampledError = 0
        returnedError = 0

        gamma = max((((14*dconst*kconst*(math.log(2/dta))))/((nconst-1)*(epsconst**2))), (27*dconst*kconst)/((nconst-1)*epsconst))
        print(f"gamma = {gamma}")

        for r in range(0, R):

            print(f"\n Processing the optimal summation result with DFT for the value n = {n}, repeat {r}.")
            from progress.bar import FillingSquaresBar
            bar = FillingSquaresBar(max=n, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

            dftRandomIndices = list()
            dftRandomisedResponse = list()
            dftSubmittedCoords = list()

            for newVector in heartbeatDataConstDVaryN[0:n]:
            
                dftVector = (rfft(newVector)).tolist()
                dftRandomIndex = random.randint(0, mconst - 1)

                if len(dftRandomIndices) < checkLength:
                    dftRandomIndices.append(dftRandomIndex)

                dftSampledCoord = (1 + dftVector[dftRandomIndex])/2
                dftRoundedCoord = math.floor(dftSampledCoord*kconst) + np.random.binomial(1, dftSampledCoord*kconst - math.floor(dftSampledCoord*kconst))
                b = np.random.binomial(1, gamma)

                if b == 0:
                    dftSubmittedCoord = dftRoundedCoord
                else:
                    dftSubmittedCoord = np.random.randint(0, kconst + 1)

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

            dftAverageVector = [idx/n for idx in totalVectorConstDVaryN]

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

            dftDescaledVector = [idx/kconst for idx in dftSubmittedVector]
            dftMergedTracker = tuple(zip(dftIndexTracker, dftDescaledVector))
            dftDebiasedVector = [2*((z - ((gamma/2)*count))/(1 - gamma)/max(count, 1))-1 for count, z in dftMergedTracker]
            paddedVector = dftDebiasedVector + [0]*(dconst - mconst)
            finalVector = (irfft(paddedVector, dconst)).tolist()

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

            exactVector = irfft(rfft(dftAverageVector).tolist()[0:mconst] + [0]*(dconst - mconst)).tolist()
            reconstructionTuple = tuple(zip(exactVector, dftAverageVector))
            reconstructionError = [(a - b)**2 for a, b in reconstructionTuple]
            totalReconstructionError.append(sum(reconstructionError))
            totalPerturbationError.append((sum(dftMeanSquaredError)) - (sum(reconstructionError)))
    
        averageDftMeanSquaredError = (sum(totalDftMeanSquaredError))/R
        averageDftSumOfSquares = dftSumOfSquares/R
        averageReconstructionError = (sum(totalReconstructionError))/R
        averagePerturbationError = (sum(totalPerturbationError))/R

        differencesDftMeanSquaredError = [(value - averageDftMeanSquaredError)**2 for value in totalDftMeanSquaredError] 
        differencesReconstructionError = [(value - averageReconstructionError)**2 for value in totalReconstructionError]
        differencesPerturbationError = [(value - averagePerturbationError)**2 for value in totalPerturbationError]
        standardDeviationDftMeanSquaredError = math.sqrt((sum(differencesDftMeanSquaredError))/R)
        standardDeviationReconstructionError = math.sqrt((sum(differencesReconstructionError))/R)
        standardDeviationPerturbationError = math.sqrt((sum(differencesPerturbationError))/R)
    
        datafile = open("fourier" + str(n) + "n.txt", "w")
        datafile.write(f"Number of vectors n used: {n} \n")
        datafile.write(f"Case 2: Fourier Summation Algorithm \n")

        dftComparison = max((((98*(1/3))*(mconst**(2/3))*(n**(1/3))*(np.log(2/dta)))/(((1-gamma)**2)*(epsconst**(4/3)))), (18*(mconst**(2/3))*(n**(1/3)))/(((1-gamma)**2)*((4*epsconst)**(2/3))))

        datafile.write(f"Theoretical upper bound for perturbation error: {round(dftComparison, 4)} \n")
        datafile.write(f"Experimental perturbation error: {round(averagePerturbationError, 4)} \n")
        error3 = round((100)*((averagePerturbationError)/dftComparison), 1)
        datafile.write(f"Experimental perturbation error was {error3}% of the theoretical upper bound for perturbation error. \n")
        datafile.write(f"Standard deviation of perturbation error: {round(standardDeviationPerturbationError, 5)} \n")
        datafile.write(f"Experimental reconstruction error: {round(averageReconstructionError, 5)} \n")

        perErrors.append(Decimal(averagePerturbationError))
        recErrors.append(Decimal(averageReconstructionError))
        totalDftErrors.append(Decimal(averageDftMeanSquaredError))
        totalDftStandardDeviation.append(Decimal(standardDeviationDftMeanSquaredError))

        datafile.write(f"Total experimental MSE: {round(averageDftMeanSquaredError, 4)} \n")
        error4 = round((100)*((averageReconstructionError)/(averageDftMeanSquaredError)), 1)
        datafile.write(f"Reconstruction error was {error4}% of the total experimental MSE. \n")
        datafile.write(f"Standard deviation of reconstruction error: {round(standardDeviationReconstructionError, 5)} \n")
        datafile.write(f"Sum of squares of the average vector: {round(averageDftSumOfSquares, 2)} \n")
        error5 = round((100)*((averageDftMeanSquaredError)/(averageDftSumOfSquares)), 3)
        datafile.write(f"Total experimental MSE was {error5}% of the sum of squares of the average vector. \n\n")

        plt.style.use('seaborn-white')
        plt.tight_layout()
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 2)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.draw()
        plt.savefig("fourier" + str(n) + "n.png")
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
        plt.savefig("fourier" + str(n) + "n.png")
        plt.clf()
        plt.cla()

        loopTotal.append(time.perf_counter() - loopTime)
        casetime = round(loopTotal[int((n/10000)-2)])
        casemins = math.floor(casetime/60)
        datafile.write(f"Total time for case n = {n}: {casemins}m {casetime - (casemins*60)}s")

    errorfile = open("dfterrorvaryn.txt", "w")

    for n in nset:
        if n != nmax:
            errorfile.write(f"{n} {perErrors[int((n/10000)-2)]} {recErrors[int((n/10000)-2)]} {totalDftErrors[int((n/10000)-2)]} {totalDftStandardDeviation[int((n/10000)-2)]} \n")
        else:
            errorfile.write(f"{n} {perErrors[int((n/10000)-2)]} {recErrors[int((n/10000)-2)]} {totalDftErrors[int((n/10000)-2)]} {totalDftStandardDeviation[int((n/10000)-2)]}")

    errorfile.close()

    avgtime = round((sum(loopTotal))/(len(loopTotal)))
    avgmins = math.floor(avgtime/60)
    datafile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
    totaltime = round(time.perf_counter() - startTime)
    totalmins = math.floor(totaltime/60)
    totalhrs = math.floor(totalmins/60)
    datafile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
    datafile.close()

readTestingDataConstDConstN()
readTrainingDataConstDConstN()
readTestingDataVaryDConstN()
readTrainingDataVaryDConstN()

readTestingDataConstDVaryN()
readTrainingDataConstDVaryN()
readAbnormalDataConstDVaryN()
readNormalDataConstDVaryN()

runBasicVaryT(rset, tset)
runBasicVaryK(rset, kset)
runBasicVaryD(rset, dset)
runBasicVaryEps(rset, epsset)
runBasicVaryN(rset, nset)

runDftVaryT(rset, tset)
runDftVaryK(rset, kset)
runDftVaryM(rset, mset)
runDftVaryD(rset, dset)
runDftVaryEps(rset, epsset)
runDftVaryN(rset, nset)

print("Thank you for using the Shuffle Model for Vectors.")