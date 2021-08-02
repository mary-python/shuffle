# IMPORTING RELEVANT PACKAGES
import random, math, time
import numpy as np
from decimal import *
from scipy.fftpack import rfft, irfft
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FixedFormatter, FixedLocator

# INITIALISING SEEDS AND START TIME
random.seed(2196018)
np.random.seed(2196018)
startTime = time.perf_counter()

# INITIALISING PARAMETERS/CONSTANTS OF THE ALGORITHM
tset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
tconst = tset[0]
kset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
kconst = kset[2]
mset = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
mconst = mset[8]

# INITIALISING PARAMETERS/CONSTANTS OF THE DATA
dset = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
dconst = dset[4]
dmax = dset[9]
epsset1 = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
epsset2 = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
epsconst = epsset1[9]
nset = [10000, 11000, 14000, 17000, 20000, 30000, 40000, 50000, 60000, 70000]
nconst = nset[7]
nmax = nset[9]

# INITIALISING OTHER PARAMETERS/CONSTANTS
parset = ['t', 'k', 'm', 'd', 'eps', 'eps', 'n']
rset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
R = len(rset)
dta = 0.5
n1 = 21892
n2const = 28108
n2vary = 38108

# INITIALISING GLOBAL VARIABLES
heartbeatDataConstDConstN = np.zeros((nconst, dconst))
heartbeatDataVaryDConstN = np.zeros((nconst, dmax))
heartbeatDataConstDVaryN = np.zeros((nmax, dconst))
totalVectorConstDConstN = np.zeros(dconst)
totalVectorVaryDConstN = np.zeros(dmax)
totalVectorConstDVaryN = np.zeros(dconst)

# INNER LOOP: ADDITION OF VECTORS TO GLOBAL VARIABLES
def addToGlobalVariables(data, start, vector, total):
    data[start] = vector
    total += vector

# MAIN DATA READING LOOP
def readData(dimension, start, number, data, total):
    newVector = np.zeros(dimension)
    coordCount = 0
    rowCount = start

    from progress.bar import FillingSquaresBar
    bar = FillingSquaresBar(max=number-1, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

    # READING EACH LINE OF THE DATA FILE
    for line in range(0, number):
        coordCount = 0

        # SPLITTING THE LINE INTO INDIVIDUAL COORDINATES
        for coord in range(0, dimension):
            newCoord = (math.cos((coord/dimension)**2))*(random.random())/2

            # MOVING ALL COORDINATES OUTSIDE OF THE RANGE -1 to 1 TO AN ENDPOINT OF THIS RANGE
            if newCoord > 1:
                clippedCoord = 1
            elif newCoord < -1:
                clippedCoord = -1
            else:
                clippedCoord = newCoord

            # ADDING THIS COORDINATE TO THE NEW VECTOR
            newVector[coordCount] = clippedCoord
            coordCount += 1

        # ADDING THE NEW VECTOR TO THE GLOBAL VARIABLES
        addToGlobalVariables(data, rowCount, newVector, total)
        rowCount += 1

        bar.next()
    bar.finish()

# KEEPING BOTH D AND N CONSTANT: APPLICABLE FOR CHANGING ANY VARIABLE EXCEPT D OR N
def readDataConstDConstN():

    print(f"\nReading in the testing data file for constant d and constant n...")
    readData(dconst, 0, n1, heartbeatDataConstDConstN, totalVectorConstDConstN)

    print(f"\nReading in the training data file for constant d and constant n...")
    readData(dconst, n1, n2const, heartbeatDataConstDConstN, totalVectorConstDConstN)

# VARYING D AND KEEPING N CONSTANT: APPLICABLE FOR CHANGING D
def readDataVaryDConstN():

    print(f"\nReading in the testing data file for varying d and constant n...")
    readData(dmax, 0, n1, heartbeatDataVaryDConstN, totalVectorVaryDConstN)

    print(f"\nReading in the training data file for varying d and constant n...")
    readData(dmax, n1, n2const, heartbeatDataVaryDConstN, totalVectorVaryDConstN)

# KEEPING D CONSTANT AND VARYING N: APPLICABLE FOR CHANGING N
def readDataConstDVaryN():

    print(f"\nReading in the testing data file for constant d and varying n...")
    readData(dconst, 0, n1, heartbeatDataConstDVaryN, totalVectorConstDVaryN)

    print(f"\nReading in the training data file for constant d and varying n...")
    readData(dconst, n1, n2vary, heartbeatDataConstDVaryN, totalVectorConstDVaryN)
    

# WRITING IN ERROR FILE AFTER THE MAIN BASIC LOOP
def afterBasicLoopStats(index, var, varset, multiplier, offset, totalErrors, totalStandardDeviation, loopTotal, gammas):

    errorfile = open("syntherrorvary" + str(index) + "%s.txt" % parset[index], "w")

    for var in varset:
        if index == 6:
            if var == 10000:
                errorfile.write(f"{int(var*multiplier)} {totalErrors[int((var*multiplier)-offset)]} {totalStandardDeviation[int((var*multiplier)-offset)]} {gammas[int((var*multiplier)-offset)]} \n")
            elif var <= 20000:
                errorfile.write(f"{int(var*multiplier)} {totalErrors[int(((var*multiplier)/3)-(8/3))]} {totalStandardDeviation[int(((var*multiplier)/3)-(8/3))]} {gammas[int(((var*multiplier)/3)-(8/3))]} \n")
            else:
                errorfile.write(f"{int(var*multiplier)} {totalErrors[int((var*multiplier*0.1) + 2)]} {totalStandardDeviation[int((var*multiplier*0.1) + 2)]} {gammas[int((var*multiplier*0.1) + 2)]} \n")
        else:
            errorfile.write(f"{var} {totalErrors[int((var*multiplier)-offset)]} {totalStandardDeviation[int((var*multiplier)-offset)]} {gammas[int((var*multiplier)-offset)]} \n")

    avgtime = round((sum(loopTotal))/(len(loopTotal)))
    avgmins = math.floor(avgtime/60)
    errorfile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
    totaltime = round(time.perf_counter() - startTime)
    totalmins = math.floor(totaltime/60)
    totalhrs = math.floor(totalmins/60)
    errorfile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
    errorfile.close()

# MAIN SHUFFLING ALGORITHM WITHOUT DISCRETE FOURIER TRANSFORM
def runBasic(index, var, varset, tchoice, kchoice, dchoice, epschoice, nchoice, data, total, multiplier, offset, totalErrors, totalStandardDeviation, loopTotal, gammas):

    loopTime = time.perf_counter()
    numBuckets = 40
    inputVector = [0]*(numBuckets)
    outputVector = [0]*(numBuckets)
    totalMeanSquaredError = list()
    sumOfSquares = 0

    # SETTING GAMMA: PROBABILITY OF A FALSE VALUE
    if tchoice == 1:
        if epschoice < 1:
            gamma = max((((14*dchoice*kchoice*(math.log(2/dta))))/((nchoice-1)*(epschoice**2))), (27*dchoice*kchoice)/((nchoice-1)*epschoice))
        else:
            gamma = max((((32*dchoice*kchoice*(math.log(2/dta))))/((nchoice-1)*(epschoice**2))), (21*dchoice*kchoice)/(4*(nchoice-1)*epschoice))

    else:
        if epschoice < 1:
            gamma = (((56*dchoice*kchoice*(math.log(1/dta))*(math.log((2*tchoice)/dta))))/((nchoice-1)*(epschoice**2)))
        else:
            gamma = (((4608*dchoice*kchoice*(math.log(1/dta))*(math.log((2*tchoice)/dta))))/((nchoice-1)*(epschoice**2)))

    print(f"\ngamma = {round(gamma, 4)}")

    from progress.bar import FillingSquaresBar
    bar = FillingSquaresBar(max=R, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

    # REPEATING THE ALGORITHM
    for r in rset:

        indexTracker = [0]*dchoice
        submittedVector = [0]*dchoice

        # APPLYING THE ALGORITHM TO EACH NEW VECTOR
        for newVector in data:
    
            # SELECTING T RANDOM COORDINATES OUT OF D TOTAL COORDINATES
            for a in range(0, tchoice):
                randomIndex = random.randint(0, dchoice - 1)

                # LINEAR TRANSFORM TO ENSURE COORDINATES ARE BETWEEN 0 AND 1
                sampledCoord = (1 + newVector[randomIndex])/2

                # ROUNDING COORDINATES TO ONE OF TWO NEAREST BUCKETS OF WHICH THERE ARE K IN TOTAL
                roundedCoord = math.floor(sampledCoord*kchoice) + np.random.binomial(1, sampledCoord*kchoice - math.floor(sampledCoord*kchoice))
                    
                # RANDOMISED RESPONSE USING GAMMA AS PROBABILITY
                b = np.random.binomial(1, gamma)
                if b == 0:
                    submittedCoord = roundedCoord
                else:
                    submittedCoord = np.random.randint(0, kchoice + 1)

                # SUBMITTING EITHER A TRUE OR FALSE VECTOR
                submittedVector[randomIndex] += submittedCoord
                indexTracker[randomIndex] += 1
    
        # GENERATING STATISTICS FOR THE TRUE AVERAGE VECTORS
        if index == 6:
            averageVector = [idx/nmax for idx in total]
        else:
            averageVector = [idx/nchoice for idx in total]

        for vector in averageVector:
            inputBucketCoord = math.floor(numBuckets*vector)
            inputVector[min(inputBucketCoord, numBuckets - 1)] += 1

        descaledVector = [idx/kchoice for idx in submittedVector]
        mergedTracker = tuple(zip(indexTracker, descaledVector))
        debiasedVector = [2*((z - ((gamma/2)*count))/(1 - gamma)/max(count, 1))-1 for count, z in mergedTracker]

        # GENERATING STATISTICS FOR THE RECONSTRUCTED UNBIASED VECTORS
        for vector in debiasedVector:
            outputBucketCoord = math.floor(numBuckets*vector)
            outputVector[min(outputBucketCoord, numBuckets - 1)] += 1

        errorTuple = tuple(zip(debiasedVector, averageVector))
        meanSquaredError = [(a - b)**2 for a, b in errorTuple]
        totalMeanSquaredError.append(sum(meanSquaredError))

        averageSquares = [idx**2 for idx in averageVector]
        sumOfSquares += sum(averageSquares)

        bar.next()
    bar.finish()

    # AVERAGING OUT THE STATISTICS OVER THE REPEATS
    averageMeanSquaredError = (sum(totalMeanSquaredError))/R
    averageSumOfSquares = sumOfSquares/R
    differencesMeanSquaredError = [(value - averageMeanSquaredError)**2 for value in totalMeanSquaredError] 
    standardDeviationMeanSquaredError = math.sqrt((sum(differencesMeanSquaredError))/R)
    totalErrors.append(Decimal(averageMeanSquaredError))
    totalStandardDeviation.append(Decimal(standardDeviationMeanSquaredError))
    gammas.append(Decimal(gamma))

    # WRITING THE STATISTICS ON A DATAFILE
    datafile = open("synthbasic" + str(index) + "%s" % parset[index] + str(var) + ".txt", "w")
    datafile.write(f"Case 1: Optimal Summation in the Shuffle Model \n")

    if tchoice == 1:
        if epschoice < 1:
            comparison = max((((98*(1/3))*(dchoice**(8/3))*((np.log(2/dta))**(2/3)))/(((1-gamma)**2)*(nchoice**(5/3))*(epschoice**(4/3)))), (18*(dchoice**(8/3)))/(((1-gamma)**2)*(nchoice**(5/3))*((4*epschoice)**(2/3))))
        else:
            comparison = max(((8*(dchoice**(8/3))*((np.log(2/dta))**(2/3)))/(((1-gamma)**2)*(nchoice**(5/3))*(epschoice**(4/3)))), ((21**(2/3))*(dchoice**(8/3)))/(2*((1-gamma)**2)*(nchoice**(5/3))*((2*epschoice)**(2/3))))

    else:
        if epschoice < 1:
            comparison = (2*tchoice*(dchoice**(8/3))*((14*(np.log(1/dta))*(np.log((2*tchoice)/dta)))**(2/3)))/(((1-gamma)**2)*(nchoice**(5/3))*(epschoice**(4/3)))
        else:
            comparison = (32*tchoice*(dchoice**(8/3))*((18*(np.log(1/dta))*(np.log((2*tchoice)/dta)))**(2/3)))/(((1-gamma)**2)*(nchoice**(5/3))*(epschoice**(4/3)))

    if comparison < 1:
        datafile.write(f"Theoretical Upper Bound for MSE: {round(comparison, 4)} \n")
    elif comparison < 10:
        datafile.write(f"Theoretical Upper Bound for MSE: {round(comparison, 2)} \n")
    else:
        datafile.write(f"Theoretical Upper Bound for MSE: {round(comparison, 1)} \n")

    datafile.write(f"Experimental MSE: {round(averageMeanSquaredError, 4)} \n")
    error1 = round((100)*((averageMeanSquaredError)/comparison), 1)
    datafile.write(f"Experimental MSE was {error1}% of the theoretical upper bound for MSE. \n")
    datafile.write(f"Sum of squares of the average vector: {round(averageSumOfSquares, 2)} \n")
    error2 = round((100)*((averageMeanSquaredError)/(averageSumOfSquares)), 2)
    datafile.write(f"Total experimental MSE was {error2}% of the sum of squares of the average vector. \n\n")

    # PLOTTING THE DISTRIBUTION OF THE TRUE AVERAGE VECTORS
    plt.style.use('seaborn-white')
    plt.tight_layout()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 2)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.draw()
    plt.savefig("synthbasic" + str(index) + "%s" % parset[index] + str(var) + ".png")
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

    # PLOTTING THE DISTRIBUTION OF THE RECONSTRUCTED UNBIASED VECTORS
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
    plt.savefig("synthbasic" + str(index) + "%s" % parset[index] + str(var) + ".png")
    plt.clf()
    plt.cla()

    # COMPUTING THE TIME TAKEN FOR EACH CASE
    casetime = time.perf_counter() - loopTime
    loopTotal.append(casetime)
    casemins = math.floor(casetime/60)
    datafile.write(f"\nTotal time for case {parset[index]} = {var}: {casemins}m {math.floor(casetime - (casemins*60))}s")

# VARYING THE NUMBER OF COORDINATES T RETAINED
def runBasicVaryT():
    totalErrors = list()
    totalStandardDeviation = list()
    loopTotal = list()
    gammas = list()

    for t in tset:
        print(f"\nProcessing the basic optimal summation result for the value t = {t}.")
        runBasic(0, t, tset, t, kconst, dconst, epsconst, nconst, heartbeatDataConstDConstN, totalVectorConstDConstN, 1, 1, totalErrors, totalStandardDeviation, loopTotal, gammas)

    afterBasicLoopStats(0, t, tset, 1, 1, totalErrors, totalStandardDeviation, loopTotal, gammas)

# VARYING THE NUMBER OF BUCKETS K USED
def runBasicVaryK():
    totalErrors = list()
    totalStandardDeviation = list()
    loopTotal = list()
    gammas = list()

    for k in kset:
        print(f"\nProcessing the basic optimal summation result for the value k = {k}.")
        runBasic(1, k, kset, tconst, k, dconst, epsconst, nconst, heartbeatDataConstDConstN, totalVectorConstDConstN, 1, 1, totalErrors, totalStandardDeviation, loopTotal, gammas)

    afterBasicLoopStats(1, k, kset, 1, 1, totalErrors, totalStandardDeviation, loopTotal, gammas)

# VARYING THE VECTOR DIMENSION D
def runBasicVaryD():
    totalErrors = list()
    totalStandardDeviation = list()
    loopTotal = list()
    gammas = list()

    for d in dset:
        print(f"\nProcessing the basic optimal summation result for the value d = {d}.")
        runBasic(3, d, dset, tconst, kconst, d, epsconst, nconst, heartbeatDataVaryDConstN, totalVectorVaryDConstN, 0.1, 6, totalErrors, totalStandardDeviation, loopTotal, gammas)
    
    afterBasicLoopStats(3, d, dset, 0.1, 6, totalErrors, totalStandardDeviation, loopTotal, gammas)

# VARYING THE VALUE OF EPSILON: LESS THAN OR EQUAL TO 1
def runBasicVaryEps1():
    totalErrors = list()
    totalStandardDeviation = list()
    loopTotal = list()
    gammas = list()

    for eps in epsset1:
        print(f"\nProcessing the basic optimal summation result for the value eps = {eps}.")
        runBasic(4, eps, epsset1, tconst, kconst, dconst, eps, nconst, heartbeatDataConstDConstN, totalVectorConstDConstN, 20, 10, totalErrors, totalStandardDeviation, loopTotal, gammas)

    afterBasicLoopStats(4, eps, epsset1, 20, 10, totalErrors, totalStandardDeviation, loopTotal, gammas)

# VARYING THE VALUE OF EPSILON: GREATER THAN 1
def runBasicVaryEps2():
    totalErrors = list()
    totalStandardDeviation = list()
    loopTotal = list()
    gammas = list()

    for eps in epsset2:
        print(f"\nProcessing the basic optimal summation result for the value eps = {eps}.")
        runBasic(5, eps, epsset2, tconst, kconst, dconst, eps, nconst, heartbeatDataConstDConstN, totalVectorConstDConstN, 2, 2, totalErrors, totalStandardDeviation, loopTotal, gammas)

    afterBasicLoopStats(5, eps, epsset2, 2, 2, totalErrors, totalStandardDeviation, loopTotal, gammas)

# VARYING THE NUMBER OF VECTORS N USED
def runBasicVaryN():
    totalErrors = list()
    totalStandardDeviation = list()
    loopTotal = list()
    gammas = list()

    for n in nset:
        print(f"\nProcessing the basic optimal summation result for the value n = {n}.")
        runBasic(6, n, nset, tconst, kconst, dconst, epsconst, n, heartbeatDataConstDVaryN, totalVectorConstDVaryN, 0.001, 10, totalErrors, totalStandardDeviation, loopTotal, gammas)

    afterBasicLoopStats(6, n, nset, 0.001, 10, totalErrors, totalStandardDeviation, loopTotal, gammas)

# WRITING IN ERROR FILE AFTER THE MAIN DFT LOOP
def afterDftLoopStats(index, var, varset, multiplier, offset, perErrors, recErrors, totalDftErrors, totalDftStandardDeviation, perStandardDeviation, loopTotal, gammas):
    
    errorfile = open("synthdfterrorvary" + str(index) + "%s.txt" % parset[index], "w")

    for var in varset:
        if index == 6:
            if var == 10000:
                errorfile.write(f"{int(var*multiplier)} {perErrors[int((var*multiplier)-offset)]} {recErrors[int((var*multiplier)-offset)]} {totalDftErrors[int((var*multiplier)-offset)]} {totalDftStandardDeviation[int((var*multiplier)-offset)]} {perStandardDeviation[int((var*multiplier)-offset)]} {gammas[int((var*multiplier)-offset)]} \n")
            elif var <= 20000:
                errorfile.write(f"{int(var*multiplier)} {perErrors[int(((var*multiplier)/3)-(8/3))]} {recErrors[int(((var*multiplier)/3)-(8/3))]} {totalDftErrors[int(((var*multiplier)/3)-(8/3))]} {totalDftStandardDeviation[int(((var*multiplier)/3)-(8/3))]} {perStandardDeviation[int(((var*multiplier)/3)-(8/3))]} {gammas[int(((var*multiplier)/3)-(8/3))]} \n")
            else:
                errorfile.write(f"{int(var*multiplier)} {perErrors[int((var*multiplier*0.1) + 2)]} {recErrors[int((var*multiplier*0.1) + 2)]} {totalDftErrors[int((var*multiplier*0.1) + 2)]} {totalDftStandardDeviation[int((var*multiplier*0.1) + 2)]} {perStandardDeviation[int((var*multiplier*0.1) + 2)]} {gammas[int((var*multiplier*0.1) + 2)]} \n")
        else:
            errorfile.write(f"{var} {perErrors[int((var*multiplier)-offset)]} {recErrors[int((var*multiplier)-offset)]} {totalDftErrors[int((var*multiplier)-offset)]} {totalDftStandardDeviation[int((var*multiplier)-offset)]} {perStandardDeviation[int((var*multiplier)-offset)]} {gammas[int((var*multiplier)-offset)]} \n")

    avgtime = round((sum(loopTotal))/(len(loopTotal)))
    avgmins = math.floor(avgtime/60)
    errorfile.write(f"\nAverage time for each case: {avgmins}m {avgtime - (avgmins*60)}s \n")
    totaltime = round(time.perf_counter() - startTime)
    totalmins = math.floor(totaltime/60)
    totalhrs = math.floor(totalmins/60)
    errorfile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
    errorfile.close()

# MAIN SHUFFLING ALGORITHM WITH DISCRETE FOURIER TRANSFORM
def runDft(index, var, varset, tchoice, kchoice, mchoice, epschoice, nchoice, data, total, multiplier, offset, perErrors, recErrors, totalDftErrors, totalDftStandardDeviation, perStandardDeviation, loopTotal, gammas):

    loopTime = time.perf_counter()
    numBuckets = 40
    dftInputVector = [0]*(numBuckets)
    dftOutputVector = [0]*(numBuckets)
    dftDebiasedVector = list()
    totalReconstructionError = list()
    totalPerturbationError = list()
    totalDftMeanSquaredError = list()
    dftSumOfSquares = 0
    sampledError = 0
    returnedError = 0

    # SETTING GAMMA: PROBABILITY OF A FALSE VALUE
    if tchoice == 1:
        if epschoice < 1:
            gamma = max((((14*mchoice*kchoice*(math.log(2/dta))))/((nchoice-1)*(epschoice**2))), (27*mchoice*kchoice)/((nchoice-1)*epschoice))
        else:
            gamma = max((((32*mchoice*kchoice*(math.log(2/dta))))/((nchoice-1)*(epschoice**2))), (21*mchoice*kchoice)/(4*(nchoice-1)*epschoice))

    else:
        if epschoice < 1:
            gamma = (((56*mchoice*kchoice*(math.log(1/dta))*(math.log((2*tchoice)/dta))))/((nchoice-1)*(epschoice**2)))
        else:
            gamma = (((4608*mchoice*kchoice*(math.log(1/dta))*(math.log((2*tchoice)/dta))))/((nchoice-1)*(epschoice**2)))
    
    print(f"\ngamma = {round(gamma, 4)}")

    from progress.bar import FillingSquaresBar
    bar = FillingSquaresBar(max=R, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

    # REPEATING THE ALGORITHM
    for r in rset:

        dftIndexTracker = [0]*mchoice
        dftSubmittedVector = [0]*mchoice

        # APPLYING THE ALGORITHM TO EACH NEW VECTOR
        for newVector in data:
    
            # DISCRETE FOURIER TRANSFORM APPLIED TO NEW VECTOR
            dftVector = (rfft(newVector)).tolist()

            # SELECTING T RANDOM COORDINATES OUT OF M TOTAL COORDINATES
            for a in range(0, tchoice):
                dftRandomIndex = random.randint(0, mchoice - 1)

                # LINEAR TRANSFORM TO ENSURE COORDINATES ARE BETWEEN 0 AND 1
                dftSampledCoord = (1 + dftVector[dftRandomIndex])/2

                # ROUNDING COORDINATES TO ONE OF TWO NEAREST BUCKETS OF WHICH THERE ARE K IN TOTAL
                dftRoundedCoord = math.floor(dftSampledCoord*kchoice) + np.random.binomial(1, dftSampledCoord*kchoice - math.floor(dftSampledCoord*kchoice))
                    
                # RANDOMISED RESPONSE USING GAMMA AS PROBABILITY
                b = np.random.binomial(1, gamma)
                if b == 0:
                    dftSubmittedCoord = dftRoundedCoord
                else:
                    dftSubmittedCoord = np.random.randint(0, kchoice + 1)

                # SUBMITTING EITHER A TRUE OR FALSE VECTOR
                dftSubmittedVector[dftRandomIndex] += dftSubmittedCoord
                dftIndexTracker[dftRandomIndex] += 1
    
        # GENERATING STATISTICS FOR THE TRUE AVERAGE VECTORS
        if index == 6:
            dftAverageVector = [idx/nmax for idx in total]
        else:
            dftAverageVector = [idx/nchoice for idx in total]

        for vector in dftAverageVector:
            dftInputBucketCoord = math.floor(numBuckets*vector)
            dftInputVector[min(dftInputBucketCoord, numBuckets - 1)] += 1

        dftDescaledVector = [idx/kchoice for idx in dftSubmittedVector]
        dftMergedTracker = tuple(zip(dftIndexTracker, dftDescaledVector))
        dftDebiasedVector = [2*((z - ((gamma/2)*count))/(1 - gamma)/max(count, 1))-1 for count, z in dftMergedTracker]
        paddedVector = dftDebiasedVector + [0]*(dconst - mchoice)
        finalVector = (irfft(paddedVector, dconst)).tolist()

        # GENERATING STATISTICS FOR THE RECONSTRUCTED UNBIASED VECTORS
        for vector in finalVector:
            dftOutputBucketCoord = math.floor(numBuckets*vector)
            dftOutputVector[min(dftOutputBucketCoord, numBuckets - 1)] += 1

        dftErrorTuple = tuple(zip(finalVector, dftAverageVector))
        dftMeanSquaredError = [(a - b)**2 for a, b in dftErrorTuple]
        totalDftMeanSquaredError.append(sum(dftMeanSquaredError))
        
        dftAverageSquares = [idx**2 for idx in dftAverageVector]
        dftSumOfSquares += sum(dftAverageSquares)

        exactVector = irfft(rfft(dftAverageVector).tolist()[0:mchoice] + [0]*(dconst-mchoice)).tolist()
        reconstructionTuple = tuple(zip(exactVector, dftAverageVector))
        reconstructionError = [(a - b)**2 for a, b in reconstructionTuple]
        totalReconstructionError.append(sum(reconstructionError))
        totalPerturbationError.append((sum(dftMeanSquaredError)) - (sum(reconstructionError)))

        bar.next()
    bar.finish()

    # AVERAGING OUT THE STATISTICS OVER THE REPEATS
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

    perErrors.append(Decimal(averagePerturbationError))
    recErrors.append(Decimal(averageReconstructionError))   
    totalDftErrors.append(Decimal(averageDftMeanSquaredError))
    totalDftStandardDeviation.append(Decimal(standardDeviationDftMeanSquaredError))
    perStandardDeviation.append(Decimal(standardDeviationPerturbationError))
    gammas.append(Decimal(gamma))

    # WRITING THE STATISTICS ON A DATAFILE
    datafile = open("synthfourier" + str(index) + "%s" % parset[index] + str(var) + ".txt", "w")

    if index == 0:
        datafile.write(f"Number of coordinates t retained: {var} \n")
    elif index == 1:
        datafile.write(f"Number of buckets k used: {var} \n")
    elif index == 2:
        datafile.write(f"Number of Fourier coefficients m: {var} \n") 
    elif index == 3:
        datafile.write(f"Vector dimension d: {var} \n")
    elif index == 4:
        datafile.write(f"Value of epsilon: {var} \n")
    else:
        datafile.write(f"Number of vectors n used: {var} \n") 

    datafile.write(f"Case 2: Fourier Summation Algorithm \n")

    if tchoice == 1:
        if epschoice < 1:
            dftComparison = max((((98*(1/3))*(mchoice**(8/3))*((np.log(2/dta))**(2/3)))/(((1-gamma)**2)*(nchoice**(5/3))*(epschoice**(4/3)))), (18*(mchoice**(8/3)))/(((1-gamma)**2)*(nchoice**(5/3))*((4*epschoice)**(2/3))))
        else:
            dftComparison = max(((8*(mchoice**(8/3))*((np.log(2/dta))**(2/3)))/(((1-gamma)**2)*(nchoice**(5/3))*(epschoice**(4/3)))), ((21**(2/3))*(mchoice**(8/3)))/(2*((1-gamma)**2)*(nchoice**(5/3))*((2*epschoice)**(2/3))))

    else:
        if epschoice < 1:
            dftComparison = (2*tchoice*(mchoice**(8/3))*((14*(np.log(1/dta))*(np.log((2*tchoice)/dta)))**(2/3)))/(((1-gamma)**2)*(nchoice**(5/3))*(epschoice**(4/3)))
        else:
            dftComparison = (32*tchoice*(mchoice**(8/3))*((18*(np.log(1/dta))*(np.log((2*tchoice)/dta)))**(2/3)))/(((1-gamma)**2)*(nchoice**(5/3))*(epschoice**(4/3)))

    if dftComparison < 1:
        datafile.write(f"Theoretical upper bound for perturbation error: {round(dftComparison, 4)} \n")
    elif dftComparison < 10:
        datafile.write(f"Theoretical upper bound for perturbation error: {round(dftComparison, 2)} \n")
    else:
        datafile.write(f"Theoretical upper bound for perturbation error: {round(dftComparison, 1)} \n")

    datafile.write(f"Experimental perturbation error: {round(averagePerturbationError, 4)} \n")
    error3 = round((100)*((averagePerturbationError)/dftComparison), 1)
    datafile.write(f"Experimental perturbation error was {error3}% of the theoretical upper bound for perturbation error. \n")
    datafile.write(f"Standard deviation of perturbation error: {round(standardDeviationPerturbationError, 5)} \n")

    datafile.write(f"Experimental reconstruction error: {round(averageReconstructionError, 5)} \n")
    datafile.write(f"Total experimental MSE: {round(averageDftMeanSquaredError, 4)} \n")
    error4 = round((100)*((averageReconstructionError)/(averageDftMeanSquaredError)), 1)
    datafile.write(f"Reconstruction error was {error4}% of the total experimental MSE. \n")
    datafile.write(f"Standard deviation of reconstruction error: {round(standardDeviationReconstructionError, 10)} \n")

    datafile.write(f"Sum of squares of the average vector: {round(averageDftSumOfSquares, 2)} \n")
    error5 = round((100)*((averageDftMeanSquaredError)/(averageDftSumOfSquares)), 3)
    datafile.write(f"Total experimental MSE was {error5}% of the sum of squares of the average vector. \n\n")

    # PLOTTING THE DISTRIBUTION OF THE TRUE AVERAGE VECTORS
    plt.style.use('seaborn-white')
    plt.tight_layout()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 2)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.draw()
    plt.savefig("synthfourier" + str(index) + "%s" % parset[index] + str(var) + ".png")
    plt.clf()
    plt.cla()

    plt.subplot(1, 2, 1)
    dftInputBarIntervals = ['0 to 0.025', '0.025 to 0.05', '0.05 to 0.075', '0.075 to 0.1', '0.1 to 0.125', '0.125 to 0.15', '0.15 to 0.175', '0.175 to 0.2', '0.2 to 0.225', '0.225 to 0.25', '0.25 to 0.275', '0.275 to 0.3', '0.3 to 0.325', '0.325 to 0.35', '0.35 to 0.375', '0.375 to 0.4', '0.4 to 0.425', '0.425 to 0.45', '0.45 to 0.475', '0.475 to 0.5', '0.5 to 0.525', '0.525 to 0.55', '0.55 to 0.575', '0.575 to 0.6', '0.6 to 0.625', '0.625 to 0.65', '0.65 to 0.675', '0.675 to 0.7', '0.7 to 0.725', '0.725 to 0.75', '0.75 to 0.775', '0.775 to 0.8', '0.8 to 0.825', '0.825 to 0.85', '0.85 to 0.875', '0.875 to 0.9', '0.9 to 0.925', '0.925 to 0.95', '0.95 to 0.975', '0.975 to 1']
    dftInputVectorSum = sum(dftInputVector)
    dftPercentageInputVector = [coord/dftInputVectorSum for coord in dftInputVector]
    plt.bar(dftInputBarIntervals, dftPercentageInputVector, width = 1, align = 'edge', alpha = 0.4, color = 'g', edgecolor = 'k')
    plt.tick_params(length = 3)

    selectiveInputFormatter = FixedFormatter(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"])
    selectiveInputLocator = FixedLocator([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    plt.gca().xaxis.set_major_formatter(selectiveInputFormatter)
    plt.gca().xaxis.set_major_locator(selectiveInputLocator)

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set(title = 'Histogram of true average vectors in the Fourier case', xlabel = 'Value', ylabel = 'Frequency')

    datafile.write(f"Frequencies of true average vectors in the Fourier case: \n")
    datafile.write(f"{str(dftInputVector)[1:-1]} \n")
    datafile.write(f"Total: {dftInputVectorSum} \n\n")

    # PLOTTING THE DISTRIBUTION OF THE RECONSTRUCTED UNBIASED VECTORS
    plt.subplot(1, 2, 2)
    dftOutputBarIntervals = ['0 to 0.025', '0.025 to 0.05', '0.05 to 0.075', '0.075 to 0.1', '0.1 to 0.125', '0.125 to 0.15', '0.15 to 0.175', '0.175 to 0.2', '0.2 to 0.225', '0.225 to 0.25', '0.25 to 0.275', '0.275 to 0.3', '0.3 to 0.325', '0.325 to 0.35', '0.35 to 0.375', '0.375 to 0.4', '0.4 to 0.425', '0.425 to 0.45', '0.45 to 0.475', '0.475 to 0.5', '0.5 to 0.525', '0.525 to 0.55', '0.55 to 0.575', '0.575 to 0.6', '0.6 to 0.625', '0.625 to 0.65', '0.65 to 0.675', '0.675 to 0.7', '0.7 to 0.725', '0.725 to 0.75', '0.75 to 0.775', '0.775 to 0.8', '0.8 to 0.825', '0.825 to 0.85', '0.85 to 0.875', '0.875 to 0.9', '0.9 to 0.925', '0.925 to 0.95', '0.95 to 0.975', '0.975 to 1']
    dftOutputVectorSum = sum(dftOutputVector)
    dftPercentageOutputVector = [coord/dftOutputVectorSum for coord in dftOutputVector]
    plt.bar(dftOutputBarIntervals, dftPercentageOutputVector, width = 1, align = 'edge', alpha = 0.4, color = 'b', edgecolor = 'k')
    plt.tick_params(length = 3)

    selectiveOutputFormatter = FixedFormatter(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"])
    selectiveOutputLocator = FixedLocator([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    plt.gca().xaxis.set_major_formatter(selectiveOutputFormatter)
    plt.gca().xaxis.set_major_locator(selectiveOutputLocator)

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set(title = 'Histogram of reconstructed unbiased vectors in the Fourier case', xlabel = 'Value', ylabel = 'Frequency')

    datafile.write(f"Frequencies of reconstructed unbiased vectors in the Fourier case: \n")
    datafile.write(f"{str(dftOutputVector)[1:-1]} \n")
    datafile.write(f"Total: {dftOutputVectorSum} \n")

    plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.draw()
    plt.savefig("synthfourier" + str(index) + "%s" % parset[index] + str(var) + ".png")
    plt.clf()
    plt.cla()

    # COMPUTING THE TIME TAKEN FOR EACH CASE
    casetime = time.perf_counter() - loopTime
    loopTotal.append(casetime)
    casemins = math.floor(casetime/60)
    datafile.write(f"\nTotal time for case {parset[index]} = {var}: {casemins}m {math.floor(casetime - (casemins*60))}s")

# VARYING THE NUMBER OF COORDINATES T RETAINED
def runDftVaryT():
    perErrors = list()
    recErrors = list()
    totalDftErrors = list()
    totalDftStandardDeviation = list()
    perStandardDeviation = list()
    loopTotal = list()
    gammas = list()

    for t in tset:
        print(f"\nProcessing the optimal summation result with DFT for the value t = {t}.")
        runDft(0, t, tset, t, kconst, mconst, epsconst, nconst, heartbeatDataConstDConstN, totalVectorConstDConstN, 1, 1, perErrors, recErrors, totalDftErrors, totalDftStandardDeviation, perStandardDeviation, loopTotal, gammas)
    
    afterDftLoopStats(0, t, tset, 1, 1, perErrors, recErrors, totalDftErrors, totalDftStandardDeviation, perStandardDeviation, loopTotal, gammas)

# VARYING THE NUMBER OF BUCKETS K USED
def runDftVaryK():
    perErrors = list()  
    recErrors = list()
    totalDftErrors = list()
    totalDftStandardDeviation = list()
    perStandardDeviation = list()
    loopTotal = list()
    gammas = list()

    for k in kset:
        print(f"\nProcessing the optimal summation result with DFT for the value k = {k}.")
        runDft(1, k, kset, tconst, k, mconst, epsconst, nconst, heartbeatDataConstDConstN, totalVectorConstDConstN, 1, 1, perErrors, recErrors, totalDftErrors, totalDftStandardDeviation, perStandardDeviation, loopTotal, gammas)

    afterDftLoopStats(1, k, kset, 1, 1, perErrors, recErrors, totalDftErrors, totalDftStandardDeviation, perStandardDeviation, loopTotal, gammas)

# VARYING THE NUMBER OF FOURIER COEFFICIENTS M
def runDftVaryM():
    perErrors = list()  
    recErrors = list()
    totalDftErrors = list()
    totalDftStandardDeviation = list()
    perStandardDeviation = list()
    loopTotal = list()
    gammas = list()

    for m in mset:
        print(f"\nProcessing the optimal summation result with DFT for the value m = {m}.")
        runDft(2, m, mset, tconst, kconst, m, epsconst, nconst, heartbeatDataConstDConstN, totalVectorConstDConstN, 0.1, 1, perErrors, recErrors, totalDftErrors, totalDftStandardDeviation, perStandardDeviation, loopTotal, gammas)

    afterDftLoopStats(2, m, mset, 0.1, 1, perErrors, recErrors, totalDftErrors, totalDftStandardDeviation, perStandardDeviation, loopTotal, gammas)

# VARYING THE VALUE OF EPSILON: LESS THAN OR EQUAL TO 1
def runDftVaryEps1():
    perErrors = list()
    recErrors = list()
    totalDftErrors = list()
    totalDftStandardDeviation = list()
    perStandardDeviation = list()
    loopTotal = list()
    gammas = list()

    for eps in epsset1:
        print(f"\nProcessing the optimal summation result with DFT for the value eps = {eps}.")
        runDft(4, eps, epsset1, tconst, kconst, mconst, eps, nconst, heartbeatDataConstDConstN, totalVectorConstDConstN, 20, 10, perErrors, recErrors, totalDftErrors, totalDftStandardDeviation, perStandardDeviation, loopTotal, gammas)

    afterDftLoopStats(4, eps, epsset1, 20, 10, perErrors, recErrors, totalDftErrors, totalDftStandardDeviation, perStandardDeviation, loopTotal, gammas)

# VARYING THE VALUE OF EPSILON: GREATER THAN 1
def runDftVaryEps2():
    perErrors = list()
    recErrors = list()
    totalDftErrors = list()
    totalDftStandardDeviation = list()
    perStandardDeviation = list()
    loopTotal = list()
    gammas = list()

    for eps in epsset2:
        print(f"\nProcessing the optimal summation result with DFT for the value eps = {eps}.")
        runDft(5, eps, epsset2, tconst, kconst, mconst, eps, nconst, heartbeatDataConstDConstN, totalVectorConstDConstN, 2, 2, perErrors, recErrors, totalDftErrors, totalDftStandardDeviation, perStandardDeviation, loopTotal, gammas)

    afterDftLoopStats(5, eps, epsset2, 2, 2, perErrors, recErrors, totalDftErrors, totalDftStandardDeviation, perStandardDeviation, loopTotal, gammas)

# VARYING THE NUMBER OF VECTORS N USED
def runDftVaryN():
    perErrors = list()
    recErrors = list()
    totalDftErrors = list()
    totalDftStandardDeviation = list()
    perStandardDeviation = list()
    loopTotal = list()
    gammas = list()

    for n in nset:
        print(f"\nProcessing the optimal summation result with DFT for the value n = {n}.")
        runDft(6, n, nset, tconst, kconst, mconst, epsconst, n, heartbeatDataConstDVaryN, totalVectorConstDVaryN, 0.001, 18, perErrors, recErrors, totalDftErrors, totalDftStandardDeviation, perStandardDeviation, loopTotal, gammas)

    afterDftLoopStats(6, n, nset, 0.001, 18, perErrors, recErrors, totalDftErrors, totalDftStandardDeviation, perStandardDeviation, loopTotal, gammas)

# CALLING ALL OF THE ABOVE FUNCTIONS
readDataConstDConstN()
readDataVaryDConstN()
readDataConstDVaryN()

runBasicVaryT()
runBasicVaryK()
runBasicVaryD()
runBasicVaryEps1()
runBasicVaryEps2()
runBasicVaryN()

runDftVaryT()
runDftVaryK()
runDftVaryM()
runDftVaryEps1()
runDftVaryEps2()
runDftVaryN()

print("Thank you for using the Shuffle Model for Vectors.")