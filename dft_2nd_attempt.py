import random, math, time; import numpy as np
from scipy.fftpack import rfft, irfft
import matplotlib.pyplot as plt; from matplotlib.ticker import PercentFormatter
from brokenaxes import brokenaxes
# Import: 1) "random", which uses "random.SystemRandom().random()" to generate a cryptographically secure 
# random negative exponential distribution, with values stored in a list named "randomVector";
# and "random.sample()" to uniformly sample a coordinate from each list "randomVector";
# 2) "math", which uses "math.sqrt()" to calculate the value of the parameter gamma;
# "math.log()" and "math.exp()" to generate the random negative exponential distribution;
# and "math.floor()" to calculate the proximity of each coordinate to each of the two nearest uniformly spaced points,
# and to convert the time taken between various points in the program from seconds to hours, minutes and seconds;
# 3) "time", which uses "time.perf_counter()" to measure the time taken between various points in the program;
# 4) "numpy" as "np", which uses "np.log()" in the setting of parameters "gamma", "comparison" and "dftComparison";
# "np.random.binomial()" to generate random Bernoulli distributions for rounding and randomised response;
# "np.random.randint()" to return an integer between 0 and k as an alternative output for the randomised response;
# and "np.ones()" to set the weights of the frequencies of each histogram to plot them as percentages of the total frequency;
# 5) "scipy.fftpack", which uses "scipy.fftpack.rfft()" to calculate the Discrete Fourier Transform (DFT) of each list "randomVector";
# and "scipy.fftpack.irfft()" to calculate the Inverse Discrete Fourier Transform (IDFT) of the padded, debiased output of the algorithm.
# 6) "matplotlib.pyplot" as "plt", which uses "plt.style.use()" to set the style of the histograms;
# "plt.subplot()" to split the layout of the plot into four subplots, each containing a histogram;
# "plt.xlim()" to define the upper and lower limits of the x-axis of each plot/histogram;
# "plt.gca()", which uses "plt.gca().yaxis.set_major_formatter()" to define the limits of the y-axis of each plot/histogram;
# "plt.gca().set()" to set the title, label the x-axis and label the y-axis of each plot/histogram;
# "plt.tight_layout()" to adjust the positioning of subplot axes to achieve a good layout for the plot;
# "plt.draw()" to redraw the current figure and "plt.savefig()" to save the current figure in the current folder;
# "plt.clf()" to clear the current figure and "plt.cla()" to clear the current axes so a new figure can be drawn;
# 7) "matplotlib.ticker", which uses "PercentFormatter" to format the labels of the y-axis of each plot/histogram as a percentage.
# and 8) "brokenaxes", which uses "brokenaxes" to specify breaks in the y-axis of the stacked bar chart.

startTime = time.perf_counter()
# This records the start time of the program, to be used when calculating the total time between particular points in the program.

d = 1000; k = 10; n = 100000; eps = 0.1; dta = 0.479; R = 1; V = 10; s = 25; v = 5
# These are the main variables that have been subject to change during the process of creating this algorithm. The above values
# are now set throughout the program to ensure an optimal result.

print("The purpose of this Python program is to compare the errors of two algorithms that extend the Shuffle Model of Differential ")
print("Privacy to Vectors. \n")

print("The first algorithm, Optimal Summation in the Shuffle Model (OSS), addresses the problem of computing the sum of n real ")
print("vectors in a way that guarantees (eps, dta)-Differential Privacy. Each of n users inputs a d-dimensional vector, from a ")
print("negative exponential distribution with scaling factor and steepness s. This d-dimensional vector consists of real ")
print("coordinates between 0 and 1, of which a small number t coordinates are uniformly sampled from each vector. It has been ")
print(f"established that the choice t = 1 produces the optimal error for the algorithm. \n")

print("The selected coordinates are rounded to one of k uniformly spaced points that cover the entire domain of the coordinates. ")
print("Each coordinate is moved to one of the two nearest such points, determined by a random Bernoulli distribution. A ")
print("randomised response mechanism is applied to each rounded coordinate, which determines whether the rounded coordinate ")
print("will be changed before it is submitted. This choice is determined by a parameter gamma which ensures the algorithm ")
print("satisfies (eps, dta)-Differential Privacy. The indices of the submitted coordinates are recorded, as well as the total ")
print("of the submitted coordinates for every index group. The submitted coordinates are aggregated, descaled and debiased in ")
print("their index groups, forming a d-dimensional vector output. \n")

print("The Mean Squared Error (MSE) will be used to measure the average squared difference in the comparison between the inputs ")
print("and outputs of the algorithm. The experimental MSE will be compared with a theoretical upper bound, which was established ")
print("using advanced composition results. \n")

print("The second algorithm, Fourier Summation Algorithm (FSA), transforms OSS to the Fourier domain. The Fourier domain has ")
print("the property that selecting a small number m coefficients of each transformed vector retains most of the data from the ")
print("users. This means that OSS only needs to be applied to m-dimensional vectors instead of d-dimensional vectors, which ")
print("theoretically improves the MSE. \n")

print("However, the MSE of OSS, or the perturbation error, forms only one part of the total error. To return to the original ")
print("domain with a d-dimensional vector, the outputted m-dimensional vector is padded with zeros before the inverse transform ")
print("is applied. The error caused by the transform to the Fourier domain and back is called the reconstruction error. ")
print("This program will explore several different values of m, investigating their effect on the perturbation and reconstruction ")
print("errors. \n")

print(f"Throughout the program, the values d = {d}, k = {k}, n = {n}, eps = {eps} and dta = {dta} will be set. These choices ")
print(f"ensure that the parameter gamma is set appropriately for the subsequent experiments. There will be {R} repeats of {V} ")
print(f"different values of m, and {v} different values of s.")

gamma = max((((14*k*(np.log(2/dta))))/((n-1)*(eps**2))), (27*k)/((n-1)*eps))
# The parameter gamma depends on k, n, eps and dta. It has already been shown that setting gamma as above guarantees
# (eps, dta)-Differential Privacy, using advanced composition results.

loopTotal = list()
# Each cell of this list will be used to record the total time for the algorithm to investigate each value of m.

perErrors = list()
recErrors = list()
labels = list()
# These lists will be used to record the perturbation and reconstruction errors of the algorithm, as well as the percentage
# of Fourier coefficients retained in each case, to plot on a bar graph at the end of the program.

for value in range(0, V):
# This loop is completed over all V different values of m.

    loopTime = time.perf_counter(); m = (value + 1)*(10)
    # At the start of this "for loop", the current time is recorded, and will be subtracted from the time recorded at the
    # end of the loop. For the first loop, the value of m is set to 10, and is increased by 10 for every loop thereafter.

    minIndexTracker = 0; maxIndexTracker = 0; avgIndexTracker = 0
    minSubmittedTotal = 0; maxSubmittedTotal = 0; avgSubmittedTotal = 0
    minDebiasedTotal = 0; maxDebiasedTotal = 0; avgDebiasedTotal = 0
    totalMeanSquaredError = 0
    # The purpose of the above variables are to store the total of the numerical statistics for the Optimal Summation algorithm.

    minDftIndexTracker = 0; maxDftIndexTracker = 0; avgDftIndexTracker = 0
    minDftSubmittedTotal = 0; maxDftSubmittedTotal = 0; avgDftSubmittedTotal = 0
    minFinalTotal = 0; maxFinalTotal = 0; avgFinalTotal = 0
    totalDftMeanSquaredError = 0; totalReconstructionError = 0
    # The purpose of the above variables are to store the total of the numerical statistics for the Fourier Summation Algorithm.
    # By setting all of these variables to 0 when each value of m is changed, it is possible to append each statistic to the
    # corresponding variable, so that a "running total" is kept that can subsequently be divided by the number of repeats.
    
    sampledList = list(); dftSampledList = list()
    debiasedList = list(); dftDebiasedList = list()
    # These lists will be used to store the values of the sampled and returned coordinates in the original and Fourier domains.
    # The distribution of these values will be plotted on separate histograms for ease of comparison. Again, each histogram will
    # display the average distribution over all repeats for each value of m, and will be reset when the value of m changes.

    for j in range(0, R):
    # This loop is completed over all R repeats of the V different values of m.

        indexTracker = [0]*d; dftIndexTracker = [0]*m
        submittedTotal = [0]*d; dftSubmittedTotal = [0]*m
        # These lists will be used to record the indices of the submitted coordinates in the original and Fourier domains, 
        # as well as the total of the submitted coordinates in the original and Fourier domains for every index group.

        meanSquaredError = 0; dftMeanSquaredError = 0
        reconstructionError = 0; sampledError = 0; returnedError = 0
        # These variables will aggregate the various differences between the input and output of each user. The Mean Squared
        # Error (MSE) measures the average squared difference between the sampled and returned coordinates in the Optimal
        # Summation in the Shuffle Model (OSS). The reconstruction error measures the additional error in the Fourier Summation
        # Algorithm (FSA), caused by the transform to the Fourier domain and back. The sampled error will measure the total 
        # absolute difference in the frequency distributions of the sampled coordinates in the original and Fourier domains. 
        # These frequency distributions will be plotted as histograms, where the sampled error can be represented by the total 
        # absolute difference in the height of the bars, where each bar represents an interval of 0.05.

        print(f"\n Processing repeat {j+1} for the value m = {m}.")

        from progress.bar import FillingSquaresBar
        bar = FillingSquaresBar(max=n, suffix = '%(percent) d%% : %(elapsed)ds elapsed')
        # This "progress bar" displays a series of boxes being filled to indicate the current progress of a long running operation,
        # in this case the progress of each repeat of the program.

        for i in range(0, n):
        # This loop is completed over all n users in the R repeats of the V different values of m.

            randomVector = list(); sampledCoord = int(); submittedCoord = int()
            descaledCoord = int(); debiasedCoord = int()
            # These lists will be used to store the inputted vector from each user into OSS, as well as each coordinate resulting 
            # from each of the subsequent steps of sampling, submitting, descaling and debiasing.

            dftSampledCoord = int(); dftSubmittedCoord = int()
            dftDescaledCoord = int(); dftDebiasedCoord = int()
            # In the FSA, each inputted vector is transformed to the Fourier domain, after which OSS begins. The above lists will be 
            # used to store the coordinate resulting from each of the subsequent steps of sampling, submitting, descaling and debiasing.

            for i in range(0, d):
                randomVector.append(-(math.log(1 - (1 - math.exp(-s))*(random.SystemRandom().random())))/s)
            # First, the program applies the steps of OSS. This loop represents the creation of a cryptographically secure random 
            # d-dimensional vector for each user, in which its coordinates form a negative exponential distribution, which is 
            # suitable for the calculation of Fourier coefficients.

            sampledPair = random.sample(list(enumerate(randomVector)), 1)
            # One coordinate is uniformly sampled from each vector, with the index and value of each coordinate recorded in an
            # enumerated list.

            for idx, smp in list(enumerate(sampledPair)):
                sampledCoord = smp[1]
                sampledList.append(smp[1])
            # The value of each sampled coordinate is stored in two different lists. The former will be used in the calculation of 
            # the MSE for each particular user, before being reset for the next user. The latter will not be reset until the sampled
            # coordinates from all users are aggregated and plotted on a histogram.

            roundedPair = [(idx, (math.floor(smp*k) + np.random.binomial(1, smp*k - math.floor(smp*k)))) for idx, smp in sampledPair]
            # Each sampled coordinate is scaled by a factor of k, then it is rounded to one of the two nearest integers. The closer
            # the coordinate is to an integer, the more likely it will be rounded to that integer.

            b = np.random.binomial(1, gamma)
            # A Bernoulli distribution is generated with parameter gamma. This parameter is approximately 0.2, which indicates
            # the probability that b == 1.

            if b == 0:
                submittedPair = roundedPair
            else:
                submittedPair = [(idx, (np.random.randint(0, k+1))) for idx, rnd in roundedPair]
            # This "if-else" statement represents a randomised response mechanism, which is applied to each rounded coordinate.
            # This mechanism uses the above Bernoulli distribution to determine whether the rounded coordinate is changed
            # before it is submitted. If b == 0, the rounded coordinate is submitted, but if b == 1, the rounded coordinate
            # is changed to a random integer picked uniformly between 0 and k (the original domain scaled by a factor of k).

            for idx, sbm in submittedPair:
                submittedTotal[idx] += sbm
                submittedCoord = sbm
                indexTracker[idx] += 1
            # The value of each submitted coordinate is stored in two different lists. In the former, the appropriate index is 
            # incremented by the value which has been submitted, and will eventually hold all submitted values for all users in 
            # this particular repeat. The latter will be used in the calculation of the MSE for each particular user, before 
            # being reset for the next user. The appropriate index of the index tracker is incremented by 1.

            descaledCoord = submittedCoord/k
            debiasedCoord = (5/6)*(abs((descaledCoord - gamma/2)/(1 - gamma)))
            # The scaling and bias applied to each sampled coordinate in the rounding step is removed. Some coordinates end up
            # just outside the original domain, so the result is scaled by a factor of 5/6 to fix this issue. The result will be 
            # used in the calculation of the MSE for each particular user, before being reset for the next user. It is much more
            # efficient to run the descaling and debiasing steps both within this loop and after the submitted coordinates from all
            # users have been aggregated, so that the MSE can be calculated on one pair of coordinates at a time, separately from
            # the calculation of the d-dimensional vector output.

            debiasedList.append(debiasedCoord)
            # This list aggregates all descaled and debiased coordinates from all users to be plotted on a histogram.

            meanSquaredError += (debiasedCoord - sampledCoord)**2
            # The MSE variable above aggregates the MSE calculated for all users, and is reset only after a repeat has been completed.
            
            dftVector = ((abs(rfft(randomVector)))/7).tolist()
            # Now, the program applies the steps of the FSA. The same random vector used for OSS earlier is now transformed to the 
            # Fourier domain by a DFT. The absolute value is taken and the resulting vector is scaled down by a factor of 7 to 
            # ensure its distribution is optimally comparable to that of the OSS.

            slicedDftVector = dftVector[0:m]
            # Only m Fourier coefficients are kept out of the d Fourier coefficients calculated above. This means that OSS
            # only needs to be applied to m-dimensional vectors instead of d-dimensional vectors, which theoretically
            # improves the MSE.
            
            dftSampledPair = random.sample(list(enumerate(slicedDftVector)), 1)
            # The steps of OSS are now applied to each sliced vector, with all subsequent steps almost identical to the steps of the
            # standalone OSS, except the names of the lists have an additional "dft" added to distinguish them from those of the
            # standalone OSS.

            for idx, smp in list(enumerate(dftSampledPair)):
                dftSampledCoord = smp[1]
                dftSampledList.append(smp[1])
            # The value of each sampled coordinate is stored in two different lists. The former will be used in the calculation of 
            # the MSE for each particular user, before being reset for the next user. The latter will not be reset until the sampled
            # coordinates from all users are aggregated and plotted on a histogram.

            dftRoundedPair = [(idx, (math.floor(smp*k) + np.random.binomial(1, smp*k - math.floor(smp*k))))\
                for idx, smp in dftSampledPair]
            # Each sampled coordinate is scaled by a factor of k, then it is rounded to one of the two nearest integers. The closer
            # the coordinate is to an integer, the more likely it will be rounded to that integer.

            b = np.random.binomial(1, gamma)
            # A Bernoulli distribution is generated with parameter gamma. This parameter is approximately 0.2, which indicates
            # the probability that b == 1.

            if b == 0:
                dftSubmittedPair = dftRoundedPair
            else:
                dftSubmittedPair = [(idx, (np.random.randint(0, k+1))) for idx, rnd in dftRoundedPair]
            # This "if-else" statement represents a randomised response mechanism, which is applied to each rounded coordinate.
            # This mechanism uses the above Bernoulli distribution to determine whether the rounded coordinate is changed
            # before it is submitted. If b == 0, the rounded coordinate is submitted, but if b == 1, the rounded coordinate
            # is changed to a random integer picked uniformly between 0 and k (the original domain scaled by a factor of k).

            for idx, sbm in dftSubmittedPair:
                dftSubmittedTotal[idx] += sbm
                dftSubmittedCoord = sbm
                dftIndexTracker[idx] += 1
            # The value of each submitted coordinate is stored in two different lists. In the former, the appropriate index is 
            # incremented by the value which has been submitted, and will eventually hold all submitted values for all users in 
            # this particular repeat. The latter will be used in the calculation of the MSE for each particular user, before 
            # being reset for the next user. The appropriate index of the index tracker is incremented by 1.

            dftDescaledCoord = dftSubmittedCoord/k
            dftDebiasedCoord = (5/6)*(abs((dftDescaledCoord - gamma/2)/(1 - gamma)))
            # The scaling and bias applied to each sampled coordinate in the rounding step is removed. Some coordinates end up
            # just outside the original domain, so the result is scaled by a factor of 5/6 to fix this issue. The result will be 
            # used in the calculation of the MSE for each particular user, before being reset for the next user. It is much more
            # efficient to run the descaling and debiasing steps both within this loop and after the submitted coordinates from all
            # users have been aggregated, so that the MSE can be calculated on one pair of coordinates at a time, separately from
            # the calculation of the d-dimensional vector output.

            dftDebiasedList.append(dftDebiasedCoord)
            # This list aggregates all descaled and debiased coordinates from all users to be plotted on a histogram.

            dftMeanSquaredError += (dftDebiasedCoord - dftSampledCoord)**2
            # The MSE variable above aggregates the MSE calculated for all users, and is reset only after a repeat has been completed.
    
            bar.next()
        bar.finish()
        # The two lines above advance and finish the operation of the "progress bar" created at the beginning of the loop.
        # Now that all data has been collected from all n users, the next step is to aggregate this data.

        minIndexTracker += min(indexTracker)
        maxIndexTracker += max(indexTracker)
        avgIndexTracker += (sum(indexTracker))/d
        # The maximum, minimum and average frequency of the original indices sampled by OSS are aggregated over all repeats 
        # processed for the current value of m.

        minSubmittedTotal += min(submittedTotal)
        maxSubmittedTotal += max(submittedTotal)
        avgSubmittedTotal += (sum(submittedTotal))/d
        # The maximum, minimum and average total submission by each original index group in OSS are aggregated over all repeats 
        # processed for the current value of m.

        descaledTotal = [idx/k for idx in submittedTotal]
        mergedTracker = tuple(zip(indexTracker, descaledTotal))
        debiasedTotal = [((5/6)*(abs((z - gamma/2)/(1 - gamma)))) for count, z in mergedTracker]
        # The submitted coordinates in OSS, now aggregated from all users, are descaled and debiased in their index groups, 
        # forming a d-dimensional vector output.

        minDebiasedTotal += min(debiasedTotal)
        maxDebiasedTotal += max(debiasedTotal)
        avgDebiasedTotal += (sum(debiasedTotal))/d
        # The maximum, minimum and average output of OSS in the original domain are aggregated over all repeats processed for 
        # the current value of m.

        totalMeanSquaredError += meanSquaredError
        # The MSE of OSS is aggregated over all repeats processed for the current value of m.

        minDftIndexTracker += min(dftIndexTracker)
        maxDftIndexTracker += max(dftIndexTracker)
        avgDftIndexTracker += (sum(dftIndexTracker))/m
        # The maximum, minimum and average frequency of the original indices sampled by the FSA are aggregated over all repeats 
        # processed for the current value of m.

        minDftSubmittedTotal += min(dftSubmittedTotal)
        maxDftSubmittedTotal += max(dftSubmittedTotal)
        avgDftSubmittedTotal += (sum(dftSubmittedTotal))/m
        # The maximum, minimum and average total submission by each original index group in the FSA are aggregated over all repeats 
        # processed for the current value of m.

        dftDescaledTotal = [idx/k for idx in dftSubmittedTotal]
        dftMergedTracker = tuple(zip(dftIndexTracker, dftDescaledTotal))
        dftDebiasedTotal = [((5/6)*(abs((z - gamma/2)/(1 - gamma)))) for count, z in dftMergedTracker]
        # The submitted coordinates in the FSA, now aggregated from all users, are descaled and debiased in their index groups, 
        # forming an m-dimensional vector output.

        paddedTotal = dftDebiasedTotal + [0]*(d-m)
        finalTotal = (irfft(paddedTotal)).tolist()
        # To return to the original domain with a d-dimensional vector, the outputted m-dimensional vector is padded with zeros
        # before the IDFT is applied.

        minFinalTotal += min(finalTotal)
        maxFinalTotal += max(finalTotal)
        avgFinalTotal += (sum(finalTotal))/d
        # The maximum, minimum and average output of the FSA in the original domain are aggregated over all repeats processed for 
        # the current value of m.

        totalDftMeanSquaredError += dftMeanSquaredError
        # The MSE of the FSA is aggregated over all repeats processed for the current value of m.

        for a, b in zip(debiasedTotal, finalTotal):
            reconstructionError += (a - b)**2
        totalReconstructionError += reconstructionError
        # The output of the FSA is compared in a MSE fashion with the output of the OSS to form the reconstruction error of the FSA.

    if s == 5:
        datafile = open("fourier" + str(m) + "factor0" + str(s) + ".txt", "w")
    else:
        datafile = open("fourier" + str(m) + "factor" + str(s) + ".txt", "w")

    datafile.write(f"Number of Fourier coefficients m: {m} \n")
    datafile.write(f"Below are the average figures across the {R} repeats that were just performed. \n\n")

    datafile.write(f"Case 1: Optimal Summation in the Shuffle Model \n")
    min1 = round(minIndexTracker/R); max1 = round(maxIndexTracker/R); avg1 = round(avgIndexTracker/R)
    datafile.write(f"Frequency of original indices sampled: min {min1}, max {max1}, avg {avg1} \n")
    min2 = round(minSubmittedTotal/R); max2 = round(maxSubmittedTotal/R); avg2 = round(avgSubmittedTotal/R)
    datafile.write(f"Total submission by each original index group: min {min2}, max {max2}, avg {avg2} \n")
    min3 = round(minDebiasedTotal/R, 1); max3 = round(maxDebiasedTotal/R, 1); avg3 = round(avgDebiasedTotal/R, 1)
    datafile.write(f"Output in the original domain: min {min3}, max {max3}, avg {avg3} \n\n")

    comparison = (2*(14**(2/3))*(d**(2/3))*(n**(1/3))*(np.log(1/dta))*(np.log(2/dta)))/(((1-gamma)**2)*(eps**(4/3)))
    datafile.write(f"Theoretical Upper Bound for MSE: {round(float(comparison)/10)*10} \n")
    datafile.write(f"Experimental MSE: {round(float(totalMeanSquaredError/R)/10)*10} \n")
    error1 = round((100)*((totalMeanSquaredError/R)/comparison), 2)
    datafile.write(f"Experimental MSE was {error1}% of the theoretical upper bound for MSE. \n\n")

    datafile.write(f"Case 2: Fourier Summation Algorithm \n")
    min4 = round(float(minDftIndexTracker/R)/10)*10; max4 = round(float(maxDftIndexTracker/R)/10)*10
    avg4 = round(float(avgDftIndexTracker/R)/10)*10
    datafile.write(f"Frequency of original indices sampled: min {min4}, max {max4}, avg {avg4} \n")
    min5 = round(float(minDftSubmittedTotal/R)/10)*10; max5 = round(float(maxDftSubmittedTotal/R)/1000)*1000
    avg5 = round(float(avgDftSubmittedTotal/R)/10)*10
    datafile.write(f"Total submission by each original index group: min {min5}, max {max5}, avg {avg5} \n")

    min6 = round(minFinalTotal/R, 2); max6 = round(maxFinalTotal/R); avg6 = round(avgFinalTotal/R, 1)
    datafile.write(f"Output in the original domain: min {min6}, max {max6}, avg {avg6} \n\n")

    dftComparison = (2*(14**(2/3))*(m**(2/3))*(n**(1/3))*(np.log(1/dta))*(np.log(2/dta)))/(((1-gamma)**2)*(eps**(4/3)))
    datafile.write(f"Theoretical upper bound for perturbation error: {round(float(dftComparison)/10)*10} \n")
    datafile.write(f"Experimental perturbation error: {round(float(totalDftMeanSquaredError/R)/1000)*1000} \n")
    error2 = round((100)*((totalDftMeanSquaredError/R)/dftComparison), 1)
    datafile.write(f"Experimental perturbation error was {error2}% of the theoretical upper bound for perturbation error. \n")

    datafile.write(f"Experimental reconstruction error: {round(float(totalReconstructionError/R)/1000)*1000} \n")
    
    perErrors.append(totalDftMeanSquaredError/R)
    recErrors.append(totalReconstructionError/R)
    labels.append(f'{round(m/10)}%')

    datafile.write(f"Total experimental MSE: {round(float((totalDftMeanSquaredError/R) + (totalReconstructionError/R))/1000)*1000} \n")
    error3 = round((100)*((totalReconstructionError/R)/((totalDftMeanSquaredError/R) + (totalReconstructionError/R))), 1)
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
    # The purpose of the lines above are to plot four empty subplots and save the resulting figure, before resetting both the 
    # figure and the axes. This is a solution which I have come up with to fix an inconsistency with the size of the subplots:
    # the subplots of the first figure produced in the loop have been consistently smaller than those of the other figures in
    # the loop. The purpose of the "tight layout" and "zoomed window" commands are to adjust the subplot params so that the 
    # subplots fit into the figure area, and to maximise the size of the figure.

    plt.subplot(2, 2, 1)
    (freq1, bins1, patches) = plt.hist(sampledList, weights = np.ones(len(sampledList)) / len(sampledList),\
        bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],\
            alpha = 0.3, histtype = 'bar', color = 'g', edgecolor = 'k')
    # Plot the first subplot, containing a histogram detailing the frequencies of sampled coordinates in the original domain.
    # The frequencies are weighted as a fraction of the total number of sampled coordinates, so that they can be displayed as
    # a percentage of the total number of sampled coordinates.

    plt.xlim(0, 1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set(title = 'Histogram of sampled coordinates in the original domain', xlabel = 'Value', ylabel = 'Frequency')
    # Define the limits of the x-axis, the formatting of the y-axis, and the labels of the subplot.

    listFreq1 = freq1.tolist(); formattedFreq1 = list()
    for item in listFreq1:
        formattedFreq1.append(int(float(item*(len(sampledList))/R)))
    # For the purposes of displaying the average numerical frequencies, undo the weighting applied for the plot, and divide 
    # by the number of repeats to convert from a total to an average.

    datafile.write(f"Frequencies of sampled coordinates in the original domain: \n")
    datafile.write(f"{str(formattedFreq1)[1:-1]} \n")
    datafile.write(f"Total: {sum(formattedFreq1)} \n")
    datafile.write(f"Percentage of sampled coordinates between 0 and 1: {round((100)*(sum(formattedFreq1))/(sum(indexTracker)))}% \n\n")

    plt.subplot(2, 2, 2)
    (freq2, bins2, patches) = plt.hist(debiasedList, weights = np.ones(len(debiasedList)) / len(debiasedList),\
        bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],\
            alpha = 0.3, histtype = 'bar', color = 'b', edgecolor = 'k')
    # Plot the second subplot, containing a histogram detailing the frequencies of returned coordinates in the original domain.
    # The frequencies are weighted as a fraction of the total number of returned coordinates, so that they can be displayed as
    # a percentage of the total number of returned coordinates.
    
    plt.xlim(0, 1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set(title = 'Histogram of returned coordinates in the original domain', xlabel = 'Value', ylabel = 'Frequency')
    # Define the limits of the x-axis, the formatting of the y-axis, and the labels of the subplot.

    listFreq2 = freq2.tolist(); formattedFreq2 = list()
    for item in listFreq2:
        formattedFreq2.append(int(float(item*(len(debiasedList))/R)))
    # For the purposes of displaying the average numerical frequencies, undo the weighting applied for the plot, and divide 
    # by the number of repeats to convert from a total to an average.

    datafile.write(f"Frequencies of returned coordinates in the original domain: \n")
    datafile.write(f"{str(formattedFreq2)[1:-1]} \n")
    datafile.write(f"Total: {sum(formattedFreq2)} \n")
    datafile.write(f"Percentage of returned coordinates between 0 and 1: {round((100)*(sum(formattedFreq2))/(sum(indexTracker)))}% \n\n")

    plt.subplot(2, 2, 3)
    (freq3, bins3, patches) = plt.hist(dftSampledList, weights = np.ones(len(dftSampledList)) / len(dftSampledList),\
        bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],\
            alpha = 0.5, histtype = 'bar', color = 'g', edgecolor = 'k')
    # Plot the third subplot, containing a histogram detailing the frequencies of sampled coordinates in the Fourier domain.
    # The frequencies are weighted as a fraction of the total number of sampled coordinates, so that they can be displayed as
    # a percentage of the total number of sampled coordinates.

    plt.xlim(0, 1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set(title = 'Histogram of sampled coordinates in the Fourier domain', xlabel = 'Value', ylabel = 'Frequency')
    # Define the limits of the x-axis, the formatting of the y-axis, and the labels of the subplot.

    listFreq3 = freq3.tolist(); formattedFreq3 = list()
    for item in listFreq3:
        formattedFreq3.append(int(float(item*(len(dftSampledList))/R)))
    # For the purposes of displaying the average numerical frequencies, undo the weighting applied for the plot, and divide 
    # by the number of repeats to convert from a total to an average.
    
    datafile.write(f"Frequencies of sampled coordinates in the Fourier domain: \n")
    datafile.write(f"{str(formattedFreq3)[1:-1]} \n")
    datafile.write(f"Total: {sum(formattedFreq3)} \n")
    perc1 = round((100)*(sum(formattedFreq3))/(sum(dftIndexTracker)), 1)
    datafile.write(f"Percentage of sampled coordinates between 0 and 1: {perc1}% \n\n")

    plt.subplot(2, 2, 4)
    (freq4, bins4, patches) = plt.hist(dftDebiasedList, weights = np.ones(len(dftDebiasedList)) / len(dftDebiasedList),\
        bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],\
            alpha = 0.5, histtype = 'bar', color = 'b', edgecolor = 'k')
    # Plot the fourth subplot, containing a histogram detailing the frequencies of returned coordinates in the Fourier domain.
    # The frequencies are weighted as a fraction of the total number of returned coordinates, so that they can be displayed as
    # a percentage of the total number of returned coordinates.

    plt.xlim(0, 1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set(title = 'Histogram of returned coordinates in the Fourier domain', xlabel = 'Value', ylabel = 'Frequency')
    # Define the limits of the x-axis, the formatting of the y-axis, and the labels of the subplot.

    listFreq4 = freq4.tolist(); formattedFreq4 = list()
    for item in listFreq4:
        formattedFreq4.append(int(float(item*(len(dftDebiasedList))/R)))
    # For the purposes of displaying the average numerical frequencies, undo the weighting applied for the plot, and divide 
    # by the number of repeats to convert from a total to an average.

    datafile.write(f"Frequencies of returned coordinates in the Fourier domain: \n")
    datafile.write(f"{str(formattedFreq4)[1:-1]} \n")
    datafile.write(f"Total: {sum(formattedFreq4)} \n")
    perc2 = round((100)*(sum(formattedFreq4))/(sum(dftIndexTracker)), 1)
    datafile.write(f"Percentage of returned coordinates between 0 and 1: {perc2}% \n\n")

    for a, b in zip(formattedFreq1, formattedFreq3):
        sampledError += abs(a - b)
    datafile.write(f"Total difference between frequencies of sampled coordinates: {round(sampledError)} \n")
    datafile.write(f"Percentage difference: {round((100)*(sampledError)/(max(sum(formattedFreq1), sum(formattedFreq3))), 1)}% \n")

    for a, b in zip(formattedFreq2, formattedFreq4):
        returnedError += abs(a - b)
    datafile.write(f"Total difference between frequencies of returned coordinates: {round(returnedError)} \n")
    datafile.write(f"Percentage difference: {round((100)*(returnedError)/(max(sum(formattedFreq2), sum(formattedFreq4))), 1)}% \n\n")

    plt.tight_layout(); mng = plt.get_current_fig_manager(); mng.window.state('zoomed')
    plt.draw()
    
    if s == 5:
        plt.savefig("fourier" + str(m) + "factor0" + str(s) + ".png")
    else:
        plt.savefig("fourier" + str(m) + "factor" + str(s) + ".png")
    
    plt.clf(); plt.cla()
    # Finally, re-draw the four subplots and save the resulting figure, before resetting both the figure and the axes for the 
    # next loop. The purpose of the "tight layout" and "zoomed window" commands are to adjust the subplot params so that the 
    # subplots fit into the figure area, and to maximise the size of the figure.

    loopTotal.append(time.perf_counter() - loopTime)
    casetime = round(loopTotal[value]); casemins = math.floor(casetime/60); casehrs = math.floor(casemins/60)
    datafile.write(f"Total time for case m = {m}: {casehrs}h {casemins - (casehrs*60)}m {casetime - (casemins*60)}s")
    # Calculate the total time for the current case, and convert the result from seconds to hours, minutes and seconds.

width = 0.35
# Define the width of the bars in the final chart displaying the ratio between the experimental errors over all values of m
# in the algorithm.

plotPerErrors = [a/(10**5) for a in perErrors]
plotRecErrors = [b/(10**5) for b in recErrors]
# Remove the scientific multiplier from the bars to avoid any overlapping issues, instead adding it to the y-axis label.

if s == 5:
    limit1 = 16
    limit2 = math.floor((plotPerErrors[2] + plotRecErrors[2])*10)/10 - 0.2
    limit3 = limit2 + 0.5
    limit4 = math.floor((plotPerErrors[1] + plotRecErrors[1])*10)/10 - 0.1
    limit5 = limit4 + 0.5
    limit6 = math.floor((plotPerErrors[0] + plotRecErrors[0])*10)/10 - 0.2
    limit7 = limit6 + 0.5
    # Set the limits for three breaks in the y-axis based on the heights of the largest three bars. I chose to use a non-logarithmic
    # y-axis to preserve the scale of the ratios displayed in the stacked bar chart, but I had to chop off the top parts of the largest 
    # three bars to fit them on the graph whilst ensuring that the ratios of the smaller bars are visible.

    fig = plt.figure()
    bax = brokenaxes(ylims = ((0, limit1), (limit2, limit3), (limit4, limit5), (limit6, limit7)), hspace = .05)
    # Use "brokenaxes" from the package "brokenaxes" to construct three breaks in the y-axis using the limits set above.

elif s == 10:
    limit1 = 4
    limit2 = math.floor((plotPerErrors[2] + plotRecErrors[2])*10)/10 - 0.2
    limit3 = limit2 + 0.8
    limit4 = math.floor((plotPerErrors[1] + plotRecErrors[1])*10)/10 - 0.1
    limit5 = limit4 + 0.8
    limit6 = math.floor((plotPerErrors[0] + plotRecErrors[0])*10)/10 - 0.2
    limit7 = limit6 + 0.8
    # Do the same as the above but to suit the distribution when s = 10.

    fig = plt.figure()
    bax = brokenaxes(ylims = ((0, limit1), (limit2, limit3), (limit4, limit5), (limit6, limit7)), hspace = .05)

elif s == 15:
    limit1 = math.ceil((plotPerErrors[2] + plotRecErrors[2])*10)/10
    limit2 = math.floor((plotPerErrors[1] + plotRecErrors[1])*10)/10 - 0.1
    limit3 = limit2 + 0.4
    limit4 = math.floor((plotPerErrors[0] + plotRecErrors[0])*10)/10 - 0.2
    limit5 = limit4 + 0.4
    # Do the same as the above but to suit the distribution when s = 15.

    fig = plt.figure()
    bax = brokenaxes(ylims = ((0, limit1), (limit2, limit3), (limit4, limit5)), hspace = .05)
    # Use "brokenaxes" from the package "brokenaxes" to construct two breaks in the y-axis using the limits set above.

elif s == 20:
    limit1 = 1.3
    limit2 = math.floor((plotPerErrors[1] + plotRecErrors[1])*10)/10 - 0.1
    limit3 = limit2 + 0.3
    limit4 = math.floor((plotPerErrors[0] + plotRecErrors[0])*10)/10 - 0.2
    limit5 = limit4 + 0.3
    # Do the same as the above but to suit the distribution when s = 20.

    fig = plt.figure()
    bax = brokenaxes(ylims = ((0, limit1), (limit2, limit3), (limit4, limit5)), hspace = .05)

else:
    limit1 = 1.1
    limit2 = math.floor((plotPerErrors[1] + plotRecErrors[1])*10)/10 - 0.1
    limit3 = limit2 + 0.4
    limit4 = math.floor((plotPerErrors[0] + plotRecErrors[0])*10)/10 - 0.2
    limit5 = limit4 + 0.4
    # Do the same as the above but to suit the distribution when s = 25.

    fig = plt.figure()
    bax = brokenaxes(ylims = ((0, limit1), (limit2, limit3), (limit4, limit5)), hspace = .05) 

bax.bar(labels, plotPerErrors, width, label = 'Perturbation error', alpha = 0.6, color = 'r', edgecolor = 'k')
bax.bar(labels, plotRecErrors, width, bottom = plotPerErrors, label = 'Reconstruction error', alpha = 0.6, color = 'c', edgecolor = 'k')
# Plot the bars representing the perturbation errors for each value of m, then stack the bars representing the reconstruction
# errors on top.

bax.ticklabel_format(axis = 'y', style = 'plain')
bax.set_xticks(['1%', '2%', '3%', '4%', '5%', '6%', '7%', '8%', '9%', '10%'])
bax.set_ylabel('Total experimental MSE' + ' ' + 'x' + ' ' + '$10^5$')
bax.set_xlabel('% of Fourier coefficients retained', labelpad = 20)
# There is now no need to use a scientific style for the numbers on the y-axis. Add the scientific multiplier to the y-axis label, 
# and move the x-axis label downwards to avoid any overlapping issues.

bax.set_title('Ratio between experimental errors by % of Fourier coefficients retained')
bax.legend()
plt.draw()

if s == 5:
    plt.savefig("errorchartfactor0" + str(s) + ".png")
else:
    plt.savefig("errorchartfactor" + str(s) + ".png")

avgtime = round((sum(loopTotal))/(V)); avgmins = math.floor(avgtime/60); avghrs = math.floor(avgmins/60)
datafile.write(f"\nAverage time for each case: {avghrs}h {avgmins - (avghrs*60)}m {avgtime - (avgmins*60)}s \n")
# Calculate the average time for each case, and convert the result from seconds to hours, minutes and seconds.

totaltime = round(time.perf_counter() - startTime); totalmins = math.floor(totaltime/60); totalhrs = math.floor(totalmins/60)
datafile.write(f"Total time elapsed: {totalhrs}h {totalmins - (totalhrs*60)}m {totaltime - (totalmins*60)}s")
# Calculate the total time elapsed, and convert the result from seconds to hours, minutes and seconds.

datafile.close()
print("Thank you for using the Shuffle Model for Vectors.")