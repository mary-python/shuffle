import math, re; import matplotlib.pyplot as plt; from decimal import *
d = 150; t = 1; width = 0.35
labels = list(); perErrors = list(); recErrors = list(); totalErrors = list(); totalStandardDeviation = list()

with open("errortemp.txt") as reader:
    for line in reader:
        tab = line.split()
        labels.append(f'{int(tab[0])}%')
        perErrors.append((Decimal(tab[1])))
        recErrors.append((Decimal(tab[2])))
        totalErrors.append((Decimal(tab[3])))
        totalStandardDeviation.append((Decimal(tab[4])))

plt.bar(labels, perErrors, width, label = 'Perturbation error', alpha = 0.6, color = 'r', edgecolor = 'k')
plt.bar(labels, recErrors, width, bottom = perErrors, label = 'Reconstruction error', alpha = 0.6, color = 'c', edgecolor = 'k')
plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

plt.ticklabel_format(axis = 'y', style = 'plain')
plt.xticks(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
plt.ylabel('Total experimental MSE')
plt.xlabel('% of Fourier coefficients retained', labelpad = 8)

plt.title('Ratio between experimental errors by % of Fourier coefficients retained')
plt.legend(); plt.tight_layout; plt.draw(); plt.savefig("errorchart" + str(d) + "d" + str(t) + "t.png")