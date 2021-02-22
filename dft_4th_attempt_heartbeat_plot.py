import math, re; import matplotlib.pyplot as plt; from decimal import *
d = 188; t = 2; V = 10; width = 0.35
labels = list(); perErrors = list(); recErrors = list(); totalErrors = list(); totalStandardDeviation = list()

with open("errortemp.txt") as reader:
    for line in reader:
        tab = line.split()
        labels.append(f'{round(float(tab[0]), 1)}%')
        perErrors.append((Decimal(tab[1])))
        recErrors.append((Decimal(tab[2])))
        totalErrors.append((Decimal(tab[3])))
        totalStandardDeviation.append((Decimal(tab[4])))

plt.bar(labels, perErrors, width, label = 'Perturbation error', alpha = 0.6, color = 'r', edgecolor = 'k')
plt.bar(labels, recErrors, width, bottom = perErrors, label = 'Reconstruction error', alpha = 0.6, color = 'c', edgecolor = 'k')
plt.errorbar(labels, totalErrors, totalStandardDeviation, linestyle = 'None', capsize = 2, color = 'g')

plt.ticklabel_format(axis = 'y', style = 'plain')
plt.xticks(['2.1%', '4.3%', '6.4%', '8.5%', '10.6%', '12.8%', '14.9%', '17.0%', '19.1%', '21.3%'])
plt.ylabel('Total experimental MSE')
plt.xlabel('% of Fourier coefficients retained', labelpad = 8)

plt.title('Ratio between experimental errors by % of Fourier coefficients retained')
plt.legend(); plt.tight_layout; plt.draw(); plt.savefig("errorchart" + str(d) + "d" + str(t) + "t.png")