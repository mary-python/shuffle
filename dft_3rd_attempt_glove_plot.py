import math, re; import matplotlib.pyplot as plt; from decimal import *
V = 10; width = 0.35
labels = list(); perErrors = list(); recErrors = list()

with open("errortemp.txt") as reader:
    for line in reader:
        tab = line.split()
        labels.append(f'{int(tab[0])}%')
        perErrors.append((Decimal(tab[1])))
        recErrors.append((Decimal(tab[2])))

plt.bar(labels, perErrors, width, label = 'Perturbation error', alpha = 0.6, color = 'r', edgecolor = 'k')
plt.bar(labels, recErrors, width, bottom = perErrors, label = 'Reconstruction error', alpha = 0.6, color = 'c', edgecolor = 'k')

plt.ticklabel_format(axis = 'y', style = 'plain')
plt.xticks(['1%', '2%', '3%', '4%', '5%', '6%', '7%', '8%', '9%', '10%'])
plt.ylabel('Total experimental MSE')
plt.xlabel('% of Fourier coefficients retained', labelpad = 8)

plt.title('Ratio between experimental errors by % of Fourier coefficients retained')
plt.legend(); plt.tight_layout; plt.draw(); plt.savefig("errorchart.png")