import math, re; import matplotlib.pyplot as plt
V = 10; s = 15; width = 0.35
perErrors = list(); recErrors = list(); labels = list()

with open("errortemp" + str(s) + ".txt") as reader:
    for value in range(0, V):
        line1 = reader.readline()
        perErrors.append(line1)
        line2 = reader.readline()
        recErrors.append(line2)
        line3 = reader.readline()
        labels.append(line3)

print(perErrors)
print(recErrors)
print(labels)

plt.bar(labels, perErrors, width, label = 'Perturbation error', alpha = 0.6, color = 'r', edgecolor = 'k')
plt.bar(labels, recErrors, width, bottom = perErrors, label = 'Reconstruction error', alpha = 0.6, color = 'c', edgecolor = 'k')

plt.ticklabel_format(axis = 'y', style = 'plain')
plt.set_xticks(['1%', '2%', '3%', '4%', '5%', '6%', '7%', '8%', '9%', '10%'])
plt.set_ylabel('Total experimental MSE')
plt.set_xlabel('% of Fourier coefficients retained', labelpad = 20)

plt.set_title('Ratio between experimental errors by % of Fourier coefficients retained')
plt.legend()
plt.draw()

if s == 5:
    plt.savefig("errorchartfactor0" + str(s) + ".png")
else:
    plt.savefig("errorchartfactor" + str(s) + ".png")