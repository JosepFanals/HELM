import numpy as np
import math
import matplotlib.pyplot as plt

U = 5  # equival a l'E
R = 2  # equival a R1
R2 = 3
P = 1.2
Vt = 0.026
Is = 0.000005

n = 200  # profunditat

Vd = np.zeros(n)  # sèries
Vl = np.zeros(n)
I1 = np.zeros(n)


I1[0] = U / R  # inicialització de les sèries
Vd[0] = Vt * math.log(1 + I1[0] / Is)
Vl[0] = P / I1[0]


def convVd(Vd, I, i):  # convolució pel càlcul de Vd[i]
    suma = 0
    for k in range(1, i):
        suma += k * Vd[k] * I[i - k]
    return suma


def convVlI(Vl, I1, i):  # convolució pel càlcul de Vl[i]
    suma = 0
    for k in range(i):
        suma = suma + Vl[k] * I1[i - k]
    return suma


for i in range(1, n):  # càlcul dels coeficients
    I1[i] = (1 / R + 1 / R2) * (-Vd[i - 1] - Vl[i - 1])
    Vd[i] = (i * Vt * I1[i] - convVd(Vd, I1, i)) / (i * (Is + I1[0]))
    Vl[i] = -convVlI(Vl, I1, i) / I1[0]

If = sum(I1)
Vdf = sum(Vd)
Vlf = sum(Vl)

print('I1: ' + str(If))
print('Vd: ' + str(Vdf))
print('Vl: ' + str(Vlf))
print('P: ' + str(Vlf * If))

Vdfinal = np.zeros(n)  # per tal de veure com evoluciona la tensió del díode
for j in range(n):
    Vdfinal[j] = np.sum([Vd[:(j+1)]])

print(Vdfinal)



