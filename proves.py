import numpy as np

## DEFININT POLICY...

a = np.zeros((2, 91, 31, 31))

# la primera dimensio 0..1
# la segona dimensio 0..90
# la tercera dimensio 0..30
# la quarta dimensio 0..30

print(a.shape)

a[0][12][13][0] = +1 # aixo és l'accio per quan estem a l'estat 1 jvb, 12 users, carregat a 65% el 1r, 0 % el segon...

print(a[1][40][20][20])


number= 36
number_decimal = float(number)/5
number_rounded = round(number_decimal)
number_final = int(number_rounded*5)
print(number_final)


# importing package
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

ax.set(xlabel='Trainning rounds', ylabel='Price after a week(€)',title='Learning rate = 0.1')

# create data
#x =  [500000,1000000,2000000,4000000]
x =  [16000000,24000000,32000000,40000000,48000000,56000000,64000000,72000000,80000000,88000000,96000000,104000000,112000000,120000000]


y2 = [27.13, 27.13, 27.13, 27.13, 27.13, 27.13, 27.13, 27.13, 27.13, 27.13, 27.13, 27.13, 27.13, 27.13]
y3 = [26.78, 26.78, 26.78, 26.78, 26.78, 26.78, 26.78, 26.78, 26.78, 26.78, 26.78, 26.78, 26.78, 26.78]
y4 = [27.06, 27.06, 27.06, 27.06, 27.06, 27.06, 27.06, 27.06, 27.06, 27.06, 27.06, 27.06, 27.06, 27.06]

y_rl = [29.47,27.44,26.25,26.19]

y_rl_2 = [90.80,62.22,48.68,43.35,42.84,37.77,38.84,35.12,34.54,35.02,33.85,32.05,31.05,30.05]

y_rl_3 = [31.56,29.94,28.25,28.18,27.62,26.84,27.08, 26.53, 26.83, 26.77,26.33,27.41,26.25,26.14]

#y_rl_3 = [90.80,90.80,90.80,90.80,90.80,90.80,90.80,90.80,26.83,26.77,26.33]



# plot lines

ax.plot(x, y2, label = "Threshold 1", linestyle=":")
ax.plot(x, y3, label = "Threshold 2", linestyle=":")
ax.plot(x, y4, label = "Threshold 3", linestyle=":")

#ax.plot(x, y_rl, label = "RL", linestyle="-")

ax.plot(x, y_rl_2, label = "RL Formulation A", linestyle="-")

ax.plot(x, y_rl_3, label = "RL Formulation B", linestyle="-")




plt.legend()
plt.show()


fig.savefig("test.png")


from random import *
list_rand=[]
for ind in range(4000):
    list_rand.append(randint(15, 20))

print (list_rand)



