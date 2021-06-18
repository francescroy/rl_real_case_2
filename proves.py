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

ax.set(xlabel='Rounds', ylabel='Price(€)',title='Learning rate = 0.1')

# create data
x =  [25000,50000,100000,200000,1000000]
y1 = [28.1,28.1,28.1,28.1,28.1]
y2 = [33.75,33.75,33.75,33.75,33.75]
y3 = [37.21,33.47,26.95,26.02,25.32]
y4 = [56.94,46.38,37.05,31.00,26.93]

# plot lines
ax.plot(x, y1, label = "Threshold 1", linestyle="-")
ax.plot(x, y2, label = "Threshold 2", linestyle="--")
ax.plot(x, y3, label = "Rl 1", linestyle="-.")
ax.plot(x, y4, label = "Rl 2", linestyle=":")


plt.legend()
plt.show()


fig.savefig("test.png")









