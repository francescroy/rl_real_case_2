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
x =  [500000,1000000,2000000,4000000]


y2 = [27.13, 27.13, 27.13, 27.13]
y3 = [26.78, 26.78, 26.78, 26.78]
y4 = [27.06, 27.06, 27.06, 27.06]

y_rl = [29.47,27.44,26.25,26.19]

y_rl_2 = [27.84,26.90,26.80,26.80]

# plot lines

ax.plot(x, y2, label = "Threshold 1", linestyle=":")
ax.plot(x, y3, label = "Threshold 2", linestyle=":")
ax.plot(x, y4, label = "Threshold 3", linestyle=":")

ax.plot(x, y_rl, label = "RL", linestyle="-")

#ax.plot(x, y_rl_2, label = "Rl 2", linestyle="-")


plt.legend()
plt.show()


fig.savefig("test.png")









