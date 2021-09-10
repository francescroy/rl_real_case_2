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


number= 39
number_decimal = float(number)/10
number_rounded = round(number_decimal)
number_final = int(number_rounded*10)
print(number_final)











"""
# importing package
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

ax.set(xlabel='Trainning rounds', ylabel='Price after a week(€)',title='Learning rate = 0.05')

# create data
x = [100000,200000,300000,500000,800000,1000000,2000000]
x_b = [500000,1000000,2000000,4000000,6000000]
x_c = [500000,1000000,2000000,4000000]

y1 = [40.75, 40.75, 40.75, 40.75, 40.75, 40.75, 40.75]
y2 = [39.23, 39.23, 39.23, 39.23, 39.23, 39.23, 39.23]
y3 = [37.71, 37.71, 37.71, 37.71, 37.71, 37.71, 37.71]
y4 = [36.17, 36.17, 36.17, 36.17, 36.17, 36.17, 36.17]

y1_b = [44.75, 44.75, 44.75, 44.75, 44.75]
y2_b = [40.84, 40.84, 40.84, 40.84, 40.84]
y3_b = [43.79, 43.79, 43.79, 43.79, 43.79]
y4_b = [37.28, 37.28, 37.28, 37.28, 37.28]

y1_c = [773.52, 773.52, 773.52, 773.52]
y2_c = [770.63, 770.63, 770.63, 770.63]
y3_c = [782.96, 782.96, 782.96, 782.96]
y4_c = [780.91, 780.91, 780.91, 780.91]


y_rl = [46.71,42.93,38.00,34.47,31.73,31.68,31.75]

y_rl_b = [48.71,44.77,36.07,31.63,31.42]

y_rl_c = [801.59,765.42,763.36,762.36]

ax.plot(x_c, y1_c, label = "Threshold tbm-50-20", linestyle=":")
ax.plot(x_c, y2_c, label = "Threshold tbm-50-30", linestyle=":")
ax.plot(x_c, y3_c, label = "Threshold tbm-60-20", linestyle=":")
ax.plot(x_c, y4_c, label = "Threshold tbm-60-30", linestyle=":")

ax.plot(x_c, y_rl_c, label = "RL", linestyle="-")

plt.legend()
plt.show()

fig.savefig("test.png")

"""

















from random import *
list_rand=[]
for ind in range(150):
    list_rand.append(randint(15, 3600-15))

for ind in range(150):
    list_rand.append(randint(15, 115))


print (list_rand)




