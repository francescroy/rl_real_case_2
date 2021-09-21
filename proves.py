import numpy as np



number= 39
number_decimal = float(number)/10
number_rounded = round(number_decimal)
number_final = int(number_rounded*10)
print(number_final)












# importing package
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

ax.set(xlabel='Trainning rounds', ylabel='Price after a week(€)',title='Learning rate = 0.15')

# create data
x = []
for i in range(50000, 255000, 5000):
  x.append(i)


y1 = [39.77] *41
y2 = [40.24] *41
y3 = [40.43] *41
y4 = [40.66] *41

y_rl = [75.79, 85.37, 69.35, 66.71, 65.07, 52.19, 52.39, 53.91, 51.96, 55.00, 43.67, 47.82, 54.61, 46.86, 44.81, 41.60, 47.46, 47.25, 42.10, 40.61, 42.03, 42.58, 37.23, 39.16, 36.75, 36.38, 38.38, 37.80, 38.37, 37.93, 38.16, 37.67, 37.33, 38.71, 37.75, 37.05, 36.68, 37.20, 36.96, 36.69, 37.78, ]


ax.plot(x, y1, label = "Threshold tbm 55-25", linestyle=":")
ax.plot(x, y2, label = "Threshold tbm 60-25", linestyle=":")
ax.plot(x, y3, label = "Threshold tbm 55-30", linestyle=":")
ax.plot(x, y4, label = "Threshold tbm 60-30", linestyle=":")

ax.plot(x, y_rl, label = "RL", linestyle="-")

plt.legend()
plt.show()

fig.savefig("test.png")



















from random import *
list_rand=[]
for ind in range(2000):
    list_rand.append(randint(15, 600-15))
    list_rand.append(randint(15, 65))


print (list_rand)




