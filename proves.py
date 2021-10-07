import numpy as np



number= 39
number_decimal = float(number)/10
number_rounded = round(number_decimal)
number_final = int(number_rounded*10)
#print(number_final)



from random import *
list_rand=[]
for ind in range(4000):
    list_rand.append(randint(15, 3600-15))
    #list_rand.append(randint(15, 315))


print (list_rand)


















# importing package
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

ax.set(xlabel='Trainning rounds', ylabel='Price after a week(€)',title='Learning rate = 0.15')

# create data
x = []
for i in range(50000, 255000, 5000):
  x.append(i)

"""
y1 = [39.77] *41
y2 = [40.24] *41
y3 = [40.43] *41
y4 = [40.66] *41

y_rl = [75.79, 85.37, 69.35, 66.71, 65.07, 52.19, 52.39, 53.91, 51.96, 55.00, 43.67, 47.82, 54.61, 46.86, 44.81, 41.60, 47.46, 47.25, 42.10, 40.61, 42.03, 42.58, 37.23, 39.16, 36.75, 36.38, 38.38, 37.80, 38.37, 37.93, 38.16, 37.67, 37.33, 38.71, 37.75, 37.05, 36.68, 37.20, 36.96, 36.69, 37.78, ]


y1 = [55.46] *41
y2 = [55.64] *41
y3 = [56.39] *41
y4 = [56.44] *41

y_rl = [80.99, 75.24, 70.35, 70.08, 65.35, 50.48, 47.66, 45.50, 48.20, 40.61, 37.99, 38.36, 34.55, 36.76, 35.98, 34.50, 35.69, 34.61, 34.46, 34.47, 34.45, 34.43, 34.44, 34.42, 34.53, 34.44, 34.41, 34.44, 34.42, 34.47, 34.40, 34.50, 34.40, 34.44, 34.40, 34.39, 34.44, 34.42, 34.39, 34.45, 34.39]
"""

# create data
x = []
for i in range(200000, 1020000, 20000):
  x.append(i)

print(len(x))

y1 = [38.93] *41
y2 = [39.05] *41
y3 = [39.53] *41
y4 = [39.73] *41

y_rl = [90.99, 93.40, 86.68, 90.05, 86.49, 82.67, 74.13, 72.25, 67.08, 63.18, 58.36, 54.47, 51.44, 50.09, 49.24, 43.51, 47.81, 43.05, 40.46, 42.70, 39.43, 38.67, 38.30, 37.69, 37.08, 37.20, 37.16, 37.41, 36.64, 36.27, 37.11, 36.06, 36.11, 35.72, 37.86, 35.74, 35.93, 36.16, 35.64, 36.03, 35.79]



# create data
x = []
for i in range(600000, 1520000, 20000):
  x.append(i)
print(len(x))

y1 = [89.75] *46
y2 = [90.45] *46
y3 = [91.62] *46
y4 = [92.14] *46

y_rl = [110.69, 113.51, 107.56, 106.88, 106.40, 96.60, 100.41, 87.85, 88.91, 97.22, 99.70, 91.46, 89.47, 86.66, 84.89, 84.22, 86.16, 88.66, 85.13, 83.60, 82.68, 94.00, 82.28, 84.17, 82.28, 82.24, 82.46, 81.92, 84.21, 82.07, 84.17, 85.24, 84.35, 82.16, 82.02, 81.92, 82.04, 81.81, 81.86, 81.92, 81.95, 81.91, 81.92, 82.14, 82.33, 81.93]




ax.plot(x, y1, label = "Threshold tbm 60-50", linestyle=":")
ax.plot(x, y2, label = "Threshold tbm 65-50", linestyle=":")
ax.plot(x, y3, label = "Threshold tbm 60-45", linestyle=":")
ax.plot(x, y4, label = "Threshold tbm 65-40", linestyle=":")

ax.plot(x, y_rl, label = "RL", linestyle="-")

plt.legend()
plt.show()

fig.savefig("test.png")



