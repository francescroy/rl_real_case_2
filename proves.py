import numpy as np

## DEFININT POLICY...

a = np.zeros((2, 41, 21, 21))

# la primera dimensio 0..1
# la segona dimensio 0..40
# la tercera dimensio 0..20
# la quarta dimensio 0..20

print(a.shape)

a[0][12][13][0] = +1 # aixo és l'accio per quan estem a l'estat 1 jvb, 12 users, carregat a 65% el 1r, 0 % el segon...

print(a[1][40][20][20])

