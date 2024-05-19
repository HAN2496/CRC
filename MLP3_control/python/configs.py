import numpy as np

g = 9.81
m = 1 # total mass of person
l = 1 # total height of the person
lc = 1 # start of the link
m1 = m * 22.36 / 100
m2 = m * 8.78/100
m3 = m * 3.66 / 100
l1 = l * 26 /100
l2 = l * 24 / 100
l3 = l * 12 / 100
lc1 = lc * 43.3 / 100 # location of the c.o.m for shank
lc2 = lc * 43.3 / 100 # location of the c.o.m for thigh
lc3 = lc * 42.9 / 100 # location of the c.o.m for foot
Izzi = lambda mi, li, lci: mi * (li * lci)**2  # Inertia calculation, needs to be adjusted based on actual model
izz1 = m1 * lc1 * lc1
izz2 = m2 * lc2 * lc2
izz3 = m3 * lc3 * lc3
