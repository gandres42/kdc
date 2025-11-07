import modern_robotics as mr
import numpy as np
import math

# h = 0.20
# r = 0.02
# m = math.pi * math.pow(r, 2) * h * 7500

# I = m * (((3 * math.pow(r, 2)) + math.pow(h, 2)) / 12)
# I = (m * math.pow(r, 2)) / 2
# print(I)

r = 0.1
m = 7500 * ((4/3) * math.pi * math.pow(r, 3))
I = m * (math.pow(r, 2) + math.pow(r, 2))/5

print(m * math.pow(0.1, 2))