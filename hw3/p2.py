import modern_robotics as mr
import numpy as np
import math

h = 0.20
r = 0.02
m = math.pi * math.pow(r, 2) * h * 7500
print(m)

# # I = m * (((3 * math.pow(r, 2)) + math.pow(h, 2)) / 12)
# # I = (m * math.pow(r, 2)) / 2
# # print(I)

# # I = np.array([
# #     [0.1256, 0, 0],
# #     [0, 0.1256, 0],
# #     [0., 0, .000376]
# # ])

# r = 0.1
# m = 7500 * ((4/3) * math.pi * math.pow(r, 3))
# dI = m * math.pow(0.2, 2)
# print(dI)

# # print(I + dI + dI)
# # # print(m * math.pow(0.1, 2))

