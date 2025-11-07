
import numpy as np

t1 = -np.pi / 4
t2 = np.pi / 2

J = np.array([
    [-np.sin(t1) - np.sin(t1 + t2), -np.sin(t1 + t2)],
    [np.cos(t1) + np.cos(t1 + t2), np.cos(t1 + t2)]
])

print(J)

v1 = J @ np.array([1, 0]).T
v2 = J @ np.array([0, 1]).T
v3 = J @ np.array([1, 1]).T

print(v1)
print(v2)
print(v3)

a1 = np.degrees(np.arctan2(v1[1], v1[0]))
a2 = np.degrees(np.arctan2(v2[1], v2[0]))
a3 = np.degrees(np.arctan2(v3[1], v3[0]))

print(a1)
print(a2)
print(a3)

m1 = np.sqrt(v1[0]**2 + v1[1]**2)
m2 = np.sqrt(v2[0]**2 + v2[1]**2)
m3 = np.sqrt(v3[0]**2 + v3[1]**2)

print()
print(m1)
print(m2)
print(m3)