import numpy as np
import modern_robotics as mr
import math

w2 = np.array([-1, 0, 0])
q2 = np.array([0, 0, -3])
print(np.cross(-w2, q2))

w1 = np.array([0, 1, 0])
q1 = np.array([0, 0, -3])
print(np.cross(-w1, q1))


M = np.array([
    [-1, 0, 0, 0],
    [ 0, 0, 1, 3],
    [ 0, 1, 0, 2],
    [ 0, 0, 0, 1]
])

B = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 0],
    [3, 0, 0],
    [0, 3, 0],
    [0, 0, 1]
])

T = np.array([
    math.radians(90),
    math.radians(90),
    1
])

print(mr.FKinBody(M, B, T).astype(np.int32))

print(mr.JacobianBody(B, T).astype(np.int32))