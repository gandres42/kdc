import numpy as np
import modern_robotics as mr
import math

w = np.array([-1, 0, 0])
q = np.array([0, 0, -3])


M = np.array([
    [-1, 0, 0, 0],
    [ 0, 0, 1, 3],
    [ 0, 1, 0, 2],
    [ 0, 0, 0, 1]
])

S = np.array([
    [0, 1, 0],
    [0, 0, 0],
    [1, 0, 0],
    [0, 0, 0],
    [0, 2, 1],
    [0, 0, 0]
])

T = np.array([
    math.radians(90),
    math.radians(90),
    1
])

print(np.cross(-w, q))

# print(mr.FKinSpace(M, S, T).astype(np.int32))

# print(mr.JacobianSpace(S, T).astype(np.int32))
