import numpy as np
import modern_robotics as mr

w = np.array([0, 0, 1])
q = np.array([0, 2, 0])
print(np.cross(-w, q))

M = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 2],
    [0, 0, 1, 1],
    [0, 0, 0, 1]
])

S = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 1, 1, 0],
    [0, 1, 2, 0],
    [0, 0, 2, 0],
    [0, 0, 0, 1]
])

T = np.array([
    0,
    np.pi/2,
    -np.pi/2,
    1
])

print(mr.FKinSpace(M, S, T).astype(np.int32))

