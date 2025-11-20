import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt

# ARM PARAMETERS --------------------------------------------------------------
# link lengths
L1 = 1.0
L2 = 1.0
m1 = 2.0
m2 = 2.0

# center of mass
p1 = np.array([L1/2, 0, 0])
p2 = np.array([L2/2, 0, 0])

# link transforms
m01 = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

m12 = np.array([[1, 0, 0, 0],
                [0, 1, 0, -L1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

m23 = np.array([[1, 0, 0, 0],
                [0, 1, 0, -L1 - L2],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

m_list = np.array([m01, m12, m23])

# intertial matrices
g1 = np.zeros((6, 6))
g1[0:3, 0:3] = np.array([[0, 0, 0],
                         [0, 4, 0],
                         [0, 0, 4]])
g1[3:6, 3:6] = np.eye(3) * m1

g2 = np.zeros((6, 6))
g2[0:3, 0:3] = np.array([[4, 0, 0],
                         [0, 4, 0],
                         [0, 0, 0]])
g2[3:6, 3:6] = np.eye(3) * m2

g_list = np.array([g1, g2])

# screw axes
s1 = np.array([0, 0, 1, 0, L1, 0])
s2 = np.array([0, 0, 1, 0, L2, 0])
s_list = np.array([s1, s2]).T

# joint states (position, velocity, and accelerations)
# thetalist = np.array([np.pi/4, np.pi/4])
thetalist = np.array([0, 0])
dthetalist = np.array([0.0, 0.0])
ddthetalist = np.array([0.0, 0.0])

# gravity
g = np.array([0, 0, -10])

# no external forces on tip
Ftip = np.array([0, 0, 0, 0, 0, 0])


torques = mr.InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, m_list, g_list, s_list)
print(f"T1: {round(torques[0], 4)}")
print(f"T2: {round(torques[1], 4)}")

M = mr.MassMatrix(thetalist, m_list, g_list, s_list)
