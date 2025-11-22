import modern_robotics as mr
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ModernArm:
    def __init__(self):
        # link lengths
        self.L1 = 1.0
        self.L2 = 1.0
        self.m1 = 2.0
        self.m2 = 2.0

        # center of mass
        self.p1 = np.array([self.L1/2, 0, 0])
        self.p2 = np.array([self.L2/2, 0, 0])

        # link transforms
        self.m01 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        self.m12 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, -self.L1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        self.m23 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, -self.L1 - self.L2],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        self.m_list = np.array([self.m01, self.m12, self.m23])

        # intertial matrices
        self.g1 = np.zeros((6, 6))
        self.g1[0:3, 0:3] = np.array([[0, 0, 0],
                                [0, 4, 0],
                                [0, 0, 4]])
        self.g1[3:6, 3:6] = np.eye(3) * self.m1

        self.g2 = np.zeros((6, 6))
        self.g2[0:3, 0:3] = np.array([[4, 0, 0],
                                [0, 4, 0],
                                [0, 0, 0]])
        self.g2[3:6, 3:6] = np.eye(3) * self.m2

        self.g_list = np.array([self.g1, self.g2])

        # screw axes
        self.s1 = np.array([0, 0, 1, 0, self.L1, 0])
        self.s2 = np.array([0, 0, 1, 0, self.L2, 0])
        self.s_list = np.array([self.s1, self.s2]).T

        # joint states (position, velocity, and accelerations)
        self.thetalist = np.array([0, 0])
        self.dthetalist = np.array([0.0, 0.0])
        self.ddthetalist = np.array([0.0, 0.0])

        # gravity
        self.g = np.array([0, 0, -10])

        # no external forces on tip
        self.Ftip = np.array([0, 0, 0, 0, 0, 0])

    def inverse_dynamics(self, thetalist, dthetalist, ddthetalist):
        torques = mr.InverseDynamics(thetalist, dthetalist, ddthetalist, self.g, self.Ftip, self.m_list, self.g_list, self.s_list)
        print(f"T1: {round(torques[0], 4)}")
        print(f"T2: {round(torques[1], 4)}")
        # mr.ForwardDynamics()

class ClassyArm:
    def __init__(self, Li = 1, ri = 0.5, m1 = 3, m2 = 2, I1 = 2, I2 = 1, g=9.81):
        # arm properties
        self.L = Li
        self.r1 = ri
        self.r2 = ri        
        self.m1 = m1
        self.m2 = m2
        self.I1 = I1
        self.I2 = I2

        # world properties
        self.g = g

    def inverse_dynamics(self, thetalist: np.ndarray, dthetalist, ddthetalist):
        theta1, theta2 = thetalist.astype(tuple)
        dtheta1, dtheta2 = dthetalist.astype(tuple)
        M = np.matrix([
            [(self.m1 * np.pow(self.r1, 2)) + self.I1 + self.I2 + (self.m2 * (np.pow(self.L, 2) + np.pow(self.r2, 2) + (2 * self.L * self.r2 * np.cos(theta2)))), (self.m2 * self.L * self.r2 * np.cos(theta2)) + (self.m2 * np.pow(self.r2, 2)) + (self.I2 * self.m2 * np.pow(self.r2, 2)) + self.I2],
            [(self.m2 * np.pow(self.r2, 2)) + (self.m2 * self.L * self.r2 * np.cos(theta2)) + self.I2, (self.m2 * np.pow(self.r2, 2))]
        ])
        c = np.array([
            [-self.m2 * self.L * self.r2 * dtheta2 * (2 * dtheta2 + dtheta2) * np.sin(dtheta2)],
            [-self.m2 * self.L * self.r2 * dtheta1 * dtheta2 * np.sin(theta2)]
        ])
        g = np.array([
            [(self.m1 * self.r1 * np.sin(theta1)) + (self.m2 * self.L * np.sin(theta1)) + (self.m2 * self.r2 * np.sin(theta1 + theta2))],
            [self.m2 * self.r2 * np.sin(theta1 + theta2)]
        ])

        # print(np.dot(M, ddthetalist).T + c + g)
        return np.dot(M, ddthetalist).T + c + g

class PendulumVisualizer:
    def __init__(self, L1=1.0, L2=1.0, dt=0.01):
        """
        Initialize the 2-link pendulum visualizer.
        
        Parameters:
        -----------
        L1 : float
            Length of first link
        L2 : float
            Length of second link
        dt : float
            Time step for integration
        """
        self.L1 = L1
        self.L2 = L2
        self.dt = dt
        
        # State variables (angles and angular velocities)
        self.theta1 = 0.0  # First joint angle
        self.theta2 = 0.0  # Second joint angle
        self.omega1 = 0.0  # First joint angular velocity
        self.omega2 = 0.0  # Second joint angular velocity
        self.alpha1 = 0.0  # First joint angular acceleration
        self.alpha2 = 0.0  # Second joint angular acceleration
        
        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-2.5, 2.5)
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('2-Link Pendulum (Press Q to quit)')
        
        # Create line objects for the pendulum
        self.line, = self.ax.plot([], [], 'o-', lw=3, markersize=10)
        self.trace, = self.ax.plot([], [], 'r-', lw=1, alpha=0.3)
        
        # Trace history
        self.trace_x = []
        self.trace_y = []
        
        # Flag for quitting
        self.running = True
        
        # Connect key press event
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
    def update_position(self, alpha1, alpha2):
        """
        Update the pendulum position based on angular accelerations.
        
        Parameters:
        -----------
        alpha1 : float
            Angular acceleration of first joint (rad/s^2)
        alpha2 : float
            Angular acceleration of second joint (rad/s^2)
        """
        # Store accelerations
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        # Update angular velocities using Euler integration
        self.omega1 += alpha1 * self.dt
        self.omega2 += alpha2 * self.dt
        
        # Update angles
        self.theta1 += self.omega1 * self.dt
        self.theta2 += self.omega2 * self.dt
    
    def _on_key_press(self, event):
        """Handle key press events."""
        if event.key == 'q':
            self.running = False
            plt.close(self.fig)
        
    def get_positions(self):
        """
        Calculate the Cartesian positions of the joints.
        
        Returns:
        --------
        tuple
            (x_positions, y_positions) for [base, joint1, joint2]
        """
        # Base is at origin
        x0, y0 = 0, 0
        
        # First joint position
        x1 = self.L1 * np.sin(self.theta1)
        y1 = -self.L1 * np.cos(self.theta1)
        
        # Second joint (end effector) position
        x2 = x1 + self.L2 * np.sin(self.theta1 + self.theta2)
        y2 = y1 - self.L2 * np.cos(self.theta1 + self.theta2)
        
        return [x0, x1, x2], [y0, y1, y2]
    
    def get_state(self):
        """
        Get the current state of the pendulum.
        
        Returns:
        --------
        dict
            Dictionary containing:
            - 'rotations': (theta1, theta2) - joint angles in radians
            - 'angular_velocities': (omega1, omega2) - angular velocities in rad/s
            - 'angular_accelerations': (alpha1, alpha2) - angular accelerations in rad/s^2
        """
        return [(self.theta1, self.theta2), (self.omega1, self.omega2), (self.alpha1, self.alpha2)]
    
    def set_state(self, theta1, theta2, omega1=0.0, omega2=0.0):
        """
        Set the state of the pendulum directly.
        
        Parameters:
        -----------
        theta1 : float
            First joint angle (rad)
        theta2 : float
            Second joint angle (rad)
        omega1 : float
            First joint angular velocity (rad/s)
        omega2 : float
            Second joint angular velocity (rad/s)
        """
        self.theta1 = theta1
        self.theta2 = theta2
        self.omega1 = omega1
        self.omega2 = omega2
        
    def draw(self, show_trace=True):
        """
        Draw the current state of the pendulum.
        
        Parameters:
        -----------
        show_trace : bool
            Whether to show the trace of the end effector
        """
        x, y = self.get_positions()
        self.line.set_data(x, y)
        
        if show_trace:
            self.trace_x.append(x[-1])
            self.trace_y.append(y[-1])
            # Keep only last 500 points
            if len(self.trace_x) > 500:
                self.trace_x.pop(0)
                self.trace_y.pop(0)
            self.trace.set_data(self.trace_x, self.trace_y)
        
        plt.draw()
        plt.pause(0.001)
        
    def clear_trace(self):
        """Clear the end effector trace."""
        self.trace_x = []
        self.trace_y = []

viz = PendulumVisualizer()
arm = ClassyArm()
viz.set_state(np.pi/4, 0)

while viz.running:
    accel = arm.inverse_dynamics(
        np.array(viz.get_state()[0]),
        np.array(viz.get_state()[1]),
        np.array(viz.get_state()[2])
    )
    print(viz.get_state())
    viz.update_position(accel[0, 0], accel[1, 0])
    viz.draw()
    time.sleep(0.1)
print("done")