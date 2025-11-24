import modern_robotics as mr
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DT = 0.01

class Arm:
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
    
    def inverse_dynamics(self, thetalist, dthetalist, ddthetalist):
        theta1, theta2 = thetalist
        dtheta1, dtheta2 = dthetalist
        ddtheta1, ddtheta2 = ddthetalist
        
        M = np.matrix([
            [(self.m1 * np.pow(self.r1, 2)) + self.I1 + self.I2 + (self.m2 * (np.pow(self.L, 2) + np.pow(self.r2, 2) + (2 * self.L * self.r2 * np.cos(theta2)))), (self.m2 * self.L * self.r2 * np.cos(theta2)) + (self.m2 * np.pow(self.r2, 2)) + self.I2],
            [(self.m2 * np.pow(self.r2, 2)) + (self.m2 * self.L * self.r2 * np.cos(theta2)) + self.I2, (self.m2 * np.pow(self.r2, 2)) + self.I2]
        ])
        c = np.array([
            [-self.m2 * self.L * self.r2 * dtheta2 * (2 * dtheta1 + dtheta2) * np.sin(theta2)],
            [self.m2 * self.L * self.r2 * np.pow(dtheta1, 2) * np.sin(theta2)]
        ])
        g_vec = np.array([
            [self.g * ((self.m1 * self.r1 * np.sin(theta1)) + (self.m2 * self.L * np.sin(theta1)) + (self.m2 * self.r2 * np.sin(theta1 + theta2)))],
            [self.g * self.m2 * self.r2 * np.sin(theta1 + theta2)]
        ])
        
        ddtheta_vec = np.array([[ddtheta1], [ddtheta2]])
        tau = np.dot(M, ddtheta_vec) + c + g_vec
        
        return tau
    
    def forward_dynamics(self, thetalist, dthetalist, taulist):
        theta1, theta2 = thetalist
        dtheta1, dtheta2 = dthetalist
        tau1, tau2 = taulist
        
        M = np.matrix([
            [(self.m1 * np.pow(self.r1, 2)) + self.I1 + self.I2 + (self.m2 * (np.pow(self.L, 2) + np.pow(self.r2, 2) + (2 * self.L * self.r2 * np.cos(theta2)))), (self.m2 * self.L * self.r2 * np.cos(theta2)) + (self.m2 * np.pow(self.r2, 2)) + self.I2],
            [(self.m2 * np.pow(self.r2, 2)) + (self.m2 * self.L * self.r2 * np.cos(theta2)) + self.I2, (self.m2 * np.pow(self.r2, 2)) + self.I2]
        ])
        c = np.array([
            [-self.m2 * self.L * self.r2 * dtheta2 * (2 * dtheta1 + dtheta2) * np.sin(theta2)],
            [self.m2 * self.L * self.r2 * np.pow(dtheta1, 2) * np.sin(theta2)]
        ])
        g_vec = np.array([
            [self.g * ((self.m1 * self.r1 * np.sin(theta1)) + (self.m2 * self.L * np.sin(theta1)) + (self.m2 * self.r2 * np.sin(theta1 + theta2)))],
            [self.g * self.m2 * self.r2 * np.sin(theta1 + theta2)]
        ])
        
        tau_vec = np.array([[tau1], [tau2]])
        ddtheta = np.linalg.solve(M, tau_vec - c - g_vec)
        
        return ddtheta.flatten()
    
    def plan(self, start, goal, plan_time=2.0):
        # move arm from start joint angles to goal joint angles using 5th order polynomial trajectory and return position, velocity, and acceleration of joint for all timesteps
        N = int(plan_time / DT)  # Number of timesteps
        
        # Initialize arrays for both joints
        positions = np.zeros((N, 2))
        velocities = np.zeros((N, 2))
        accelerations = np.zeros((N, 2))
        
        # Generate trajectory for each joint independently
        for joint_idx in range(2):
            theta_start = start[joint_idx]
            theta_goal = goal[joint_idx]
            
            # 5th order polynomial trajectory
            # Boundary conditions: theta(0) = start, theta(T) = goal
            # dtheta(0) = 0, dtheta(T) = 0
            # ddtheta(0) = 0, ddtheta(T) = 0
            for i in range(N):
                t = i * DT
                s = t / plan_time  # Normalized time [0, 1]
                
                # 5th order polynomial: 10s^3 - 15s^4 + 6s^5
                # This gives s(0)=0, s(1)=1 with zero velocity and acceleration at endpoints
                s_t = 10 * s**3 - 15 * s**4 + 6 * s**5
                ds_t = (30 * s**2 - 60 * s**3 + 30 * s**4) / plan_time
                dds_t = (60 * s - 180 * s**2 + 120 * s**3) / (plan_time**2)
                
                # Interpolate between start and goal
                positions[i, joint_idx] = theta_start + (theta_goal - theta_start) * s_t
                velocities[i, joint_idx] = (theta_goal - theta_start) * ds_t
                accelerations[i, joint_idx] = (theta_goal - theta_start) * dds_t
        
        return positions, velocities, accelerations

class Sim:
    def __init__(self, L1=1.0, L2=1.0, dt=DT):
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
        self.ax.set_title('2-Link Pendulum')
        
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
        return [(self.theta1, self.theta2), (self.omega1, self.omega2), (self.alpha1, self.alpha2)]
    
    def set_state(self, theta1, theta2, omega1=0.0, omega2=0.0):
        self.theta1 = theta1
        self.theta2 = theta2
        self.omega1 = omega1
        self.omega2 = omega2
        
    def draw(self, show_trace=True):
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
        plt.pause(self.dt)
        
    def clear_trace(self):
        self.trace_x = []
        self.trace_y = []

viz = Sim()
arm = Arm()

pos, vel, accels = arm.plan((-np.pi/4, 0), (np.pi/4, np.pi/2), 2.0)
viz.set_state(-np.pi/4, 0)

# PID controller gains
Kp = np.array([800.0, 800.0])  # Lower proportional gain
Kd = np.array([100.0, 100.0])    # Higher derivative gain for better damping
Ki = np.array([50.0, 50.0])    # Moderate integral gain

# Initialize integral error
integral_error = np.array([0.0, 0.0])

t = 0
for i in range(pos.shape[0]):
    t += DT
    print(f"time: {round(t, 1)}/2.0      ", end="\r")
    if viz.running:
        # Get current state
        positions, velocities, _ = viz.get_state()
        current_pos = np.array(positions)
        current_vel = np.array(velocities)
        
        # Desired state from trajectory
        desired_pos = pos[i]
        desired_vel = vel[i]
        
        # Compute errors
        position_error = desired_pos - current_pos
        velocity_error = desired_vel - current_vel
        integral_error += position_error * viz.dt
        
        # PID control law: tau = Kp * e_pos + Kd * e_vel + Ki * integral_error
        controls = Kp * position_error + Kd * velocity_error + Ki * integral_error
        
        # Compute forward dynamics to get accelerations
        dynamics = arm.forward_dynamics(
            current_pos,
            current_vel,
            controls
        )
        
        # Update the visualization using forward dynamics
        viz.update_position(dynamics[0], dynamics[1])
        viz.draw()
    else:
        break
