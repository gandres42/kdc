import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple

dt = 0.01

class ArmState(NamedTuple):
    position: np.ndarray[float, float]
    velocity: np.ndarray[float, float]
    acceleration: np.ndarray[float, float]

class Arm:
    def __init__(self, Li = 1, ri = 0.5, m1 = 3, m2 = 2, I1 = 2, I2 = 1, g=9.81):
        self.L = Li
        self.r1 = ri
        self.r2 = ri        
        self.m1 = m1
        self.m2 = m2
        self.I1 = I1
        self.I2 = I2
        self.g = g
    
    def inverse_dynamics(self, state: ArmState):
        theta1, theta2 = state.position
        dtheta1, dtheta2 = state.velocity
        ddtheta1, ddtheta2 = state.acceleration
        
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
    
    def forward_dynamics(self, state: ArmState, torques):
        theta1, theta2 = state.position
        dtheta1, dtheta2 = state.velocity
        tau1, tau2 = torques
        
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
    
    def plan(self, start, goal, plan_time=2.0) -> list[ArmState]:
        # number of timestemps
        n_steps = int(plan_time / dt)
        
        # Initialize list of states
        states = []
        
        # Generate trajectory for each timestep
        for i in range(n_steps):
            t = i * dt
            s = t / plan_time  # Normalized time [0, 1]
            
            # 5th order polynomial: 10s^3 - 15s^4 + 6s^5
            # This gives s(0)=0, s(1)=1 with zero velocity and acceleration at endpoints
            s_t = 10 * s**3 - 15 * s**4 + 6 * s**5
            ds_t = (30 * s**2 - 60 * s**3 + 30 * s**4) / plan_time
            dds_t = (60 * s - 180 * s**2 + 120 * s**3) / (plan_time**2)
            
            # Compute position, velocity, acceleration for both joints
            start_arr = np.array(start)
            goal_arr = np.array(goal)
            positions = start_arr + (goal_arr - start_arr) * s_t
            velocities = (goal_arr - start_arr) * ds_t
            accelerations = (goal_arr - start_arr) * dds_t
            
            states.append(ArmState(positions, velocities, accelerations))
        
        return states

    def embiggen(self):
        self.m1 = self.m1 * 1.2
        self.m2 = self.m2 * 1.2

class Sim:
    def __init__(self, L1=1.0, L2=1.0, dt=dt):
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
        
    def update_position(self, accels):
        # Store accelerations
        self.alpha1 = accels[0]
        self.alpha2 = accels[1]
        
        # Update angular velocities using Euler integration
        self.omega1 += accels[0] * self.dt
        self.omega2 += accels[1] * self.dt
        
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
        return ArmState(np.array([self.theta1, self.theta2]), np.array([self.omega1, self.omega2]), np.array([self.alpha1, self.alpha2]))
    
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

sim = Sim()
arm = Arm()

# generate motion plan
states = arm.plan((-np.pi/4, 0), (np.pi/4, np.pi/2), 2.0)
sim.set_state(-np.pi/4, 0)

# make link masses 20% bigger
arm.embiggen()

# pid controller gains
Kp = np.array([800.0, 800.0])
Kd = np.array([100.0, 100.0])
Ki = np.array([50.0, 50.0])
integral_error = np.array([0.0, 0.0])

t = 0
for target_state in states:
    t += dt
    print(f"time: {round(t, 1)}/2.0      ", end="\r")
    if sim.running:
        # get current state
        current_state = sim.get_state()
        
        # compute error
        position_error = target_state.position - current_state.position
        velocity_error = target_state.velocity - current_state.velocity
        integral_error += position_error * sim.dt
        
        # update torques based on pid error
        controls = Kp * position_error + Kd * velocity_error + Ki * integral_error
        
        # update sim with forward dynamics
        accels = arm.forward_dynamics(current_state, controls)
        sim.update_position(accels)
        sim.draw()
    else:
        break
print()