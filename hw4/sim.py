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
    
    def forward_dynamics(self, state: ArmState, torques):
        """
        Computes joint accelerations based on current arm state and applied torques.  Uses dynamics from class slides.

        :param state: current arm state
        :param torques: applied joint torques
        """
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
    
    def trajectory(self, start: ArmState, goal: ArmState, T) -> list[ArmState]:
        """
        Generates 5th order polynomial trajectory from start to start state to goal state in time T.

        :param start: starting arm state
        :param goal: desired ending arm state
        :param T: duration of arm movement from start to goal

        """
        n_steps = int(T / dt)
        
        states = []
        
        pf = np.array(goal.position)
        vf = np.array(goal.velocity)
        af = np.array(goal.acceleration)
        c0 = np.array(start.position)
        c1 = np.array(start.velocity)
        c2 = np.array(start.acceleration) / 2

        A = np.array([
            [T**3, T**4, T**5],
            [3*T**2, 4*T**3, 5*T**4],
            [6*T, 12*T**2, 20*T**3]
        ])
        
        b = np.array([
            pf - c0 - c1*T - c2*T**2,
            vf - c1 - 2*c2*T,
            af - 2*c2
        ])
        
        coeffs = np.linalg.solve(A, b)
        c3 = coeffs[0]
        c4 = coeffs[1]
        c5 = coeffs[2]
        
        for i in range(n_steps):
            t = i * dt    
            # compute position, velocity, acceleration using quintic polynomial
            positions = c0 + c1*t + c2*t**2 + c3*t**3 + c4*t**4 + c5*t**5
            velocities = c1 + 2*c2*t + 3*c3*t**2 + 4*c4*t**3 + 5*c5*t**4
            accelerations = 2*c2 + 6*c3*t + 12*c4*t**2 + 20*c5*t**3
            
            states.append(ArmState(positions, velocities, accelerations))
        
        return states

    def embiggen(self):
        """
        Increases arm weights by 20%.
        """
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
        return ArmState(
            np.array([self.theta1, self.theta2]),
            np.array([self.omega1, self.omega2]),
            np.array([self.alpha1, self.alpha2])
        )
    
    def set_state(self, state: ArmState):
        self.theta1, self.theta2 = state.position
        self.omega1, self.omega2 = state.velocity
        self.alpha1, self.alpha2 = state.acceleration
        
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
start_state = ArmState(
    np.array([-np.pi/4, 0]),
    np.array([0, 0]),
    np.array([0, 0])
)
target_state = ArmState(
    np.array([np.pi/4, np.pi/2]),
    np.array([0, 0]),
    np.array([0, 0])
)
duration = 2.00

states = arm.trajectory(start_state, target_state, duration)
sim.set_state(start_state)

# make link masses 20% bigger
arm.embiggen()

# pid controller gains
factor = 120
Kp = np.array(np.full(2, 8 * factor))
Ki = np.array(np.full(2, 4 * factor))
Kd = np.array(np.full(2, 1 * factor))

integral_error = np.array([0.0, 0.0])

t = 0
for state in states:
    t += dt
    print(f"time: {t:.2f}/{duration:.2f}", end="\r")
    if sim.running:
        # get current state
        current_state = sim.get_state()
        
        # compute error
        position_error = state.position - current_state.position
        velocity_error = state.velocity - current_state.velocity
        integral_error += position_error * sim.dt
        
        # update torques based on pid error
        torques = Kp * position_error + Kd * velocity_error + Ki * integral_error
        
        # update sim with forward dynamics
        accels = arm.forward_dynamics(current_state, torques)
        sim.update_position(accels)
        sim.draw()
    else:
        break
print()
final_state = sim.get_state()

# Error summary
print(f"Target Position:  [{target_state.position[0]:.4f}, {target_state.position[1]:.4f}]")
print(f"Final Position:   [{final_state.position[0]:.4f}, {final_state.position[1]:.4f}]")
position_error = target_state.position - final_state.position
print(f"Position Error:   [{position_error[0]:.4f}, {position_error[1]:.4f}]")
print("-" * 50)
print(f"Target Velocity:  [{target_state.velocity[0]:.4f}, {target_state.velocity[1]:.4f}]")
print(f"Final Velocity:   [{final_state.velocity[0]:.4f}, {final_state.velocity[1]:.4f}]")
velocity_error = target_state.velocity - final_state.velocity
print(f"Velocity Error:   [{velocity_error[0]:.4f}, {velocity_error[1]:.4f}]")