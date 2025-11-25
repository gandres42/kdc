# Collaborated with Clement Cantil
# Generative AI was used to aid in writing some matplotlib visualizations
# Run with flag -s to enable sim visualization
# Code downloadable at https://drive.google.com/file/d/1IrPz05IUdQgwLePOrYkmWCoNDXE7sIMc/view?usp=drive_link

import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple
import argparse

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

    def embiggen_links(self):
        """
        Increases arm weights by 20%.
        """
        self.m1 = self.m1 * 1.2
        self.m2 = self.m2 * 1.2

class Sim:
    def __init__(self, L1=1.0, L2=1.0):
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

        # Connect key press event
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            
        # Trace history
        self.trace_x = []
        self.trace_y = []
        
        # Flag for quitting
        self.running = True

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

    def angles_to_positions(self, t1, t2):
        # Base is at origin
        x0, y0 = 0, 0
        
        # First joint position
        x1 = self.L1 * np.sin(t1)
        y1 = -self.L1 * np.cos(t1)
        
        # Second joint (end effector) position
        x2 = x1 + self.L2 * np.sin(t1 + t2)
        y2 = y1 - self.L2 * np.cos(t1 + t2)
        
        return (x1, y1), (x2, y2)
    
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

def main(viz=False):
    # region: setup

    # create arm and sim
    sim = Sim()
    arm = Arm()

    # set starting configuration and goal configuration
    start_state = ArmState(
        np.array([-np.pi/4, 0]),
        np.array([0, 0]),
        np.array([0, 0])
    )
    sim.set_state(start_state)
    target_state = ArmState(
        np.array([np.pi/4, np.pi/2]),
        np.array([0, 0]),
        np.array([0, 0])
    )

    # generate motion plan
    duration = 2.00
    planned_states = arm.trajectory(start_state, target_state, duration)
    
    # make link masses 20% bigger
    arm.embiggen_links()

    # pid controller gains
    gain = 2500
    Kp = np.array(np.full(2, 1.0 * gain))
    Ki = np.array(np.full(2, 0.1 * gain))
    Kd = np.array(np.full(2, 0.025 * gain))
    print(f"pid gains - Kp: {Kp}, Ki: {Ki}, Kd: {Kd}")
    
    # variable storage for plotting
    actual_states = []
    torques = []
    ee_positions = []

    # endregion
    
    # region: simulation
    t = 0
    for state in planned_states:
        t += dt
        print(f"time: {t:.2f}/{duration:.2f}", end="\r")
        if sim.running:
            # get current state
            current_state = sim.get_state()
            
            # compute error
            error = state.position - current_state.position
            derivative = state.velocity - current_state.velocity
            integral = error * sim.dt
            
            # update torques based on pid error
            applied_torques = Kp * error + Kd * derivative + Ki * integral
            
            # update sim with forward dynamics
            accels = arm.forward_dynamics(current_state, applied_torques)
            sim.update_position(accels)

            # optional visualization
            if viz:
                sim.draw()
                print(f"time: {t:.2f}/{duration:.2f}", end="\r")

            # save variables for plotting
            actual_states.append(sim.get_state())
            torques.append(applied_torques)
            ee_positions.append(sim.get_positions())
        else:
            break
    print()

    # endregion

    # region: plotting
    plt.close('all')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Joint positions over time
    time = np.arange(len(actual_states)) * dt
    theta1_actual = [s.position[0] for s in actual_states]
    theta2_actual = [s.position[1] for s in actual_states]
    theta1_planned = [s.position[0] for s in planned_states[:len(actual_states)]]
    theta2_planned = [s.position[1] for s in planned_states[:len(actual_states)]]

    axes[0].plot(time, theta1_actual, label='θ1 actual')
    axes[0].plot(time, theta2_actual, label='θ2 actual')
    axes[0].plot(time, theta1_planned, '--', label='θ1 planned')
    axes[0].plot(time, theta2_planned, '--', label='θ2 planned')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Position (rad)')
    axes[0].set_title('Joint Positions')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Applied torques over time
    tau1 = [t[0] for t in torques]
    tau2 = [t[1] for t in torques]
    axes[1].plot(time, tau1, label='τ1')
    axes[1].plot(time, tau2, label='τ2')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Torque (Nm)')
    axes[1].set_title('Applied Torques')
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: End effector trajectory
    actual_ee_2_x = [pos[0][2] for pos in ee_positions]
    actual_ee_2_y = [pos[1][2] for pos in ee_positions]
    expected_ee_2_x = [sim.angles_to_positions(pos.position[0], pos.position[1])[1][0] for pos in planned_states[:len(actual_states)]]
    expected_ee_2_y = [sim.angles_to_positions(pos.position[0], pos.position[1])[1][1] for pos in planned_states[:len(actual_states)]]
    axes[2].plot(actual_ee_2_x, actual_ee_2_y)
    axes[2].plot(expected_ee_2_x, expected_ee_2_y, '--')
    axes[2].set_xlabel('X (m)')
    axes[2].set_ylabel('Y (m)')
    axes[2].set_title('End Effector Trajectory')
    axes[2].set_xlim(-2, 2)
    axes[2].set_ylim(-2, 2)
    axes[2].set_aspect('equal')
    axes[2].grid(True)
    axes[2].legend(['Actual joint 2 position', 'Planned joint 2 position'])
    axes[2].plot(actual_ee_2_x[-1], actual_ee_2_y[-1], 'o', color='C0', markersize=8)
    axes[2].plot(expected_ee_2_x[-1], expected_ee_2_y[-1], 'o', color='C1', markersize=8)

    # Plot starting and ending arm configurations
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))

    # Starting configuration
    start_j1, start_j2 = sim.angles_to_positions(start_state.position[0], start_state.position[1])
    axes2[0].plot([0, start_j1[0], start_j2[0]], [0, start_j1[1], start_j2[1]], 'o-', lw=3, markersize=10)
    axes2[0].set_xlim(-2.5, 2.5)
    axes2[0].set_ylim(-2.5, 2.5)
    axes2[0].set_aspect('equal')
    axes2[0].grid(True)
    axes2[0].set_xlabel('X (m)')
    axes2[0].set_ylabel('Y (m)')
    axes2[0].set_title('Starting Configuration')

    # Ending configuration
    end_j1, end_j2 = sim.angles_to_positions(actual_states[-1].position[0], actual_states[-1].position[1])
    axes2[1].plot([0, end_j1[0], end_j2[0]], [0, end_j1[1], end_j2[1]], 'o-', lw=3, markersize=10)
    axes2[1].set_xlim(-2.5, 2.5)
    axes2[1].set_ylim(-2.5, 2.5)
    axes2[1].set_aspect('equal')
    axes2[1].grid(True)
    axes2[1].set_xlabel('X (m)')
    axes2[1].set_ylabel('Y (m)')
    axes2[1].set_title('Ending Configuration')

    plt.tight_layout()
    plt.show()

    # endregion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2-Link Arm Simulation')
    parser.add_argument('-s', '--sim', action='store_true', help='show visualization')
    args = parser.parse_args()
    viz = False
    if args.sim:
        viz = True
    main(viz)