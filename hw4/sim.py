# Collaborated with Clement Cantil
# Generative AI was used to aid in writing matplotlib visualizations

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
    def __init__(self, L1=1.0, L2=1.0, dt=dt, viz=True):
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
        
        if viz:
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

def sim():
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
    arm.embiggen_links()

    # pid controller gains
    factor = 120
    Kp = np.array(np.full(2, 8 * factor))
    Ki = np.array(np.full(2, 4 * factor))
    Kd = np.array(np.full(2, 0.5 * factor))
    print(f"PID Gains - Kp: {Kp}, Ki: {Ki}, Kd: {Kd}")

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

def plots():
    sim = Sim(viz=False)
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
    arm.embiggen_links()

    # pid controller gains
    factor = 120
    Kp = np.array(np.full(2, 8 * factor))
    Ki = np.array(np.full(2, 4 * factor))
    Kd = np.array(np.full(2, 0.5 * factor))
    print(f"PID Gains - Kp: {Kp}, Ki: {Ki}, Kd: {Kd}")

    integral_error = np.array([0.0, 0.0])

    # store recorded data
    time_history = []
    expected_theta1 = []
    expected_theta2 = []
    actual_theta1 = []
    actual_theta2 = []
    expected_ee_x = []
    expected_ee_y = []
    actual_ee_x = []
    actual_ee_y = []
    torque1_history = []
    torque2_history = []

    # copied from sim get_arm_positions
    def get_arm_positions(state, L1=1.0, L2=1.0):
        theta1, theta2 = state.position
        x0, y0 = 0, 0
        x1 = L1 * np.sin(theta1)
        y1 = -L1 * np.cos(theta1)
        x2 = x1 + L2 * np.sin(theta1 + theta2)
        y2 = y1 - L2 * np.cos(theta1 + theta2)
        return [x0, x1, x2], [y0, y1, y2]

    start_state = sim.get_state()
    t = 0
    for state in states:
        # get current state
        current_state = sim.get_state()
        
        # record expected and actual positions
        time_history.append(t)
        expected_theta1.append(state.position[0])
        expected_theta2.append(state.position[1])
        actual_theta1.append(current_state.position[0])
        actual_theta2.append(current_state.position[1])
        
        # record expected and actual end effector positions
        exp_x, exp_y = get_arm_positions(state)
        act_x, act_y = get_arm_positions(current_state)
        expected_ee_x.append(exp_x[-1])
        expected_ee_y.append(exp_y[-1])
        actual_ee_x.append(act_x[-1])
        actual_ee_y.append(act_y[-1])
        
        # compute error
        position_error = state.position - current_state.position
        velocity_error = state.velocity - current_state.velocity
        integral_error += position_error * sim.dt
        
        # update torques based on pid error
        torques = Kp * position_error + Kd * velocity_error + Ki * integral_error
        torque1_history.append(torques[0])
        torque2_history.append(torques[1])
        
        # update sim with forward dynamics
        accels = arm.forward_dynamics(current_state, torques)
        sim.update_position(accels)
        t += dt

    # region: plotting

    # get final state
    final_state = sim.get_state()
    
    # create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ax1, ax2, ax5 = axes[0]
    ax3, ax4, ax6 = axes[1]
    
    # plot start state
    x_start, y_start = get_arm_positions(start_state)
    ax1.plot(x_start, y_start, 'o-', lw=3, markersize=10, color='blue')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Start')
    
    # plot final state
    x_final, y_final = get_arm_positions(final_state)
    ax2.plot(x_final, y_final, 'o-', lw=3, markersize=10, color='blue')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('End')
    
    # plot expected vs actual for both joints on one plot
    ax3.plot(time_history, expected_theta1, 'g--', label='Joint 1 Expected')
    ax3.plot(time_history, actual_theta1, 'g-', label='Joint 1 Actual')
    ax3.plot(time_history, expected_theta2, 'r--', label='Joint 2 Expected')
    ax3.plot(time_history, actual_theta2, 'r-', label='Joint 2 Actual')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angle (rad)')
    ax3.set_title('Expected vs Actual Joint Angles')
    ax3.legend()
    ax3.grid(True)
    
    # plot expected vs actual end effector position
    ax4.plot(expected_ee_x, expected_ee_y, 'g--', label='Expected', linewidth=2)
    ax4.plot(actual_ee_x, actual_ee_y, 'r-', label='Actual', linewidth=2)
    ax4.plot(expected_ee_x[0], expected_ee_y[0], 'go', markersize=10, label='Start')
    ax4.plot(expected_ee_x[-1], expected_ee_y[-1], 'g^', markersize=10, label='Expected End')
    ax4.plot(actual_ee_x[-1], actual_ee_y[-1], 'r^', markersize=10, label='Actual End')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('End Effector Position')
    ax4.legend()
    ax4.grid(True)
    
    # hide unused subplot
    ax5.axis('off')
    
    # plot computed torques over time
    ax6.plot(time_history, torque1_history, 'g-', label='Joint 1')
    ax6.plot(time_history, torque2_history, 'r-', label='Joint 2')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Torque (Nm)')
    ax6.set_title('Computed Torques')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.show()

    # endregion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2-Link Arm Simulation')
    parser.add_argument('-s', '--sim', action='store_true', help='Run simulation')
    parser.add_argument('-c', '--csv', action='store_true', help='Generate trajectory and save to a csv')
    parser.add_argument('-p', '--plots', action='store_true', help='Show plots')
    args = parser.parse_args()
    
    if args.sim:
        sim()
    elif args.trajectory:
        print("Running trajectory mode")
    elif args.plots:
        plots()
    else:
        parser.print_help()